import asyncio
import concurrent.futures
import contextvars
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import request_control


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class MemoryTests(unittest.TestCase):
    def test_emotions_bypass_llm_in_both_languages(self):
        commands = {'посмейся': 'amused', 'вздохни': 'sigh', 'пошепчи': 'whisper',
                    'порадуйся': 'excited', 'погрусти': 'sad', 'говори обычно': 'normal'}
        from voice_style import delivery, speech_language
        for language in ('Russian', 'Japanese'):
            self.llm.set_reply_language('a', language)
            for command, style in commands.items():
                with self.subTest(language=language, command=command):
                    answer = list(self.llm.stream_sentences('a', 'Ари, ' + command + ', пожалуйста!'))
                    self.assertEqual(len(answer), 1)
                    spoken, actual_style = delivery(answer[0])
                    self.assertEqual(actual_style, style)
                    self.assertEqual(speech_language(spoken), language)
                    self.assertNotIn('もう一度', spoken)
        self.client.chat.assert_not_called()

    def test_emotion_requests_with_text_and_negation(self):
        from voice_style import emotion_request
        for text in ('Ари, не смейся', 'Мне грустно', 'Почему ты вздохнула?',
                     'Ари, расскажи про смех', 'Перестань говорить шепотом'):
            self.assertIsNone(emotion_request(text))
        self.assertEqual(emotion_request('Ари, скажи шепотом привет'), ('whisper', 'привет'))
        self.llm.set_reply_language('a', 'Japanese')
        self.client.chat.return_value = iter([{'message': {'content': 'こんにちは。'}, 'done_reason': 'stop'}])
        answer = list(self.llm.stream_sentences('a', 'Ари, скажи шепотом привет'))
        self.assertEqual(answer, ['[whisper] こんにちは。'])
        self.assertEqual(self.client.chat.call_args.kwargs['tools'], [])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = types.SimpleNamespace(MEMORY_FILE=str(Path(self.tmp.name)/'memory.json'),
            SHORT_HISTORY_LEN=2, SUMMARIZE_AFTER=4, CHARACTER_FILE='missing-persona',
            OLLAMA_HOST='', OLLAMA_TIMEOUT_SEC=1, OLLAMA_MODEL='test',
            OLLAMA_NUM_THREAD=1, OLLAMA_NUM_CTX=4096, LLM_MAX_TOKENS=20,
            OLLAMA_KEEP_ALIVE='1m', LLM_THINK=False)
        self.client = Mock()
        self.modules = patch.dict(sys.modules, config=self.cfg,
            ollama=types.SimpleNamespace(Client=lambda **kw: self.client),
            discord_tools=types.SimpleNamespace(TEXT_TOOLS=[], VOICE_TOOLS=[], MAX_TOOL_ROUNDS=3,
                list_voice_channels=lambda: [], list_soundboard_sounds=lambda: []))
        self.modules.start()
        self.memory = module('test_memory', 'memory.py')
        sys.modules['memory'] = self.memory
        self.llm = module('test_llm', 'llm.py')
        for i in range(4):
            self.memory.append('a', 'user', str(i))

    def tearDown(self):
        sys.modules.pop('memory', None)
        self.modules.stop()
        self.tmp.cleanup()

    def test_failed_compression_preserves_history(self):
        before = self.memory.entry('a')
        self.client.chat.side_effect = RuntimeError('offline')
        self.llm.compress_history('a')
        self.assertEqual(self.memory.entry('a'), before)

    def test_empty_compression_preserves_history(self):
        before = self.memory.entry('a')
        self.client.chat.return_value = {'message': {'content': '  '}}
        self.llm.compress_history('a')
        self.assertEqual(self.memory.entry('a'), before)

    def test_compression_preserves_new_tail(self):
        old = self.memory.old_messages('a')
        self.memory.append('a', 'user', 'new')
        self.assertTrue(self.memory.commit_compression('a', old, 'summary'))
        self.assertEqual([m['content'] for m in self.memory.entry('a')['recent']], ['2','3','new'])
        self.assertFalse(self.memory.commit_compression('a', old, 'duplicate'))

    def test_atomic_replace_failure_leaves_previous_file(self):
        self.memory.save()
        before = Path(self.cfg.MEMORY_FILE).read_bytes()
        self.memory.append('a','user','new')
        with patch.object(self.memory.os,'replace',side_effect=OSError('locked')):
            with self.assertRaises(OSError): self.memory.save()
        self.assertEqual(Path(self.cfg.MEMORY_FILE).read_bytes(), before)
        self.assertEqual(list(Path(self.tmp.name).glob('*.tmp')), [])

    def test_concurrent_writes_and_snapshot(self):
        def write(i):
            self.memory.append(str(i),'user',str(i))
            self.memory.save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write,range(50)))
        data=json.loads(Path(self.cfg.MEMORY_FILE).read_text(encoding='utf-8'))
        self.assertEqual(len(data),51)
        snapshot=self.memory.entry('a'); snapshot['recent'].clear()
        self.assertEqual(len(self.memory.entry('a')['recent']),4)

    def test_other_user_summary_not_in_prompt(self):
        self.memory.commit_compression('a',self.memory.old_messages('a'),'SECRET_OTHER_USER')
        self.assertNotIn('SECRET_OTHER_USER',self.llm._system_prompt('b',False))

    def test_russian_requests_switch_response_language_per_user(self):
        self.llm.update_reply_language('a', 'Ари, говори по-японски')
        self.assertIn('отвечай только по-японски', self.llm._system_prompt('a', True))
        self.assertIn('Отвечай по-русски', self.llm._system_prompt('b', True))
        self.llm.update_reply_language('a', 'Как у тебя дела?')
        self.assertEqual(self.llm._reply_languages['a'], 'Japanese')
        self.llm.update_reply_language('a', 'Ари, говори по-русски')
        self.assertEqual(self.llm._reply_languages['a'], 'Russian')

    def test_japanese_mode_filters_answer_and_saved_history(self):
        self.llm.update_reply_language('a', 'Ари, говори по-японски')
        mixed = 'こんにちは！ Это значит привет. Как дела?'
        self.assertEqual(self.llm.prepare_voice_answer('a', mixed), 'こんにちは！')
        self.assertEqual(self.llm.prepare_voice_answer('b', 'Привет!'), 'Привет!')
        self.assertEqual(self.llm.prepare_voice_answer('a', ''), '')
        self.assertEqual(self.llm.prepare_voice_answer('a', 'Привет!'),
                         'すみません。もう一度お願いします。')
        self.client.chat.return_value = iter([{'message': {'content': mixed}}])
        with patch.object(self.llm, 'compress_history'):
            list(self.llm.stream_sentences('a', 'привет'))
        self.assertEqual(self.memory.entry('a')['recent'][-1]['content'], 'こんにちは！')

    def test_cancelled_stream_cannot_commit_or_yield(self):
        request=request_control.Request()
        def chunks():
            request.cancel()
            yield {'message': {'content':'This is a late reply.'}}
        self.client.chat.return_value=chunks()
        token=request_control.current.set(request)
        try:
            with self.assertRaises(request_control.RequestCancelled):
                list(self.llm.stream_sentences('a','question'))
        finally: request_control.current.reset(token)
        self.assertFalse(any(m['role']=='assistant' for m in self.memory.entry('a')['recent']))

    def test_token_limit_retries_before_speech_and_history(self):
        self.client.chat.side_effect = [
            iter([{'message': {'content': 'История оборвалась на сло'}},
                  {'done': True, 'done_reason': 'length'}]),
            iter([{'message': {'content': 'Я заблудилась, но нашла дорогу домой.'}},
                  {'done': True, 'done_reason': 'stop'}])]
        with patch.object(self.llm, 'compress_history'):
            answer = ' '.join(self.llm.stream_sentences('a', 'расскажи историю'))
        self.assertEqual(answer, 'Я заблудилась, но нашла дорогу домой.')
        self.assertEqual(self.memory.entry('a')['recent'][-1]['content'], answer)
        self.assertEqual(self.client.chat.call_count, 2)
        self.assertGreater(self.client.chat.call_args.kwargs['options']['num_predict'],
                           self.cfg.LLM_MAX_TOKENS)

    def test_japanese_translation_uses_only_request_without_history_or_tools(self):
        self.client.chat.return_value = iter([
            {'message': {'content': 'いい天気ですね。'}, 'done_reason': 'stop'}])
        with patch.object(self.llm, 'compress_history'):
            answer = ' '.join(self.llm.stream_sentences(
                'a', 'Ари, как будет на японском прекрасная погода?'))
        self.assertEqual(answer, 'いい天気ですね。')
        call = self.client.chat.call_args.kwargs
        self.assertEqual(call['tools'], [])
        self.assertEqual(len(call['messages']), 2)
        self.assertIn('ТОЛЬКО японский перевод', call['messages'][0]['content'])
        self.assertFalse(self.llm._is_japanese_translation('Ари, говори по-японски'))
        self.assertFalse(self.llm._is_japanese_translation('Расскажи про японскую культуру'))
        self.assertEqual(call['messages'][1]['content'], 'прекрасная погода?')

    def test_translation_extracts_phrase_from_recognized_voice_variants(self):
        cases = {
            'Ари, как бы на японском «привет»': 'привет',
            'Ари, как будто на японском, привет!': 'привет!',
            'Ари, как быть на японском хорошее погода?': 'хорошее погода?',
            'Ари, скажи привет по-японски': 'привет',
            'Переведи на японский: спокойной ночи': 'спокойной ночи',
        }
        for request, phrase in cases.items():
            with self.subTest(request=request):
                self.assertEqual(self.llm._japanese_translation_phrase(request), phrase)
        for request in ('Ари, говори по-японски', 'Расскажи о японской культуре',
                        'Не переводи на японский привет', 'Как будет погода завтра?'):
            self.assertIsNone(self.llm._japanese_translation_phrase(request))

    def test_repeated_token_limit_has_bounded_retry_and_no_fragment(self):
        self.client.chat.side_effect = [iter([
            {'message': {'content': 'Обрезанный фрагмент'}},
            {'done_reason': 'length'}]) for _ in range(2)]
        with patch.object(self.llm, 'compress_history'):
            answer = ' '.join(self.llm.stream_sentences('a', 'расскажи историю'))
        self.assertNotIn('Обрезанный', answer)
        self.assertIn('Попробуй спросить ещё раз.', answer)
        self.assertEqual(self.client.chat.call_count, 2)

    def test_cancelled_compression_does_not_remove_messages(self):
        before=self.memory.entry('a')
        request=request_control.Request()
        def response(**kwargs):
            request.cancel()
            return {'message':{'content':'late summary'}}
        self.client.chat.side_effect=response
        token=request_control.current.set(request)
        try:
            with self.assertRaises(request_control.RequestCancelled): self.llm.compress_history('a')
        finally: request_control.current.reset(token)
        self.assertEqual(self.memory.entry('a'),before)

    def test_voice_cues_stream_but_do_not_enter_memory(self):
        self.client.chat.return_value = iter([
            {'message': {'content': '[whis'}},
            {'message': {'content': 'per] Только между нами.'}},
        ])
        with patch.object(self.llm, 'compress_history'):
            sentences = list(self.llm.stream_sentences('a', 'скажи тихо'))
        self.assertEqual(sentences, ['[whisper] Только между нами.'])
        self.assertEqual(self.memory.entry('a')['recent'][-1],
                         {'role': 'assistant', 'content': 'Только между нами.'})

    def test_incomplete_fake_tool_call_is_not_spoken_or_saved(self):
        self.client.chat.return_value = iter([
            {'message': {'content': 'Привет, слушаю тебя. function_'}},
            {'message': {'content': 'call{name: play_sound, arguments:{ sound_type'}},
            {'message': {'content': ': music. more technical content.'}},
        ])
        with patch.object(self.llm, 'compress_history'):
            sentences = list(self.llm.stream_sentences('a', 'привет'))
        self.assertEqual(' '.join(sentences), 'Привет, слушаю тебя.')
        self.assertNotIn('function_call', self.memory.entry('a')['recent'][-1]['content'])

    def test_idle_removes_braced_tool_calls_and_stage_directions(self):
        self.assertEqual(self.llm._sanitize_idle_chatter(
            'Как дела? (Я решила использовать инструмент для имитации звука) '
            'function_call{name: play_sound, arguments:{ sound_type'), 'Как дела?')

    def test_japanese_song_language_is_requested_and_filtered(self):
        self.client.chat.return_value = {'message': {'content': '星が光る。 Вот перевод.'}}
        result = self.llm.generate_idle_chatter(singing=True, language='Japanese')
        self.assertEqual(result, '[singing] 星が光る。')
        self.assertIn('только японский', self.client.chat.call_args.kwargs['messages'][0]['content'])
        self.llm.update_reply_language('a', 'Ари, спой на японском')
        self.assertEqual(self.llm.get_reply_language('a'), 'Japanese')

    def test_idle_song_has_hidden_singing_style_and_no_tool_call(self):
        self.client.chat.return_value = {'message': {'content':
            '[amused] Тихо звёзды светят нам. function_call{name: play_sound'}}
        result = self.llm.generate_idle_chatter(singing=True)
        self.assertEqual(result, '[singing] Тихо звёзды светят нам.')
        prompt = self.client.chat.call_args.kwargs['messages'][-1]['content']
        self.assertIn('пять минут', prompt)


class SpeakerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with patch.dict(sys.modules, discord=types.SimpleNamespace(AudioSource=object),
                        tts=types.SimpleNamespace(synthesize_pcm=lambda text: b"1234")):
            self.module = module('test_speaker', 'speaker.py')
        self.vc = Mock()
        self.vc.is_connected.return_value = True
        self.vc.is_playing.return_value = False
        self.speaker = self.module.Speaker(self.vc, asyncio.get_running_loop())

    async def asyncTearDown(self):
        await self.speaker.close()
        await asyncio.gather(self.speaker._worker_task, return_exceptions=True)

    async def test_missing_callback_stops_playback_and_recovers(self):
        original = asyncio.wait_for
        async def short_wait(awaitable, timeout):
            # Only shorten playback completion, not the native play/stop calls.
            return await original(awaitable, .02 if timeout > 9 else timeout)
        with patch.object(self.module.asyncio, 'wait_for', short_wait):
            await self.speaker.say('first')
            await asyncio.sleep(.1)
            self.assertTrue(self.vc.stop.called)
            self.assertFalse(self.speaker.speaking)
            self.assertFalse(self.speaker._worker_task.done())

    async def test_interrupt_discards_synthesis_already_in_progress(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def synthesize(text):
            started.set()
            await release.wait()
            self.speaker.queue.task_done()
            return b'1234'
        self.speaker._synthesize = synthesize
        await self.speaker.say('old')
        await started.wait()
        await self.speaker.interrupt()
        release.set()
        await asyncio.sleep(.02)
        self.vc.play.assert_not_called()

    async def test_interrupt_rejects_late_enqueue_callback(self):
        self.speaker.say_threadsafe('obsolete')
        await self.speaker.interrupt()
        await asyncio.sleep(.02)
        self.vc.play.assert_not_called()
        self.assertTrue(self.speaker.queue.empty())

    async def test_interrupt_cancels_blocking_synth_and_allows_new_reply(self):
        started = threading.Event()
        def synthesize(text):
            if text == 'old':
                started.set()
                request = request_control.current.get()
                if not request.cancelled.wait(1):
                    raise AssertionError('synthesis was not cancelled')
                request_control.check()
            return b'1234'
        self.module.tts.synthesize_pcm = synthesize
        self.vc.play.side_effect = lambda source, after: after(None)
        await self.speaker.say('old')
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        await self.speaker.say('queued old')
        await self.speaker.interrupt()
        await self.speaker.say('new')
        await asyncio.wait_for(self.speaker.queue.join(), 1)
        await asyncio.sleep(.03)
        self.vc.play.assert_called_once()


if __name__=='__main__': unittest.main()
