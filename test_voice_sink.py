import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import voice_sink


class FakeVad:
    def __init__(self, mode):
        self.frames = []

    def is_speech(self, frame, rate):
        self.frames.append(frame)
        return frame[:2] == b'\x01\x00'


class VoiceSinkTests(unittest.IsolatedAsyncioTestCase):
    def test_voice_singing_request_detection(self):
        import re
        tree = ast.parse(Path('bot.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_is_singing_request')
        scope = {'re': re}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'bot.py', 'exec'), scope)
        detect = scope['_is_singing_request']
        for text in ('Ари, спой песенку', 'Ари, напой что-нибудь', 'Ари, можешь спеть?'):
            self.assertTrue(detect(text), text)
        for text in ('Ари, не пой', 'Ари, не надо петь', 'Ари, не спой случайно',
                     'Мне нравится петь', 'Что такое пение?'):
            self.assertFalse(detect(text), text)

    async def asyncSetUp(self):
        self.results = []
        self.playing = False
        self.patches = [patch.object(voice_sink.webrtcvad, 'Vad', FakeVad),
                        patch.object(voice_sink.config, 'MIN_UTTERANCE_SEC', .04),
                        patch.object(voice_sink.config, 'SILENCE_LIMIT_SEC', 1.0)]
        for p in self.patches:
            p.start()

        async def received(uid, pcm):
            self.results.append((uid, pcm))

        self.sink = voice_sink.VoiceSink(asyncio.get_running_loop(), received,
                                        lambda: self.playing)
        self.speech = b'\x01\x00' * 320
        self.quiet = b'\x02\x00' * 320

    async def asyncTearDown(self):
        self.sink.cleanup()
        await asyncio.sleep(0)
        for p in reversed(self.patches):
            p.stop()

    async def drain(self):
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def test_preserves_lead_internal_pause_and_tail(self):
        frames = [self.quiet, self.speech, self.quiet, self.speech, self.quiet]
        for i, frame in enumerate(frames):
            self.sink._accept_frame(1, frame, i * .02)
        self.sink._expire(2)
        await self.drain()
        self.assertEqual(self.results, [(1, b''.join(frames))])

    async def test_users_have_independent_detectors_and_buffers(self):
        for uid in (1, 2, 1, 2):
            self.sink._accept_frame(uid, self.speech, .1)
        self.assertIsNot(self.sink.vads[1], self.sink.vads[2])
        self.assertEqual(len(self.sink.vads[1].frames), 2)
        self.sink._expire(2)
        await self.drain()
        self.assertEqual(self.results, [(1, self.speech * 2), (2, self.speech * 2)])

    async def test_padding_does_not_make_short_noise_a_valid_phrase(self):
        for _ in range(20):
            self.sink._accept_frame(1, self.quiet, 0)
        self.sink._accept_frame(1, self.speech, .1)
        self.sink._accept_frame(1, self.quiet, .2)
        self.sink._expire(2)
        await self.drain()
        self.assertEqual(self.results, [])

    async def test_timer_finalizes_once_without_more_packets(self):
        for _ in range(2):
            self.sink._accept_frame(1, self.speech, 0)
        await asyncio.sleep(.15)
        self.sink._expire(100)
        await self.drain()
        self.assertEqual(self.results, [(1, self.speech * 2)])

    async def test_playback_flushes_accepted_phrase_and_discards_new_audio(self):
        for _ in range(2):
            self.sink._accept_frame(1, self.speech, 0)
        self.playing = True
        self.sink._handle(SimpleNamespace(pcm=b'\x01\x00' * 1920), 1)
        self.sink._pause_capture()
        await self.drain()
        self.assertEqual(self.results, [(1, self.speech * 2)])
        self.assertFalse(self.sink.vads)

    async def test_packet_fragments_are_processed_as_complete_frames(self):
        # Four 10 ms stereo packets become two 20 ms mono VAD frames.
        for _ in range(4):
            self.sink._handle(SimpleNamespace(pcm=b'\x01\x00' * 960), 1)
        self.assertEqual(len(self.sink.vads[1].frames), 2)
        self.sink._expire(float('inf'))
        await self.drain()
        self.assertEqual(self.results, [(1, self.speech * 2)])

    async def test_bot_callback_allows_synthesis_but_blocks_playback(self):
        # Load only the factory: importing bot.py would load models/connectors.
        tree = ast.parse(Path('bot.py').read_text(encoding='utf-8'))
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == '_make_sink')
        sp = SimpleNamespace(busy=True, speaking=False)
        scope = dict(discord=SimpleNamespace(Guild=object),
                     speakers={1: sp}, bot=SimpleNamespace(loop=None),
                     VoiceSink=lambda loop, callback, playing: playing)
        exec(compile(ast.Module(body=[factory], type_ignores=[]), 'bot.py', 'exec'), scope)
        playing = scope['_make_sink'](SimpleNamespace(id=1))
        self.assertFalse(playing())
        sp.speaking = True
        self.assertTrue(playing())

    async def test_sing_command_queues_one_song_and_respects_busy(self):
        from unittest.mock import AsyncMock
        tree = ast.parse(Path('bot.py').read_text(encoding='utf-8'))
        command = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                       and n.name == 'sing_cmd')
        command.decorator_list = []
        speaker = SimpleNamespace(busy=False, vc=SimpleNamespace(is_connected=lambda: True),
                                  say=AsyncMock())
        ctx = SimpleNamespace(guild=SimpleNamespace(id=1), send=AsyncMock())
        activity = {}
        scope = dict(commands=SimpleNamespace(Context=object), speakers={1: speaker},
                     config=SimpleNamespace(TTS_ENGINE='qwen'), asyncio=asyncio,
                     time=SimpleNamespace(monotonic=lambda: 10), utterance_locks={},
                     last_activity=activity, last_human_activity={}, last_idle_song={},
                     pending_utterances={}, handle_utterance=AsyncMock(),
                     _touch_activity=lambda gid: activity.update({gid: 10}),
                     llm=SimpleNamespace(generate_idle_chatter=lambda singing: '[singing] Ла-ла!'))
        exec(compile(ast.Module(body=[command], type_ignores=[]), 'bot.py', 'exec'), scope)
        await scope['sing_cmd'](ctx)
        speaker.say.assert_awaited_once_with('[singing] Ла-ла!')
        speaker.say.reset_mock()
        speaker.busy = True
        await scope['sing_cmd'](ctx)
        speaker.say.assert_not_awaited()

    async def test_complete_answer_is_queued_once_after_stream_ends(self):
        from unittest.mock import Mock
        import request_control
        tree = ast.parse(Path('bot.py').read_text(encoding='utf-8'))
        process = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                       and n.name == '_process_utterance')
        worker = next(n for n in process.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_stream_worker')
        speaker = SimpleNamespace(say_threadsafe=Mock())
        def stream(*args):
            yield 'Первое предложение.'
            speaker.say_threadsafe.assert_not_called()
            yield 'Второе предложение.'
            speaker.say_threadsafe.assert_not_called()
        scope = dict(speaker=speaker, llm=SimpleNamespace(stream_sentences=stream),
                     uid=1, text='вопрос', status='', name='', t1=0,
                     time=SimpleNamespace(monotonic=lambda: 1, strftime=lambda fmt: 'now'),
                     request_control=request_control)
        exec(compile(ast.Module(body=[worker], type_ignores=[]), 'bot.py', 'exec'), scope)
        scope['_stream_worker']()
        speaker.say_threadsafe.assert_called_once_with(
            'Первое предложение. Второе предложение.')
        speaker.say_threadsafe.reset_mock()
        def broken_stream(*args):
            yield 'Незаконченный ответ.'
            raise RuntimeError('stream failed')
        scope['llm'].stream_sentences = broken_stream
        with self.assertRaises(RuntimeError):
            scope['_stream_worker']()
        speaker.say_threadsafe.assert_not_called()

    async def test_speech_during_idle_generation_is_handled_and_idle_discarded(self):
        from unittest.mock import AsyncMock
        tree = ast.parse(Path('bot.py').read_text(encoding='utf-8'))
        watcher = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                       and n.name == 'idle_watcher')
        sessions, activity, pending = {1: True}, {1: 0}, {}
        speaker = SimpleNamespace(busy=False, say=AsyncMock())
        handler = AsyncMock()
        def generate(singing=False):
            activity[1] = 100
            pending[1] = {2: b'speech'}
            sessions[1] = False
            return 'obsolete idle'
        scope = dict(discord=SimpleNamespace(Guild=object),
                     asyncio=SimpleNamespace(sleep=AsyncMock(), Lock=asyncio.Lock,
                                             to_thread=asyncio.to_thread),
                     time=SimpleNamespace(monotonic=lambda: 100),
                     config=SimpleNamespace(IDLE_CHATTER_SEC=1, IDLE_SONG_SEC=300),
                     last_human_activity={1: 0}, last_idle_song={},
                     voice_sessions=sessions, last_activity=activity,
                     pending_utterances=pending, utterance_locks={}, speakers={1: speaker},
                     bot=SimpleNamespace(user=SimpleNamespace(id=99)),
                     llm=SimpleNamespace(generate_idle_chatter=generate),
                     handle_utterance=handler, _touch_activity=lambda gid: None)
        exec(compile(ast.Module(body=[watcher], type_ignores=[]), 'bot.py', 'exec'), scope)
        guild = SimpleNamespace(id=1, voice_client=SimpleNamespace(
            is_connected=lambda: True, channel=SimpleNamespace(voice_states={2: None})))
        await scope['idle_watcher'](guild)
        speaker.say.assert_not_awaited()
        handler.assert_awaited_once_with(1, 2, b'speech')


if __name__ == '__main__':
    unittest.main()
