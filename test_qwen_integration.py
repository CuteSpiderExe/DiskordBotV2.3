import base64
import importlib.util
import io
from pathlib import Path
import queue
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from audio_pcm import stereo48k
from voice_style import delivery, delivery_instruction, strip_html_markup, speech_language, japanese_only


class AudioTests(unittest.TestCase):
    def test_embedded_japanese_translation_reaches_tts_once(self):
        text = ('Прекрасной погодой будет 日本語で「いい天気ですね」(Ii tenki desu ne). '
                'Это значит Хорошая погода. Ну что скажешь о самой фразе?')
        expected = '日本語で「いい天気ですね」.'
        self.assertEqual(japanese_only(text), expected)
        self.assertEqual(japanese_only('[whisper] По-японски: こんにちは。'),
                         '[whisper] こんにちは。')
        backend = Mock()
        backend.synthesize.return_value = b'audio'
        with patch.dict(sys.modules, config=types.SimpleNamespace(TTS_ENGINE='qwen'),
                        qwen_backend=types.SimpleNamespace(backend=backend)):
            spec = importlib.util.spec_from_file_location('test_tts', Path(__file__).with_name('tts.py'))
            tts = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tts)
            self.assertEqual(tts.synthesize_pcm(text), b'audio')
        backend.synthesize.assert_called_once_with(expected, 'normal')

    def test_japanese_answer_discards_russian_translation_and_commentary(self):
        text = ('じゃあね、あなた。また会いたいな？ '
                '(Я сказала Джаа нэ, аната. Мата айтай на? — '
                'Ну ладно, милый/милая (обращение). Хочу увидеться снова? .) '
                'Как я угадала настроение? Мне нравится твой интерес к языкам...')
        self.assertEqual(japanese_only(text), 'じゃあね、あなた。 また会いたいな？')
        self.assertEqual(japanese_only('[whisper] またね。'), '[whisper] またね。')
        self.assertEqual(japanese_only('Только русский ответ.'), '')

    def test_japanese_voice_language_and_instructions(self):
        self.assertEqual(speech_language('こんにちは！今日は元気です。'), 'Japanese')
        self.assertEqual(speech_language('Привет, как дела?'), 'Russian')
        self.assertIn('Japanese', delivery_instruction('whisper', 'Japanese'))
        self.assertNotIn('Russian', delivery_instruction('whisper', 'Japanese'))
    def test_html_is_removed_but_quoted_text_is_preserved(self):
        self.assertEqual(strip_html_markup('</blockquote>'), '')
        self.assertEqual(strip_html_markup('&lt;/blockquote&gt;'), '')
        self.assertEqual(strip_html_markup('<blockquote>Привет!</blockquote>'), 'Привет!')
        self.assertEqual(strip_html_markup('Привет!</blockquote'), 'Привет!')
        self.assertEqual(strip_html_markup('2 < 3'), '2 < 3')

    def test_warmup_runs_generation_and_conversion_without_playback(self):
        from qwen_worker import warmup_model
        model, convert = Mock(), Mock()
        model.generate_custom_voice.return_value = ([b'audio'], 24000)
        warmup_model(model, 'Ono_Anna', 'natural', convert)
        self.assertEqual(model.generate_custom_voice.call_args.kwargs['speaker'], 'Ono_Anna')
        self.assertEqual(model.generate_custom_voice.call_args.kwargs['max_new_tokens'], 64)
        convert.assert_called_once_with(b'audio', 24000)

    def test_style_is_not_spoken(self):
        text, style = delivery(" [whisper] Тихо. [amused] Да!")
        self.assertEqual(text, "Тихо.  Да!")
        self.assertEqual(style, {"segments": [
            {"text": "Тихо.", "style": "whisper"}, {"text": "Да!", "style": "amused"}]})
        self.assertEqual(delivery("Обычная речь."), ("Обычная речь.", "normal"))

    def test_mid_answer_emotion_and_return_to_normal(self):
        text, style = delivery('Привет. [whisper] Скажу по секрету. [normal] Пока!')
        self.assertNotIn('[', text)
        self.assertEqual([part['style'] for part in style['segments']],
                         ['normal', 'whisper', 'normal'])
        instruction = delivery_instruction(style)
        self.assertIn('Скажу по секрету.', instruction)
        self.assertIn('Whisper', instruction)
        self.assertIn('one continuous utterance', instruction)
        self.assertEqual(delivery('[whisper] [sad] Грустно.'), ('Грустно.', 'sad'))
        self.assertEqual(delivery('Привет. [whisper]'), ('Привет.', 'normal'))

    def test_pcm_duration_channels_and_fade(self):
        pcm = stereo48k(np.ones(2400, dtype=np.float32) * .5, 24000)
        stereo = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2)
        self.assertEqual(len(stereo), 4800)
        np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1])
        self.assertEqual(int(stereo[-1, 0]), 0)

    def test_invalid_audio_rejected(self):
        with self.assertRaises(ValueError):
            stereo48k(np.array([float("nan")]), 24000)


class BackendTests(unittest.TestCase):
    def setUp(self):
        cfg = types.SimpleNamespace(QWEN_TIMEOUT_SEC=.01, QWEN_SPEAKER="Ono_Anna")
        with patch.dict(sys.modules, config=cfg):
            spec = importlib.util.spec_from_file_location("test_backend", Path(__file__).with_name("qwen_backend.py"))
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.backend = self.module.QwenBackend()

    def tearDown(self):
        self.backend.close()

    def test_utf8_request_and_pcm_response(self):
        proc = Mock()
        proc.poll.return_value = None
        proc.stdin = io.StringIO()
        self.backend._process = proc
        self.backend._responses = queue.Queue()
        self.backend._responses.put({"pcm": base64.b64encode(b"\x01\x02\x03\x04").decode()})
        self.assertEqual(self.backend.synthesize("Привет", "whisper"), b"\x01\x02\x03\x04")
        import json
        self.assertEqual(json.loads(proc.stdin.getvalue()), {"text": "Привет", "style": "whisper", "speaker": "Ono_Anna", "language": "Russian"})

    def test_japanese_language_sent_to_worker(self):
        proc = Mock()
        proc.poll.return_value = None
        proc.stdin = io.StringIO()
        self.backend._process = proc
        self.backend._responses = queue.Queue()
        self.backend._responses.put({'pcm': base64.b64encode(b'1234').decode()})
        self.backend.synthesize('こんにちは！')
        import json
        self.assertEqual(json.loads(proc.stdin.getvalue())['language'], 'Japanese')

    def test_timeout_kills_worker(self):
        proc = Mock()
        proc.poll.return_value = None
        self.backend._process = proc
        self.backend._responses = queue.Queue()
        with self.assertRaises(TimeoutError):
            self.backend._receive()
        proc.kill.assert_called_once()
        self.assertIsNone(self.backend._process)

    def test_cancelled_request_kills_synthesis_worker(self):
        import request_control
        request = request_control.Request()
        request.cancel()
        proc = Mock()
        proc.poll.return_value = None
        self.backend._process = proc
        self.backend._responses = queue.Queue()
        token = request_control.current.set(request)
        try:
            with self.assertRaises(request_control.RequestCancelled):
                self.backend._receive()
        finally:
            request_control.current.reset(token)
        proc.kill.assert_called_once()
        self.assertIsNone(self.backend._process)

    def test_worker_eof_reports_error(self):
        responses = queue.Queue()
        self.backend._responses = responses
        self.backend._read_output(io.StringIO(""), responses)
        with self.assertRaisesRegex(RuntimeError, "exited"):
            self.backend._receive()


if __name__ == "__main__":
    unittest.main()
