import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from scripts.training.audio import audio_features, read_audio


class AudioContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'silence.wav'
        with wave.open(str(self.path), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(struct.pack('<3200h', *([0] * 3200)))
        self.record = {'present': True, 'path': self.path.name,
                       'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                       'sample_rate': 16000, 'channels': 1, 'encoding': 'pcm_s16le',
                       'clock_id': 'episode-clock', 'window_start_ns': 800_000_000,
                       'window_end_ns': 1_000_000_000, 'available_ns': 1_000_000_000}

    def read(self, record=None, captured=1_000_000_000):
        return read_audio(record or self.record, self.root, captured_ns=captured,
                          clock_id='episode-clock')

    def test_silence_is_present_and_not_the_missing_mask(self):
        actual = self.read()
        missing = self.read({'present': False, 'missing_reason': 'sensor_unavailable'})
        self.assertTrue(actual['present'])
        self.assertFalse(missing['present'])
        self.assertEqual(actual['missing_reason'], None)
        self.assertEqual(len(audio_features(actual)), 10)
        self.assertEqual(audio_features(actual), audio_features(missing))
        self.assertEqual(missing['waveform'], [])

    def test_future_unavailable_stale_and_foreign_clock_are_explicitly_masked(self):
        cases = [({'window_start_ns': 810_000_000, 'window_end_ns': 1_010_000_000,
                   'available_ns': 1_010_000_000}, 'future_window'),
                 ({'available_ns': 1_000_000_001}, 'not_yet_available'),
                 ({'clock_id': 'other'}, 'clock_mismatch')]
        for update, reason in cases:
            with self.subTest(reason=reason):
                value = self.read({**self.record, **update})
                self.assertFalse(value['present'])
                self.assertEqual(value['missing_reason'], reason)
                self.assertEqual(value['waveform'], [])
        stale = self.read(captured=1_050_000_001)
        self.assertEqual(stale['missing_reason'], 'stale_window')
        self.assertTrue(self.read(captured=1_050_000_000)['present'])

    def test_rejects_hash_duration_encoding_and_path_mismatches(self):
        updates = [{'sha256': '0' * 64}, {'sample_rate': 8000}, {'channels': 2},
                   {'encoding': 'float32'}, {'window_start_ns': 900_000_000},
                   {'path': '../silence.wav'}, {'path': str(self.path)},
                   {'available_ns': 999_999_999}, {'present': 1}]
        for update in updates:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.read({**self.record, **update})

    def test_missing_audio_requires_an_explanation(self):
        with self.assertRaises(ValueError):
            self.read({'present': False})

    def test_single_sample_window_never_produces_nonfinite_features(self):
        path = self.root / 'one.wav'
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(struct.pack('<h', 1000))
        record = {**self.record, 'path': path.name,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'window_start_ns': 999_937_500}
        features = audio_features(self.read(record))
        self.assertTrue(all(math.isfinite(v) for v in features))
        self.assertEqual(features[-1], 0.)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch CPU runtime required')
    def test_fusion_has_real_gradients_and_missing_mask_preserves_rgb(self):
        import torch
        from scripts.training.audio_smoke import make_fusion
        torch.manual_seed(7)
        rgb = torch.nn.Linear(3, 2)
        model = make_fusion(rgb)
        pixels, features = torch.randn(4, 3), torch.randn(4, 10)
        no_audio = torch.zeros(4, dtype=torch.bool)
        self.assertTrue(torch.equal(model(pixels, features, no_audio), rgb(pixels)))
        self.assertTrue(all(not p.requires_grad for p in rgb.parameters()))
        model(pixels, features, ~no_audio).sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.audio.parameters()))
        self.assertTrue(all(p.grad is None for p in rgb.parameters()))


if __name__ == '__main__':
    unittest.main()
