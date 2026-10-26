from pathlib import Path
import tempfile
import unittest

from scripts.training.audio import read_audio
from scripts.training.audio_cue import AudioCueGame, waveform_bytes
from scripts.training.audio_vision_eval import condition_input, evaluation_schedule


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds*1e9)


class AudioEvaluationTests(unittest.TestCase):
    def test_condition_schedule_is_frozen_and_pairs_stay_together(self):
        primary = evaluation_schedule('primary')
        self.assertEqual(len(primary), 16)
        self.assertEqual({s['context'] for s in primary}, set(range(23000, 23004)))
        secondary = evaluation_schedule('secondary')
        self.assertEqual(len(secondary), 8)
        self.assertTrue(all(s['variant'] == 'slalom' for s in secondary))

    def test_swap_noise_silence_and_future_change_sensor_not_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clock = Clock()
            game = AudioCueGame(context=5, colors=0, tone=0, clock_ns=clock.ns, sleep=clock.sleep)
            observed = game.observe(root)
            row = {'system_prompt': game.system_prompt(), 'rgb': {'path': 'unchanged.png'},
                   'audio': observed['audio'], 'captured_ns': observed['captured_ns'],
                   'clock_id': game.base.clock_id}
            before = game.base.snapshot()
            for condition in ('swapped', 'noise', 'silence', 'future', 'missing', 'rgb-only', 'rgb-blank', 'aligned'):
                changed, flags, evidence = condition_input(row, root, condition, game.spec)
                self.assertEqual(changed['rgb'], row['rgb'])
                self.assertEqual(game.base.snapshot(), before)
                if condition == 'future':
                    value = read_audio(changed['audio'], root, captured_ns=row['captured_ns'], clock_id=row['clock_id'])
                    self.assertFalse(value['present'])
                    self.assertFalse((root/changed['audio']['path']).exists())
                if condition == 'swapped':
                    self.assertEqual((root/changed['audio']['path']).read_bytes(), waveform_bytes(5, 1))
                if condition == 'silence':
                    value = read_audio(changed['audio'], root, captured_ns=row['captured_ns'], clock_id=row['clock_id'])
                    self.assertTrue(value['present'])
                    self.assertEqual(set(value['waveform']), {0.})
                self.assertEqual(evidence['condition'], condition)


if __name__ == '__main__':
    unittest.main()
