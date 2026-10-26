import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.training.audio import read_audio
from scripts.training.audio_cue import AudioCueGame, spectral_image
from scripts.training.audio_cue_data import audit_audio_episodes, collect, load_audio_dataset
from scripts.training.audio_vision import observation_messages


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1_000_000_000)


class AudioVisionTests(unittest.TestCase):
    def game(self, **kwargs):
        clock = Clock()
        return AudioCueGame(clock_ns=clock.ns, sleep=clock.sleep, **kwargs)

    def test_paired_rgb_is_identical_but_audio_and_correct_actions_differ(self):
        with tempfile.TemporaryDirectory() as folder:
            games = [self.game(context=20000, colors=0, tone=t) for t in (0, 1)]
            observations = [g.observe(Path(folder)) for g in games]
            self.assertEqual(observations[0]['image'].tobytes(), observations[1]['image'].tobytes())
            audio = [read_audio(o['audio'], folder, captured_ns=o['captured_ns'], clock_id=g.base.clock_id)
                     for o, g in zip(observations, games)]
            self.assertNotEqual(audio[0]['sha256'], audio[1]['sha256'])
            rms = [(sum(x*x for x in a['waveform']) / len(a['waveform'])) ** .5 for a in audio]
            self.assertAlmostEqual(*rms, places=5)
            actions = [g.base.expert_action().steps[0] for g in games]
            self.assertEqual(actions, [('A',), ('D',)])
            mirror = self.game(context=20000, colors=1, tone=0)
            self.assertEqual(mirror.base.expert_action().steps[0], ('D',))
            self.assertNotEqual(mirror.render().tobytes(), observations[0]['image'].tobytes())

    def test_spectrogram_preserves_present_silence_and_masks_future_without_open(self):
        with tempfile.TemporaryDirectory() as folder:
            game = self.game(context=1, colors=0, tone=0)
            observation = game.observe(Path(folder))
            audio = read_audio(observation['audio'], folder, captured_ns=observation['captured_ns'],
                               clock_id=game.base.clock_id)
            first = spectral_image(audio)
            silent = spectral_image({**audio, 'waveform': [0.] * 3200})
            future = {**observation['audio'], 'window_start_ns': observation['captured_ns'] + 1,
                      'window_end_ns': observation['captured_ns'] + 200_000_001,
                      'available_ns': observation['captured_ns'] + 200_000_001,
                      'path': 'does-not-exist.wav'}
            masked = read_audio(future, folder, captured_ns=observation['captured_ns'], clock_id=game.base.clock_id)
            self.assertFalse(masked['present'])
            self.assertEqual(masked['missing_reason'], 'future_window')
            self.assertEqual(silent.tobytes(), spectral_image(masked).tobytes())
            self.assertNotEqual(first.tobytes(), silent.tobytes())
            self.assertLessEqual(observation['audio']['available_ns'], observation['captured_ns'])
            self.assertGreaterEqual(observation['audio']['window_end_ns'] - observation['audio']['window_start_ns'], 200_000_000)

    def test_two_images_are_one_current_observation_and_metadata_is_not_text(self):
        row = {'system_prompt': 'fixed rules', 'rgb': {'path': 'rgb.png'},
               'teacher': 'SECRET-TEACHER', 'goal': 'SECRET-GOAL', 'action': 'SECRET-LABEL'}
        messages, paths = observation_messages(row, 'spectral.png', present=True)
        self.assertEqual(paths, ['rgb.png', 'spectral.png'])
        users = [m for m in messages if m['role'] == 'user']
        self.assertEqual(len(users), 1)
        self.assertEqual(sum(p['type'] == 'image' for p in users[0]['content']), 2)
        self.assertNotIn('SECRET', str(messages))
        missing, _ = observation_messages(row, 'spectral.png', present=False)
        self.assertIn('present=0', str(missing))
        self.assertIn('present=1', str(messages))

    def test_audio_availability_and_capture_precede_slow_evidence_persistence(self):
        from unittest.mock import patch
        clock = Clock()
        game = AudioCueGame(context=2, colors=0, tone=0, clock_ns=clock.ns, sleep=clock.sleep)
        original = Path.write_bytes
        def slow_write(path, raw):
            clock.sleep(.1)
            return original(path, raw)
        with tempfile.TemporaryDirectory() as folder, patch.object(Path, 'write_bytes', slow_write):
            observation = game.observe(Path(folder))
            audio = read_audio(observation['audio'], folder, captured_ns=observation['captured_ns'],
                               clock_id=game.base.clock_id)
            self.assertTrue(audio['present'])
            self.assertLess(observation['captured_ns'], clock.ns())

    def test_collection_audits_projection_actual_rewards_and_audio_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'corpus'
            report = collect(root, contexts={'train': 1, 'validation': 1, 'test': 1}, clock_factory=Clock)
            self.assertEqual(report['episodes'], 12)
            self.assertEqual(report['successes'], 12)
            dataset, _, _, audio = load_audio_dataset(root / 'converted')
            self.assertEqual(len(dataset['train']), 4)
            self.assertEqual(len({r['rgb']['pixel_sha256'] for r in dataset['train']}), 2)
            self.assertEqual(len(audio), sum(map(len, dataset.values())))
            self.assertNotIn('state_before', str(dataset))
            index_path = root / 'converted/audio-index.json'
            index_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'audio index'):
                load_audio_dataset(root / 'converted')
            # Resigning the sidecar must not turn false audio bindings into truth.
            import json
            extra_path = root / 'converted/audio-manifest.json'
            extra = json.loads(extra_path.read_text())
            extra['audio_index_sha256'] = hashlib.sha256(index_path.read_bytes()).hexdigest()
            extra_path.write_text(json.dumps(extra))
            with self.assertRaisesRegex(ValueError, 'freshly audited'):
                load_audio_dataset(root / 'converted')

    def test_dedicated_auditor_rejects_rehashed_wrong_sensor_and_projection(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'corpus'
            collect(root, contexts={'train': 1, 'validation': 1, 'test': 1}, clock_factory=Clock)
            episodes = [json.loads(line) for line in (root / 'episodes.jsonl').read_text().splitlines()]
            changed = copy.deepcopy(episodes)
            source = root / changed[0]['frames'][0]['audio']['path']
            other = root / changed[1]['frames'][0]['audio']['path']
            source.write_bytes(other.read_bytes())
            changed[0]['frames'][0]['audio']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'cue waveform'):
                audit_audio_episodes(changed, root)

    def test_rehashed_rgb_replacement_cannot_bypass_projection_audit(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'corpus'
            collect(root, contexts={'train': 1, 'validation': 1, 'test': 1}, clock_factory=Clock)
            episodes = [json.loads(line) for line in (root / 'episodes.jsonl').read_text().splitlines()]
            episodes[0]['frames'][0]['rgb'] = episodes[2]['frames'][0]['rgb']
            with self.assertRaisesRegex(ValueError, 'RGB projection'):
                audit_audio_episodes(episodes, root)


if __name__ == '__main__':
    unittest.main()
