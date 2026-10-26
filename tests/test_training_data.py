"""Synthetic contract tests; never read real gameplay or model weights."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.training.data import audit_episodes, convert, strict_action
from scripts.training.hf_backend import load_dataset


def fixture(root):
    from PIL import Image
    episodes = []
    for game_index, game in enumerate(('train-game', 'validation-game', 'test-game')):
        frames = []
        for i in range(3):
            name = f'{game}-{i}.png'
            Image.new('RGB', (32, 32), (game_index * 40 + i, i, 0)).save(root / name)
            data = (root / name).read_bytes()
            frames.append({
                'frame_id': i, 'captured_ns': 1_000_000_000 + i * 300_000_000,
                'rgb': {'path': name, 'sha256': hashlib.sha256(data).hexdigest()},
                'actual_action': {'text': '<|action_start|>0 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>',
                                  'start_ns': 1_010_000_000 + i * 300_000_000,
                                  'end_ns': 1_210_000_000 + i * 300_000_000,
                                  'completed': True},
                'terminated': i == 2, 'truncated': False,
                'terminal_reason': 'success' if i == 2 else None,
            })
        episodes.append({'schema_version': 1, 'episode_id': f'episode-{game}',
                         'game_id': game, 'game_build': 'synthetic-v1',
                         'task': 'Hold W for 200ms.', 'system_prompt': 'Synthetic test only.',
                         'source': {'id': 'synthetic-fixture', 'license': 'generated-for-test',
                                    'training_allowed': True, 'kind': 'synthetic'},
                         'clock': {'id': 'fixture', 'unit': 'ns', 'kind': 'monotonic'},
                         'action_contract': {'version': 'gamescaling-200ms-v1',
                                             'executor': 'synthetic', 'screen_size': [32, 32]},
                         'input_policy': 'player_observation', 'frames': frames})
    splits = {'train': ['train-game'], 'validation': ['validation-game'], 'test': ['test-game']}
    return episodes, splits


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episodes, self.splits = fixture(self.root)

    def audit(self):
        return audit_episodes(self.episodes, self.splits, self.root)

    def test_conversion_uses_causal_actual_history_and_explicit_terminal(self):
        # Proposed actions are deliberately different; they must never be targets/history.
        self.episodes[0]['frames'][0]['proposed_action'] = 'BAD PROPOSAL'
        samples, report = self.audit()
        self.assertEqual(report['samples'], 9)
        self.assertEqual(samples[0]['history'], [])
        self.assertEqual(samples[1]['history'][0]['action'], samples[0]['action'])
        self.assertNotIn('BAD PROPOSAL', json.dumps(samples))
        self.assertTrue(samples[2]['terminated'])
        self.assertEqual(samples[2]['split'], 'train')
        self.assertEqual(report['source_kinds'], ['synthetic'])

    def test_incomplete_execution_is_excluded_and_never_enters_history(self):
        self.episodes[0]['frames'][0]['actual_action']['completed'] = False
        samples, report = self.audit()
        self.assertEqual(report['excluded_incomplete'], 1)
        self.assertEqual(len(samples), 8)
        self.assertEqual(samples[0]['history'], [])

    def test_rejects_game_episode_and_image_leakage(self):
        original = copy.deepcopy(self.episodes)
        for mutate in ('game', 'episode', 'image'):
            with self.subTest(mutate=mutate):
                self.episodes = copy.deepcopy(original)
                splits = copy.deepcopy(self.splits)
                if mutate == 'game':
                    splits['test'].append('train-game')
                elif mutate == 'episode':
                    self.episodes[1]['episode_id'] = self.episodes[0]['episode_id']
                else:
                    self.episodes[1]['frames'][0]['rgb'] = self.episodes[0]['frames'][0]['rgb']
                with self.assertRaisesRegex(ValueError, 'overlap|duplicate|leak'):
                    audit_episodes(self.episodes, splits, self.root)

    def test_same_pixels_reencoded_cannot_cross_splits(self):
        from PIL import Image
        source = self.root / self.episodes[0]['frames'][0]['rgb']['path']
        duplicate = self.root / 'reencoded.png'
        with Image.open(source) as image:
            image.save(duplicate, compress_level=0)
        metadata = {'path': duplicate.name, 'sha256': hashlib.sha256(duplicate.read_bytes()).hexdigest()}
        self.assertNotEqual(metadata['sha256'], self.episodes[0]['frames'][0]['rgb']['sha256'])
        self.episodes[1]['frames'][0]['rgb'] = metadata
        with self.assertRaisesRegex(ValueError, 'leak'):
            self.audit()

    def test_partial_execution_resets_history_and_retains_episode_outcome(self):
        self.episodes[0]['frames'][1]['actual_action']['completed'] = False
        samples, report = self.audit()
        self.assertEqual(samples[1]['history'], [])
        self.assertEqual(report['episode_outcomes'][0]['terminal_reason'], 'success')
        self.assertEqual(report['excluded_frames'][0]['frame_id'], 1)

    def test_rejects_invalid_timing_hash_terminal_and_license(self):
        original = copy.deepcopy(self.episodes)
        mutations = [
            lambda e: e[0]['frames'][0]['actual_action'].update(start_ns=0),
            lambda e: e[0]['frames'][0]['actual_action'].update(end_ns=1_010_000_001),
            lambda e: e[0]['frames'][1].update(captured_ns=1_010_000_000),
            lambda e: e[0]['frames'][0]['rgb'].update(sha256='0' * 64),
            lambda e: e[0]['frames'][-1].update(terminated=False, terminal_reason=None),
            lambda e: e[0]['frames'][0].update(terminated=True, terminal_reason='early'),
            lambda e: e[0]['source'].update(training_allowed=False),
            lambda e: e[0].update(input_policy='privileged'),
        ]
        for mutate in mutations:
            self.episodes = copy.deepcopy(original)
            mutate(self.episodes)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.audit()

    def test_paths_cannot_escape_data_root_and_missing_games_are_rejected(self):
        self.episodes[0]['frames'][0]['rgb']['path'] = '../outside.rgb'
        with self.assertRaises(ValueError):
            self.audit()
        self.episodes, self.splits = fixture(self.root)
        del self.splits['test']
        with self.assertRaises(ValueError):
            self.audit()

    def test_strict_action_never_repairs_malformed_training_targets(self):
        valid = '<|action_start|>-1000 1000 -5 ; W LB ; ; Shift ; ; ; Space<|action_end|>'
        self.assertEqual(strict_action(valid), valid)
        for bad in (valid.replace('-1000', 'nan'), valid.replace('-1000', '-1001'),
                    valid.replace('W LB', 'W BAD'), valid.replace('W LB', 'W W'),
                    valid.replace('W LB', 'W A S D LB'), valid.replace(' ; Space', ''),
                    valid + 'suffix', '<think>reasoning</think>' + valid):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                strict_action(bad)

    def test_conversion_outputs_manifest_hashes_and_refuses_overwrite(self):
        source = self.root / 'episodes.jsonl'
        source.write_text(''.join(json.dumps(e) + '\n' for e in self.episodes), encoding='utf-8')
        split_path = self.root / 'splits.json'
        split_path.write_text(json.dumps(self.splits), encoding='utf-8')
        output = self.root / 'converted'
        convert(source, split_path, self.root, output)
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(manifest['split_counts'], {'train': 3, 'validation': 3, 'test': 3})
        for split in self.splits:
            self.assertEqual(manifest['files'][split]['sha256'],
                             hashlib.sha256((output / f'{split}.jsonl').read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            convert(source, split_path, self.root, output)

    def test_loader_reaudits_source_even_if_converted_hash_is_forged(self):
        source, splits = self.root / 'raw.jsonl', self.root / 'split.json'
        source.write_text(''.join(json.dumps(e) + '\n' for e in self.episodes), encoding='utf-8')
        splits.write_text(json.dumps(self.splits), encoding='utf-8')
        output = self.root / 'converted'
        convert(source, splits, self.root, output)
        loaded, _, _ = load_dataset(output)
        self.assertEqual(len(loaded['train']), 3)
        path = output / 'train.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[0]['action'] = rows[0]['action'].replace('W', 'S')
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        manifest_path = output / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['files']['train']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'audited source'):
            load_dataset(output)


if __name__ == '__main__':
    unittest.main()
