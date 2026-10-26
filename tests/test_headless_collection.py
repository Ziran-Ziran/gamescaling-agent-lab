import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.training.collect_headless import collect
from scripts.training.feedback import audit_feedback_episodes
from scripts.training.hf_backend import load_dataset


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1_000_000_000)


class HeadlessCollectionTests(unittest.TestCase):
    def test_one_step_corpus_keeps_impossible_keydoor_outcome_honest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'one-step'
            report = collect(root, counts={'train': 1, 'validation': 1, 'test': 1},
                             clock_factory=Clock, max_steps=1)
            self.assertEqual(report['frames'], 3)
            self.assertEqual(report['observed_successes'], 2)
            dataset, _, _ = load_dataset(root / 'converted')
            self.assertEqual(len(dataset['test']), 1)
            episodes = [json.loads(line) for line in (root / 'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(episodes[-1]['frames'][-1]['terminal_reason'], 'horizon')
            self.assertTrue(episodes[-1]['frames'][-1]['truncated'])

    def test_expert_collection_passes_both_feedback_and_training_audits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'run'
            result = collect(root, counts={'train': 2, 'validation': 2, 'test': 2}, clock_factory=Clock)
            self.assertEqual(result['episodes'], 6)
            self.assertTrue(all(v['reason'] == 'goal' for v in result['outcomes'].values()))
            dataset, manifest, _ = load_dataset(root / 'converted')
            self.assertEqual(set(manifest['source_kinds']), {'synthetic'})
            self.assertEqual(len(dataset['train']), 2)
            self.assertGreater(len(dataset['test']), 2)
            # Full engine truth stays only in original feedback/annotations, not SFT rows.
            self.assertNotIn('state_before', json.dumps(dataset))
            self.assertNotIn('state_after', json.dumps(dataset))
            self.assertEqual(manifest['excluded_incomplete'], 0)
            self.assertEqual({row['game_id'] for row in dataset['train']}, {'headless-reach-v1'})
            self.assertEqual({row['game_id'] for row in dataset['test']}, {'headless-keydoor-v1'})

    def test_ledger_rejects_misbound_frames_and_replaced_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'run'
            collect(root, counts={'train': 1, 'validation': 1, 'test': 1}, clock_factory=Clock)
            episodes = [json.loads(line) for line in (root / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            changed = copy.deepcopy(episodes)
            changed[0]['frames'][0]['reward'] = 100.
            with self.assertRaises(ValueError):
                audit_feedback_episodes(changed, root)
            changed = copy.deepcopy(episodes)
            changed[0]['frames'][0]['rgb'] = changed[1]['frames'][0]['rgb']
            with self.assertRaisesRegex(ValueError, 'RGB'):
                audit_feedback_episodes(changed, root)
            changed = copy.deepcopy(episodes)
            changed[2]['frames'][1]['feedback']['state_before_sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                audit_feedback_episodes(changed, root)

    def test_truncated_episode_prefix_cannot_inflate_success_return(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'run'
            collect(root, counts={'train': 1, 'validation': 1, 'test': 1}, clock_factory=Clock)
            episodes = [json.loads(line) for line in (root / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            changed = copy.deepcopy(episodes[-1])
            self.assertGreater(len(changed['frames']), 1)
            changed['frames'] = changed['frames'][-1:]
            with self.assertRaisesRegex(ValueError, 'reset'):
                audit_feedback_episodes([changed], root)


if __name__ == '__main__':
    unittest.main()
