import copy
import json
from pathlib import Path
import tempfile
import unittest

from controller.control_core import SafeAction, to_action_text
from scripts.training.feedback import audit_feedback_episodes
from scripts.training.rollout import filter_completion, make_schedule, run_episode, summarize_episodes


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1_000_000_000)


def text(key=None):
    return to_action_text(SafeAction((0., 0., 0.), ((key,) if key else (),) * 6))


class RolloutTests(unittest.TestCase):
    def test_selected_games_keep_global_seed_offsets_and_no_duplicate_tasks(self):
        self.assertEqual(make_schedule(['hazard'], 12000, 2), [('hazard', 13000), ('hazard', 13001)])
        self.assertEqual(make_schedule(['reach', 'keydoor'], 12000, 1), [('reach', 12000), ('keydoor', 14000)])
        for games in ([], ['reach', 'reach'], ['unknown']):
            with self.subTest(games=games), self.assertRaises(ValueError):
                make_schedule(games, 0, 1)

    def test_tagged_training_resets_have_unique_episode_ids_without_changed_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            episodes = [run_episode(lambda row: {'completion': text('W'), 'generated_tokens': 0, 'timing_ms': {}},
                                    root, variant='reach', seed=0, max_steps=1, clock=Clock(),
                                    episode_tag=f'update-{index}') for index in range(2)]
            self.assertNotEqual(episodes[0]['episode_id'], episodes[1]['episode_id'])
            self.assertEqual(episodes[0]['frames'][0]['rgb']['sha256'], episodes[1]['frames'][0]['rgb']['sha256'])
            audit_feedback_episodes(episodes, root)
            with self.assertRaises(ValueError):
                run_episode(None, root, variant='reach', seed=0, episode_tag='../escape')

    def test_strict_parse_filter_and_failure_are_separate(self):
        action, info = filter_completion(text('W'))
        self.assertTrue(info['strict_valid'])
        self.assertFalse(info['fallback'])
        self.assertFalse(info['filtered'])
        action, info = filter_completion(text('W').replace('0 0 0', '20 0 1').replace('W', 'LB'))
        self.assertTrue(info['strict_valid'])
        self.assertTrue(info['filtered'])
        self.assertEqual(to_action_text(action), text())
        for invalid in ('reasoning ' + text('W'), text('W')[:-4], 'bad', text('W') + text('S')):
            with self.subTest(invalid=invalid):
                action, info = filter_completion(invalid)
                self.assertTrue(info['fallback'])
                self.assertFalse(info['strict_valid'])
                self.assertEqual(to_action_text(action), text())

    def test_actor_sees_only_rgb_prompt_and_actual_completed_history(self):
        seen = []

        def policy(row):
            seen.append(copy.deepcopy(row))
            return {'completion': text('A') if len(seen) == 1 else text('D'), 'generated_tokens': 1,
                    'timing_ms': {'inference': 1.}}

        with tempfile.TemporaryDirectory() as folder:
            episode = run_episode(policy, Path(folder), variant='reach', seed=3,
                                  max_steps=3, history_len=2, clock=Clock())
            self.assertEqual(episode['frames'][-1]['terminal_reason'], 'goal')
            self.assertEqual(len(seen), 3)
            self.assertEqual(set(seen[0]), {'system_prompt', 'rgb', 'history'})
            self.assertEqual(seen[0]['history'], [])
            self.assertEqual(seen[1]['history'][0]['action'], text('A'))
            self.assertNotIn('state_before', json.dumps(seen))
            self.assertNotIn('reward', json.dumps({k: v for k, v in seen[0].items() if k != 'system_prompt'}))
            audit_feedback_episodes([episode], folder)
            summary = summarize_episodes([episode])
            self.assertEqual(summary['reach']['episodes'], 1)
            self.assertEqual(summary['reach']['success_rate'], 1.)
            self.assertAlmostEqual(summary['reach']['mean_return'], .97)

    def test_invalid_proposal_falls_back_and_is_counted_without_fake_success(self):
        with tempfile.TemporaryDirectory() as folder:
            episode = run_episode(lambda row: {'completion': 'not an action', 'generated_tokens': 3,
                                               'timing_ms': {}}, Path(folder), variant='reach', seed=0,
                                  max_steps=2, history_len=0, clock=Clock())
            summary = summarize_episodes([episode])['reach']
            self.assertEqual(summary['success_rate'], 0.)
            self.assertEqual(summary['fallback_rate'], 1.)
            self.assertEqual(summary['strict_valid_rate'], 0.)
            self.assertEqual(summary['reasons'], {'horizon': 1})
            self.assertAlmostEqual(summary['mean_return'], -.02)
            self.assertEqual(episode['source']['kind'], 'synthetic')

    def test_failed_policy_keeps_completed_transition_and_incomplete_status(self):
        calls = []

        def policy(row):
            calls.append(row)
            if len(calls) == 2:
                raise RuntimeError('injected inference failure')
            return {'completion': text('A'), 'generated_tokens': 1, 'timing_ms': {}}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                run_episode(policy, root, variant='reach', seed=0, max_steps=3, clock=Clock())
            transitions = [json.loads(line) for line in (root / 'transitions.jsonl').read_text().splitlines()]
            self.assertEqual(len(transitions), 1)
            self.assertTrue(transitions[0]['frame']['actual_action']['completed'])
            statuses = [json.loads(line) for line in (root / 'episode-status.jsonl').read_text().splitlines()]
            self.assertEqual(statuses[-1]['status'], 'failed')
            self.assertEqual(statuses[-1]['completed_actions'], 1)
            self.assertNotEqual(transitions[0]['frame']['terminal_reason'], 'goal')


if __name__ == '__main__':
    unittest.main()
