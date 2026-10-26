"""Virtual single-worker scheduling; no online execution or quality claims."""
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest


class QueueReplayTests(unittest.TestCase):
    def simulate(self, services, **kwargs):
        from scripts.training.queue_replay import simulate_queue
        return simulate_queue(services, interval_ns=10, action_ns=5,
                              start_budget_ns=10, end_budget_ns=15, **kwargs)

    def test_fifo_accounts_for_serial_action_occupancy(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.training.queue_replay'),
                             'queue replay is not implemented')
        result = self.simulate([8, 8, 8], policy='fifo')
        self.assertEqual([e['worker_start_ns'] for e in result['events']], [0, 13, 26])
        self.assertEqual([e['queue_ns'] for e in result['events']], [0, 3, 6])
        self.assertEqual(result['on_time_start_arrivals'], 1)
        self.assertEqual(result['on_time_end_arrivals'], 1)
        self.assertEqual(result['processed'], 3)
        self.assertEqual(result['dropped'], [])

    def test_latest_drops_pending_only_and_drains_last_frame(self):
        result = self.simulate([28, 1, 1, 1, 1], policy='latest')
        self.assertEqual([e['frame_index'] for e in result['events']], [0, 3, 4])
        self.assertEqual(result['dropped'], [1, 2])
        self.assertEqual([e['worker_start_ns'] for e in result['events']], [0, 33, 40])
        self.assertEqual(result['arrivals'], 5)
        self.assertEqual(result['on_time_start_arrivals'], 2)
        self.assertEqual(result['on_time_start_fraction_all_arrivals'], 2 / 5)
        self.assertEqual(result['drain_end_ns'], 46)

    def test_arrival_at_release_is_eligible_and_future_service_is_not_a_selection_oracle(self):
        result = self.simulate([25, 1, 1000, 1000, 2], policy='latest')
        self.assertEqual(result['events'][1]['frame_index'], 3)
        self.assertEqual(result['events'][1]['worker_start_ns'], 30)
        self.assertEqual(result['dropped'], [1, 2])
        self.assertEqual(result['events'][-1]['frame_index'], 4)

    def test_idle_worker_waits_for_capture_and_budget_equality_is_on_time(self):
        result = self.simulate([5, 5, 5], policy='latest')
        self.assertEqual([e['worker_start_ns'] for e in result['events']], [0, 10, 20])
        self.assertEqual(result['on_time_end_fraction_all_arrivals'], 1)
        self.assertEqual(result['dropped'], [])

    def test_invalid_or_ambiguous_inputs_fail_closed(self):
        from scripts.training.queue_replay import simulate_queue
        for values in ([], [0], [-1], [True], [1.5]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.simulate(values, policy='fifo')
        for changes in ({'interval_ns': 0}, {'action_ns': -1}, {'policy': 'oracle'},
                        {'start_budget_ns': True}, {'end_budget_ns': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                simulate_queue([1], **{'interval_ns': 10, 'action_ns': 5, 'policy': 'fifo',
                                      'start_budget_ns': 10, 'end_budget_ns': 15, **changes})


class SavedTraceTests(unittest.TestCase):
    def setUp(self):
        from tests.test_decoding_benchmark import DecodingBenchmarkTests
        from scripts.training.decoding_benchmark import summarize_pairs
        from scripts.training.data import sha256_file
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = DecodingBenchmarkTests().rows()
        self.rows = []
        for i in range(3):
            pair = rows[2*i:2*i+2]
            if i % 2:
                pair.reverse()
            for row in pair:
                row['pair_id'] = f'0:{i}'
            self.rows.extend(pair)
        self.config = {'samples': [{'id': str(i), 'rgb_sha256': 'same'} for i in range(3)],
                       'repeats': 1, 'model': {'fingerprint': 'test'}, 'adapter': {'fingerprint': 'test'},
                       'environment': {'gpu': 'synthetic-test'},
                       'source_code_sha256': {'decoding_benchmark.py': sha256_file(
                           Path('scripts/training/decoding_benchmark.py'))}}
        self.summary = summarize_pairs(self.rows)
        self.write()

    def write(self):
        (self.root / 'config.json').write_text(json.dumps(self.config), encoding='utf-8')
        (self.root / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows), encoding='utf-8')
        (self.root / 'summary.json').write_text(json.dumps(self.summary), encoding='utf-8')

    def test_saved_trace_replays_every_pair_in_both_fixed_policies(self):
        from scripts.training.queue_replay import replay_benchmark
        result = replay_benchmark(self.root)
        self.assertEqual(set(result['cells']), {'full', 'first-key'})
        self.assertEqual(result['cells']['full']['fifo']['arrivals'], 3)
        self.assertEqual(result['cells']['first-key']['latest']['arrivals'], 3)
        self.assertIn('simulation', result['claim'])
        self.assertNotIn('task_success', result)

    def test_reordered_missing_or_misbound_requests_are_rejected(self):
        from scripts.training.queue_replay import replay_benchmark
        original = copy.deepcopy(self.rows)
        for mutation in ('order', 'missing', 'binding'):
            self.rows = copy.deepcopy(original)
            if mutation == 'order':
                self.rows[:2] = reversed(self.rows[:2])
            elif mutation == 'missing':
                self.rows.pop()
            else:
                self.config['samples'][0]['id'] = 'not-the-measured-input'
            self.write()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                replay_benchmark(self.root)

    def test_source_code_drift_is_rejected_before_any_simulated_output(self):
        from scripts.training.queue_replay import replay_benchmark
        self.config['source_code_sha256']['decoding_benchmark.py'] = '0' * 64
        self.write()
        with self.assertRaisesRegex(ValueError, 'source'):
            replay_benchmark(self.root)


if __name__ == '__main__':
    unittest.main()
