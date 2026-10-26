import unittest

from scripts.training.metrics import action_signature, summarize


class MetricsTests(unittest.TestCase):
    def test_action_metrics_use_strict_structure_and_semantic_keys(self):
        expected = '<|action_start|>0 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>'
        self.assertEqual(action_signature(expected), action_signature(expected.replace(' ; ', ';')))
        self.assertIsNone(action_signature(expected.replace('W', 'BAD')))
        self.assertIsNone(action_signature('invalid output</think>' + expected))
        rows = [{'strict_valid': True, 'action_match': True, 'loss': 1., 'target_tokens': 10,
                 'timing_ms': {'offline_total': 10., 'inference': 8.}},
                {'strict_valid': False, 'action_match': False, 'loss': 3., 'target_tokens': 30,
                 'timing_ms': {'offline_total': 100., 'inference': 90.}}]
        result = summarize(rows)
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['action_exact_match_rate'], .5)
        self.assertEqual(result['token_weighted_loss'], 2.5)
        self.assertEqual(result['timing_ms']['offline_total']['p95'], 100.)
        self.assertEqual(result['timing_ms']['offline_total']['p50'], 55.)


if __name__ == '__main__':
    unittest.main()
