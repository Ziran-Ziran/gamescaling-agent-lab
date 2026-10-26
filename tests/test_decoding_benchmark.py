import copy
import math
import unittest

from scripts.training.decoding_benchmark import summarize_pairs


NOOP = '<|action_start|>0 0 0 ;  ;  ;  ;  ;  ; <|action_end|>'
UP = '<|action_start|>0 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>'


class DecodingBenchmarkTests(unittest.TestCase):
    def rows(self):
        return [{'pair_id': str(i), 'sample_id': str(i), 'rgb_sha256': 'same',
                 'arm': arm, 'error': None, 'strict_valid': True, 'completion': NOOP,
                 'actual_action': NOOP, 'timing_ms': {'offline_total': ms, 'inference': ms - 10}}
                for i in range(3) for arm, ms in (('full', 300.), ('first-key', 150.))]

    def test_paired_latency_and_agreement_are_separate_from_actual_task_quality(self):
        result = summarize_pairs(self.rows())
        self.assertEqual(result['paired_median_speedup_observed'], 2.)
        self.assertEqual(result['executed_action_agreement_rate'], 1.)
        self.assertEqual(result['arms']['full']['processing_budget_overrun_rate'], 1.)
        self.assertEqual(result['arms']['first-key']['processing_budget_overrun_rate'], 0.)
        self.assertNotIn('task_success', result)
        rows = self.rows()
        rows[1].update(completion=UP, actual_action=UP)
        result = summarize_pairs(rows)
        self.assertEqual(result['paired_median_speedup_observed'], 2.)
        self.assertEqual(result['executed_action_agreement_rate'], 2 / 3)
        self.assertFalse(result['all_paired_actions_agree'])

    def test_failure_keeps_denominator_but_blocks_speedup_claim(self):
        rows = self.rows()
        rows[1].update(error='model failure', strict_valid=False, completion='', actual_action=None)
        result = summarize_pairs(rows)
        self.assertIsNone(result['paired_median_speedup_observed'])
        self.assertEqual(result['arms']['first-key']['requests'], 3)
        self.assertEqual(result['arms']['first-key']['errors'], 1)
        self.assertEqual(result['executed_action_agreement_rate'], 2 / 3)

    def test_incomplete_duplicate_misbound_and_invalid_timings_are_rejected(self):
        rows = self.rows()
        invalid = [rows[:-1], rows + [rows[0]]]
        for field in ('sample_id', 'rgb_sha256'):
            changed = copy.deepcopy(rows)
            changed[1][field] = 'other'
            invalid.append(changed)
        for latency in (-1., 0., math.nan):
            changed = copy.deepcopy(rows)
            changed[1]['timing_ms']['offline_total'] = latency
            invalid.append(changed)
        for changed in invalid:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                summarize_pairs(changed)


if __name__ == '__main__':
    unittest.main()
