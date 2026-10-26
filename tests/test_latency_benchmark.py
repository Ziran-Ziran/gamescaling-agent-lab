import unittest

from scripts.training.latency import summarize_pairs


class LatencyComparisonTests(unittest.TestCase):
    def rows(self):
        return [{'pair_id': str(i), 'sample_id': str(i), 'arm': arm, 'error': None,
                 'strict_valid': True, 'action_match': True, 'completion': 'same',
                 'timing_ms': {'offline_total': latency, 'inference': latency - 5}}
                for i in range(3) for arm, latency in (('unmerged', 300.), ('merged', 150.))]

    def test_reports_paired_speed_and_processing_budget_with_equal_quality(self):
        result = summarize_pairs(self.rows(), budget_ms=200)
        self.assertEqual(result['paired_median_speedup'], 2.)
        self.assertEqual(result['arms']['unmerged']['budget_overrun_rate'], 1.)
        self.assertEqual(result['arms']['merged']['budget_overrun_rate'], 0.)
        self.assertEqual(result['completion_agreement_rate'], 1.)
        self.assertTrue(result['quality_preserved_on_static_probes'])

    def test_does_not_claim_quality_preserving_speed_when_requests_fail_or_mismatch(self):
        for reason in ('error', 'match'):
            rows = self.rows()
            if reason == 'error':
                rows[1]['error'] = 'inference failed'
                rows[1]['strict_valid'] = False
            else:
                rows[1]['action_match'] = False
            result = summarize_pairs(rows)
            self.assertFalse(result['quality_preserved_on_static_probes'])
            self.assertIsNone(result['paired_median_speedup'])
            self.assertEqual(result['arms']['merged']['requests'], 3)
            if reason == 'error':
                self.assertEqual(result['completion_agreement_rate'], 2 / 3)

    def test_rejects_missing_or_duplicate_pair_arms(self):
        rows = self.rows()
        for invalid in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                summarize_pairs(invalid)


if __name__ == '__main__':
    unittest.main()
