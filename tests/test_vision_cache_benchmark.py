import copy
import unittest

from scripts.training.vision_cache_benchmark import make_pair_schedule,summarize_pairs


class CachePairTests(unittest.TestCase):
    def rows(self):
        return [{'pair_id':pair,'arm':arm,'sample_id':str(pair),'inputs':{'pixel':'same'},
                 'evidence_features':{'rgb':'r','audio':'a','logits':'l','head_input':'h'},
                 'selected_class':'A','completion':'safe-action','error':None,
                 'timing_ms':{'processing':10. if arm=='uncached' else 5.,'transport':None,'queue':None}}
                for pair in range(2) for arm in ('uncached','cached')]

    def test_fixed_alternating_orders_and_three_repeats(self):
        schedule=make_pair_schedule([{'sample_id':'one'},{'sample_id':'two'}],repeats=3)
        self.assertEqual(len(schedule),6)
        self.assertEqual([row['order'] for row in schedule],
                         [['uncached','cached'],['cached','uncached']]*3)
        self.assertEqual(len({row['pair_id'] for row in schedule}),6)

    def test_complete_pairs_exact_equality_and_no_discarded_errors(self):
        report=summarize_pairs(self.rows(),expected_pairs=2,primary='processing')
        self.assertEqual(report['paired_median_speedup_observed'],2.)
        self.assertTrue(report['all_exactly_equal'])
        self.assertEqual(report['arms']['cached']['requests'],2)
        failed=self.rows()
        failed[1]['error']='deliberate-test-failure'
        report=summarize_pairs(failed,expected_pairs=2,primary='processing')
        self.assertEqual(report['arms']['cached']['errors'],1)
        self.assertIsNone(report['paired_median_speedup_observed'])
        self.assertFalse(report['all_exactly_equal'])

    def test_missing_duplicate_or_changed_feature_and_negative_time_rejected(self):
        rows=self.rows()
        for changed in (rows[:-1],rows+[copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError):
                summarize_pairs(changed,expected_pairs=2,primary='processing')
        changed=copy.deepcopy(rows)
        changed[1]['evidence_features']['audio']='different'
        with self.assertRaisesRegex(ValueError,'feature'):
            summarize_pairs(changed,expected_pairs=2,primary='processing')
        changed=copy.deepcopy(rows)
        changed[0]['timing_ms']['processing']=-1.
        with self.assertRaisesRegex(ValueError,'timing'):
            summarize_pairs(changed,expected_pairs=2,primary='processing')


if __name__=='__main__':
    unittest.main()
