import importlib.util
import copy
import math
import unittest
from collections import Counter
from tests import test_audio_replay as fixtures
from scripts.training import audio_replay
from scripts.training import audio_replay_audit as module

@unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch CPU runtime')
class ReplayRecordAuditTests(unittest.TestCase):
    def fixture(self,arm):
        groups,rgb=fixtures.ReplayScheduleTests().fixture();records=[]
        for step in range(64):
            examples=audio_replay.examples_for_step(groups,rgb,arm=arm,step=step,seed=20261101)
            rows=[{'kind':e['kind'],'sample_id':e['row']['sample_id'],'target_class':e['target_class'],
                   'raw_class_logits':[0.]*5,'loss':math.log(5),'sensor_evidence':{'sample':e['row']['sample_id'],'missing':e['missing']}} for e in examples]
            records.append({'step':step+1,'mask_dropout':step%8==7,'objective':len(rows)*math.log(5)/4,
                            'audio_mean_ce':math.log(5),'rgb_mean_ce':math.log(5) if arm=='rgb-replay' else None,
                            'gradient_norm':2.,'gradient_clip_factor':1/(2+1e-6),'step_ms':1.,'peak_cuda_allocated_mib':1.,'examples':rows})
        return groups,rgb,records
    def test_full_budgets_and_common_audio_histogram(self):
        reports=[]
        for arm in audio_replay.ARMS:
            g,r,records=self.fixture(arm)
            report=module.audit_records(records,g,r,arm=arm,expected_inputs=lambda e:{'sample':e['row']['sample_id'],'missing':e['missing']})
            self.assertEqual(report['updates'],64);self.assertEqual(report['audio_label_uses'],128)
            self.assertEqual(report['missing_audio_label_uses'],16);self.assertEqual(report['rgb_label_uses'],128 if arm=='rgb-replay' else 0)
            reports.append(report['per_sample_audio_exposure'])
        self.assertEqual(*reports)
    def test_wrong_objective_coefficient_and_sensor_record_rejected(self):
        g,r,records=self.fixture('audio-only-control')
        bad=copy.deepcopy(records);bad[-1]['objective']*=2
        with self.assertRaisesRegex(ValueError,'objective'):module.audit_records(bad,g,r,arm='audio-only-control',expected_inputs=lambda e:{'sample':e['row']['sample_id'],'missing':e['missing']})
        bad=copy.deepcopy(records);bad[-1]['examples'][0]['sensor_evidence']['teacher']=1
        with self.assertRaisesRegex(ValueError,'sensor'):module.audit_records(bad,g,r,arm='audio-only-control',expected_inputs=lambda e:{'sample':e['row']['sample_id'],'missing':e['missing']})

if __name__=='__main__':unittest.main()
