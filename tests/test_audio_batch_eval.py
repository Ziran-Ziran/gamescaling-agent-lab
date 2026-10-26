import copy
import hashlib
import json
import unittest

from scripts.training.audio_batch_eval import POLICIES,frozen_schedule,validate_cohort
from scripts.training.retention_compare import ADAPTER_IDENTITIES,MODEL_FINGERPRINT


class AudioBatchEvaluationProtocolTests(unittest.TestCase):
    def test_two_new_frozen_suites_have_complete_counterfactual_pairs(self):
        for suite,variant,start in (('transfer','hazard',35000),('iid','reach',36000)):
            schedule=frozen_schedule(suite)
            self.assertEqual(len(schedule),16)
            self.assertEqual(schedule,[{'variant':variant,'context':context,'colors':colors,'tone':tone}
                for context in range(start,start+4) for colors in (0,1) for tone in (0,1)])
        with self.assertRaises(ValueError):
            frozen_schedule('old')

    def fixture(self):
        files={'adapter_config.json':'a'*64,'adapter_model.safetensors':'b'*64}
        adapter={'files':files,'fingerprint':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
        identity={'model':MODEL_FINGERPRINT,'initialization_fingerprint':'init','packages':{'fixture':'version1'}}
        training={arm:{'identity':{**identity,'arm':arm},'audited_adapter_identity':copy.deepcopy(adapter),
            'export_vs_checkpoint_all24_exact':True,'actual_training':{
            'updates':200,'label_uses':800,'missing_label_uses':100,
            'label_counts':{'present':{'1':350,'3':350},'missing':{'1':50,'3':50}},
            'per_sample_exposure':[{'sample_id':str(i),'missing':missing,'count':
                (11 if i<60 else 10) if not missing else (2 if i<36 else 1)}
                for i in range(64) for missing in (False,True)]}}
                  for arm in ('paired-context','balanced-unpaired')}
        return {'schema_version':1,'model_fingerprint':MODEL_FINGERPRINT,
            'policies':{'original':None,'legacy128':copy.deepcopy(ADAPTER_IDENTITIES['audio-lora128']),
                        'paired-context':copy.deepcopy(adapter),'balanced-unpaired':copy.deepcopy(adapter)},
            'training':training}

    def test_all4_policies_fixed_and_exposure_matched_before_any_new_metrics(self):
        value=self.fixture()
        validate_cohort(value)
        self.assertEqual(set(value['policies']),set(POLICIES))
        for change in (
            lambda v:v['policies'].pop('legacy128'),
            lambda v:v['policies']['legacy128']['files'].update({'adapter_model.safetensors':'c'*64}),
            lambda v:v['training']['paired-context']['actual_training'].update(updates=128),
            lambda v:v['policies'].update({'paired-context':copy.deepcopy(ADAPTER_IDENTITIES['audio-lora128'])}),
            lambda v:v['training']['balanced-unpaired']['identity'].update(initialization_fingerprint='different'),
            lambda v:v['training']['balanced-unpaired']['actual_training'].update(per_sample_exposure=[])):
            bad=copy.deepcopy(value)
            change(bad)
            with self.assertRaises(ValueError):
                validate_cohort(bad)


if __name__=='__main__':
    unittest.main()
