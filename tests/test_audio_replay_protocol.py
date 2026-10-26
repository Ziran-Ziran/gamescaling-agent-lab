import copy
import hashlib
import json
import unittest
from scripts.training import audio_replay_protocol as module
from scripts.training.audio_replay_train import PARENT_COHORT
from scripts.training.retention_compare import MODEL_FINGERPRINT

class RehearsalProtocolTests(unittest.TestCase):
    def fixture(self):
        parent={'fingerprint':'a0f855a986117d92babdec515f0ba20b6c13a254f4d61d8a3c4225a28d6aa6ea','files':{'adapter_config.json':'2dc33d71032002e5ed0d6000e5ec8555d1de33062a4570a9b4900168d1a8f316','adapter_model.safetensors':'e017302b535a028872b5899493259ee7429020ffb5da53b0933f1736e6f7fa44'}}
        policies={'original':None,'parent200':parent};training={}
        for arm in ('audio-only-control','rgb-replay'):
            files={'adapter_config.json':'1'*64,'adapter_model.safetensors':('2' if arm=='rgb-replay' else '3')*64}
            adapter={'files':files,'fingerprint':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()};policies[arm]=adapter
            identity={'arm':arm,'parent_adapter':parent,'initialization_fingerprint':'warm','parent_cohort':PARENT_COHORT,'rgb_terms':2 if arm=='rgb-replay' else 0,'packages':{'fixture':'version1'}}
            exposure=[{'sample_id':str(i),'missing':i<8,'count':2} for i in range(64)]
            actual={'updates':64,'audio_label_uses':128,'missing_audio_label_uses':16,'rgb_label_uses':128 if arm=='rgb-replay' else 0,'all64_gradients_finite_nonzero':True,'audio_label_counts':{'present':{'1':56,'3':56},'missing':{'1':8,'3':8}},'rgb_label_counts':{'0':32,'1':32,'2':32,'3':32} if arm=='rgb-replay' else {},'per_sample_audio_exposure':exposure}
            training[arm]={'identity':identity,'actual_training':actual,'checkpoint_step':64,'trainable_tensors':24,'export_vs_checkpoint_all24_exact':True,'parent24_copy_initial_fingerprint_exact':True,'audited_adapter_identity':adapter}
        return {'schema_version':1,'model_fingerprint':MODEL_FINGERPRINT,'parent_cohort_sha256':PARENT_COHORT,'parent_initialization_fingerprint':'warm','policies':policies,'training':training}
    def test_all_new_seed_suites_and_final_cohort_exposure_required(self):
        cohort=self.fixture();module.validate_cohort(cohort)
        for suite,start,game in (('transfer',47000,'hazard'),('iid',48000,'reach')):
            rows=module.frozen_schedule(suite);self.assertEqual(len(rows),8);self.assertEqual({r['context'] for r in rows},{start,start+1});self.assertEqual({r['variant'] for r in rows},{game})
        self.assertEqual(module.CONDITIONS,('aligned','missing','swapped','rgb-only','rgb-blank'))
        for key,value in (('audio_label_uses',127),('missing_audio_label_uses',15),('rgb_label_uses',127)):
            bad=copy.deepcopy(cohort);bad['training']['rgb-replay']['actual_training'][key]=value
            with self.assertRaises(ValueError):module.validate_cohort(bad)
        bad=copy.deepcopy(cohort);bad['training']['rgb-replay']['actual_training']['per_sample_audio_exposure'][0]['count']=1
        with self.assertRaises(ValueError):module.validate_cohort(bad)
        bad=copy.deepcopy(cohort);bad['policies']['rgb-replay']['fingerprint']='f'*64;bad['training']['rgb-replay']['audited_adapter_identity']['fingerprint']='f'*64
        with self.assertRaises(ValueError):module.validate_cohort(bad)
    def test_partial_audio_and_rgb_matrices_fail_before_file_reads(self):
        from scripts.training.audio_replay_compare import compare_suite
        from scripts.training.audio_replay_retention import compare_matrix
        with self.assertRaises(ValueError):compare_suite({'parent200':'unread'},'unread',model='unread')
        with self.assertRaises(ValueError):compare_matrix({'original':'unread'},'unread',suite='primary')

    def test_complete_three_policy_aggregation_uses_control_to_replay(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tests import test_audio_batch_compare as fixtures
        from scripts.training import audio_replay_compare as compare
        fields=('model','environment','cohort_sha256','suite','conditions','schedule','image_processor','spectrogram_packages','decoder','history_len','learning_during_evaluation','source_code_sha256')
        base={key:'fixed' for key in fields};base['suite']='transfer'
        def read_run(path,*args,**kwargs):
            config={**base,'policy':path};episodes=fixtures.AudioBatchPairedEvidenceTests().fixture()
            if path=='rgb-replay':episodes[0]['frames'][0].update(reward=.99,terminal_reason='goal')
            return config,{condition:(copy.deepcopy(episodes),{'successes':int(path=='rgb-replay')}) for condition in module.CONDITIONS}
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'cohort.json';file.write_text(json.dumps(self.fixture()))
            with patch.object(compare,'cpu_backend',return_value=None),patch.object(compare,'read_run',side_effect=read_run):
                report=compare.compare_suite({p:p for p in module.AUDIO_POLICIES},file,model='unused')
        self.assertEqual(set(report['groups']),set(module.CONDITIONS))
        for group in report['groups'].values():
            self.assertEqual(group['additional_rgb_objective']['return_wins_ties_losses'],[1,0,0])
            self.assertEqual(set(group['versus_parent']),{'audio-only-control','rgb-replay'})

if __name__=='__main__':unittest.main()
