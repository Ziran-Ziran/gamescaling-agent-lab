import copy
import importlib.util
import unittest
from pathlib import Path
import json
import tempfile

from scripts.training.audio_batch import batch_for_step,group_training_rows,target
from tests import test_audio_batch as fixtures
from scripts.training.audio_batch_audit import audit_export,audit_training_records


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional torch CPU runtime required')
class AudioBatchEvidenceTests(unittest.TestCase):
    def fixture(self):
        import torch
        groups=group_training_rows(*fixtures.BalancedAudioBatchTests().fixture())
        rows=[]
        for step in range(8):
            batch,missing=batch_for_step(groups,arm='paired-context',step=step,seed=20261031)
            examples=[]
            for row in batch:
                loss=float(torch.nn.functional.cross_entropy(torch.zeros(1,5),torch.tensor([target(row)])))
                examples.append({'sample_id':row['sample_id'],'target_class':target(row),
                    'raw_class_logits':[0.]*5,'loss':loss,
                    'sensor_evidence':{'present':not missing,'pixel':row['sample_id'],'prepare_ms':1.}})
            rows.append({'step':step+1,'mask_dropout':missing,'target_counts':{'1':2,'3':2},
                         'loss':sum(e['loss'] for e in examples)/4,'gradient_norm':1.,'step_ms':1.,
                         'peak_cuda_allocated_mib':1.,'examples':examples})
        return rows,groups,lambda row,missing:{'present':not missing,'pixel':row['sample_id']}

    def test_recomputes_all_losses_and_binds_missing_sample_exposure(self):
        rows,groups,expected=self.fixture()
        report=audit_training_records(rows,groups,arm='paired-context',expected_inputs=expected)
        self.assertEqual(report['label_uses'],32)
        self.assertEqual(report['missing_label_uses'],4)
        self.assertEqual(report['label_counts'],{'present':{'1':14,'3':14},'missing':{'1':2,'3':2}})
        self.assertLess(report['maximum_ce_difference'],1e-6)

    def test_wrong_sample_mask_logits_loss_or_step_rejected(self):
        rows,groups,expected=self.fixture()
        changes=(lambda r:r[0].update(step=2),
                 lambda r:r[7].update(mask_dropout=False),
                 lambda r:r[0]['examples'][0].update(sample_id='wrong'),
                 lambda r:r[0]['examples'][0].update(loss=0.),
                 lambda r:r[0]['examples'][0]['sensor_evidence'].update(present=False),
                 lambda r:r[0]['examples'][0].update(raw_class_logits=[float('nan')]*5))
        for change in changes:
            bad=copy.deepcopy(rows)
            change(bad)
            with self.assertRaises(ValueError):
                audit_training_records(bad,groups,arm='paired-context',expected_inputs=expected)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),
                     'Optional torch/safetensors CPU runtime required')
class AudioAdapterExportTests(unittest.TestCase):
    def test_export_values_and_configuration_must_match_audited_training_state(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'adapter').mkdir()
            trainables={f'base.layer.{i}.lora_A.default.weight':torch.full((2,2),float(i)) for i in range(24)}
            exported={name.replace('.default.','.'):value.clone() for name,value in trainables.items()}
            config={'base_model_name_or_path':'/local/frozen-snapshot','r':8,'lora_alpha':16,'lora_dropout':0.,
                    'target_modules':['v_proj','q_proj'],'bias':'none','task_type':'CAUSAL_LM','peft_type':'LORA',
                    'inference_mode':True,'fan_in_fan_out':False,'use_dora':False,'use_rslora':False,
                    'use_qalora':False,'lora_bias':False,'rank_pattern':{},'alpha_pattern':{},
                    'modules_to_save':None,'target_parameters':None,'layer_replication':None}
            config_path=root/'adapter/adapter_config.json'
            weights=root/'adapter/adapter_model.safetensors'
            config_path.write_text(json.dumps(config))
            save_file(exported,str(weights))
            identity=audit_export(root,{'trainables':trainables},base_snapshot='frozen-snapshot')
            self.assertIn('fingerprint',identity)
            bad=dict(exported)
            bad[next(iter(bad))]=torch.ones((2,2))*99
            save_file(bad,str(weights))
            with self.assertRaisesRegex(ValueError,'tensor'):
                audit_export(root,{'trainables':trainables},base_snapshot='frozen-snapshot')
            save_file(exported,str(weights))
            for changed in ({'r':16},{'lora_alpha':32},{'use_dora':True},
                            {'target_modules':['q_proj']},{'base_model_name_or_path':'/local/other'}):
                config_path.write_text(json.dumps({**config,**changed}))
                with self.assertRaisesRegex(ValueError,'configuration|base'):
                    audit_export(root,{'trainables':trainables},base_snapshot='frozen-snapshot')


if __name__=='__main__':
    unittest.main()
