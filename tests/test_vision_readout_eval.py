import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts.training.vision_readout_eval import load_readout,readout_schedule


class ReadoutScheduleTests(unittest.TestCase):
    def test_class_readout_contract_reports_no_language_token_prediction(self):
        from PIL import Image
        from scripts.training.action_head import choose_class,template_action
        from scripts.training.audio import read_audio
        from scripts.training.audio_cue import AudioCueGame,spectral_image
        from scripts.training.audio_cue_data import record_episode
        from scripts.training.audio_vision_compare import audit_policy_inputs
        from scripts.training.vision_features import image_digest
        from controller.control_core import to_action_text
        class Clock:
            now=1_000_000_000
            def ns(self):
                return self.now
            def sleep(self,seconds):
                self.now+=round(seconds*1e9)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            clock=Clock()
            game=AudioCueGame(context=25000,colors=0,tone=0,variant='hazard',clock_ns=clock.ns,sleep=clock.sleep)
            def policy(row):
                observation=read_audio(row['audio'],root,captured_ns=row['captured_ns'],clock_id=row['clock_id'])
                with Image.open(root/row['rgb']['path']) as source:
                    rgb=image_digest(source)
                label,details=choose_class([0.,1.,0.,0.,0.])
                return {'completion':to_action_text(template_action(label)),'evidence':{
                    'condition':'aligned','sensor_intervention':{'condition':'aligned',
                        'source_audio_sha256':row['audio']['sha256'],'input_audio_record':row['audio']},
                    'inputs':{'present':True,'missing_reason':None,'image_count':2,'audio_sha256':row['audio']['sha256'],
                        'input_rgb_pixel_sha256':rgb,'input_spectral_pixel_sha256':image_digest(spectral_image(observation))},
                    'class_record':details,'selected_class':label,'generated_tokens':0,'predicted_tokens':0,'predicted_classes':1}}
            episode=record_episode(game,root,clock_injected=True,policy=policy)
            audit_policy_inputs(episode,root,'aligned',prediction_kind='feature-readout')
            with self.assertRaises(ValueError):
                audit_policy_inputs(episode,root,'aligned')

    def test_confirmation_uses_fresh_frozen_contexts_and_full_counterbalanced_groups(self):
        schedule=readout_schedule()
        self.assertEqual(len(schedule),32)
        self.assertEqual({s['context'] for s in schedule},set(range(25000,25008)))
        self.assertTrue(all(s['variant']=='hazard' for s in schedule))
        for context in range(25000,25008):
            self.assertEqual({(s['colors'],s['tone']) for s in schedule if s['context']==context},
                             {(0,0),(0,1),(1,0),(1,1)})
        iid=readout_schedule(suite='iid')
        self.assertEqual(len(iid),32)
        self.assertEqual({s['context'] for s in iid},set(range(26000,26008)))
        self.assertTrue(all(s['variant']=='reach' for s in iid))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Optional torch CPU runtime required')
    def test_inference_rejects_partial_or_corrupted_head_artifact(self):
        import torch
        from scripts.training.data import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            torch.save({'schema_version':1,'step':4},root/'readout.pt')
            (root/'result.json').write_text(json.dumps({'readout_sha256':sha256_file(root/'readout.pt'),
                                                       'step':4}))
            with self.assertRaisesRegex(ValueError,'200'):
                load_readout(root)
            (root/'result.json').write_text(json.dumps({'readout_sha256':'0'*64,'step':200}))
            with self.assertRaisesRegex(ValueError,'content hash'):
                load_readout(root)


if __name__=='__main__':
    unittest.main()
