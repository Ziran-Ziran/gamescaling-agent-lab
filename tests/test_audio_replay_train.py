import unittest
from types import SimpleNamespace
from scripts.training.audio_replay_train import rgb_evidence

class ReplaySensorAllowlistTests(unittest.TestCase):
    def test_rgb_teacher_reset_reward_never_enter_actor_or_prompt(self):
        captured={}
        class Processor:
            def apply_chat_template(self,messages,**kwargs):
                captured['messages']=messages;return 'fixed prompt'
        def prepare(row,root,**kwargs):
            captured['row']=row;return {'input_ids':SimpleNamespace(shape=(1,9))},0,{'decode':1.,'encode':2.}
        backend=SimpleNamespace(processor=Processor(),prepare=prepare)
        row={'system_prompt':'visible rules','rgb':{'path':'one.png','pixel_sha256':'pixels'},'history':[{'private':1}],
             'action':'teacher','reward':99,'reset':{'hidden':1},'engine_state':'private'}
        _,evidence=rgb_evidence(backend,row,'.')
        self.assertEqual(set(captured['row']),{'system_prompt','rgb','history'});self.assertEqual(captured['row']['history'],[])
        self.assertNotIn('private',str(captured['messages']));self.assertNotIn('teacher',str(captured['messages']))
        self.assertEqual(evidence['image_count'],1);self.assertEqual(evidence['input_tokens'],9);self.assertEqual(evidence['prepare_ms'],3.)

    def test_numpy_version_drift_rejected_in_parent_restore_contract(self):
        import copy
        from scripts.training.audio_replay_train import require_parent_dependencies
        parent={'packages':{'torch':'same'},'image_processor':{'type':'same'},'spectrogram_packages':{'numpy':'2.2.6','Pillow':'13.0.0'}}
        require_parent_dependencies(copy.deepcopy(parent),parent)
        changed=copy.deepcopy(parent);changed['spectrogram_packages']['numpy']='changed'
        with self.assertRaisesRegex(ValueError,'dependencies'):
            require_parent_dependencies(changed,parent)

if __name__=='__main__':unittest.main()
