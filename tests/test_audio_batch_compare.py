import copy
import unittest

from scripts.training.audio_batch_compare import paired_audio_results


class AudioBatchPairedEvidenceTests(unittest.TestCase):
    def fixture(self):
        inputs={key:'fixed' for key in ('present','missing_reason','image_count','audio_sha256',
                'input_rgb_pixel_sha256','input_spectral_pixel_sha256','prompt_sha256','input_tokens')}
        return [{'episode_id':'one','system_prompt':'rules','action_contract':{'version':1},
            'frames':[{'rgb':{'path':'one.png','sha256':'fixed'},'audio':{'sha256':'source'},
                       'feedback':{'state_before':{'layout':'fixed'}},'policy':{'inputs':inputs},
                       'reward':-.01,'terminal_reason':'horizon'}]}]

    def test_actual_paired_return_gain_and_input_fairness(self):
        before=self.fixture()
        after=copy.deepcopy(before)
        after[0]['frames'][0].update(reward=.99,terminal_reason='goal')
        report=paired_audio_results(before,after)
        self.assertEqual(report['return_wins_ties_losses'],[1,0,0])
        self.assertEqual(report['paired_mean_return_delta'],1.)
        after[0]['frames'][0]['policy']['inputs']['prompt_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'sensor'):
            paired_audio_results(before,after)

    def test_causal_control_allows_only_declared_intervention_and_preserves_reset(self):
        before=self.fixture()
        after=copy.deepcopy(before)
        after[0]['frames'][0]['policy']['inputs']['present']=False
        paired_audio_results(before,after,check_inputs=False)
        after[0]['frames'][0]['feedback']['state_before']['layout']='different'
        with self.assertRaisesRegex(ValueError,'initial'):
            paired_audio_results(before,after,check_inputs=False)
        with self.assertRaises(ValueError):
            paired_audio_results(before,[])


if __name__=='__main__':
    unittest.main()
