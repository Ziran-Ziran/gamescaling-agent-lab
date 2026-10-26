import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.training.action_head import choose_class, template_action
from scripts.training.audio import read_audio
from scripts.training.audio_cue import AudioCueGame, spectral_image
from scripts.training.audio_cue_data import record_episode
from scripts.training.audio_vision_compare import audit_policy_inputs, compare, require_complete, verify_summary
from scripts.training.audio_vision_eval import condition_input
from controller.control_core import to_action_text


class Clock:
    now = 1_000_000_000

    def ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds*1e9)


def pixel(image):
    return hashlib.sha256(f'RGB:{image.width}:{image.height}:'.encode()+image.tobytes()).hexdigest()


class AudioComparisonTests(unittest.TestCase):
    def test_pairing_rejects_changed_execution_runtime_or_gpu(self):
        from unittest.mock import patch
        config = {'model':{},'spectrogram_packages':{},'suite':'primary','conditions':[],
                  'schedule':[],'decoder':'safe','history_len':0,'learning_during_evaluation':False,
                  'source_code_sha256':{},'adapter':None,
                  'environment':{'packages':{'torch':'version1'},'gpu':'device1','torch_cuda':'12.8','host':'host1'}}
        for field,new in (('packages',{'torch':'version2'}),('gpu','device2'),('torch_cuda','13'),('host','host2')):
            changed = copy.deepcopy(config)
            changed['environment'][field] = new
            with self.subTest(field=field), patch('scripts.training.audio_vision_compare.read_run',
                    side_effect=[(config,{}),(changed,{})]), self.assertRaises(ValueError):
                compare('before','after')

    def test_published_latency_mask_and_step_summaries_are_recomputed_from_raw_frames(self):
        frames = [{'captured_ns': 1_000_000_000, 'audio': {'window_start_ns': 799_000_000},
                   'actual_action': {'start_ns': start, 'end_ns': start+200_000_000},
                   'policy': {'inputs': {'present': present}, 'inference_ms': inference}}
                  for start,present,inference in ((1_400_000_000,True,50.),(1_600_000_000,False,150.))]
        episodes = [{'frames':[frame]} for frame in frames]
        audit = {'frames':2,'successes':1,'outcomes':[{'return':.99},{'return':-.01}]}
        declared = {'episodes':2,'actions':2,'successes':1,'mean_return':.49,'mean_steps':1.,
                    'inference_ms_p50':100.,'observation_to_action_start_ms_p50':500.,
                    'observation_to_action_end_ms_p50':700.,'sensor_start_to_action_end_ms_p50':901.,
                    'present_inputs':1,'episodes_sha256':'d'*64}
        verify_summary(declared,episodes,audit,'d'*64)
        for key in ('mean_steps','present_inputs','inference_ms_p50','observation_to_action_start_ms_p50',
                    'observation_to_action_end_ms_p50','sensor_start_to_action_end_ms_p50'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_summary({**declared,key:declared[key]+1},episodes,audit,'d'*64)

    def test_missing_or_failed_episodes_cannot_be_published_as_complete_comparison(self):
        for changed in ({'completed_episodes': 1}, {'failed_episodes': 1}, {'active_episode': 'still-running'},
                        {'state': 'running'}):
            status = {'state': 'completed', 'planned_episodes': 2, 'completed_episodes': 2,
                      'failed_episodes': 0, 'active_episode': None, **changed}
            with self.assertRaises(ValueError):
                require_complete(status, 2)

    def test_sensor_input_audit_recomputes_actual_intervention_and_rejects_fake_hash(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clock = Clock()
            game = AudioCueGame(context=23000, colors=0, tone=0, variant='hazard',
                                clock_ns=clock.ns, sleep=clock.sleep)
            def policy(row):
                changed, flags, intervention = condition_input(row, root, 'swapped', game.spec)
                obs = read_audio(changed['audio'], root, captured_ns=row['captured_ns'], clock_id=row['clock_id'])
                with Image.open(root/row['rgb']['path']) as rgb:
                    rgb_hash = pixel(rgb.convert('RGB'))
                label, details = choose_class([0., 1., 0., 0., 0.])
                return {'completion': to_action_text(template_action(label)), 'evidence': {
                    'condition': 'swapped', 'sensor_intervention': intervention,
                    'inputs': {'present': True, 'missing_reason': None, 'image_count': 2,
                               'audio_sha256': obs['sha256'], 'input_rgb_pixel_sha256': rgb_hash,
                               'input_spectral_pixel_sha256': pixel(spectral_image(obs))},
                    'class_record': details, 'selected_class': label, 'generated_tokens': 0, 'predicted_tokens': 1}}
            episode = record_episode(game, root, clock_injected=True, policy=policy)
            audit_policy_inputs(episode, root, 'swapped')
            import os
            previous = Path.cwd()
            try:
                os.chdir(root)
                audit_policy_inputs(episode, Path('.'), 'swapped')
            finally:
                os.chdir(previous)
            changed = copy.deepcopy(episode)
            changed['frames'][0]['policy']['inputs']['input_spectral_pixel_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'spectral pixels'):
                audit_policy_inputs(changed, root, 'swapped')


if __name__ == '__main__':
    unittest.main()
