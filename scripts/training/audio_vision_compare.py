"""CPU-only audit and paired comparison of complete frozen audio policy runs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from controller.control_core import to_action_text
from scripts.training.action_head import choose_class, template_action
from scripts.training.audio import read_audio
from scripts.training.audio_cue import AudioCueGame, spectral_image, waveform_bytes
from scripts.training.audio_cue_data import audit_audio_episodes
from scripts.training.data import checked_rgb, read_jsonl, require, sha256_file
from scripts.training.audio_vision_eval import CONDITIONS, evaluation_schedule


def pixel_hash(image):
    return hashlib.sha256(f'RGB:{image.width}:{image.height}:'.encode()+image.tobytes()).hexdigest()


def require_complete(status, planned):
    require(status['state'] == 'completed' and status['planned_episodes'] == planned
            and status['completed_episodes'] == planned and status['failed_episodes'] == 0
            and status['active_episode'] is None, 'comparison requires every planned episode complete and no failures')


def audit_policy_inputs(episode, root, condition, *, prediction_kind='first-key-token'):
    """Recompute actual sensor interventions independently of inference logs."""
    from PIL import Image
    root = Path(root).resolve()
    require(prediction_kind in ('first-key-token','feature-readout'),'unsupported prediction contract')
    spec = episode['cue_reset']
    for frame in episode['frames']:
        policy = frame['policy']
        require(policy['condition'] == condition and policy['strict_valid'] is True
                and policy['fallback'] is False and policy['filtered'] is False,
                'invalid audio policy condition/action')
        intervention = policy['sensor_intervention']
        require(intervention['condition'] == condition
                and intervention['source_audio_sha256'] == frame['audio']['sha256'], 'source/intervention binding mismatch')
        record = intervention['input_audio_record']
        expected = dict(frame['audio'])
        if condition in ('swapped', 'noise', 'silence'):
            expected.update(path=frame['audio']['path']+f'.{condition}.wav',
                sha256=hashlib.sha256(waveform_bytes(spec['context'], spec['tone']^1 if condition == 'swapped' else 0,
                    kind='tone' if condition == 'swapped' else condition)).hexdigest())
        elif condition == 'future':
            expected.update(path='intentionally-unavailable-future.wav',
                window_start_ns=frame['captured_ns']+200_000_000,
                window_end_ns=frame['captured_ns']+400_000_000,
                available_ns=frame['captured_ns']+400_000_000)
        require(record == expected, 'input audio intervention differs from frozen condition')
        missing = condition in ('missing', 'rgb-only')
        observation = read_audio({'present':False, 'missing_reason':'controlled_mask'} if missing else record,
                                 root, captured_ns=frame['captured_ns'], clock_id=episode['clock']['id'])
        inputs = policy['inputs']
        require(inputs['present'] == observation['present']
                and inputs['missing_reason'] == observation['missing_reason']
                and inputs['audio_sha256'] == observation.get('sha256'), 'model input audio/mask evidence mismatch')
        require(inputs['image_count'] == (1 if condition == 'rgb-only' else 2), 'wrong current sensor image count')
        rgb = checked_rgb(frame['rgb'], root)
        actual_rgb = pixel_hash(Image.new('RGB', (256,256), (0,0,0))) if condition == 'rgb-blank' else rgb['pixel_sha256']
        require(inputs['input_rgb_pixel_sha256'] == actual_rgb, 'model input RGB pixels differ from condition')
        spectral = None if condition == 'rgb-only' else pixel_hash(spectral_image(observation))
        require(inputs['input_spectral_pixel_sha256'] == spectral, 'model input spectral pixels differ from waveform/mask')
        values = policy['class_record']['raw_class_logits']
        label, _ = choose_class([values[k] for k in ('W','A','S','D','noop')])
        require(policy['selected_class'] == label
                and policy['generated_tokens'] == 0
                and policy['predicted_tokens'] == (1 if prediction_kind=='first-key-token' else 0)
                and (prediction_kind=='first-key-token' or policy.get('predicted_classes')==1)
                and frame['actual_action']['text'] == to_action_text(template_action(label)),
                'recorded actual action differs from five-class policy')


def verify_summary(declared, episodes, audit, digest):
    frames = [f for e in episodes for f in e['frames']]
    measured = {'episodes':len(episodes), 'actions':len(frames), 'successes':audit['successes'],
                'mean_return':statistics.mean(v['return'] for v in audit['outcomes']),
                'mean_steps':len(frames)/len(episodes),
                'inference_ms_p50':statistics.median(f['policy']['inference_ms'] for f in frames),
                'observation_to_action_start_ms_p50':statistics.median(
                    (f['actual_action']['start_ns']-f['captured_ns'])/1e6 for f in frames),
                'observation_to_action_end_ms_p50':statistics.median(
                    (f['actual_action']['end_ns']-f['captured_ns'])/1e6 for f in frames),
                'sensor_start_to_action_end_ms_p50':statistics.median(
                    (f['actual_action']['end_ns']-f['audio']['window_start_ns'])/1e6 for f in frames),
                'present_inputs':sum(f['policy']['inputs']['present'] for f in frames), 'episodes_sha256':digest}
    require(len(frames) == audit['frames'] and set(declared) == set(measured), 'audio summary fields differ')
    for key,value in measured.items():
        require(abs(declared[key]-value) < 1e-9 if isinstance(value,float) else declared[key] == value,
                f'declared audio summary differs from raw records: {key}')
    return measured


def read_run(directory):
    directory = Path(directory).resolve()
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    schedule = evaluation_schedule(config['suite'])
    require(config['schedule'] == schedule, 'audio evaluation schedule changed')
    require(config['conditions'] == (list(CONDITIONS) if config['suite'] == 'primary' else ['aligned','missing']),
            'complete frozen condition set required for comparison')
    require_complete(json.loads((directory/'run-status.json').read_text()), len(schedule)*len(config['conditions']))
    summary = json.loads((directory/'summary.json').read_text())
    result = {}
    for condition in config['conditions']:
        root = directory/condition
        episodes = list(read_jsonl(root/'episodes.jsonl'))
        require([e['cue_reset'] for e in episodes] == schedule, 'audio episode order/reset differs from frozen schedule')
        audit = audit_audio_episodes(episodes, root)
        for episode in episodes:
            require(episode['source']['teacher_inputs'] == [], 'teacher rollout cannot stand in for model evaluation')
            audit_policy_inputs(episode, root, condition)
        declared = summary['results'][condition]
        measured = verify_summary(declared, episodes, audit, sha256_file(root/'episodes.jsonl'))
        result[condition] = (episodes, measured)
    return config, result


def compare(before, after):
    config_a, a = read_run(before)
    config_b, b = read_run(after)
    for key in ('model', 'environment', 'spectrogram_packages', 'suite', 'conditions', 'schedule', 'decoder',
                'history_len', 'learning_during_evaluation', 'source_code_sha256'):
        require(config_a[key] == config_b[key], f'paired audio fairness mismatch: {key}')
    groups = {}
    for condition in config_a['conditions']:
        deltas = []
        for ea, eb in zip(a[condition][0], b[condition][0]):
            require(ea['system_prompt'] == eb['system_prompt'], 'paired player rules differ')
            fa, fb = ea['frames'][0], eb['frames'][0]
            require(fa['rgb'] == fb['rgb'] and fa['feedback']['state_before'] == fb['feedback']['state_before'],
                    'paired initial RGB/state differ')
            for key in ('present','missing_reason','image_count','audio_sha256','input_rgb_pixel_sha256',
                        'input_spectral_pixel_sha256','prompt_sha256','input_tokens'):
                require(fa['policy']['inputs'][key] == fb['policy']['inputs'][key], f'paired actual sensor inputs differ: {key}')
            deltas.append(sum(f['reward'] for f in eb['frames'])-sum(f['reward'] for f in ea['frames']))
        groups[condition] = {'before':a[condition][1], 'after':b[condition][1],
                             'paired_mean_return_delta':statistics.mean(deltas),
                             'return_wins_ties_losses':[sum(d>1e-9 for d in deltas),sum(abs(d)<=1e-9 for d in deltas),sum(d< -1e-9 for d in deltas)]}
    return {'schema_version':1,'paired_fairness_and_input_feedback_audits':True,
            'model':config_a['model'],'adapter_before':config_a['adapter'],'adapter_after':config_b['adapter'],
            'suite':config_a['suite'],'groups':groups,
            'claim':'fixed procedural audio-derived vision fusion test; not native audio or real-game generalization'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.before,args.after)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({k:{'before_successes':v['before']['successes'],'after_successes':v['after']['successes'],
                         'paired_return_delta':v['paired_mean_return_delta']} for k,v in report['groups'].items()},indent=2))
