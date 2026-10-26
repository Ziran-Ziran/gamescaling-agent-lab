"""Frozen sensor interventions and actual rewards for audio-derived vision fusion."""
import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import statistics
import time

from controller.control_core import to_action_text
from scripts.training.action_head import choose_class, forward_action_logits, template_action
from scripts.training.audio import read_audio
from scripts.training.audio_cue import AudioCueGame, waveform_bytes
from scripts.training.audio_cue_data import audit_audio_episodes, record_episode
from scripts.training.audio_vision import prepare_audio
from scripts.training.data import require, sha256_file
from scripts.training.hf_backend import RGBBackend, environment

CONDITIONS = ('aligned', 'rgb-only', 'missing', 'swapped', 'future', 'noise', 'silence', 'rgb-blank')


def evaluation_schedule(suite):
    require(suite in ('primary', 'secondary'), 'unsupported frozen audio evaluation suite')
    variant, start, count = ('hazard', 23000, 4) if suite == 'primary' else ('slalom', 24000, 2)
    return [{'variant': variant, 'context': context, 'colors': colors, 'tone': tone}
            for context in range(start, start+count) for colors in (0, 1) for tone in (0, 1)]


def condition_input(row, root, condition, spec):
    """Construct a controlled sensor intervention outside the learner, never a label.

    Reset metadata is used only to generate the counterfactual waveform. Neither
    reset metadata nor goal/action targets are passed to the encoder or decoder.
    """
    require(condition in CONDITIONS, 'unsupported audio evaluation condition')
    read_audio(row['audio'], root, captured_ns=row['captured_ns'], clock_id=row['clock_id'])
    changed = {**row, 'audio': dict(row['audio'])}
    flags = {'force_missing': condition == 'missing', 'rgb_only': condition == 'rgb-only',
             'rgb_blank': condition == 'rgb-blank'}
    evidence = {'condition': condition, 'source_audio_sha256': row['audio']['sha256']}
    if condition in ('swapped', 'noise', 'silence'):
        name = row['audio']['path'] + f'.{condition}.wav'
        path = Path(root)/name
        require(not path.exists(), 'intervention output already exists')
        raw = waveform_bytes(spec['context'], spec['tone'] ^ 1 if condition == 'swapped' else 0,
                             kind='tone' if condition == 'swapped' else condition)
        path.write_bytes(raw)
        changed['audio'].update(path=name, sha256=sha256_file(path))
    elif condition == 'future':
        changed['audio'].update(path='intentionally-unavailable-future.wav',
                               window_start_ns=row['captured_ns']+200_000_000,
                               window_end_ns=row['captured_ns']+400_000_000,
                               available_ns=row['captured_ns']+400_000_000)
    evidence['input_audio_record'] = dict(changed['audio'])
    return changed, flags, evidence


def _status(root, value):
    temporary = root/'run-status.tmp'
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(root/'run-status.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=('primary', 'secondary'), required=True)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS)
    args = parser.parse_args()
    conditions = args.conditions or (list(CONDITIONS) if args.suite == 'primary' else ['aligned', 'missing'])
    require(len(set(conditions)) == len(conditions), 'duplicate audio condition')
    args.output.mkdir(parents=True, exist_ok=False)
    schedule = evaluation_schedule(args.suite)
    status = {'state': 'started', 'planned_episodes': len(schedule)*len(conditions),
              'completed_episodes': 0, 'failed_episodes': 0, 'active_episode': None,
              'started_utc': datetime.now(timezone.utc).isoformat()}
    _status(args.output, status)
    try:
        backend = RGBBackend(args.model, adapter=args.adapter)
        backend.model.eval()
        config = {'schema_version': 1, 'model': backend.identity, 'adapter': backend.adapter_identity,
                  'environment': environment(), 'suite': args.suite, 'conditions': conditions,
                  'spectrogram_packages': {name: importlib.metadata.version(name) for name in ('numpy', 'Pillow')},
                  'schedule': schedule, 'decoder': 'argmax5class first-key safe template,0generated/1predicted',
                  'history_len': 0, 'learning_during_evaluation': False,
                  'source_code_sha256': {p.name: sha256_file(p) for p in (
                      Path(__file__), *(Path(__file__).with_name(n) for n in (
                      'audio_cue.py', 'audio_cue_data.py', 'audio_vision.py', 'audio.py',
                      'action_head.py', 'hf_backend.py', 'headless.py', 'feedback.py', 'data.py', 'formatting.py',
                      'metrics.py', 'rollout.py')), Path('controller/control_core.py'))},
                  'claims': 'own paused procedural-audio sensor task, not native waveform/general audio understanding'}
        (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        reports = {}
        status['state'] = 'running'
        for condition in conditions:
            root = args.output/condition
            root.mkdir()
            episodes = []
            for spec in schedule:
                game = AudioCueGame(**spec)
                status['active_episode'] = f'{condition}/{game.base.episode_id}'
                _status(args.output, status)

                def policy(row):
                    changed, flags, intervention = condition_input(row, root, condition, spec)
                    batch, inputs = prepare_audio(backend, changed, root, root, **flags)
                    backend.synchronize()
                    began = time.perf_counter()
                    with backend.torch.inference_mode():
                        raw, candidates = forward_action_logits(backend, batch)
                        values = raw[candidates].float().tolist()
                    backend.synchronize()
                    inference_ms = (time.perf_counter()-began)*1000
                    label, details = choose_class(values)
                    return {'completion': to_action_text(template_action(label)),
                            'evidence': {'condition': condition, 'sensor_intervention': intervention,
                                         'inputs': inputs, 'class_record': details, 'selected_class': label,
                                         'generated_tokens': 0, 'predicted_tokens': 1,
                                         'inference_ms': inference_ms}}

                episode = record_episode(game, root, policy=policy)
                episodes.append(episode)
                with (root/'episodes.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(episode)+'\n')
                status.update(completed_episodes=status['completed_episodes']+1, active_episode=None)
                _status(args.output, status)
                print(json.dumps({'condition': condition, 'episode': episode['episode_id'],
                                  'reason': episode['frames'][-1]['terminal_reason'],
                                  'return': sum(f['reward'] for f in episode['frames'])}), flush=True)
            audit = audit_audio_episodes(episodes, root)
            frames = [f for e in episodes for f in e['frames']]
            reports[condition] = {'episodes': len(episodes), 'actions': len(frames), 'successes': audit['successes'],
                                  'mean_return': statistics.mean(r['return'] for r in audit['outcomes']),
                                  'mean_steps': len(frames)/len(episodes),
                                  'inference_ms_p50': statistics.median(f['policy']['inference_ms'] for f in frames),
                                  'observation_to_action_start_ms_p50': statistics.median(
                                      (f['actual_action']['start_ns']-f['captured_ns'])/1e6 for f in frames),
                                  'observation_to_action_end_ms_p50': statistics.median(
                                      (f['actual_action']['end_ns']-f['captured_ns'])/1e6 for f in frames),
                                  'sensor_start_to_action_end_ms_p50': statistics.median(
                                      (f['actual_action']['end_ns']-f['audio']['window_start_ns'])/1e6 for f in frames),
                                  'present_inputs': sum(f['policy']['inputs']['present'] for f in frames),
                                  'episodes_sha256': sha256_file(root/'episodes.jsonl')}
            (root/'summary.json').write_text(json.dumps(reports[condition], indent=2), encoding='utf-8')
        summary = {'results': reports, 'peak_cuda_allocated_mib': backend.torch.cuda.max_memory_allocated()/1024**2,
                   'finished_utc': datetime.now(timezone.utc).isoformat(), 'claim': config['claims']}
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        status['state'] = 'completed'
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        status.update(state='failed', error_type=type(error).__name__,
                      failed_episodes=int(status['active_episode'] is not None))
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        _status(args.output, status)


if __name__ == '__main__':
    main()
