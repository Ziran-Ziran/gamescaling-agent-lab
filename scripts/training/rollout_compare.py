"""Re-audit paired RGB rollouts and report task-grounded retention failures."""
import argparse
import json
import math
from pathlib import Path
import statistics

from controller.control_core import to_action_text
from scripts.training.action_head import choose_class,template_action
from scripts.training.data import read_jsonl,require,sha256_file
from scripts.training.feedback import audit_feedback_episodes
from scripts.training.headless import GAME_IDS,HeadlessGame
from scripts.training import rollout


def audit_first_key(frame):
    policy=frame['policy']
    require(policy['strict_valid'] is True and policy['filtered'] is False and policy['fallback'] is False
            and policy['generated_tokens']==0,'first-key policy fallback/filter contract differs')
    record=policy['decoder_record']
    label,details=choose_class([record['raw_class_logits'][key] for key in ('W','A','S','D','noop')])
    expected=to_action_text(template_action(label))
    probabilities=record['restricted_softmax_not_calibrated']
    recomputed=details['restricted_softmax_not_calibrated']
    # exp() may differ by one ULP across Windows/Linux libm. This tolerance
    # applies only to diagnostic probabilities, never logits/argmax/actions.
    probabilities_match=(set(probabilities)==set(recomputed) and all(
        type(probabilities[key]) in (int,float) and 0<=probabilities[key]<=1
        and math.isclose(probabilities[key],value,rel_tol=1e-12,abs_tol=1e-15)
        for key,value in recomputed.items()))
    require(record['selected_class']==label and record['predicted_tokens']==1
            and probabilities_match
            and frame['proposed_action']==frame['filtered_action']==frame['actual_action']['text']==expected,
            'first-key logits/class/template/actual action mismatch')


def paired_outcomes(before,after):
    require(len(before)==len(after)>0,'complete nonempty paired episodes required')
    differences=[]
    regressions,improvements=[],[]
    original_successes=0
    for first,second in zip(before,after):
        require(first['episode_id']==second['episode_id'],'paired episode order/reset differs')
        original=first['frames'][-1]['terminal_reason']=='goal'
        adapted=second['frames'][-1]['terminal_reason']=='goal'
        original_successes+=original
        if original and not adapted:
            regressions.append(first['episode_id'])
        if adapted and not original:
            improvements.append(first['episode_id'])
        differences.append(sum(f['reward'] for f in second['frames'])-sum(f['reward'] for f in first['frames']))
    return {'paired_mean_return_delta':statistics.mean(differences),
            'return_wins_ties_losses':[sum(d>1e-9 for d in differences),sum(abs(d)<=1e-9 for d in differences),
                                      sum(d< -1e-9 for d in differences)],
            'original_successes':original_successes,'original_success_to_failure':regressions,
            'original_failure_to_success':improvements,
            'retention_fraction_of_original_successes':
                (original_successes-len(regressions))/original_successes if original_successes else None,
            'retention_informative':bool(original_successes)}


def audit_timing(frame):
    durations=frame['timing_ms']
    require(all((key in ('transport','queue') and value is None) or
                (type(value) in (int,float) and math.isfinite(value) and value>=0)
                for key,value in durations.items()),'RGB duration must be finite and nonnegative')
    action=frame['actual_action']
    measured={'execution':(action['end_ns']-action['start_ns'])/1e6,
              'observation_to_action_start':(action['start_ns']-frame['captured_ns'])/1e6,
              'observation_to_action_end':(action['end_ns']-frame['captured_ns'])/1e6}
    require(all(key in durations and abs(durations[key]-value)<1e-9 for key,value in measured.items()),
            'RGB duration differs from actual action/observation timestamps')


def source_paths():
    directory=Path(rollout.__file__).resolve().parent
    names=('rollout.py','headless.py','feedback.py','hf_backend.py','formatting.py','metrics.py','action_head.py')
    return {p.name:p for p in [*(directory/name for name in names),directory.parents[1]/'controller/control_core.py']}


def read_run(directory):
    root=Path(directory).resolve()
    config=json.loads((root/'config.json').read_text())
    arguments=config['arguments']
    require(arguments['decoder']=='first-key' and config['generation']['decoder']=='first-key'
            and config['generation']['effective_predicted_tokens']==1
            and config['generation']['do_sample'] is False,'retention comparison requires fixed first-key decoding')
    require(config['teacher_or_reward_inputs'] is False and config['learning_during_rollout'] is False,
            'evaluation cannot use teacher feedback or learn during rollout')
    paths=source_paths()
    require(config['source_code_sha256']=={name:sha256_file(path) for name,path in paths.items()},
            'RGB rollout source versions differ from current auditor')
    schedule=rollout.make_schedule(arguments['games'],arguments['seed_start'],arguments['episodes_per_game'])
    expected=[f'{GAME_IDS[game]}-seed-{seed}' for game,seed in schedule]
    status=json.loads((root/'run-status.json').read_text())
    require(status['state']=='completed' and status['failed_episodes']==0 and status['active_episode'] is None
            and status['planned_episodes']==status['completed_episodes']==len(expected),'incomplete actual RGB rollout')
    episodes=list(read_jsonl(root/'episodes.jsonl'))
    require([e['episode_id'] for e in episodes]==expected,'RGB rollout does not match complete declared reset schedule')
    for episode,(game,seed) in zip(episodes,schedule):
        require(episode['task']==game and episode['game_id']==GAME_IDS[game]
                and episode['source']['kind']=='recorded'
                and episode['source']['clock_scope']=='real perf_counter_ns'
                and episode['frames'][0]['feedback']['state_before']['max_steps']==arguments['max_steps'],
                'RGB retention requires actual recorded policy task/horizon')
        default=HeadlessGame(variant=game,seed=seed,max_steps=arguments['max_steps'])
        require(episode['system_prompt']==default.system_prompt(),'actual player prompt differs from default visible rules')
        require(episode['frames'][0]['feedback']['state_before']==default.snapshot(),
                'actual reset differs from declared default RGB game/seed')
        for frame in episode['frames']:
            audit_first_key(frame)
            audit_timing(frame)
    audit=audit_feedback_episodes(episodes,root)
    measured=rollout.summarize_episodes(episodes)
    summary=json.loads((root/'summary.json').read_text())
    require(summary['episodes_sha256']==sha256_file(root/'episodes.jsonl')
            and summary['feedback_audit']==audit and summary['results']==measured,
            'declared RGB feedback/metrics differ from audited actual records')
    return config,episodes,measured


def compare_loaded(before,after):
    first,a,summary_a=before
    second,b,summary_b=after
    for key in ('model','environment','source_code_sha256','generation','cache','task_boundary','actor_inputs',
                'teacher_or_reward_inputs','learning_during_rollout','timing_boundary'):
        require(first[key]==second[key],f'paired RGB fairness mismatch: {key}')
    arg_a={k:v for k,v in first['arguments'].items() if k not in ('adapter','output')}
    arg_b={k:v for k,v in second['arguments'].items() if k not in ('adapter','output')}
    require(arg_a==arg_b,'paired RGB arguments differ beyond adapter/output')
    require(len(a)==len(b),'paired RGB episode count differs')
    for x,y in zip(a,b):
        require(x['episode_id']==y['episode_id'] and x['system_prompt']==y['system_prompt']
                and x['action_contract']==y['action_contract'],'paired RGB episode/prompt/action contract differs')
        fa,fb=x['frames'][0],y['frames'][0]
        require(fa['rgb']==fb['rgb'] and fa['feedback']['state_before']==fb['feedback']['state_before'],
                'paired initial RGB/privileged audit state differs')
    groups={}
    for game in first['arguments']['games']:
        subset_a=[e for e in a if e['task']==game]
        subset_b=[e for e in b if e['task']==game]
        groups[game]={'before':summary_a[game],'after':summary_b[game],**paired_outcomes(subset_a,subset_b)}
    return {'schema_version':1,'paired_source_input_action_feedback_summary_audits':True,
            'model':first['model'],'adapter_before':first['adapter'],'adapter_after':second['adapter'],
            'groups':groups,'claim':'actual reward retention on own paused RGB toy rules; not commercial/general capability retention'}


def compare(before,after):
    return compare_loaded(read_run(before),read_run(after))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',required=True,type=Path)
    parser.add_argument('--after',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    report=compare(args.before,args.after)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({game:{k:v for k,v in values.items() if k not in ('before','after')}
                      for game,values in report['groups'].items()},indent=2))
