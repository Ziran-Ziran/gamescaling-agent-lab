"""Paired spectral-feature caching with one frozen vision encoder and CPU head."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import time

from controller.control_core import to_action_text
from scripts.training.action_head import choose_class,template_action
from scripts.training.audio_cue import AudioCueGame
from scripts.training.audio_cue_data import audit_audio_episodes,record_episode
from scripts.training.audio_vision_compare import audit_policy_inputs,require_complete,verify_summary
from scripts.training.data import read_jsonl,require,sha256_file
from scripts.training.hf_backend import RGBBackend,environment
from scripts.training.latency import percentile
from scripts.training.rollout import filter_completion
from scripts.training.spectral_cache import SpectralFeatureCache,feature_from_audio
from scripts.training.vision_features import FrozenVisionEncoder,image_digest,tensor_digest
from scripts.training.vision_readout_compare import audit_feature_file
from scripts.training.vision_readout_eval import load_readout,readout_schedule

ARMS=('uncached','cached')


def make_pair_schedule(rows,*,repeats):
    require(rows and type(repeats) is int and 1<=repeats<=3
            and len({row['sample_id'] for row in rows})==len(rows),'unique source samples and fixed repeat count required')
    return [{'pair_id':f'{repeat}-{index}','sample_id':row['sample_id'],
             'order':list(ARMS if (repeat*len(rows)+index)%2==0 else reversed(ARMS))}
            for repeat in range(repeats) for index,row in enumerate(rows)]


def summarize_pairs(rows,*,expected_pairs,primary):
    require(type(expected_pairs) is int and expected_pairs>0 and len(rows)==expected_pairs*2,
            'all planned paired requests required')
    pairs=defaultdict(dict)
    for row in rows:
        require(row['arm'] in ARMS and row['arm'] not in pairs[row['pair_id']],'duplicate/unknown cache arm')
        require(row['timing_ms'].get(primary,0)>0 and all(value is None or
                type(value) in (int,float) and math.isfinite(value) and value>=0
                for value in row['timing_ms'].values()),'invalid cache timing')
        pairs[row['pair_id']][row['arm']]=row
    require(len(pairs)==expected_pairs and all(set(pair)==set(ARMS) for pair in pairs.values()),
            'missing cache pair')
    equal=0
    for pair in pairs.values():
        a,b=pair['uncached'],pair['cached']
        require(a['sample_id']==b['sample_id'],'cache pair source differs')
        if not a['error'] and not b['error']:
            require(all(a[key]==b[key] for key in ('inputs','evidence_features','selected_class','completion')),
                    'cache pair input/feature/logit/action differs')
            equal+=1
    report={'pairs':expected_pairs,'primary_metric':primary,'processing_budget_ms':200.,'arms':{}}
    for arm in ARMS:
        group=[row for row in rows if row['arm']==arm]
        keys=sorted({key for row in group for key in row['timing_ms']})
        timings={}
        for key in keys:
            values=[row['timing_ms'][key] for row in group if row['timing_ms'].get(key) is not None]
            timings[key]={'p50':statistics.median(values),'p95':percentile(values,.95)} if values else None
        report['arms'][arm]={'requests':len(group),'errors':sum(bool(row['error']) for row in group),
            'primary_over_200ms':sum(row['timing_ms'][primary]>200 for row in group),'timing_ms':timings}
    valid=all(not row['error'] for row in rows)
    report.update(all_exactly_equal=valid and equal==expected_pairs,
        paired_feature_logit_action_agreement_rate=equal/expected_pairs,
        paired_median_speedup_observed=statistics.median(
            pair['uncached']['timing_ms'][primary]/pair['cached']['timing_ms'][primary]
            for pair in pairs.values()) if valid else None,
        p95_method='nearest-rank empirical; repeated observations are not independent contexts',
        claim='fixed task-specific vision/head caching only; not original language decoder or real-time gameplay')
    return report


def source_rows(directory,policy,metadata,missing,*,pilot):
    root=Path(directory).resolve()
    config=json.loads((root/'config.json').read_text())
    require(config['pilot'] is pilot and (pilot or config['suite']=='iid')
            and config['schedule']==readout_schedule(suite=config['suite'],pilot=pilot),
            'cache source must match frozen TRAIN pilot or IID32 confirmation schedule')
    require(config['readout']==metadata and config['model']==metadata['identity']['encoder_identity']
            and config['model_text_used'] is False,'cache source policy/backbone differs from fixed head')
    require_complete(json.loads((root/'run-status.json').read_text()),1 if pilot else 256)
    for name,digest in config['source_code_sha256'].items():
        path=(Path(__file__).resolve().parents[2]/'controller/control_core.py' if name=='control_core.py'
              else Path(__file__).with_name(name))
        require(sha256_file(path)==digest,f'cache source inference code changed: {name}')
    folder=root/'aligned'
    episodes=list(read_jsonl(folder/'episodes.jsonl'))
    require([e['cue_reset'] for e in episodes]==config['schedule'],'cache source reset order differs')
    audit=audit_audio_episodes(episodes,folder,require_pairs=not pilot)
    summary=json.loads((root/'summary.json').read_text())
    verify_summary(summary['results']['aligned'],episodes,audit,sha256_file(folder/'episodes.jsonl'))
    result=[]
    for episode in episodes:
        require(episode['source']['teacher_inputs']==[] and len(episode['frames'])==1,
                'cache source requires one-step actual policy episodes without teacher inputs')
        audit_policy_inputs(episode,folder,'aligned',prediction_kind='feature-readout')
        frame=episode['frames'][0]
        audit_feature_file(frame['policy'],folder,policy,missing,metadata['identity']['feature_dim'])
        result.append({'sample_id':f'{episode["episode_id"]}:0','spec':episode['cue_reset'],
            'expected_inputs':{key:value for key,value in frame['policy']['inputs'].items()
                               if key not in ('actual_visual_calls','feature_prepare_ms')},
            'row':{'rgb':frame['rgb'],'audio':frame['audio'],'captured_ns':frame['captured_ns'],
                   'clock_id':episode['clock']['id']}})
    return result,folder,{'config_sha256':sha256_file(root/'config.json'),
        'episodes_sha256':sha256_file(folder/'episodes.jsonl'),'source':str(root),'samples':len(result)}


def prepare(encoder,head,cache,missing,row,root,*,use_cache):
    from PIL import Image
    began_ns=time.perf_counter_ns()
    began=began_ns/1e9
    require(sha256_file(Path(root)/row['rgb']['path'])==row['rgb']['sha256'],'cache request RGB file hash mismatch')
    with Image.open(Path(root)/row['rgb']['path']) as source:
        rgb_image=source.convert('RGB')
    loaded=time.perf_counter()
    rgb,rgb_record=encoder.encode(rgb_image)
    audio,observation,audio_record=feature_from_audio(row['audio'],root,captured_ns=row['captured_ns'],
        clock_id=row['clock_id'],encoder=encoder,cache=cache if use_cache else None,missing_feature=missing)
    head_began=time.perf_counter()
    logits,head_record=head.predict(rgb,audio,observation['present'])
    head_finished=time.perf_counter()
    label,details=choose_class(logits.tolist())
    completion=to_action_text(template_action(label))
    action,flags=filter_completion(completion)
    require(flags=={'strict_valid':True,'fallback':False,'filtered':False}
            and to_action_text(action)==completion,'cache head must preserve strict safe action')
    finished_ns=time.perf_counter_ns()
    finished=finished_ns/1e9
    inputs={'present':observation['present'],'missing_reason':observation['missing_reason'],'image_count':2,
        'audio_sha256':observation.get('sha256'),'input_rgb_pixel_sha256':image_digest(rgb_image),
        'input_spectral_pixel_sha256':audio_record['input_pixel_sha256'],'prompt_sha256':None,
        'input_tokens':0,'model_text_used':False,'actual_visual_calls':1+audio_record['actual_visual_calls']}
    evidence={'inputs':inputs,'encoder_records':{'rgb':rgb_record,'audio':audio_record},'head_record':head_record,
        'class_record':details,'selected_class':label,'generated_tokens':0,'predicted_tokens':0,'predicted_classes':1,
        'inference_ms':(finished_ns-began_ns)/1e6,'processing_start_ns':began_ns,'processing_end_ns':finished_ns,
        'timing_ms':{'processing':(finished_ns-began_ns)/1e6,'rgb_file_load':(loaded-began)*1000,
            'rgb_visual':rgb_record['total_ms'],'audio_feature_total':audio_record['audio_feature_total_ms'],
            'audio_causal_read':audio_record['causal_read_ms'],'spectral':audio_record['spectral_ms'],
            'audio_visual_or_cache':audio_record['feature_call_ms'],'head':(head_finished-head_began)*1000,
            'parse_filter':(finished-head_finished)*1000,'transport':None,'queue':None,
            'png_persist':None,'wav_persist':None}}
    bundle={'rgb':rgb,'audio':audio,'present':observation['present'],'logits':logits,'head_record':head_record}
    return completion,evidence,bundle


def persist(root,name,bundle):
    import torch
    began=time.perf_counter()
    path=Path(root)/(name+'.features.pt')
    require(not path.exists(),'cache feature evidence already exists')
    torch.save(bundle,path)
    return {'path':path.name,'sha256':sha256_file(path)},(time.perf_counter()-began)*1000


def request_record(pair_id,arm,sample_id,completion,evidence,bundle):
    inputs={key:value for key,value in evidence['inputs'].items() if key!='actual_visual_calls'}
    return {'pair_id':pair_id,'arm':arm,'sample_id':sample_id,'completion':completion,'error':None,
        'inputs':inputs,'evidence_features':{'rgb':tensor_digest(bundle['rgb']),'audio':tensor_digest(bundle['audio']),
            'logits':tensor_digest(bundle['logits']),'head_input':bundle['head_record']['head_input_sha256']},
        'selected_class':evidence['selected_class'],'timing_ms':dict(evidence['timing_ms']),'evidence':evidence}


def append(path,value):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')


def write_status(root,status):
    temporary=Path(root)/'run-status.tmp'
    temporary.write_text(json.dumps(status,indent=2))
    temporary.replace(Path(root)/'run-status.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--readout',type=Path,required=True)
    parser.add_argument('--expected-readout-sha256',required=True,help='freeze artifact before timing')
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--phase',choices=('pilot','offline','live'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    planned=3 if args.phase=='pilot' else 192 if args.phase=='offline' else 64
    status={'state':'loading','planned_requests':planned,'completed_requests':0,'failed_requests':0,
            'active_request':None,'started_utc':datetime.now(timezone.utc).isoformat()}
    write_status(args.output,status)
    try:
        import torch
        torch.set_num_threads(4)
        head,metadata,missing=load_readout(args.readout)
        require(metadata['readout_sha256']==args.expected_readout_sha256
                and metadata['identity']['arm']=='joint-mlp','cache experiment must use frozen joint200 artifact')
        sources,source_root,source_binding=source_rows(args.source,head,metadata,missing,pilot=args.phase=='pilot')
        encoder=FrozenVisionEncoder(RGBBackend(args.model))
        require(encoder.identity==metadata['identity']['encoder_identity'],'cache encoder differs from trained-head backbone')
        cache=SpectralFeatureCache(encoder,capacity=64)
        config={'schema_version':1,'phase':args.phase,'readout':metadata,'encoder':encoder.identity,
            'source_binding':source_binding,'source_schedule':[s['spec'] for s in sources],
            'environment':environment(),'capacity':64,'model_text_used':False,
            'expected_readout_sha256':args.expected_readout_sha256,
            'timing_boundary':'offline processing excludes evidence write/execution; live obs-start includes synchronous evidence writes',
            'source_code_sha256':{p.name:sha256_file(p) for p in (
                Path(__file__),*(Path(__file__).with_name(n) for n in (
                'spectral_cache.py','vision_features.py','vision_readout.py','vision_readout_eval.py',
                'vision_readout_compare.py','audio.py','audio_cue.py','audio_cue_data.py','audio_vision_compare.py',
                'hf_backend.py','feedback.py','headless.py','data.py','action_head.py','rollout.py','formatting.py','metrics.py')),
                Path('controller/control_core.py'))},
            'claim':'exact spectral-image feature reuse for fixed task readout; no native audio or real-time claim'}
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
        evidence_root=args.output/'features'
        evidence_root.mkdir()
        rows=[]
        def offline_request(source,arm,pair_id,name):
            completion,evidence,bundle=prepare(encoder,head,cache,missing,source['row'],source_root,use_cache=arm=='cached')
            feature,persist_ms=persist(evidence_root,name,bundle)
            evidence['feature_file']=feature
            evidence['timing_ms']['evidence_persist_outside_processing']=persist_ms
            return request_record(pair_id,arm,source['sample_id'],completion,evidence,bundle)
        if args.phase=='pilot':
            status['state']='running'
            for index,arm in enumerate(('uncached','cached','cached')):
                status['active_request']=f'pilot-{index}-{arm}'
                write_status(args.output,status)
                row=offline_request(sources[0],arm,0,f'pilot-{index}-{arm}')
                rows.append(row)
                append(args.output/'requests.jsonl',row)
                status['completed_requests']+=1
            require(all(rows[0][key]==row[key] for row in rows[1:]
                        for key in ('inputs','evidence_features','selected_class','completion')),'pilot cache changed policy')
            require([r['evidence']['encoder_records']['audio']['cache_hit'] for r in rows]==[False,False,True],
                    'pilot must demonstrate actual cold miss then hot hit')
            summary={'requests':3,'all_exactly_equal':True,'cache':cache.stats(),
                'actual_visual_calls':[r['evidence']['inputs']['actual_visual_calls'] for r in rows],
                'processing_ms':[r['timing_ms']['processing'] for r in rows],
                'claim':'saved TRAIN-reset GPU cache smoke, excluded from formal latency/quality'}
        else:
            # Two source observations, each arm warmed once; four explicit requests.
            for pair in make_pair_schedule(sources[:2],repeats=1):
                source=next(s for s in sources if s['sample_id']==pair['sample_id'])
                for arm in pair['order']:
                    warm=offline_request(source,arm,pair['pair_id'],f'warmup-{pair["pair_id"]}-{arm}')
                    append(args.output/'warmups.jsonl',warm)
            config['cache_after_warmup']=cache.stats()
            config['warmup_observations']=2
            config['warmup_requests']=4
            (args.output/'config.json').write_text(json.dumps(config,indent=2))
            status['state']='running'
            schedule=make_pair_schedule(sources,repeats=3 if args.phase=='offline' else 1)
            by_id={s['sample_id']:s for s in sources}
            for pair in schedule:
                source=by_id[pair['sample_id']]
                for arm in pair['order']:
                    status['active_request']=f'{pair["pair_id"]}-{arm}'
                    write_status(args.output,status)
                    if args.phase=='offline':
                        row=offline_request(source,arm,pair['pair_id'],f'request-{pair["pair_id"]}-{arm}')
                    else:
                        folder=args.output/arm
                        folder.mkdir(exist_ok=True)
                        def policy(observation):
                            entered=time.perf_counter_ns()
                            completion,evidence,bundle=prepare(encoder,head,cache,missing,observation,folder,use_cache=arm=='cached')
                            feature,persist_ms=persist(folder,Path(observation['rgb']['path']).stem,bundle)
                            evidence.update(condition='aligned',sensor_intervention={'condition':'aligned',
                                'source_audio_sha256':observation['audio']['sha256'],
                                'input_audio_record':dict(observation['audio'])},feature_file=feature,
                                feature_persist_ms=persist_ms,policy_started_ns=entered)
                            evidence['timing_ms']['feature_persist']=persist_ms
                            evidence['policy_finished_ns']=time.perf_counter_ns()
                            return {'completion':completion,'evidence':evidence}
                        episode=record_episode(AudioCueGame(**source['spec']),folder,policy=policy)
                        audit_policy_inputs(episode,folder,'aligned',prediction_kind='feature-readout')
                        require(len(episode['frames'])==1,'live cache confirmation requires one-step source rule')
                        frame=episode['frames'][0]
                        evidence=frame['policy']
                        bundle=torch.load(folder/evidence['feature_file']['path'],map_location='cpu',weights_only=True)
                        row=request_record(pair['pair_id'],arm,source['sample_id'],frame['actual_action']['text'],evidence,bundle)
                        action=frame['actual_action']
                        row['timing_ms'].update(
                            observation_to_action_start=(action['start_ns']-frame['captured_ns'])/1e6,
                            observation_to_action_end=(action['end_ns']-frame['captured_ns'])/1e6,
                            sensor_start_to_action_end=(action['end_ns']-frame['audio']['window_start_ns'])/1e6,
                            execution=(action['end_ns']-action['start_ns'])/1e6,
                            capture_to_policy_start=(evidence['policy_started_ns']-frame['captured_ns'])/1e6,
                            post_policy_to_action_start=(action['start_ns']-evidence['policy_finished_ns'])/1e6,
                            rgb_render_after_audio_available=(frame['captured_ns']-frame['audio']['available_ns'])/1e6,
                            sensor_window_wait=(frame['audio']['available_ns']-frame['audio']['window_start_ns'])/1e6)
                        append(folder/'episodes.jsonl',episode)
                    rows.append(row)
                    append(args.output/'requests.jsonl',row)
                    status.update(completed_requests=status['completed_requests']+1,active_request=None)
                    write_status(args.output,status)
            summary=summarize_pairs(rows,expected_pairs=len(schedule),
                primary='processing' if args.phase=='offline' else 'observation_to_action_start')
            require(summary['all_exactly_equal'],'formal cache policy disagreement/error')
            if args.phase=='live':
                summary['actual_task_results']={}
                for arm in ARMS:
                    episodes=list(read_jsonl(args.output/arm/'episodes.jsonl'))
                    audit=audit_audio_episodes(episodes,args.output/arm)
                    summary['actual_task_results'][arm]={'episodes':len(episodes),'actions':audit['frames'],
                        'successes':audit['successes'],'mean_return':statistics.mean(r['return'] for r in audit['outcomes'])}
            summary['cache']=cache.stats()
        summary['peak_cuda_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        summary['finished_utc']=datetime.now(timezone.utc).isoformat()
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        status.update(state='completed',active_request=None)
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        status.update(state='failed',error_type=type(error).__name__,failed_requests=int(status['active_request'] is not None))
        raise
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_status(args.output,status)


if __name__=='__main__':
    main()
