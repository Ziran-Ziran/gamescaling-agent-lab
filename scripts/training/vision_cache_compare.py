"""CPU reconstruction of cache transactions, saved features and actual timings."""
import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from controller.control_core import to_action_text
from scripts.training.action_head import choose_class,template_action
from scripts.training.audio_cue_data import audit_audio_episodes
from scripts.training.audio_vision_compare import audit_policy_inputs
from scripts.training.data import read_jsonl,require,sha256_file
from scripts.training.spectral_cache import identity_text
from scripts.training.vision_cache_benchmark import ARMS,make_pair_schedule,source_rows,summarize_pairs
from scripts.training.vision_features import tensor_digest
from scripts.training.vision_readout_compare import audit_feature_file
from scripts.training.vision_readout_eval import load_readout


def audit_cache_trace(rows,encoder_identity,*,capacity):
    namespace=hashlib.sha256(identity_text(encoder_identity).encode()).hexdigest()
    entries=OrderedDict()
    lookups=hits=misses=evictions=0
    expected_calls=[]
    def stats():
        return {'capacity':capacity,'entries':len(entries),'lookups':lookups,'hits':hits,
                'misses':misses,'evictions':evictions}
    for row in rows:
        require(row['arm'] in ARMS,'invalid cache trace arm')
        evidence=row['evidence']
        inputs=evidence['inputs']
        record=evidence['encoder_records']['audio']
        require(type(inputs['present']) is bool and type(record['cache_lookup']) is bool
                and type(record['cache_hit']) is bool,'boolean causal/cache flags required')
        active=row['arm']=='cached' and inputs['present']
        require(record['cache_lookup'] is active,'cache lookup differs from causal mask/arm')
        if active:
            require(record['cache_before']==stats(),'cache counter before lookup differs')
            key=hashlib.sha256((namespace+':'+record['input_pixel_sha256']).encode()).hexdigest()
            require(record['cache_namespace']==namespace and record['cache_key']==key,'cache key/encoder namespace mismatch')
            hit=key in entries
            require(record['cache_hit'] is hit,'forged cache hit/miss')
            lookups+=1
            if hit:
                hits+=1
                previous=entries.pop(key)
                require(record['feature_sha256']==previous[0] and record['origin_record']==previous[1],
                        'cache hit feature/origin differs from earlier encoded entry')
                entries[key]=previous
            else:
                misses+=1
                origin=record['origin_record']
                require(origin['feature_sha256']==record['feature_sha256']
                        and origin['input_pixel_sha256']==record['input_pixel_sha256']
                        and origin['image_count']==1,'cache miss source feature/pixels/image count differs')
                entries[key]=(record['feature_sha256'],copy.deepcopy(origin))
                if len(entries)>capacity:
                    entries.popitem(last=False)
                    evictions+=1
            require(record['cache_after']==stats(),'cache counter after lookup differs')
            audio_calls=0 if hit else 1
        else:
            require(record['cache_hit'] is False,'uncached/missing audio cannot be cache hit')
            expected_stats=stats() if row['arm']=='cached' else None
            require(record['cache_before']==record['cache_after']==expected_stats,'masked/uncached audio touched cache')
            audio_calls=1 if inputs['present'] else 0
            if inputs['present']:
                origin=record['origin_record']
                require(origin['feature_sha256']==record['feature_sha256']
                        and origin['input_pixel_sha256']==record['input_pixel_sha256']
                        and origin['image_count']==1,'uncached source visual feature differs')
            else:
                require(record['reused_fixed_missing_placeholder'] is True,'missing audio must use declared placeholder')
        require(type(record['actual_visual_calls']) is int and record['actual_visual_calls']==audio_calls
                and type(inputs['actual_visual_calls']) is int and inputs['actual_visual_calls']==audio_calls+1,
                'visual call count differs from reconstructed cache transaction')
        expected_calls.append(audio_calls+1)
    return {'cache':stats(),'expected_visual_calls':expected_calls,'namespace':namespace}


def audit_request(row,root,head,missing,feature_dim,expected_inputs,expected_calls):
    import torch
    evidence=row['evidence']
    require(row['error'] is None and row['inputs']==expected_inputs
            and row['inputs']=={k:v for k,v in evidence['inputs'].items() if k!='actual_visual_calls'},
            'cache request input source/binding differs')
    audit_feature_file(evidence,root,head,missing,feature_dim,expected_visual_calls=expected_calls)
    bundle=torch.load(Path(root)/evidence['feature_file']['path'],map_location='cpu',weights_only=True)
    measured={'rgb':tensor_digest(bundle['rgb']),'audio':tensor_digest(bundle['audio']),
              'logits':tensor_digest(bundle['logits']),'head_input':bundle['head_record']['head_input_sha256']}
    label,_=choose_class(bundle['logits'].tolist())
    require(row['evidence_features']==measured and row['selected_class']==evidence['selected_class']==label
            and row['completion']==to_action_text(template_action(label)), 'cache request saved feature/logit/action differs')
    start,end=evidence['processing_start_ns'],evidence['processing_end_ns']
    require(type(start) is type(end) is int and start<end,'invalid processing clock interval')
    require(row['timing_ms']['processing']==evidence['timing_ms']['processing']==evidence['inference_ms']==(end-start)/1e6,
            'processing time differs from raw monotonic clock')
    require(all(row['timing_ms'][key]==value for key,value in evidence['timing_ms'].items()),
            'cache request timing/evidence differs')


def compare(directory,readout,source):
    import torch
    torch.set_num_threads(4)
    root=Path(directory).resolve()
    config=json.loads((root/'config.json').read_text())
    phase=config['phase']
    require(phase in ('pilot','offline','live') and config['capacity']==64,'unsupported frozen cache phase/capacity')
    head,metadata,missing=load_readout(readout)
    require(config['readout']==metadata and config['expected_readout_sha256']==metadata['readout_sha256']
            and config['encoder']==metadata['identity']['encoder_identity'] and config['model_text_used'] is False,
            'cache experiment trained-head/backbone identity mismatch')
    for name,digest in config['source_code_sha256'].items():
        path=(Path(__file__).resolve().parents[2]/'controller/control_core.py' if name=='control_core.py'
              else Path(__file__).with_name(name))
        require(sha256_file(path)==digest,f'cache experiment source changed: {name}')
    sources,_,binding=source_rows(source,head,metadata,missing,pilot=phase=='pilot')
    require(all(binding[key]==config['source_binding'][key] for key in ('config_sha256','episodes_sha256','samples'))
            and config['source_schedule']==[s['spec'] for s in sources],'cache experiment frozen source changed')
    rows=list(read_jsonl(root/'requests.jsonl'))
    planned=3 if phase=='pilot' else 192 if phase=='offline' else 64
    status=json.loads((root/'run-status.json').read_text())
    require(status['state']=='completed' and status['planned_requests']==status['completed_requests']==planned
            and status['failed_requests']==0 and status['active_request'] is None and len(rows)==planned,
            'cache audit requires complete planned requests and no failures')
    warmups=[] if phase=='pilot' else list(read_jsonl(root/'warmups.jsonl'))
    if phase=='pilot':
        expected=[(0,arm,sources[0]['sample_id']) for arm in ('uncached','cached','cached')]
    else:
        warm_schedule=make_pair_schedule(sources[:2],repeats=1)
        require([(r['pair_id'],r['arm'],r['sample_id']) for r in warmups]==
                [(p['pair_id'],arm,p['sample_id']) for p in warm_schedule for arm in p['order']]
                and config['warmup_observations']==2 and config['warmup_requests']==4,'warmup observations/order changed')
        schedule=make_pair_schedule(sources,repeats=3 if phase=='offline' else 1)
        expected=[(p['pair_id'],arm,p['sample_id']) for p in schedule for arm in p['order']]
    require([(r['pair_id'],r['arm'],r['sample_id']) for r in rows]==expected,'measured cache pair/repeat/order changed')
    trace=audit_cache_trace(warmups+rows,config['encoder'],capacity=64)
    warm_trace=audit_cache_trace(warmups,config['encoder'],capacity=64)
    if phase!='pilot':
        require(config['cache_after_warmup']==warm_trace['cache'],'warmup cache counters changed')
    by_id={s['sample_id']:s for s in sources}
    for index,row in enumerate(warmups+rows):
        folder=root/'features' if index<len(warmups) or phase!='live' else root/row['arm']
        audit_request(row,folder,head,missing,metadata['identity']['feature_dim'],
                      by_id[row['sample_id']]['expected_inputs'],trace['expected_visual_calls'][index])
    declared=json.loads((root/'summary.json').read_text())
    require(declared['cache']==trace['cache'],'reported final cache counters differ from trace')
    if phase=='pilot':
        require(declared['requests']==3 and declared['all_exactly_equal'] is True
                and declared['actual_visual_calls']==trace['expected_visual_calls']
                and declared['processing_ms']==[r['timing_ms']['processing'] for r in rows]
                and all(rows[0][key]==row[key] for row in rows[1:]
                        for key in ('inputs','evidence_features','selected_class','completion')),
                'pilot numeric equality/counters differ')
        metrics={key:declared[key] for key in ('requests','all_exactly_equal','actual_visual_calls','processing_ms')}
    else:
        metrics=summarize_pairs(rows,expected_pairs=96 if phase=='offline' else 32,
            primary='processing' if phase=='offline' else 'observation_to_action_start')
        require(all(declared[key]==value for key,value in metrics.items()),'reported cache timing/agreement summary differs')
    if phase=='live':
        results={}
        for arm in ARMS:
            folder=root/arm
            episodes=list(read_jsonl(folder/'episodes.jsonl'))
            require([e['cue_reset'] for e in episodes]==config['source_schedule'],'live reset schedule changed')
            audit=audit_audio_episodes(episodes,folder)
            selected=[row for row in rows if row['arm']==arm]
            for episode,row in zip(episodes,selected):
                require(episode['source']['teacher_inputs']==[] and episode['source']['kind']=='recorded'
                        and len(episode['frames'])==1,'live cache confirmation needs actual policy actions')
                audit_policy_inputs(episode,folder,'aligned',prediction_kind='feature-readout')
                frame=episode['frames'][0]
                evidence=frame['policy']
                action=frame['actual_action']
                require(row['evidence']==evidence and row['completion']==action['text'],'live request/actual frame binding differs')
                require(frame['captured_ns']<=evidence['policy_started_ns']<=evidence['processing_start_ns']
                        <evidence['processing_end_ns']<=evidence['policy_finished_ns']<=action['start_ns'],
                        'live policy/processing/action raw clocks overlap')
                measured={'observation_to_action_start':(action['start_ns']-frame['captured_ns'])/1e6,
                    'observation_to_action_end':(action['end_ns']-frame['captured_ns'])/1e6,
                    'sensor_start_to_action_end':(action['end_ns']-frame['audio']['window_start_ns'])/1e6,
                    'execution':(action['end_ns']-action['start_ns'])/1e6,
                    'capture_to_policy_start':(evidence['policy_started_ns']-frame['captured_ns'])/1e6,
                    'post_policy_to_action_start':(action['start_ns']-evidence['policy_finished_ns'])/1e6,
                    'rgb_render_after_audio_available':(frame['captured_ns']-frame['audio']['available_ns'])/1e6,
                    'sensor_window_wait':(frame['audio']['available_ns']-frame['audio']['window_start_ns'])/1e6}
                require(all(abs(row['timing_ms'][key]-value)<1e-9 for key,value in measured.items()),
                        'live timing differs from actual monotonic timestamps')
            results[arm]={'episodes':len(episodes),'actions':audit['frames'],'successes':audit['successes'],
                         'mean_return':statistics.mean(r['return'] for r in audit['outcomes'])}
        require(declared['actual_task_results']==results,'reported live task quality differs from actual rewards')
    require(type(declared['peak_cuda_allocated_mib']) in (int,float)
            and math.isfinite(declared['peak_cuda_allocated_mib']) and declared['peak_cuda_allocated_mib']>0,
            'invalid reported GPU allocator peak')
    warm=warm_trace['cache']
    final=trace['cache']
    delta={key:final[key]-warm[key] for key in ('lookups','hits','misses','evictions')}
    return {'schema_version':1,'phase':phase,'source_and_all_new_feature_head_clock_feedback_audits':True,
            'maximum_exact_head_logit_difference':0.,'measured_requests':planned,'warmup_requests':len(warmups),
            'warmup_cache':warm,'measured_cache_delta':delta,'final_cache':final,'metrics':metrics,
            'claim':'frozen task readout cache experiment; artificial repeated spectra, not real-world audio hit rates or real-time gameplay'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--readout',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=compare(args.run,args.readout,args.source)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
