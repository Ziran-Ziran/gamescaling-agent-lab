"""CPU audit of matched-batch SFT inputs/losses, checkpoints and exact resume."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.training.audio_batch import ARMS,batch_for_step,group_training_rows,target
from scripts.training.audio_batch_train import image_processor_identity,sensor_payload,source_paths
from scripts.training.audio_cue_data import load_audio_dataset
from scripts.training.audio_vision import prepare_audio
from scripts.training.data import read_jsonl,require,sha256_file
from scripts.training.hf_backend import adapter_identity,model_identity

MODEL_FINGERPRINT='ed679f4172207776e6aa8e63915c3645721f62a38aef7c1ce3fd82a96241c87c'
DATA_DIGESTS={'rgb':'19702bcd209526bab4793ffa4de9e89398116f7993dbea01aafa9705251dede5',
              'audio':'d99732d3c7827fb6721e30055da6fa5e491e544d4d8259f88ef1fba840324949'}


def audit_training_records(records,groups,*,arm,expected_inputs,start_step=0):
    import torch
    require(records,'nonempty actual training records required')
    exposure=Counter()
    labels={'present':Counter(),'missing':Counter()}
    errors=[]
    norms=[]
    for index,record in enumerate(records):
        step=start_step+index
        batch,missing=batch_for_step(groups,arm=arm,step=step,seed=20261031)
        require(record['step']==step+1 and record['mask_dropout'] is missing,
                'training globalstep/mask differs from frozen schedule')
        require(record['target_counts']=={'1':2,'3':2} and len(record['examples'])==4,
                'four balanced target examples required')
        for row,example in zip(batch,record['examples']):
            require(example['sample_id']==row['sample_id'] and example['target_class']==target(row),
                    'TRAIN sample/order/target differs')
            inputs={key:value for key,value in example['sensor_evidence'].items() if key!='prepare_ms'}
            require(inputs==expected_inputs(row,missing),'actual TRAIN sensor/mask/prompt differs from source')
            values=example['raw_class_logits']
            require(isinstance(values,list) and len(values)==5 and all(
                type(value) in (int,float) and math.isfinite(value) for value in values),'finite five-class logits required')
            recomputed=float(torch.nn.functional.cross_entropy(torch.tensor([values],dtype=torch.float32)/2,
                                                               torch.tensor([target(row)])))
            require(type(example['loss']) in (int,float) and math.isfinite(example['loss']),
                    'finite raw CE required')
            error=abs(recomputed-example['loss'])
            require(error<=2e-6,'raw training CE differs from saved class logits/target/temperature')
            errors.append(error)
            exposure[(row['sample_id'],missing)]+=1
            labels['missing' if missing else 'present'][str(target(row))]+=1
        require(abs(record['loss']-sum(example['loss'] for example in record['examples'])/4)<=1e-12,
                'reported mean CE differs from four actual examples')
        require(all(type(record[key]) in (int,float) and math.isfinite(record[key]) and record[key]>=0
                    for key in ('gradient_norm','step_ms','peak_cuda_allocated_mib')),
                'finite nonnegative gradient/cost required')
        norms.append(record['gradient_norm'])
    require(any(value>0 for value in norms),'no nonzero gradient evidence')
    return {'updates':len(records),'label_uses':len(records)*4,
            'missing_label_uses':sum(labels['missing'].values()),
            'label_counts':{key:dict(value) for key,value in labels.items()},
            'per_sample_exposure':[{'sample_id':key[0],'missing':key[1],'count':value}
                                   for key,value in sorted(exposure.items())],
            'maximum_ce_difference':max(errors),'ce_absolute_tolerance':2e-6,
            'all_gradients_finite':True,'nonzero_gradient_updates':sum(value>0 for value in norms)}


def load_checkpoint(directory):
    import torch
    root=Path(directory)
    manifest=json.loads((root/'manifest.json').read_text())
    require(sha256_file(root/'state.pt')==manifest['state_sha256'],'checkpoint content hash mismatch')
    state=torch.load(root/'state.pt',map_location='cpu',weights_only=True)
    require(state['step']==manifest['step'] and state['identity']==manifest['identity']
            and sorted(state['trainables'])==manifest['trainable_names'],'checkpoint manifest/state differs')
    require(len(state['trainables'])==24 and all(value.device.type=='cpu' and bool(torch.isfinite(value).all())
            for value in state['trainables'].values()),'24 finite actual trainable tensors required')
    require(len(state['optimizer']['state'])==24 and all(
        float(value['step'])==state['step'] for value in state['optimizer']['state'].values()),
        'optimizer per-parameter step differs from completed update count')
    return state


def audit_export(checkpoint,state,*,base_snapshot):
    from safetensors.torch import load_file
    import torch
    root=Path(checkpoint)/'adapter'
    before=adapter_identity(root)
    config=json.loads((root/'adapter_config.json').read_text())
    fixed={'r':8,'lora_alpha':16,'lora_dropout':0.,'bias':'none','task_type':'CAUSAL_LM',
           'peft_type':'LORA','inference_mode':True}
    require(all(config.get(key)==value for key,value in fixed.items())
            and set(config.get('target_modules',[]))=={'q_proj','v_proj'},'exported LoRA configuration differs')
    require(all(config.get(key) in (None,False) for key in (
        'fan_in_fan_out','use_dora','use_rslora','use_qalora','lora_bias','ensure_weight_tying','use_bdlora'))
        and all(config.get(key) is None for key in (
        'alora_invocation_tokens','arrow_config','auto_mapping','corda_config','eva_config','exclude_modules',
        'kasa_config','layer_replication','layers_pattern','layers_to_transform','lora_ga_config','megatron_config',
        'modules_to_save','monteclora_config','revision','target_parameters','trainable_token_indices','velora_config'))
        and all(config.get(key,{})=={} for key in ('rank_pattern','alpha_pattern','loftq_config')),
        'exported optional LoRA configuration differs')
    name=config.get('base_model_name_or_path','').replace('\\','/').rstrip('/').split('/')[-1]
    require(name==base_snapshot,'exported adapter base snapshot differs from audited actual model')
    exported=load_file(str(root/'adapter_model.safetensors'),device='cpu')
    expected={name.replace('.default.','.'):value for name,value in state['trainables'].items()}
    require(len(expected)==len(state['trainables']) and set(exported)==set(expected)
            and all(value.dtype==expected[name].dtype and value.shape==expected[name].shape
                    and torch.equal(value,expected[name]) for name,value in exported.items()),
            'exported adapter tensor keys/dtype/shape/values differ from audited checkpoint')
    require(adapter_identity(root)==before,'adapter changed during export audit')
    return before


def validate_identity(identity,arm):
    require(arm in ARMS and identity['model']==MODEL_FINGERPRINT and identity['dataset']==DATA_DIGESTS,
            'frozen actual model/data changed')
    fixed={'arm':arm,'seed':20261031,'rank':8,'alpha':16,'dropout':0.,'target_modules':['q_proj','v_proj'],
           'dtype':'bfloat16','optimizer':'AdamW','lr':1e-4,'weight_decay':.01,'clip':1.,
           'temperature':2.,'effective_batch':4,'microbatch':1,'frozen_final_updates':200,
           'sampling':'shared8context32row-block;lastcontextmissing;per-batch2A2D;v1',
           'mask_dropout':'globalzero-based-update%8==7;all4samples;sharedsamplemaskmultiset',
           'inputs':'currentRGB+causalSTFTimage+mask;nohistory/target/reset/engine/reward'}
    require(all(identity.get(key)==value for key,value in fixed.items()),'frozen optimizer/sampling/input protocol changed')
    require(identity['code']=={path.name:sha256_file(path) for path in source_paths()},'actual training source changed')


def read_segments(segments):
    require(segments,'training segment paths required')
    configs=[]
    records=[]
    previous=None
    for directory in segments:
        root=Path(directory).resolve()
        config=json.loads((root/'config.json').read_text())
        status=json.loads((root/'run-status.json').read_text())
        result=json.loads((root/'result.json').read_text())
        require(config['start_step']==len(records),'missing/overlapping training segment updates')
        require(config['resume']==(str(previous) if previous else None),'training resume chain changed')
        validate_identity(config['identity'],config['identity']['arm'])
        require(not configs or config['identity']==configs[0]['identity'],'training segment identity changed')
        require(status['state']=='completed' and status['active_sample'] is None and
                status['completed_total_updates']==status['planned_total_updates']==config['total_steps'],
                'incomplete actual training segment')
        rows=list(read_jsonl(root/'steps.jsonl'))
        require(len(rows)==config['total_steps']-config['start_step'] and
                result['completed_optimizer_steps']==config['total_steps'] and
                result['new_updates']==len(rows) and result['new_label_uses']==4*len(rows),
                'reported actual update/label counts differ')
        checkpoint=root/result['checkpoint']
        state=load_checkpoint(checkpoint)
        require(state['identity']==config['identity'] and state['step']==config['total_steps'],
                'actual saved checkpoint identity/step differs')
        previous=checkpoint
        configs.append(config)
        records.extend(rows)
    return configs,records,state


def audit_training(segments,*,model,dataset):
    configs,records,state=read_segments(segments)
    require(len(records)==200,'complete frozen200 updates required')
    identity=configs[0]['identity']
    require(model_identity(model)['fingerprint']==identity['model'],'CPU audit model files changed')
    data,manifest,digests,audio_index=load_audio_dataset(dataset)
    require(digests==identity['dataset'],'CPU audit actual corpus differs')
    groups=group_training_rows(data['train'],list(read_jsonl(manifest['source_path'])))
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    require(image_processor_identity(processor)==identity['image_processor'],'CPU audit image processor/torchvision changed')
    backend=SimpleNamespace(processor=processor,device='cpu',synchronize=lambda:None)
    cache={}
    def expected(row,missing):
        key=(row['sample_id'],missing)
        if key not in cache:
            _,evidence=prepare_audio(backend,sensor_payload(row,audio_index),manifest['data_root'],Path('.'),
                                     force_missing=missing)
            cache[key]={key:value for key,value in evidence.items() if key!='prepare_ms'}
        return cache[key]
    report=audit_training_records(records,groups,arm=identity['arm'],expected_inputs=expected)
    last=Path(segments[-1])
    result=json.loads((last/'result.json').read_text())
    exported=audit_export(last/result['checkpoint'],state,base_snapshot=configs[0]['model']['snapshot'])
    return {'schema_version':1,'identity':identity,'segments':len(segments),'actual_training':report,
            'checkpoint_step':state['step'],'trainable_tensors':len(state['trainables']),
            'audited_adapter_identity':exported,'export_vs_checkpoint_all24_exact':True,
            'claim':'actual200update original-model SFT audit; no test metric or native-audio/general ability claim'}


def exact(left,right):
    import torch
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and set(left)==set(right) and all(exact(left[key],right[key]) for key in left)
    if isinstance(left,(list,tuple)):
        return type(left) is type(right) and len(left)==len(right) and all(exact(a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left==right


def compare_resume(resumed,continuous):
    _,a,state_a=read_segments(resumed)
    _,b,state_b=read_segments([continuous])
    require(len(a)==len(b)==4 and exact(state_a,state_b),'GPU2+2 checkpoint parameters/optimizer/RNG differ from continuous4')
    def effective(records):
        result=[]
        for record in records:
            row={key:value for key,value in record.items() if key not in ('step_ms','peak_cuda_allocated_mib')}
            row['examples']=[{**example,'sensor_evidence':{key:value for key,value in example['sensor_evidence'].items()
                                                       if key!='prepare_ms'}} for example in record['examples']]
            result.append(row)
        return result
    require(effective(a)==effective(b),'GPU2+2 actual sample/input/logit/loss/gradient sequence differs')
    return {'steps':4,'identity_equal':True,'trainable_tensors':24,'all_trainables_bitwise_equal':True,
            'max_parameter_abs_difference':0.,'optimizer_and_all_rng_bitwise_equal':True,
            'actual_inputs_logits_losses_gradients_equal':True,
            'claim':'actualGPU2+2vs4;first4updates have no scheduled missing masks; control4extra updates excluded from main budget'}


def freeze_cohort(paired,unpaired,*,model,dataset):
    from scripts.training.audio_batch_eval import validate_cohort
    from scripts.training.retention_compare import ADAPTER_IDENTITIES
    training={}
    policies={'original':None,'legacy128':ADAPTER_IDENTITIES['audio-lora128']}
    for arm,segments in (('paired-context',paired),('balanced-unpaired',unpaired)):
        report=audit_training(segments,model=model,dataset=dataset)
        require(report['identity']['arm']==arm,'cohort arm differs from actual training')
        policies[arm]=report['audited_adapter_identity']
        training[arm]=report
    cohort={'schema_version':1,'model_fingerprint':MODEL_FINGERPRINT,'policies':policies,'training':training,
            'claim':'all fixed final artifacts audited before new metrics; original/legacy/newtwo grouping arms'}
    validate_cohort(cohort)
    return cohort


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments',nargs='+',type=Path)
    parser.add_argument('--paired-segments',nargs='+',type=Path)
    parser.add_argument('--unpaired-segments',nargs='+',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--continuous',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.paired_segments:
        require(args.unpaired_segments and args.model and args.dataset,'cohort requires both arms/model/dataset')
        report=freeze_cohort(args.paired_segments,args.unpaired_segments,model=args.model,dataset=args.dataset)
    elif args.continuous:
        report=compare_resume(args.segments,args.continuous)
    else:
        require(args.segments and args.model and args.dataset,'training audit requires segments/model/dataset')
        report=audit_training(args.segments,model=args.model,dataset=args.dataset)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
