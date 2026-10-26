"""Bounded warm-parent LoRA rehearsal; audio objective is identical across arms."""
import argparse
from datetime import datetime,timezone
import json
import importlib.metadata
from pathlib import Path
import time
from scripts.training.action_head import forward_action_logits
from scripts.training.audio_batch import group_training_rows
from scripts.training.audio_batch_audit import audit_export,load_checkpoint
from scripts.training.audio_batch_eval import validate_cohort
from scripts.training.audio_batch_train import image_processor_identity,sensor_payload,source_paths as previous_paths,write_status
from scripts.training.audio_cue_data import load_audio_dataset
from scripts.training.audio_replay import ARMS,copy_parent_trainables,examples_for_step,rgb_groups,optimize_examples
from scripts.training.audio_vision import prepare_audio
from scripts.training.bandit_train import checked_resets,trainable_fingerprint
from scripts.training.checkpoint import restore_checkpoint,save_checkpoint
from scripts.training.data import read_jsonl,require,sha256_file
from scripts.training.feedback import audit_feedback_episodes
from scripts.training.feedback_learning import student_row
from scripts.training.formatting import make_messages
from scripts.training.hf_backend import RGBBackend,environment,load_dataset
PARENT_COHORT='4058e462f2eb55dcfb2278447e65977a0d3a8251635c5efbfb471417ac0a3122'
RGB_MANIFEST='d58ddde6623581a86ef59502f1e7d32253cc4d4f6bc5a40a40b6b975f3b516d2'

def source_paths():
    return [Path(__file__),Path(__file__).with_name('audio_replay.py'),Path(previous_paths()[0]).with_name('audio_batch_audit.py'),Path(previous_paths()[0]).with_name('audio_batch_eval.py'),*previous_paths()]

def spectrogram_packages():
    return {name:importlib.metadata.version(name) for name in ('numpy','Pillow')}

def require_parent_dependencies(identity,parent):
    require(all(identity[key]==parent[key] for key in ('packages','image_processor','spectrogram_packages')),
            'parent inference/spectrogram dependencies or processor changed')


def rgb_evidence(backend,row,root):
    import hashlib
    batch,_,timing=backend.prepare(student_row(row),root,history_len=0)
    messages,_=make_messages(student_row(row),history_len=0)
    prompt=backend.processor.apply_chat_template(messages,tokenize=False,continue_final_message=True,add_generation_prompt=False)
    return batch,{'input_rgb_pixel_sha256':row['rgb']['pixel_sha256'],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                 'input_tokens':int(batch['input_ids'].shape[-1]),'image_count':1,'history_len':0,'prepare_ms':sum(timing.values())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','audio-dataset','rgb-dataset','parent','cohort','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int,choices=range(1,65),default=2);p.add_argument('--save-every',type=int,choices=range(1,65),default=25)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    status={'state':'loading','planned_total_updates':a.steps,'completed_total_updates':0,'active_sample':None,'started_utc':datetime.now(timezone.utc).isoformat()};write_status(a.output,status)
    try:
        require(sha256_file(a.cohort)==PARENT_COHORT,'fixed200update parent cohort changed')
        cohort=json.loads(a.cohort.read_text());validate_cohort(cohort)
        parent=load_checkpoint(a.parent)
        require(parent['step']==200 and parent['identity']==cohort['training']['paired-context']['identity'],'named paired200 parent state required')
        require(audit_export(a.parent,parent,base_snapshot=Path(a.model).resolve().name)==cohort['policies']['paired-context'],'parent export/state differs')
        audio,am,ad,ai=load_audio_dataset(a.audio_dataset);require(ad==parent['identity']['dataset'],'audioTRAIN corpus changed')
        groups=group_training_rows(audio['train'],list(read_jsonl(am['source_path'])))
        rgb,rm,rd=load_dataset(a.rgb_dataset);require(rd==RGB_MANIFEST and rm['source_kinds']==['recorded'],'fixed expert RGBTRAIN corpus changed')
        episodes=list(read_jsonl(rm['source_path']));audit_feedback_episodes(episodes,rm['data_root']);checked_resets(rgb['train'],episodes);rg=rgb_groups(rgb['train'])
        started=time.perf_counter();backend=RGBBackend(a.model,rank=8,learning_seed=20261101,deterministic=True);torch=backend.torch
        require(backend.identity['fingerprint']==cohort['model_fingerprint'],'actual base changed')
        copy_parent_trainables(backend.model,parent['trainables']);initial=trainable_fingerprint(backend.model)
        optimizer=torch.optim.AdamW([v for v in backend.model.parameters() if v.requires_grad],lr=1e-4,weight_decay=.01)
        require(not optimizer.state,'rehearsal must start fresh optimizer')
        identity={'schema_version':1,'model':backend.identity['fingerprint'],'audio_dataset':ad,'rgb_dataset':rd,'arm':a.arm,'seed':20261101,
                  'parent_cohort':PARENT_COHORT,'parent_state_sha256':sha256_file(a.parent/'state.pt'),'parent_adapter':cohort['policies']['paired-context'],
                  'initialization_fingerprint':initial,'parent_copy':'parameters_only24;freshoptimizer+newseed','rank':8,'alpha':16,'dropout':0.,
                  'target_modules':['q_proj','v_proj'],'dtype':'bfloat16','optimizer':'AdamW','lr':1e-4,'weight_decay':.01,'clip':1.,'temperature':2.,
                  'audio_terms':2,'rgb_terms':2 if a.arm=='rgb-replay' else 0,'each_ce_coefficient':.25,'frozen_final_updates':64,
                  'mask_dropout':'globalzero-based-update%8==7;bothaudio','sampling':'shuffle32sameRGBoppositetonepairs;RGBalternatingWS_AD;v1',
                  'inputs':'sensorallowlistonly;nohistory/target/reset/engine/reward','image_processor':image_processor_identity(backend.processor),
                  'spectrogram_packages':spectrogram_packages(),'packages':environment()['packages'],'determinism':backend.determinism,'code':{v.name:sha256_file(v) for v in source_paths()}}
        require_parent_dependencies(identity,parent['identity'])
        step=restore_checkpoint(a.resume,backend.model,optimizer,identity=identity) if a.resume else 0;require(step<a.steps,'resume must precede requested total');start_step=step
        config={'schema_version':1,'identity':identity,'model':backend.identity,'environment':environment(),'start_step':step,'total_steps':a.steps,
                'resume':str(a.resume.resolve()) if a.resume else None,'parent':str(a.parent.resolve()),'audio_dataset':str(a.audio_dataset.resolve()),'rgb_dataset':str(a.rgb_dataset.resolve()),
                'claim':'additional RGB objective at matched audio terms, extra labels/compute; own artificial task, not equal-cost method ranking'}
        (a.output/'config.json').write_text(json.dumps(config,indent=2));backend.model.train();status['state']='running';update_seconds=0.
        while step<a.steps:
            examples=examples_for_step(groups,rg,arm=a.arm,step=step,seed=20261101);backend.synchronize();began=time.perf_counter()
            def forward(e):
                row=e['row'];status['active_sample']=row['sample_id']
                if e['kind']=='audio':batch,evidence=prepare_audio(backend,sensor_payload(row,ai),am['data_root'],a.output,force_missing=e['missing'])
                else:batch,evidence=rgb_evidence(backend,row,rm['data_root'])
                raw,candidates=forward_action_logits(backend,batch);return raw[candidates].float(),evidence
            record=optimize_examples(backend.model,optimizer,examples,forward);backend.synchronize();elapsed=time.perf_counter()-began;update_seconds+=elapsed;step+=1
            record.update(step=step,mask_dropout=examples[0]['missing'],step_ms=elapsed*1000,peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
            with (a.output/'steps.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
            status.update(completed_total_updates=step,active_sample=None);write_status(a.output,status);print(json.dumps({k:v for k,v in record.items() if k!='examples'}),flush=True)
            if step%a.save_every==0 or step==a.steps:
                path=a.output/f'checkpoint-{step:06}';save_checkpoint(path,backend.model,optimizer,step=step,identity=identity);backend.model.save_pretrained(path/'adapter',safe_serialization=True)
        result={'completed_optimizer_steps':step,'start_step':start_step,'new_updates':step-start_step,'new_audio_label_uses':2*(step-start_step),
                'new_rgb_label_uses':2*(step-start_step) if a.arm=='rgb-replay' else 0,'elapsed_seconds':time.perf_counter()-started,'optimizer_updates_seconds':update_seconds,
                'checkpoint':f'checkpoint-{step:06}','peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
        (a.output/'result.json').write_text(json.dumps(result,indent=2));status['state']='completed';print(json.dumps(result),flush=True)
    except BaseException as error:status.update(state='failed',error_type=type(error).__name__);raise
    finally:status['finished_utc']=datetime.now(timezone.utc).isoformat();write_status(a.output,status)

if __name__=='__main__':main()
