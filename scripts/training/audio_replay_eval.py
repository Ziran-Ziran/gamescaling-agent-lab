"""Fresh frozen control suites for the matched original-model audio LoRA study."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

from controller.control_core import to_action_text
from scripts.training.action_head import choose_class,forward_action_logits,template_action
from scripts.training.audio_batch_train import image_processor_identity
from scripts.training.audio_replay_train import spectrogram_packages
from scripts.training.audio_cue import AudioCueGame
from scripts.training.audio_cue_data import audit_audio_episodes,record_episode
from scripts.training.audio_vision import prepare_audio
from scripts.training.audio_vision_eval import condition_input,_status
from scripts.training.data import read_jsonl,require,sha256_file
from scripts.training.hf_backend import RGBBackend,adapter_identity,environment
from scripts.training.retention_compare import ADAPTER_IDENTITIES,MODEL_FINGERPRINT

from scripts.training.audio_replay_protocol import AUDIO_POLICIES as POLICIES,CONDITIONS,frozen_schedule,validate_cohort
from scripts.training.audio_batch_eval import summarize,source_paths as previous_paths


def source_paths():
    names=('audio_replay_eval.py','audio_replay_train.py','audio_replay.py','audio_replay_protocol.py')
    return [*(Path(__file__).with_name(name) for name in names),*previous_paths()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--policy',choices=POLICIES,required=True)
    parser.add_argument('--suite',choices=('transfer','iid'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cohort=json.loads(args.cohort.read_text())
    validate_cohort(cohort)
    require(adapter_identity(args.adapter)==cohort['policies'][args.policy],'requested adapter differs from frozen cohort')
    args.output.mkdir(parents=True,exist_ok=False)
    schedule=frozen_schedule(args.suite)
    status={'state':'loading','planned_episodes':40,'completed_episodes':0,'failed_episodes':0,
            'active_episode':None,'started_utc':datetime.now(timezone.utc).isoformat()}
    _status(args.output,status)
    try:
        backend=RGBBackend(args.model,adapter=args.adapter)
        require(backend.identity['fingerprint']==cohort['model_fingerprint'],'actual base model differs')
        require(environment()['packages']==cohort['training']['audio-only-control']['identity']['packages']
                and spectrogram_packages()==cohort['training']['audio-only-control']['identity']['spectrogram_packages'],
                'evaluation model/adapter dependency versions changed from training')
        processor=image_processor_identity(backend.processor)
        require(processor==cohort['training']['audio-only-control']['identity']['image_processor'],
                'evaluation image processor/torchvision changed from training')
        config={'schema_version':1,'model':backend.identity,'adapter':backend.adapter_identity,
                'environment':environment(),'cohort_sha256':sha256_file(args.cohort),'policy':args.policy,
                'suite':args.suite,'conditions':list(CONDITIONS),'schedule':schedule,
                'image_processor':processor,
                'spectrogram_packages':spectrogram_packages(),
                'decoder':'argmax5class first-key safe template,0generated/1predicted','history_len':0,
                'learning_during_evaluation':False,'source_code_sha256':{p.name:sha256_file(p) for p in source_paths()}}
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
        reports={}
        status['state']='running'
        for condition in CONDITIONS:
            root=args.output/condition
            root.mkdir()
            episodes=[]
            for spec in schedule:
                game=AudioCueGame(**spec)
                status['active_episode']=f'{condition}/{game.base.episode_id}'
                _status(args.output,status)
                def policy(row):
                    changed,flags,intervention=condition_input(row,root,condition,spec)
                    batch,inputs=prepare_audio(backend,changed,root,root,**flags)
                    backend.synchronize()
                    began=time.perf_counter()
                    with backend.torch.inference_mode():
                        raw,candidates=forward_action_logits(backend,batch)
                        values=raw[candidates].float().tolist()
                    backend.synchronize()
                    inference_ms=(time.perf_counter()-began)*1000
                    label,details=choose_class(values)
                    return {'completion':to_action_text(template_action(label)),
                            'evidence':{'condition':condition,'sensor_intervention':intervention,'inputs':inputs,
                                        'class_record':details,'selected_class':label,'generated_tokens':0,
                                        'predicted_tokens':1,'inference_ms':inference_ms}}
                episode=record_episode(game,root,policy=policy)
                episodes.append(episode)
                with (root/'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(episode)+'\n')
                status.update(completed_episodes=status['completed_episodes']+1,active_episode=None)
                _status(args.output,status)
            audit=audit_audio_episodes(episodes,root)
            reports[condition]=summarize(episodes,audit,root)
            print(json.dumps({'condition':condition,**reports[condition]}),flush=True)
        summary={'results':reports,'peak_cuda_allocated_mib':backend.torch.cuda.max_memory_allocated()/1024**2,
                 'finished_utc':datetime.now(timezone.utc).isoformat(),
                 'claim':'new fixed own virtual-sensor controls; not native audio/commercial/general capability'}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        status['state']='completed'
    except BaseException as error:
        status.update(state='failed',error_type=type(error).__name__,
                      failed_episodes=int(status['active_episode'] is not None))
        raise
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat()
        _status(args.output,status)


if __name__=='__main__':
    main()
