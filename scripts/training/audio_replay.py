"""Matched audio objective with an explicitly additional RGB rehearsal target."""
import random
from scripts.training.audio_batch import target
from scripts.training.data import require
from scripts.training.feedback_learning import class_from_action
ARMS=('audio-only-control','rgb-replay')

def rgb_groups(rows):
    require(len(rows)==64 and len({r['sample_id'] for r in rows})==64,'64 distinct RGB TRAIN rows required')
    groups={label:[] for label in range(4)}
    for row in rows:
        require(row['game_id']=='headless-reach-v1' and row['frame_id']==0 and not row['history'],'RGB reach initial TRAIN only')
        label=class_from_action(row['action'])
        require(label in groups,'RGB replay requires W/A/S/D')
        groups[label].append(row)
    require(all(len(v)==16 for v in groups.values()),'RGB TRAIN must have16 rows/class')
    return {label:tuple(sorted(values,key=lambda r:r['sample_id'])) for label,values in groups.items()}

def examples_for_step(groups,rgb,*,arm,step,seed):
    require(arm in ARMS and type(step) is int and step>=0 and type(seed) is int and seed>=0,'invalid replay arm/globalstep/seed')
    require(len(groups)==16 and all(len(g)==4 for g in groups),'complete64audio TRAIN groups required')
    pairs=[g[offset:offset+2] for g in groups for offset in (0,2)]
    epoch,within=divmod(step,32);order=list(range(32));random.Random(seed+epoch).shuffle(order)
    rows=pairs[order[within]]
    require(sorted(target(r) for r in rows)==[1,3] and rows[0]['rgb']['pixel_sha256']==rows[1]['rgb']['pixel_sha256'],'audio pair must share RGB/opposite target')
    result=[{'kind':'audio','row':row,'target_class':target(row),'missing':step%8==7} for row in rows]
    if arm=='rgb-replay':
        for label in ((0,2) if step%2==0 else (1,3)):
            epoch,index=divmod(step//2,16);order=list(range(16));random.Random(seed+900001*label+epoch).shuffle(order)
            result.append({'kind':'rgb','row':rgb[label][order[index]],'target_class':label,'missing':False})
    return result

def copy_parent_trainables(model,parent):
    import torch
    params={name:value for name,value in model.named_parameters() if value.requires_grad}
    require(len(params)==24 and set(params)==set(parent),'exact24 named parent trainables required')
    for name,value in params.items():
        source=parent[name]
        require(source.shape==value.shape and source.dtype==value.dtype and bool(torch.isfinite(source).all()),'parent shape/dtype/finite mismatch')
    with torch.no_grad():
        for name,value in params.items():value.copy_(parent[name].to(value.device))

def optimize_examples(model,optimizer,examples,forward,*,temperature=2.,clip=1.):
    import torch
    from scripts.training.feedback_learning import supervised_loss
    require(len(examples) in (2,4) and [e['kind'] for e in examples[:2]]==['audio','audio']
            and sorted(e['target_class'] for e in examples[:2])==[1,3]
            and all(e['kind']=='rgb' for e in examples[2:]),'two common audio terms, optional two RGB terms required')
    require(type(clip) in (int,float) and 0<clip<=1e6,'positive finite gradient clip required')
    parameters=[p for p in model.parameters() if p.requires_grad];require(parameters,'trainables required')
    optimizer.zero_grad(set_to_none=True);records=[]
    try:
        for example in examples:
            logits,evidence=forward(example);loss=supervised_loss(logits,example['target_class'],temperature=temperature)
            require(bool(torch.isfinite(loss)),'nonfinite replay CE')
            (loss/4).backward()
            records.append({'kind':example['kind'],'sample_id':example['row']['sample_id'],'target_class':example['target_class'],
                            'raw_class_logits':logits.detach().cpu().tolist(),'loss':float(loss.detach()),'sensor_evidence':evidence})
            del logits,loss
        norm=torch.nn.utils.clip_grad_norm_(parameters,clip);require(bool(torch.isfinite(norm)),'nonfinite replay gradient')
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True);raise
    audio=[r['loss'] for r in records if r['kind']=='audio'];rgb=[r['loss'] for r in records if r['kind']=='rgb']
    return {'objective':sum(r['loss'] for r in records)/4,'audio_mean_ce':sum(audio)/len(audio),
            'rgb_mean_ce':sum(rgb)/len(rgb) if rgb else None,'gradient_norm':float(norm),
            'gradient_clip_factor':min(1.,clip/(float(norm)+1e-6)),'examples':records}
