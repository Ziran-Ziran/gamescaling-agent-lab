"""Matched exposures and missing masks; only present minibatch grouping differs."""
import random

from scripts.training.data import require
from scripts.training.feedback_learning import class_from_action

ARMS=('paired-context','balanced-unpaired')


def target(row):
    return class_from_action(row['action'])


def group_training_rows(rows,episodes):
    require(len(rows)==64 and len({r['sample_id'] for r in rows})==64,
            'exactly64 unique frozen TRAIN observations required')
    index={episode['episode_id']:episode for episode in episodes}
    require(len(index)==len(episodes),'duplicate source episode')
    groups={}
    for row in rows:
        require(row['game_id']=='audio-cue-reach-v1' and row['frame_id']==0 and not row['history'],
                'only recorded reach initial TRAIN sensors supported')
        spec=index[row['episode_id']]['cue_reset']
        require(spec['variant']=='reach' and type(spec['context']) is int and spec['context']>=0
                and all(type(spec[key]) is int and spec[key] in (0,1) for key in ('colors','tone')),
                'invalid counterfactual context')
        key=(spec['colors'],spec['tone'])
        group=groups.setdefault(spec['context'],{})
        require(key not in group,'duplicate counterfactual row')
        group[key]=row
    require(len(groups)==16,'exactly16 complete TRAIN contexts required')
    result=[]
    for context,group in sorted(groups.items()):
        require(set(group)=={(colors,tone) for colors in (0,1) for tone in (0,1)},
                'complete two-color by two-tone group required')
        for colors in (0,1):
            a,b=group[colors,0],group[colors,1]
            require(a['rgb']['pixel_sha256']==b['rgb']['pixel_sha256'],
                    'opposite tones require identical initial RGB')
            require({target(a),target(b)}=={1,3},'opposite tones require opposite A/D labels')
        for tone in (0,1):
            a,b=group[0,tone],group[1,tone]
            require(a['rgb']['pixel_sha256']!=b['rgb']['pixel_sha256'] and {target(a),target(b)}=={1,3},
                    'mirrored RGB requires opposite A/D labels at fixed tone')
        result.append(tuple(group[key] for key in sorted(group)))
    return tuple(result)


def batch_for_step(groups,*,arm,step,seed):
    require(arm in ARMS and type(step) is int and step>=0 and type(seed) is int and seed>=0,
            'invalid frozen grouping arm/globalstep/seed')
    require(len(groups)==16 and all(len(group)==4 for group in groups),'complete64-row groups required')
    epoch,within=divmod(step,16)
    half,position=divmod(within,8)
    order=list(range(16))
    random.Random(seed+epoch).shuffle(order)
    block=[groups[index] for index in order[half*8:(half+1)*8]]
    missing=position==7
    if arm=='paired-context' or missing:
        batch=list(block[position])
    else:
        present=[row for group in block[:7] for row in group]
        buckets={label:[row for row in present if target(row)==label] for label in (1,3)}
        rng=random.Random(seed+1_000_003*epoch+97*half+17)
        for label in (1,3):
            require(len(buckets[label])==14,'shared present block must balance A/D')
            rng.shuffle(buckets[label])
        batch=[*buckets[1][position*2:position*2+2],*buckets[3][position*2:position*2+2]]
    batch.sort(key=lambda row:row['sample_id'])
    require(sorted(target(row) for row in batch)==[1,1,3,3],'minibatch must contain exactly2A2D')
    return batch,missing


def optimize_microbatch(model,optimizer,rows,forward,*,temperature=2.,clip=1.):
    """Four immediate backwards; abort a partial microbatch before stepping."""
    import torch
    from scripts.training.feedback_learning import supervised_loss
    require(len(rows)==4 and sorted(target(row) for row in rows)==[1,1,3,3],
            'four balanced TRAIN examples required')
    require(type(clip) in (float,int) and 0<clip<=1e6,'invalid gradient clip')
    parameters=[value for value in model.parameters() if value.requires_grad]
    require(parameters,'trainable parameters required')
    optimizer.zero_grad(set_to_none=True)
    examples=[]
    try:
        for row in rows:
            logits,evidence=forward(row)
            loss=supervised_loss(logits,target(row),temperature=temperature)
            require(bool(torch.isfinite(loss)),'nonfinite microbatch loss')
            (loss/4).backward()
            examples.append({'sample_id':row['sample_id'],'target_class':target(row),
                             'raw_class_logits':logits.detach().cpu().tolist(),
                             'loss':float(loss.detach()),'sensor_evidence':evidence})
            del logits,loss
        norm=torch.nn.utils.clip_grad_norm_(parameters,clip)
        require(bool(torch.isfinite(norm)),'nonfinite accumulated gradient')
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {'loss':sum(row['loss'] for row in examples)/4,'gradient_norm':float(norm),
            'examples':examples}
