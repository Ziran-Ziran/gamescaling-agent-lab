"""Explicit one-step reward-learning mathematics; no teacher in bandit objective."""
import math
import random

from scripts.training.action_head import LABELS, template_action
from scripts.training.data import require
from scripts.training.metrics import action_signature
from controller.control_core import to_action_text


def class_from_action(action):
    signature = action_signature(action)
    for index, label in enumerate(LABELS):
        if signature is not None and signature == action_signature(to_action_text(template_action(label))):
            return index
    raise ValueError('expert action must be an exact whole-window five-class action')


def student_row(row):
    """An allowlist, not a teacher-bearing training record passed to the model."""
    return {'system_prompt': row['system_prompt'], 'rgb': dict(row['rgb']), 'history': []}


def index_for_step(size, step, seed):
    require(type(size) is int and size > 0 and type(step) is int and step >= 0, 'invalid data size/step')
    indices = list(range(size))
    random.Random(seed + step // size).shuffle(indices)
    return indices[step % size]


def _log_prob(logits, selected, temperature):
    import torch
    require(logits.ndim == 1 and len(logits) == len(LABELS) and bool(torch.isfinite(logits).all()),
            'five finite model class logits required')
    require(type(selected) is int and 0 <= selected < len(LABELS), 'invalid selected class')
    require(type(temperature) in (int, float) and math.isfinite(temperature) and .1 <= temperature <= 10.,
            'temperature must be finite in [.1,10]')
    return torch.nn.functional.log_softmax(logits / temperature, dim=-1)[selected]


def supervised_loss(logits, teacher_class, *, temperature=2.):
    return -_log_prob(logits, teacher_class, temperature)


def bandit_loss(logits, sampled_class, actual_reward, *, temperature=2., baseline=.19):
    # Python scalar feedback cannot carry a learned reward graph or hidden teacher target.
    require(type(actual_reward) in (float, int) and math.isfinite(actual_reward)
            and -1.01 <= actual_reward <= .99, 'detached finite environment reward required')
    require(type(baseline) in (float, int) and math.isfinite(baseline) and -1.01 <= baseline <= .99,
            'constant baseline must be finite in reward range')
    return -(actual_reward - baseline) * _log_prob(logits, sampled_class, temperature)
