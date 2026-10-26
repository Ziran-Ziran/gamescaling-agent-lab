"""Task-specific five-way action decoding with fixed RGB model/adapter weights.

Only whole-window W/A/S/D/no-input actions are expressible. This is not a
replacement for the official mixed-key/mouse protocol or a learned format head.
"""
import math
import time

from controller.control_core import SafeAction, to_action_text
from scripts.training.data import require
from scripts.training.hf_backend import RGBBackend


LABELS = ('W', 'A', 'S', 'D', 'noop')
FORCED_PREFIX = '\n\n<|action_start|>0 0 0 ;'
EXTENSIONS = (' W', ' A', ' S', ' D', ' ;')


def first_key_contract(tokenizer, tail_ids):
    """Fail closed if real tokenizer changes a boundary or any class is ambiguous."""
    text = tokenizer.decode(tail_ids, skip_special_tokens=False)
    require(tokenizer.encode(text, add_special_tokens=False) == tail_ids, 'tail tokenizer roundtrip failed')
    extended = tokenizer.encode(text + FORCED_PREFIX, add_special_tokens=False)
    require(extended[:len(tail_ids)] == tail_ids and len(extended) > len(tail_ids),
            'forced action prefix must exactly extend prompt tokens')
    candidates = []
    for suffix in EXTENSIONS:
        ids = tokenizer.encode(text + FORCED_PREFIX + suffix, add_special_tokens=False)
        require(ids[:len(extended)] == extended and len(ids) == len(extended) + 1,
                'each action class must be one exact next token')
        candidates.append(ids[-1])
    require(len(set(candidates)) == len(LABELS), 'action class tokens must be unique')
    return extended[len(tail_ids):], candidates


def choose_class(logits):
    require(len(logits) == len(LABELS) and all(type(v) in (int, float) and math.isfinite(v) for v in logits),
            'five finite class logits required')
    selected = max(range(len(logits)), key=logits.__getitem__)
    normalizer = sum(math.exp(v - max(logits)) for v in logits)
    return LABELS[selected], {'raw_class_logits': dict(zip(LABELS, logits)),
                             'restricted_softmax_not_calibrated': {
                                 key: math.exp(v - max(logits)) / normalizer for key, v in zip(LABELS, logits)}}


def template_action(label):
    require(label in LABELS, 'unsupported template action')
    return SafeAction((0., 0., 0.), ((label,) if label != 'noop' else (),) * 6)


def forward_action_logits(backend, batch):
    """Shared prefix/forward contract; caller chooses gradient or inference context."""
    torch = backend.torch
    require(set(batch) <= {'input_ids', 'attention_mask', 'mm_token_type_ids',
                           'pixel_values', 'image_grid_thw'}, 'unsupported action-head model inputs')
    require(batch['input_ids'].shape[0] == 1, 'action head requires batch one')
    tail = batch['input_ids'][0, -64:].tolist()
    prefix, candidates = first_key_contract(backend.processor.tokenizer, tail)
    prefix_tensor = torch.tensor([prefix], device=batch['input_ids'].device,
                                 dtype=batch['input_ids'].dtype)
    extended = dict(batch)
    extended['input_ids'] = torch.cat((batch['input_ids'], prefix_tensor), dim=1)
    for key in ('attention_mask', 'mm_token_type_ids'):
        if key in batch:
            extra = torch.ones_like(prefix_tensor) if key == 'attention_mask' else torch.zeros_like(prefix_tensor)
            extended[key] = torch.cat((batch[key], extra.to(batch[key].dtype)), dim=1)
    logits = backend.model(**extended, use_cache=False, logits_to_keep=1).logits[0, -1]
    return logits, candidates


class FirstKeyBackend(RGBBackend):
    def generate(self, batch, *, max_new_tokens=64):
        # There is one forward/prediction and no autoregressive generation. max_new_tokens
        # is accepted for the shared policy interface; the effective prediction count is1.
        self.synchronize()
        started = time.perf_counter()
        torch = self.torch
        with torch.inference_mode():
            logits, candidates = forward_action_logits(self, batch)
            values = logits[candidates].float().tolist()
            top_token = int(logits.argmax())
        label, details = choose_class(values)
        completion = to_action_text(template_action(label))
        self.last_head_record = {**details, 'selected_class': label,
                                 'class_token_ids': dict(zip(LABELS, candidates)),
                                 'unrestricted_top_token_id': top_token,
                                 'unrestricted_top_is_allowed': top_token in candidates,
                                 'forced_prefix': FORCED_PREFIX, 'predicted_tokens': 1,
                                 'text_source': 'safe whole-window template, not generated action text'}
        self.synchronize()
        # A serialized action is not30generated tokens. The policy records0generated,
        # 1predicted token and this explicit source description.
        return completion, 0, (time.perf_counter() - started) * 1000
