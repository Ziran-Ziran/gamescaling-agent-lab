"""Portable action validation and interruptible keyboard execution."""
from dataclasses import dataclass
import math
import numbers
import time

ALLOWED = ('W', 'A', 'S', 'D')

@dataclass(frozen=True)
class SafeAction:
    mouse: tuple[float, float, float]
    steps: tuple[tuple[str, ...], ...]

def sanitize_action(mouse, steps, *, parsed=True, truncated=False, allow_mouse=False, max_mouse=15.0):
    """Validate structure, discard nonmovement keys, and bound camera motion."""
    if parsed is not True or truncated is not False:
        raise ValueError('Action must be fully parsed and untruncated')
    def finite(v):
        return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)
    if not finite(max_mouse) or not 0 <= max_mouse <= 15:
        raise ValueError('Invalid mouse limit')
    if not isinstance(mouse, (list, tuple)) or len(mouse) != 3 or not all(finite(v) for v in mouse):
        raise ValueError('Mouse requires three finite numbers')
    if not isinstance(steps, (list, tuple)) or len(steps) != 6:
        raise ValueError('Exactly six steps required')
    cleaned=[]
    for step in steps:
        if not isinstance(step, list) or len(step)>4 or not all(isinstance(k,str) for k in step):
            raise ValueError('Each step must be a list of up to four key strings')
        cleaned.append(tuple(dict.fromkeys(k for k in step if k in ALLOWED)))
    xy=tuple(float(max(-max_mouse,min(max_mouse,v))) for v in mouse[:2]) if allow_mouse else (0.,0.)
    return SafeAction((*xy,0.),tuple(cleaned))

def _validate(action):
    if not isinstance(action, SafeAction) or not isinstance(action.steps, tuple) or not all(isinstance(s,tuple) for s in action.steps):
        raise ValueError('Expected immutable SafeAction')
    clean=sanitize_action(action.mouse,[list(s) for s in action.steps],allow_mouse=True)
    if action != clean:
        raise ValueError('Unsafe directly constructed action')
    return clean

def to_action_text(action):
    """Serialize the exact sanitized action using GameScaling tags."""
    action=_validate(action)
    groups=[' '.join(str(int(round(v))) for v in action.mouse)]
    groups.extend(' '.join(s) for s in action.steps)
    return '<|action_start|>' + ' ; '.join(groups) + '<|action_end|>'

def execute_chunk(action, sink, should_stop, *, monotonic=time.monotonic, sleep=time.sleep, duration=.2):
    """Execute six steps; sink mouse units are normalized per-mille XY."""
    held=set()
    try:
        action=_validate(action)
        if isinstance(duration,bool) or not isinstance(duration,numbers.Real) or not math.isfinite(duration) or duration<=0:
            raise ValueError('Duration must be positive and finite')
        if should_stop(): return False
        start=monotonic()
        if action.mouse[:2] != (0.,0.): sink.move_relative(*action.mouse[:2])
        for index, step in enumerate(action.steps):
            if should_stop(): return False
            deadline=start+duration*(index+1)/6
            if monotonic() >= deadline:
                return False  # Incomplete actions must not be committed to history.
            wanted=set(step)
            for k in ALLOWED:
                if k in held and k not in wanted: sink.key_up(k)
            for k in step:
                if k not in held: sink.key_down(k)
            held=wanted
            if monotonic() > deadline:
                return False
            deadline=start+duration*(index+1)/6
            while True:
                if should_stop(): return False
                remaining=deadline-monotonic()
                if remaining<=0: break
                sleep(min(.01,remaining))
        return True
    finally:
        # Attempt every release even if one backend operation fails.
        error=None
        for key in ALLOWED:
            try: sink.key_up(key)
            except BaseException as exc:
                if error is None: error=exc
        if error is not None: raise error
