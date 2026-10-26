"""Small fully rendered procedural task variants, without any desktop input.

This is a point-agent toy environment. Physics advances during a 200ms action
window and pauses during inference. It does not emulate a commercial game.
"""
from collections import deque
import hashlib
import math
import random
import time

from controller.control_core import SafeAction, to_action_text
from scripts.training.data import require, timestamp
from scripts.training.feedback import REWARD_VERSION, content_hash, reward_from_events, validate_feedback


GAME_IDS = {'reach': 'headless-reach-v1', 'hazard': 'headless-hazard-v1', 'keydoor': 'headless-keydoor-v1'}
DIRECTIONS = (('W', 0, -1), ('A', -1, 0), ('S', 0, 1), ('D', 1, 0))


class HeadlessGame:
    def __init__(self, *, variant='reach', seed=0, max_steps=8, layout=None,
                 clock_ns=time.perf_counter_ns, sleep=time.sleep):
        require(variant in GAME_IDS and type(seed) is int and seed >= 0, 'invalid game/seed')
        require(type(max_steps) is int and 1 <= max_steps <= 32, 'invalid horizon')
        self.variant, self.seed, self.max_steps = variant, seed, max_steps
        self.clock_ns, self.sleep = clock_ns, sleep
        self.episode_id = f'{GAME_IDS[variant]}-seed-{seed}'
        self.clock_id = self.episode_id + '-perf-counter-ns'
        self.size, self.step_count, self.done, self.reason = 7, 0, False, None
        self.player = [3.5, 3.5]
        _, dx, dy = DIRECTIONS[seed % len(DIRECTIONS)]
        self.goal = (3 + dx, 3 + dy)
        self.walls = {(x, y) for x in range(7) for y in range(7) if x in (0, 6) or y in (0, 6)}
        self.hazards = {(3 - dx, 3 - dy)} if variant == 'hazard' else set()
        self.key = (3, 2) if dx else (2, 3)
        self.door = self.goal if variant == 'keydoor' else None
        if variant != 'keydoor':
            self.key = None
        self.has_key = False
        if layout:
            if 'player' in layout:
                self.player = [v + .5 for v in layout['player']]
            for name in ('goal', 'key', 'door'):
                if name in layout:
                    setattr(self, name, tuple(layout[name]) if layout[name] is not None else None)
            for name in ('walls', 'hazards'):
                if name in layout:
                    if name == 'walls':
                        self.walls.update(set(layout[name]))
                    else:
                        self.hazards = set(layout[name])
        positions = [self.goal, self.cell(), *self.hazards, *self.walls]
        positions += [p for p in (self.key, self.door) if p is not None]
        require(all(len(p) == 2 and all(type(v) is int and 0 <= v < 7 for v in p) for p in positions),
                'layout requires in-bounds integer cells')
        require(self.goal not in self.walls and self.cell() not in self.walls
                and self.goal not in self.hazards and self.cell() not in self.hazards,
                'start/goal cannot be wall or hazard')
        if variant == 'keydoor':
            require(self.key is not None and self.door is not None
                    and self.key not in self.walls and self.key not in self.hazards
                    and self.key not in (self.cell(), self.goal, self.door)
                    and self.door not in self.walls and self.door not in self.hazards
                    and self.door == self.goal and self.door != self.cell(), 'invalid key/door layout')
        else:
            require(self.key is None and self.door is None, 'key/door require the keydoor rule variant')
        rng = random.Random(seed)
        self.palette = {'floor': [rng.randrange(175, 215) for _ in range(3)],
                        'wall': [rng.randrange(35, 70) for _ in range(3)]}

    def cell(self, point=None):
        return tuple(math.floor(v + 1e-9) for v in (point if point is not None else self.player))

    def snapshot(self):
        """Privileged engine truth for expert/ledger only. Never feed to the actor."""
        return {'episode_id': self.episode_id, 'clock_id': self.clock_id, 'game_id': GAME_IDS[self.variant],
                'game_build': 'procedural-point-grid-v1', 'variant': self.variant,
                'player': [round(v, 10) for v in self.player], 'goal': list(self.goal),
                'walls': [list(p) for p in sorted(self.walls)],
                'hazards': [list(p) for p in sorted(self.hazards)],
                'key': list(self.key) if self.key else None, 'door': list(self.door) if self.door else None,
                'has_key': self.has_key, 'step_count': self.step_count, 'max_steps': self.max_steps,
                'done': self.done, 'terminal_reason': self.reason, 'palette': self.palette}

    def render(self):
        from PIL import Image, ImageDraw
        image = Image.new('RGB', (256, 256), tuple(self.palette['floor']))
        draw = ImageDraw.Draw(image)
        scale, origin = 32, 16

        def rect(cell, color, inset=1):
            x, y = cell
            draw.rectangle((origin + x * scale + inset, origin + y * scale + inset,
                            origin + (x + 1) * scale - inset, origin + (y + 1) * scale - inset), fill=color)

        for cell in self.walls:
            rect(cell, tuple(self.palette['wall']), 0)
        for cell in self.hazards:
            rect(cell, (220, 25, 25))
        rect(self.goal, (20, 200, 50), 3)
        if self.door and not self.has_key:
            rect(self.door, (175, 40, 180), 7)  # Green edge identifies the goal behind the door.
        if self.key and not self.has_key:
            rect(self.key, (235, 200, 20), 10)
        x, y = [origin + v * scale for v in self.player]
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(25, 60, 230))
        draw.rectangle((2, 2, 12, 12), fill=(235, 200, 20) if self.has_key else (90, 90, 90))
        for i in range(self.max_steps):
            x = 16 + i * 224 / self.max_steps
            draw.rectangle((x, 244, x + 224 / self.max_steps - 2, 252),
                           fill=(40, 130, 60) if i >= self.step_count else (110, 110, 110))
        return image

    @staticmethod
    def _snapshot_view(state):
        """Detached world for rendering and deterministic physical replay only."""
        view = object.__new__(HeadlessGame)
        for name in ('palette', 'has_key', 'max_steps', 'step_count', 'episode_id', 'clock_id', 'variant', 'done'):
            setattr(view, name, state[name])
        view.player = list(state['player'])
        view.reason = state['terminal_reason']
        for name in ('goal', 'key', 'door'):
            value = state[name]
            setattr(view, name, tuple(value) if value is not None else None)
        view.walls = set(map(tuple, state['walls']))
        view.hazards = set(map(tuple, state['hazards']))
        return view

    @staticmethod
    def render_snapshot(state):
        """Reproject audit truth into RGB without a running world or clock."""
        return HeadlessGame._snapshot_view(state).render()

    @staticmethod
    def replay_snapshot(state, action_text):
        """Replay the recorded six substeps without wall-clock sleeps or inference."""
        from scripts.training.data import strict_action
        strict_action(action_text)
        parts = action_text[len('<|action_start|>'):-len('<|action_end|>')].split(';')
        action = SafeAction(tuple(map(float, parts[0].split())), tuple(tuple(p.split()) for p in parts[1:]))
        require(action.mouse == (0., 0., 0.), 'physics replay requires zero mouse')
        to_action_text(action)  # Reject non-WASD or otherwise unsafe direct actions.
        view = HeadlessGame._snapshot_view(state)
        hazard, key = False, False
        for keys in action.steps:
            if not hazard:
                hazard, picked = view._move_substep(keys)
                key |= picked
        events = view._finish_window(hazard, key)
        return view.snapshot(), events

    def observe(self):
        image = self.render()
        return {'image': image, 'captured_ns': self.clock_ns(), 'episode_id': self.episode_id,
                'frame_id': self.step_count, 'clock_id': self.clock_id}

    def _blocked(self, point):
        cell = self.cell(point)
        return cell in self.walls or (cell == self.door and not self.has_key)

    def _move_substep(self, keys):
        dx = int('D' in keys) - int('A' in keys)
        dy = int('S' in keys) - int('W' in keys)
        magnitude = math.hypot(dx, dy)
        if magnitude:
            dx, dy = dx / magnitude / 6, dy / magnitude / 6
            candidate = [self.player[0] + dx, self.player[1]]
            if not self._blocked(candidate):
                self.player[0] = candidate[0]
            candidate = [self.player[0], self.player[1] + dy]
            if not self._blocked(candidate):
                self.player[1] = candidate[1]
        got_key = bool(self.key and self.cell() == self.key and not self.has_key)
        self.has_key |= got_key
        return self.cell() in self.hazards, got_key

    def _finish_window(self, touched_hazard, got_key):
        self.step_count += 1
        goal = self.cell() == self.goal and (self.variant != 'keydoor' or self.has_key)
        self.reason = 'hazard' if touched_hazard else 'goal' if goal else 'horizon' if self.step_count >= self.max_steps else None
        self.done = self.reason is not None
        return ['step'] + (['key'] if got_key else []) + ([self.reason] if self.reason else [])

    def step(self, action, *, observed_ns):
        if self.done:
            raise RuntimeError('episode already ended')
        require(isinstance(action, SafeAction) and action.mouse == (0., 0., 0.),
                'headless executor only accepts safe zero-mouse actions')
        action_text = to_action_text(action)
        timestamp(observed_ns, 'observation time')
        before = self.snapshot()
        start = self.clock_ns()
        require(start >= observed_ns, 'action precedes observation')
        touched_hazard, got_key = False, False
        for index, keys in enumerate(action.steps):
            if not touched_hazard:
                touched_hazard, picked = self._move_substep(keys)
                got_key |= picked
            deadline = start + round((index + 1) * 200_000_000 / 6)
            remaining = (deadline - self.clock_ns()) / 1_000_000_000
            if remaining > 0:
                self.sleep(remaining)
        end = self.clock_ns()
        events = self._finish_window(touched_hazard, got_key)
        terminated, truncated = self.reason in ('goal', 'hazard'), self.reason == 'horizon'
        after = self.snapshot()
        feedback = {'schema_version': 1, 'reward_contract': REWARD_VERSION,
                    'clock_id': self.clock_id,
                    'episode_id': self.episode_id, 'frame_id': before['step_count'],
                    'action_text': action_text, 'action_sha256': hashlib.sha256(action_text.encode()).hexdigest(),
                    'state_before': before, 'state_after': after,
                    'state_before_sha256': content_hash(before), 'state_after_sha256': content_hash(after),
                    'observed_ns': observed_ns, 'action_start_ns': start, 'action_end_ns': end,
                    'feedback_ns': self.clock_ns(), 'completed': True, 'completed_substeps': 6,
                    'events': events, 'reward': reward_from_events(events),
                    'terminated': terminated, 'truncated': truncated, 'terminal_reason': self.reason}
        validate_feedback(feedback)
        return {'actual_action': {'text': action_text, 'start_ns': start, 'end_ns': end, 'completed': True},
                'reward': feedback['reward'], 'terminated': terminated, 'truncated': truncated,
                'terminal_reason': self.reason, 'feedback': feedback}

    def expert_action(self):
        """BFS teacher uses declared training-only state; actor sees RGB instead."""
        if self.done:
            raise RuntimeError('episode already ended')
        start = (*self.cell(), self.has_key)
        queue, visited = deque([(start, None)]), {start}
        while queue:
            (x, y, has_key), first = queue.popleft()
            if (x, y) == self.goal and (self.variant != 'keydoor' or has_key):
                return SafeAction((0., 0., 0.), ((first,),) * 6 if first else ((),) * 6)
            for key, dx, dy in DIRECTIONS:
                cell = (x + dx, y + dy)
                if cell in self.walls or cell in self.hazards or (cell == self.door and not has_key):
                    continue
                if not all(0 <= v < 7 for v in cell):
                    continue
                state = (*cell, has_key or cell == self.key)
                if state not in visited:
                    visited.add(state)
                    queue.append((state, first or key))
        raise RuntimeError('no fully observed expert path')

    def system_prompt(self):
        return ('You control the blue circular point avatar in a fully visible generated grid task. '
                'Reach the green goal. W moves up, A left, S down, D right. Dark cells are walls. '
                'Red cells are lethal hazards. If the goal has a purple door, first collect the '
                'yellow key; the small top-left indicator is yellow when you hold it. '
                'The environment pauses during inference. Each action is 200ms with six substeps; '
                'holding one key for all six moves one cell. Mouse and scroll must be zero. '
                'Return only <|action_start|>0 0 0 ; g1 ; g2 ; g3 ; g4 ; g5 ; g6<|action_end|>, '
                'where each g is held WASD keys or empty. Never include engine state or reward.')
