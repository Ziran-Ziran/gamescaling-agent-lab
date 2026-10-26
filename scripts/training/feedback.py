"""Versioned environment reward ledger; truth is never an actor observation."""
import hashlib
import json
import math

from scripts.training.data import require, strict_action, text, timestamp


REWARD_VERSION = 'headless-grid-feedback-v1'


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def reward_from_events(events):
    require(isinstance(events, list) and len(events) == len(set(events)), 'unique event list required')
    require(set(events) <= {'step', 'goal', 'hazard', 'key', 'horizon'} and 'step' in events,
            'unsupported reward event')
    require(not ('goal' in events and 'hazard' in events), 'goal and hazard cannot coincide')
    require(not ('horizon' in events and ({'goal', 'hazard'} & set(events))), 'terminal takes priority over horizon')
    return -.01 + float('goal' in events) - float('hazard' in events)


def validate_feedback(record):
    require(record.get('schema_version') == 1 and record.get('reward_contract') == REWARD_VERSION,
            'unsupported feedback contract')
    text(record.get('episode_id'), 'feedback episode')
    timestamp(record.get('frame_id'), 'feedback frame')
    action = strict_action(record.get('action_text'))
    require(hashlib.sha256(action.encode()).hexdigest() == record.get('action_sha256'),
            'feedback action hash mismatch')
    require(record.get('completed') is True and record.get('completed_substeps') == 6,
            'this feedback contract requires a completed six-substep execution')
    times = [timestamp(record.get(name), name)
             for name in ('observed_ns', 'action_start_ns', 'action_end_ns', 'feedback_ns')]
    require(times == sorted(times), 'feedback clock order violated')
    require(180_000_000 <= times[2] - times[1] <= 220_000_000, 'action duration outside 200ms contract')
    for phase in ('before', 'after'):
        value = record.get('state_' + phase)
        require(isinstance(value, dict) and content_hash(value) == record.get('state_' + phase + '_sha256'),
                'feedback state hash mismatch')
        require(value.get('episode_id') == record['episode_id'], 'feedback episode/state mismatch')
    before, after = record['state_before'], record['state_after']
    from scripts.training.headless import GAME_IDS, HeadlessGame
    require(before.get('variant') in GAME_IDS
            and before.get('game_id') == GAME_IDS[before['variant']]
            and before.get('game_build') == 'procedural-point-grid-v1',
            'unsupported game/build for feedback physics')
    immutable = ('episode_id', 'clock_id', 'game_id', 'game_build', 'variant', 'goal',
                 'walls', 'hazards', 'key', 'door', 'max_steps', 'palette')
    require(all(before.get(k) == after.get(k) for k in immutable), 'immutable world fields changed')
    require(record.get('clock_id') == before.get('clock_id')
            and isinstance(record.get('clock_id'), str), 'feedback clock identity mismatch')
    require(before['step_count'] == record['frame_id'] and after['step_count'] == before['step_count'] + 1,
            'feedback transition/step mismatch')
    require(before['done'] is False, 'cannot transition an ended episode')
    events = record.get('events')
    expected = reward_from_events(events)
    reward = record.get('reward')
    require(type(reward) in (float, int) and math.isfinite(reward) and abs(reward - expected) < 1e-9,
            'reward differs from versioned event contract')
    terminated, truncated = record.get('terminated'), record.get('truncated')
    require(type(terminated) is bool and type(truncated) is bool and not (terminated and truncated),
            'invalid feedback end flags')
    expected_terminal = 'goal' in events or 'hazard' in events
    require(terminated == expected_terminal and truncated == ('horizon' in events),
            'feedback events/end flags mismatch')
    reason = 'goal' if 'goal' in events else 'hazard' if 'hazard' in events else 'horizon' if truncated else None
    require(record.get('terminal_reason') == reason and after['terminal_reason'] == reason
            and after['done'] == (terminated or truncated), 'feedback terminal reason/state mismatch')
    for state in (before, after):
        point = state.get('player')
        require(isinstance(point, list) and len(point) == 2 and all(
            type(v) in (int, float) and math.isfinite(v) and 0 <= v < 7 for v in point),
            'invalid player position')
    cell = [math.floor(v + 1e-9) for v in after['player']]
    at_goal = cell == after['goal'] and (after['variant'] != 'keydoor' or after['has_key'])
    at_hazard = cell in after['hazards']
    require(('goal' in events) == (at_goal and not at_hazard)
            and ('hazard' in events) == at_hazard, 'reward event disagrees with environment position')
    require(('horizon' in events) == (after['step_count'] >= after['max_steps'] and not terminated),
            'horizon event disagrees with environment step')
    require(('key' in events) == (not before['has_key'] and after['has_key']),
            'key event disagrees with inventory transition')
    require(not before['has_key'] or after['has_key'], 'key inventory cannot be lost in this game version')
    require('key' not in events or before['key'] is not None, 'key event without an environment key')
    require(math.dist(before['player'], after['player']) <= 1.000000001,
            'player displacement exceeds one-cell action budget')
    replayed, replay_events = HeadlessGame.replay_snapshot(before, action)
    require(all(replayed[k] == v for k, v in after.items() if k != 'player')
            and math.dist(replayed['player'], after['player']) <= 1e-9
            and replay_events == events, 'recorded transition differs from action physics replay')
    return record


def audit_feedback_episodes(episodes, data_root):
    """Bind rewards, actual action times, state chain and rendered RGB for each frame."""
    from pathlib import Path
    from scripts.training.data import checked_rgb
    from scripts.training.headless import HeadlessGame
    root = Path(data_root).resolve()
    total_frames, total_reward, outcomes, seen = 0, 0., {}, set()
    for episode in episodes:
        require(episode['episode_id'] not in seen, 'duplicate feedback episode')
        seen.add(episode['episode_id'])
        previous_hash = None
        frames = episode.get('frames')
        require(isinstance(frames, list) and frames, 'feedback episode requires frames')
        initial = frames[0]['feedback']['state_before']
        require(frames[0]['frame_id'] == 0 and initial['step_count'] == 0
                and initial['has_key'] is False and initial['done'] is False
                and initial['terminal_reason'] is None, 'feedback episode must start at reset')
        episode_return = 0.
        for index, frame in enumerate(frames):
            feedback = validate_feedback(frame['feedback'])
            require(feedback['episode_id'] == episode['episode_id']
                    and feedback['clock_id'] == episode['clock']['id']
                    and feedback['state_before']['game_id'] == episode['game_id'],
                    'feedback episode/game/clock binding mismatch')
            if previous_hash is not None:
                require(feedback['state_before_sha256'] == previous_hash, 'broken environment state chain')
            previous_hash = feedback['state_after_sha256']
            require(frame['frame_id'] == feedback['frame_id']
                    and frame['captured_ns'] == feedback['observed_ns'], 'frame/feedback time binding mismatch')
            action = frame['actual_action']
            require(action['text'] == feedback['action_text'] and action['completed'] is True
                    and action['start_ns'] == feedback['action_start_ns']
                    and action['end_ns'] == feedback['action_end_ns'], 'actual action/feedback binding mismatch')
            for key in ('reward', 'terminated', 'truncated', 'terminal_reason'):
                require(frame.get(key) == feedback.get(key), 'frame/feedback outcome mismatch')
            require((feedback['terminated'] or feedback['truncated']) == (index == len(frames) - 1),
                    'feedback terminal not at episode boundary')
            rgb = checked_rgb(frame['rgb'], root)
            projected = HeadlessGame.render_snapshot(feedback['state_before']).convert('RGB')
            canonical = f'RGB:{projected.width}:{projected.height}:'.encode() + projected.tobytes()
            require(hashlib.sha256(canonical).hexdigest() == rgb['pixel_sha256'],
                    'saved RGB does not match recorded observable world state')
            episode_return += feedback['reward']
            total_frames += 1
        total_reward += episode_return
        outcomes[episode['episode_id']] = {'return': episode_return, 'steps': len(frames),
                                           'reason': frames[-1]['terminal_reason']}
    require(seen, 'no feedback episodes')
    return {'schema_version': 1, 'reward_contract': REWARD_VERSION,
            'episodes': len(seen), 'frames': total_frames, 'total_return': total_reward,
            'outcomes': outcomes, 'claims': 'environment feedback ledger audit; no policy or RL success claim'}


def main():
    import argparse
    from pathlib import Path
    from scripts.training.data import read_jsonl, sha256_file
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit_feedback_episodes(read_jsonl(args.source), args.data_root)
    report['source_sha256'] = sha256_file(args.source)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'outcomes'}, indent=2))


if __name__ == '__main__':
    main()
