"""Bounded local experiments. Default mode is read-only shadow inference."""
import argparse
import base64
from datetime import datetime
import io
import json
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller.control_core import sanitize_action, execute_chunk, to_action_text
from controller.supervisor import Supervisor
from controller.telemetry import GpuSampler
from controller.windows_adapter import (GameWindow, NativeInput, list_game_windows, EmergencyHotkey,
                                        input_access_report, require_input_access)
from controller.backends import BACKENDS, backend_target, create_bridge, validate_health


def result(bridge, command, *, wait=10, **payload):
    envelope = bridge.request(dict(command=command, **payload), timeout=wait)
    if envelope.get('type') != 'result':
        raise RuntimeError(envelope.get('error', 'Invalid bridge response'))
    return envelope['result']


def encode(image):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def run_outcome(mode, records, reason):
    """Passing means the control plumbing worked, never that a goal was reached."""
    if not records or reason not in (None, 'Completed', 'Time limit'):
        return 'stopped'
    if mode == 'explore' and not any(row['executed'] for row in records):
        return 'stopped'
    return 'passed'


def execute_and_record(action, sink, guard, entry, records, log, commit, reset, *, cycle_start=None):
    """Persist input delivery before touching the fallible history transport."""
    cycle_start = time.monotonic() if cycle_start is None else cycle_start
    class TrackedSink:
        def key_down(self, key):
            sink.key_down(key)
            entry['sent_inputs'] += 1

        def key_up(self, key):
            sink.key_up(key)

        def move_relative(self, x, y):
            sink.move_relative(x, y)
            entry['sent_inputs'] += 1

    entry.update(executed=False, sent_inputs=0, partial_execution=False,
                 history_status='not_attempted')
    records.append(entry)
    try:
        try:
            entry['executed'] = execute_chunk(action, TrackedSink(), guard.stop.is_set)
        finally:
            entry['partial_execution'] = bool(entry['sent_inputs'] and not entry['executed'])
            entry['cycle_ms'] = (time.monotonic() - cycle_start) * 1000
            log('decision', **entry)
        if commit is None:
            entry['history_status'] = 'not_applicable'
        elif entry['executed']:
            commit()
            entry['history_status'] = 'committed'
        else:
            reset()
            entry['history_status'] = 'reset'
        entry['cycle_ms'] = (time.monotonic() - cycle_start) * 1000
        log('history', index=entry['index'], status=entry['history_status'], cycle_ms=entry['cycle_ms'])
    except BaseException as error:
        entry['history_status'] = 'failed'
        entry['error'] = str(error) or type(error).__name__
        guard.trigger('Execution or history failure')
        entry['cycle_ms'] = (time.monotonic() - cycle_start) * 1000
        log('decision_failure', **entry)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'capture', 'shadow', 'calibrate', 'explore'])
    parser.add_argument('--backend', choices=BACKENDS, default='local')
    parser.add_argument('--title', default='原神')
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--history', type=int, choices=[0, 1, 5], default=0)
    parser.add_argument('--allow-mouse', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Explicitly enable bounded WASD input')
    parser.add_argument('--activate', action='store_true', help='Restore and activate the unique game once at startup')
    parser.add_argument('--countdown', type=float, default=3)
    parser.add_argument('--action', choices=['forward', 'left', 'right'], default='forward')
    parser.add_argument('--max-age', type=float, default=1.0)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'local-control')
    args = parser.parse_args(argv)
    if not 0 < args.seconds <= 60 or not 1 <= args.count <= 300:
        parser.error('seconds must be in (0,60], count in [1,300]')
    if not 0 <= args.countdown <= 15 or not 0 < args.max_age <= 1:
        parser.error('countdown must be 0..15; max-age must be in (0,1]')
    if args.mode == 'shadow' and args.history:
        parser.error('Shadow mode must use history=0: suggested actions were never executed')
    if args.mode in ('explore', 'calibrate') and not args.execute:
        parser.error('This mode requires --execute; first verify capture and shadow modes')
    if args.action != 'forward' and args.mode == 'calibrate' and not args.allow_mouse:
        parser.error('Camera calibration requires --allow-mouse')

    directory = args.output / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + args.mode)
    directory.mkdir(parents=True, exist_ok=False)
    events = (directory / 'events.jsonl').open('w', encoding='utf-8', buffering=1)
    def log(event, **data):
        entry = dict(time=datetime.now().astimezone().isoformat(), event=event, **data)
        events.write(json.dumps(entry, ensure_ascii=True) + '\n')
        print(json.dumps(entry, ensure_ascii=True), flush=True)

    sampler = GpuSampler(directory / 'gpu.csv')
    bridge = guard = sink = hotkey = None
    records = []
    error = None
    started = time.monotonic()
    status = 'failed'
    try:
        log('start', mode=args.mode, target=backend_target(args.backend), host=platform.node(), python=sys.version,
            arguments=vars(args) | {'output': str(args.output)}, windows=list_game_windows(args.title))
        if args.mode == 'check':
            with create_bridge(args.backend) as checker:
                health = validate_health(result(checker, 'health'), args.backend)
                log('health', target=backend_target(args.backend), **health)
            window = GameWindow(args.title)
            if args.activate:
                window.activate()
            log('window', **window.info, foreground=window.is_foreground(), rect=window.rect())
            log('input_access', **input_access_report(window.info['pid']))
            status = 'passed'
            return 0

        window = GameWindow(args.title)
        access = input_access_report(window.info['pid'])
        log('input_access', **access)
        if args.execute and args.mode in ('calibrate', 'explore'):
            require_input_access(access)
        if args.activate:
            window.activate()
        hotkey = EmergencyHotkey()
        log('countdown', seconds=args.countdown, instruction='Keep the game foreground. F8 stops.')
        deadline = time.monotonic() + args.countdown
        while time.monotonic() < deadline:
            if hotkey():
                raise RuntimeError('F8 during countdown')
            time.sleep(.01)
        if not window.is_foreground():
            raise RuntimeError('Activate the selected game window before countdown ends')
        sampler.start()
        if args.mode == 'capture':
            original, frame, rect = window.capture()
            original.save(directory / 'original.png')
            frame.save(directory / 'frame-0000.png')
            log('capture', rect=rect, original_size=original.size, input_size=frame.size)
            status = 'passed'
            return 0

        if args.mode in ('shadow', 'explore'):
            bridge = create_bridge(args.backend)
            health = validate_health(result(bridge, 'health'), args.backend)
            log('bridge_ready', target=backend_target(args.backend), **health)
            if hotkey():
                raise RuntimeError('F8 during setup')
            original, frame, rect = window.capture()
            original.save(directory / 'original-start.png')
            log('setup_capture', target=backend_target(args.backend), window=window.info.copy(),
                rect=rect, original_size=original.size, input_size=frame.size,
                original_path='original-start.png')
        stop_event = threading.Event()
        if args.execute and args.mode in ('calibrate', 'explore'):
            sink = NativeInput(window, stop_event)
            rect = window.rect()
            sink.size = (rect[2] - rect[0], rect[3] - rect[1])
        guard = Supervisor(window.is_foreground, hotkey,
                           sink.release_all if sink else lambda: None, args.seconds)
        # Share the independently latched stop event with input methods.
        if sink:
            sink.stop = guard.stop
        guard.start()
        if args.mode == 'calibrate':
            calibration_start = time.monotonic()
            before, _, _ = window.capture()
            before.save(directory / 'before.png')
            mouse = (0, 0, 0) if args.action == 'forward' else (-10 if args.action == 'left' else 10, 0, 0)
            steps = [['W'] for _ in range(6)] if args.action == 'forward' else [[] for _ in range(6)]
            action = sanitize_action(mouse, steps, allow_mouse=args.allow_mouse)
            entry = dict(index=0, mode=args.mode, model_ms=None,
                         filtered=to_action_text(action), raw=None)
            execute_and_record(action, sink, guard, entry, records, log,
                               None, None, cycle_start=calibration_start)
            completed = entry['executed']
            time.sleep(.15)
            if window.is_foreground():
                after, _, _ = window.capture()
                after.save(directory / 'after.png')
            log('calibration', action=to_action_text(action), completed=completed)
            status = 'passed' if completed else 'stopped'
            return 0 if completed else 2

        for index in range(args.count):
            if guard.stop.is_set():
                break
            frame_time = time.monotonic()
            original, frame, rect = window.capture()
            frame.save(directory / f'frame-{index:04d}.png')
            capture_ms = (time.monotonic() - frame_time) * 1000
            decision = result(bridge, 'act', image_b64=encode(frame),
                              history_len=args.history, max_tokens=512, timeout=5, wait=7)
            if decision.get('error'):
                raise RuntimeError(decision['error'])
            action = sanitize_action(decision['mouse'], decision['steps'],
                                     parsed=decision['parsed'], truncated=decision['truncated'],
                                     allow_mouse=args.allow_mouse)
            age = time.monotonic() - frame_time
            completed = False
            entry = dict(index=index, capture_ms=capture_ms, model_ms=decision['latency_ms'],
                         target=backend_target(args.backend), window=window.info.copy(), rect=rect,
                         original_size=original.size, input_size=frame.size,
                         age_before_execution_ms=age * 1000,
                         cycle_ms=(time.monotonic() - frame_time) * 1000,
                         raw=decision['raw_text'], filtered=to_action_text(action),
                         executed=False, sent_inputs=0, partial_execution=False, mode=args.mode)
            recorded = False
            if args.mode == 'shadow':
                result(bridge, 'reset')
            elif guard.stop.is_set() or age > args.max_age:
                result(bridge, 'reset')
                guard.trigger('Stale decision' if age > args.max_age else 'Stopped before execution')
            else:
                size = (rect[2] - rect[0], rect[3] - rect[1])
                current = window.rect()
                if size != sink.size or current != rect:
                    guard.trigger('Window geometry changed')
                else:
                    execute_and_record(action, sink, guard, entry, records, log,
                        lambda: result(bridge, 'commit', pending_id=decision['pending_id'], actual_action=to_action_text(action)),
                        lambda: result(bridge, 'reset'), cycle_start=frame_time)
                    recorded = True
                if not recorded:
                    result(bridge, 'reset')
            entry['cycle_ms'] = (time.monotonic() - frame_time) * 1000
            if not recorded:
                records.append(entry)
                log('decision', **entry)
        status = run_outcome(args.mode, records, guard.reason)
        return 0 if status == 'passed' else 2
    except (Exception, KeyboardInterrupt) as exc:
        error = str(exc) or type(exc).__name__
        log('error', message=error)
        return 1
    finally:
        if guard:
            guard.close()
        if sink:
            sink.release_all()
        if bridge:
            bridge.close()
        if hotkey:
            hotkey.close()
        if hasattr(sampler, 'thread'):
            sampler.close()
        summary = dict(mode=args.mode, target=backend_target(args.backend), status=status, error=error, decisions=len(records),
                       executed=sum(r['executed'] for r in records),
                       partial_executions=sum(r.get('partial_execution', False) for r in records),
                       sent_inputs=sum(r.get('sent_inputs', 0) for r in records),
                       movement_chunks=sum(r['executed'] and any(k in r['filtered'] for k in ('W', 'A', 'S', 'D')) for r in records),
                       elapsed_seconds=time.monotonic() - started,
                       stop_reason=guard.reason if guard else None,
                       median_cycle_ms=statistics.median(r['cycle_ms'] for r in records) if records else None,
                       median_model_ms=statistics.median(r['model_ms'] for r in records if r['model_ms'] is not None) if any(r['model_ms'] is not None for r in records) else None,
                       artifacts=str(directory))
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        log('summary', **summary)
        events.close()


if __name__ == '__main__':
    raise SystemExit(main())
