"""Finite local administrator worker. JSON requests never contain commands or paths."""
import argparse
from contextlib import contextmanager
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import threading
import time
import uuid

from controller import run_agent, windows_adapter as wa
from controller.supervisor import Supervisor
from controller.backends import BACKENDS, backend_target

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / 'output' / 'local-control' / 'sessions'
MAX_SECONDS = 300
MAX_JOBS = 12
MAX_REQUEST = 4096
MODES = {'check', 'capture', 'shadow', 'calibrate', 'explore'}


def validate_request(value):
    if type(value) is not dict or type(value.get('mode')) is not str or value.get('mode') not in MODES:
        raise ValueError('Expected an object with a supported mode')
    mode = value['mode']
    allowed = {'mode'}
    if mode in {'shadow', 'calibrate', 'explore'}:
        allowed |= {'seconds', 'count'}
    if mode in {'shadow', 'explore'}:
        allowed |= {'history', 'max_age'}
    if mode in {'calibrate', 'explore'}:
        allowed.add('allow_mouse')
    if mode == 'calibrate':
        allowed.add('action')
    if value.keys() - allowed:
        raise ValueError('Unknown or inapplicable request fields')
    req = dict(mode=mode, seconds=10, count=30, history=0,
               max_age=1.0, allow_mouse=False, action='forward') | value
    for name, low, high in [('seconds', 0, 10), ('max_age', 0, 1)]:
        n = req[name]
        if type(n) not in (int, float) or not low < n <= high or not math.isfinite(n):
            raise ValueError(f'{name} must be finite in ({low}, {high}]')
    if type(req['count']) is not int or not 1 <= req['count'] <= 30:
        raise ValueError('count must be an integer in [1,30]')
    if type(req['history']) is not int or req['history'] not in (0, 1, 5):
        raise ValueError('history must be 0, 1 or 5')
    if mode == 'shadow' and req['history'] != 0:
        raise ValueError('shadow requires history=0')
    if type(req['allow_mouse']) is not bool or req['action'] not in ('forward', 'left', 'right'):
        raise ValueError('Invalid allow_mouse or action')
    if req['action'] != 'forward' and not req['allow_mouse']:
        raise ValueError('Camera calibration requires allow_mouse=true')
    return req


def decode_request(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError('Duplicate JSON field')
            obj[key] = value
        return obj
    def invalid(value):
        raise ValueError(f'Nonstandard JSON number: {value}')
    if len(raw.encode('utf-8')) > MAX_REQUEST:
        raise ValueError('Request exceeds 4096 bytes')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def plain_path(path):
    if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
        raise RuntimeError('Session paths must not be symbolic links or junctions')


def session_path(session_id):
    if uuid.UUID(session_id).hex != session_id:
        raise ValueError('Session must be a lowercase UUID hex string')
    path = SESSIONS / session_id
    for ancestor in [SESSIONS.parent.parent, SESSIONS.parent, SESSIONS, path]:
        plain_path(ancestor)
    if path.resolve() != SESSIONS.resolve() / session_id:
        raise ValueError('Session outside fixed output directory')
    return path


def atomic_json(path, value):
    plain_path(path.parent)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare_session():
    session_id = uuid.uuid4().hex
    root = session_path(session_id)
    (root / 'request').mkdir(parents=True, exist_ok=False)
    atomic_json(root / 'manifest.json', dict(session=session_id, max_seconds=MAX_SECONDS,
                max_jobs=MAX_JOBS, host=platform.node(), created=time.time()))
    return root


class RequestQueue:
    def __init__(self, root):
        self.root = root
        self.next = 1
        self.request = root / 'request'
        self.consumed = root / 'consumed'
        plain_path(self.request)
        self.consumed.mkdir(exist_ok=False)

    def claim(self):
        plain_path(self.request)
        plain_path(self.consumed)
        for i in range(1, self.next):
            if (self.request / f'req{i:04d}.json').exists():
                raise RuntimeError('Duplicate/replayed request; session stopped')
        if self.next > MAX_JOBS:
            raise RuntimeError('Maximum 12 jobs reached')
        name = f'req{self.next:04d}.json'
        source = self.request / name
        if not source.exists():
            return None
        plain_path(source)
        dest = self.consumed / name
        # One worker lock plus a destination that must not exist prevents replay.
        if dest.exists():
            raise RuntimeError('Request already consumed')
        source.rename(dest)
        seq = self.next
        self.next += 1
        with dest.open('rb') as stream:
            raw = stream.read(MAX_REQUEST + 1)
        if len(raw) > MAX_REQUEST:
            return seq, ''  # invalid JSON; consumed and reported, never retried
        try:
            return seq, raw.decode('utf-8')
        except UnicodeError:
            return seq, ''  # Invalid UTF-8 is a consumed, rejected request.


class SessionStop:
    def __init__(self, root, deadline):
        self.root, self.deadline = root, deadline
        self.latched = None
        self.window = self.hotkey = None
        self.armed = False
        self.lock = threading.Lock()

    def trigger(self, reason):
        with self.lock:
            self.latched = self.latched or reason
        return self.latched

    def reason(self):
        if self.latched:
            return self.latched
        if (self.root / 'STOP').exists():
            return self.trigger('Global STOP file')
        now = time.monotonic()
        if now >= self.deadline:
            return self.trigger('Session deadline')
        if self.hotkey and self.hotkey():
            return self.trigger('F8 emergency stop')
        if self.armed and not self.window.is_foreground():
            return self.trigger('Bound game lost foreground, identity or geometry')
        return None


def external_supervisor(reason):
    class ExternalSupervisor(Supervisor):
        def __init__(self, foreground, emergency, release, seconds, **kwargs):
            def check():
                why = reason()
                if why:
                    self.trigger(why)
                return emergency()
            super().__init__(foreground, check, release, seconds, **kwargs)
    return ExternalSupervisor


class BoundWindow:
    def __init__(self, window, probe=None):
        self.window = window
        self.info = window.info.copy()
        self.hwnd = self.info['hwnd']
        self.initial_rect = window.rect()
        self.probe = probe or wa.list_game_windows

    def rect(self):
        matches = self.probe()
        if len(matches) != 1 or any(matches[0][k] != self.info[k] for k in ('hwnd', 'pid', 'title')):
            raise RuntimeError('Bound window identity changed')
        rect = self.window.rect()
        if rect != self.initial_rect:
            raise RuntimeError('Bound window geometry changed')
        return rect

    def is_foreground(self):
        try:
            self.rect()
            return self.window.is_foreground()
        except Exception:
            return False

    def capture(self):
        self.rect()
        result = self.window.capture()
        self.rect()
        return result


def require_worker_access(report):
    wa.require_input_access(report)
    if report['controller_integrity'] < 12288:
        raise RuntimeError('Bounded worker requires administrator integrity; user must confirm Windows UAC')


def send_test_hotkey(up):
    entry = wa.Input(type=1)
    entry.ki = wa.KeyboardInput(0x87, 0, 2 if up else 0, 0, 0)  # F24 only
    wa.USER.SendInput.argtypes = [wa.w.UINT, ctypes.POINTER(wa.Input), ctypes.c_int]
    wa.USER.SendInput.restype = wa.w.UINT
    if wa.USER.SendInput(1, ctypes.byref(entry), ctypes.sizeof(entry)) != 1:
        raise RuntimeError('OS rejected synthetic F24')


def synthetic_hotkey_selftest(emergency, send=send_test_hotkey, *, timeout=1,
                             hotkey_factory=wa.EmergencyHotkey, stop_reason=lambda: None):
    """Exercise the generic OS hotkey channel with F24, never resetting F8."""
    def check_stop():
        if emergency() or emergency.error:
            raise RuntimeError('F8 emergency stop or hotkey failure during selftest')
        why = stop_reason()
        if why:
            raise RuntimeError(why)
    check_stop()
    testkey = hotkey_factory(vk=0x87)
    try:
        check_stop()
        try:
            send(False)
        finally:
            send(True)
        deadline = time.monotonic() + timeout
        while True:
            check_stop()
            if testkey.error:
                raise RuntimeError('Synthetic F24 hotkey message thread failed')
            if testkey.pressed.is_set():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('Synthetic F24 did not reach hotkey.pressed')
            testkey.pressed.wait(min(.01, remaining))
        check_stop()
    finally:
        testkey.close()
    check_stop()


@contextmanager
def agent_bindings(window, stop, hotkey):
    # This module runs as a dedicated, single-job worker; no concurrent main calls.
    names = ('GameWindow', 'Supervisor', 'EmergencyHotkey', 'NativeInput')
    old = {name: getattr(run_agent, name) for name in names}
    class HotkeyLease:
        def __call__(self): return bool(stop.reason()) or hotkey()
        def close(self): pass  # Worker owns the actual OS registration.
    class BoundedInput(old['NativeInput']):
        def key_down(self, key):
            if stop.reason():
                raise RuntimeError(stop.reason())
            return super().key_down(key)
        def move_relative(self, x, y):
            if stop.reason():
                raise RuntimeError(stop.reason())
            return super().move_relative(x, y)
    def bound_factory(title='原神'):
        if title != '原神':
            raise ValueError('Only original 原神 binding is permitted')
        window.rect()
        return window
    run_agent.GameWindow = bound_factory
    run_agent.Supervisor = external_supervisor(stop.reason)
    run_agent.EmergencyHotkey = HotkeyLease
    run_agent.NativeInput = BoundedInput
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(run_agent, name, value)


def save_original(window, directory, name):
    if window.is_foreground():
        image, _, _ = window.capture()
        image.save(directory / name)
        return True
    return False


def execute_job(req, directory, window, stop, hotkey, *, backend='local'):
    target = backend_target(backend)
    directory.mkdir(exist_ok=False)
    outcome = dict(request=req, target=target, status='failed', before=False, after=False)
    try:
        if stop.reason():
            raise RuntimeError(stop.reason())
        window.rect()
        access = wa.input_access_report(window.info['pid'])
        require_worker_access(access)
        outcome['input_access'] = access
        if req['mode'] != 'check':
            stop.armed = True
            if not window.is_foreground():
                raise RuntimeError('Keep original game foreground; automatic activation is forbidden')
        outcome['before'] = save_original(window, directory, 'before.png')
        if req['mode'] in {'calibrate', 'explore'}:
            if stop.reason():
                raise RuntimeError(stop.reason())
            synthetic_hotkey_selftest(hotkey, stop_reason=stop.reason)
            outcome['hotkey_selftest'] = 'OS synthetic F24 generic hotkey channel only; F8 delivery untested'
        stop.hotkey = hotkey
        argv = [req['mode'], '--backend', backend, '--seconds', str(req['seconds']), '--count', str(req['count']),
                '--history', str(req['history']), '--max-age', str(req['max_age']),
                '--countdown', '0', '--output', str(directory / 'agent')]
        if req['mode'] in {'calibrate', 'explore'}:
            argv += ['--execute', '--action', req['action']]
        if req['allow_mouse']:
            argv += ['--allow-mouse']
        with agent_bindings(window, stop, hotkey):
            try:
                outcome['returncode'] = run_agent.main(argv)
            finally:
                # main's finally closes the guard and releases input before return.
                # Keep armed through that entire cleanup, then permit log inspection.
                stop.armed = False
        summaries = list((directory / 'agent').glob('*/summary.json'))
        if len(summaries) > 1:
            raise RuntimeError('Expected only one agent summary for this job')
        if summaries:
            outcome['agent_summary'] = json.loads(summaries[0].read_text(encoding='utf-8'))
        outcome['status'] = 'passed' if outcome['returncode'] == 0 and not stop.reason() else 'stopped'
    except Exception as exc:
        outcome['error'] = str(exc)
        stop.trigger('Job failure')
    finally:
        # Also cover validation/selftest failures before main was entered.
        # Existing latched F8/STOP/deadline/focus reasons are never cleared.
        stop.armed = False
        try:
            outcome['after'] = save_original(window, directory, 'after.png')
        except Exception as exc:
            outcome['after_error'] = str(exc)
        outcome['stop_reason'] = stop.reason() or outcome.get('agent_summary', {}).get('stop_reason')
        if outcome['status'] != 'passed':
            stop.trigger('Job failure or stopped')
        atomic_json(directory / 'result.json', outcome)
    return outcome


def worker(session_id, *, backend='local', activate=False):
    target = backend_target(backend)
    root = session_path(session_id)
    if not root.is_dir() or not (root / 'request').is_dir():
        raise RuntimeError('Parent must prepare the random session request directory first')
    # Persistent lock makes a crashed/finished UUID unusable for a second worker.
    with (root / 'worker.lock').open('x', encoding='ascii') as stream:
        stream.write(str(os.getpid()))
    stop = SessionStop(root, time.monotonic() + MAX_SECONDS)
    hotkey = None
    try:
        if stop.reason():
            return 1
        original = wa.GameWindow()
        access = wa.input_access_report(original.info['pid'])
        require_worker_access(access)  # Refuse BEFORE hotkey registration, WSL, or input.
        if stop.reason():
            return 1
        if activate:
            original.activate()
        if stop.reason():
            return 1
        window = BoundWindow(original)  # Freeze geometry only after restoring a minimized window.
        stop.window = window
        hotkey = wa.EmergencyHotkey()
        stop.hotkey = hotkey  # Real F8 remains independently registered and latched throughout.
        queue = RequestQueue(root)
        results = root / 'results'
        results.mkdir(exist_ok=False)
        atomic_json(root / 'ready.json', dict(pid=os.getpid(), window=window.info,
                    target=target, rect=window.initial_rect, input_access=access, deadline_seconds=MAX_SECONDS))
        # Automatic bootstrap is read-only check; never synthesizes hotkeys or game input.
        check = execute_job(validate_request({'mode':'check'}), results / 'bootstrap', window, stop, hotkey, backend=backend)
        if check['status'] != 'passed':
            return 1
        while not stop.reason():
            claimed = queue.claim()
            if claimed is None:
                time.sleep(.02)
                continue
            seq, raw = claimed
            directory = results / f'req{seq:04d}'
            try:
                req = validate_request(decode_request(raw))
            except (ValueError, TypeError, UnicodeError) as exc:
                directory.mkdir()
                atomic_json(directory / 'result.json', dict(status='rejected', target=target, error=str(exc)))
                stop.trigger('Invalid request')
                break
            outcome = execute_job(req, directory, window, stop, hotkey, backend=backend)
            if outcome['status'] != 'passed':
                break
            if seq == MAX_JOBS:
                stop.trigger('Maximum 12 jobs reached')
        return 0
    except Exception as exc:
        stop.trigger(str(exc))
        return 1
    finally:
        if hotkey:
            hotkey.close()
        atomic_json(root / 'closed.json', dict(reason=stop.reason() or 'Completed', target=target, pid=os.getpid()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'worker'])
    parser.add_argument('--session', help='32 lowercase UUID hex digits; never a filesystem path')
    parser.add_argument('--backend', choices=BACKENDS, default='local', help='Fixed for the entire worker lifetime')
    parser.add_argument('--activate', action='store_true', help='Restore and activate the unique game once before binding')
    args = parser.parse_args(argv)
    if args.operation == 'prepare':
        if args.session:
            parser.error('prepare creates a fresh random UUID')
        print(prepare_session())
        return 0
    if not args.session:
        parser.error('worker requires --session')
    return worker(args.session, backend=args.backend, activate=args.activate)


if __name__ == '__main__':
    raise SystemExit(main())
