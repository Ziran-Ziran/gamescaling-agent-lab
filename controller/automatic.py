"""Run one bounded automatic round; input requires --execute and Windows UAC."""
import argparse
import ctypes
from ctypes import wintypes as w
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from controller import bounded_session as b, run_agent
from controller.backends import BACKENDS, backend_target

SHADOW_OUTPUT = b.ROOT / 'output' / 'local-control' / 'automatic'
UAC_TIMEOUT = 90
RESULT_TIMEOUT = 120
CLEANUP_TIMEOUT = 10
POLL_SECONDS = .05


class ShellExecuteInfo(ctypes.Structure):
    _fields_ = [('cbSize', w.DWORD), ('fMask', w.ULONG), ('hwnd', w.HWND),
                ('lpVerb', w.LPCWSTR), ('lpFile', w.LPCWSTR), ('lpParameters', w.LPCWSTR),
                ('lpDirectory', w.LPCWSTR), ('nShow', ctypes.c_int), ('hInstApp', w.HINSTANCE),
                ('lpIDList', ctypes.c_void_p), ('lpClass', w.LPCWSTR), ('hkeyClass', w.HKEY),
                ('dwHotKey', w.DWORD), ('hIcon', w.HANDLE), ('hProcess', w.HANDLE)]


class WorkerProcess:
    def __init__(self, handle, kernel):
        self.handle, self.kernel = handle, kernel
        kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        kernel.WaitForSingleObject.restype = w.DWORD
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.CloseHandle.restype = w.BOOL
        kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        kernel.GetExitCodeProcess.restype = w.BOOL

    def exited(self):
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result == 0xffffffff:
            raise ctypes.WinError(ctypes.get_last_error())
        return result == 0

    def returncode(self):
        code = w.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _shell_execute_worker(root, backend):
    # All arguments are fixed or bounded enums/UUIDs; the shell never sees user commands.
    backend_target(backend)
    if uuid.UUID(root.name).hex != root.name:
        raise ValueError('Invalid session UUID')
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    ole = ctypes.WinDLL('ole32', use_last_error=True)
    ole.CoInitializeEx.argtypes = [ctypes.c_void_p, w.DWORD]
    ole.CoInitializeEx.restype = ctypes.c_long
    ole.CoUninitialize.argtypes = []
    ole.CoUninitialize.restype = None
    initialized = ole.CoInitializeEx(None, 0x2 | 0x4)  # STA, disable legacy OLE1 DDE.
    if initialized < 0:
        raise OSError(f'COM initialization failed: {initialized}')
    try:
        info = ShellExecuteInfo()
        info.cbSize = ctypes.sizeof(info)
        info.fMask = 0x40 | 0x100 | 0x400  # process handle, synchronous shell launch, no error UI.
        info.lpVerb = 'runas'
        info.lpFile = sys.executable
        info.lpParameters = subprocess.list2cmdline([
            '-m', 'controller.bounded_session', 'worker', '--session', root.name,
            '--backend', backend, '--activate'])
        info.lpDirectory = str(b.ROOT)
        info.nShow = 0  # SW_HIDE; UAC's security prompt remains visible.
        shell.ShellExecuteExW.argtypes = [ctypes.POINTER(ShellExecuteInfo)]
        shell.ShellExecuteExW.restype = w.BOOL
        if not shell.ShellExecuteExW(ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not info.hProcess:
            raise RuntimeError('Windows did not return a worker process handle')
        return WorkerProcess(info.hProcess, kernel)
    finally:
        ole.CoUninitialize()


def launch_worker(root, backend):
    """Keep the parent responsive while ShellExecuteEx waits for the human UAC answer."""
    done, lock = threading.Event(), threading.Lock()
    state = {'abandoned': False}

    def launch():
        try:
            process = _shell_execute_worker(root, backend)
            with lock:
                if state['abandoned']:
                    process.close()  # Late worker sees persistent STOP before activation.
                else:
                    state['process'] = process
        except BaseException as exc:
            with lock:
                state['error'] = exc
        finally:
            done.set()

    try:
        threading.Thread(target=launch, name='automatic-uac', daemon=True).start()
        if not done.wait(UAC_TIMEOUT):
            raise TimeoutError('UAC launch exceeded 90 seconds; STOP blocks late worker input')
        with lock:
            if 'error' in state:
                raise state['error']
            return state.pop('process')
    except BaseException:
        # STOP precedes abandoning ownership, including a result/timeout race.
        try:
            (root / 'STOP').touch(exist_ok=True)
        finally:
            # A STOP filesystem failure must propagate, but cannot leak a late handle.
            with lock:
                state['abandoned'] = True
                process = state.pop('process', None)
                if process:
                    process.close()
        raise


def wait_result(root, process):
    deadline = time.monotonic() + RESULT_TIMEOUT
    path = root / 'results' / 'req0001' / 'result.json'
    while True:
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        if process.exited():
            raise RuntimeError('Worker exited without req0001 result; inspect closed.json')
        if time.monotonic() >= deadline:
            raise TimeoutError('Worker result exceeded 120 seconds')
        time.sleep(POLL_SECONDS)


def stop_worker(root, process):
    # Never terminate the input worker: its guard must release held inputs itself.
    try:
        (root / 'STOP').touch(exist_ok=True)
        if process is None:
            return None
        deadline = time.monotonic() + CLEANUP_TIMEOUT
        while not process.exited():
            if time.monotonic() >= deadline:
                return None
            time.sleep(POLL_SECONDS)
        return process.returncode()
    finally:
        if process is not None:
            process.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=BACKENDS, default='cloud')
    parser.add_argument('--mode', choices=['shadow', 'explore', 'calibrate'], default='shadow')
    parser.add_argument('--seconds', type=float, default=10, help='Finite round duration, (0,10]')
    parser.add_argument('--count', type=int, default=10, help='Maximum decisions, [1,30]')
    parser.add_argument('--execute', action='store_true', help='Authorize bounded WASD input')
    parser.add_argument('--elevated', action='store_true',
                        help='Explicit UAC worker for shadow when game activation requires administrator access')
    args = parser.parse_args(argv)
    if args.elevated and args.mode != 'shadow':
        parser.error('--elevated is only supported with --mode shadow')
    if args.mode != 'shadow' and not args.execute:
        parser.error('explore/calibrate require explicit --execute')
    request = dict(mode=args.mode, seconds=args.seconds, count=args.count)
    try:
        b.validate_request(request)
    except ValueError as exc:
        parser.error(str(exc))
    use_worker = args.mode != 'shadow' or args.elevated
    state = dict(mode=args.mode, elevated_shadow=args.elevated, target=backend_target(args.backend), request=request,
                 status='failed', worker_closed=False, error=None)
    root = process = None
    code = 1
    try:
        if not use_worker:
            root = SHADOW_OUTPUT / uuid.uuid4().hex
            root.mkdir(parents=True, exist_ok=False)
            state['artifacts'] = str(root)
            print(f'Automatic shadow artifacts: {root}', flush=True)
            code = run_agent.main(['shadow', '--backend', args.backend, '--seconds', str(args.seconds),
                                   '--count', str(args.count), '--history', '0', '--max-age', '1',
                                   '--activate', '--countdown', '0', '--output', str(root)])
            state['status'] = 'passed' if code == 0 else 'failed'
        else:
            root = b.prepare_session()
            state['session'] = root.name
            state['artifacts'] = str(root)
            b.atomic_json(root / 'request' / 'req0001.json', request)
            print(f'Automatic {args.mode} worker session: {root}\nConfirm Windows UAC within 90 seconds. '
                  'Result budget: 120 seconds; STOP cleanup: 10 seconds.', flush=True)
            process = launch_worker(root, args.backend)
            state['result'] = outcome = wait_result(root, process)
            code = 0 if outcome.get('status') == 'passed' and outcome.get('returncode') == 0 else 1
            state['status'] = 'passed' if code == 0 else 'failed'
    except KeyboardInterrupt:
        state.update(status='interrupted', error='Parent interrupted; STOP requested')
        code = 130
    except Exception as exc:
        state['error'] = str(exc)
        code = 1
    finally:
        if root is not None:
            if use_worker:
                try:
                    worker_code = stop_worker(root, process)
                    state['worker_closed'] = worker_code is not None
                    state['worker_returncode'] = worker_code
                    if process is not None and not state['worker_closed']:
                        code = 1
                        state.update(status='failed', cleanup_error='Worker did not exit within 10 seconds; persistent STOP remains')
                    elif worker_code not in (None, 0):
                        code = 1
                        state.update(status='failed', cleanup_error=f'Worker exited with code {worker_code}')
                except (Exception, KeyboardInterrupt) as exc:
                    code = 1
                    state.update(status='failed', cleanup_error=str(exc) or type(exc).__name__)
            state['returncode'] = code
            b.atomic_json(root / 'automatic.json', state)
            print(json.dumps(state, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
