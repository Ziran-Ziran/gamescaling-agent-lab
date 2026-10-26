"""Exclusive cross-chat heavy-task lock and bounded process-tree lifecycle.

The lock is created exclusively; age and process metadata never authorize deletion.
WSL tasks share the Windows lock by passing its /mnt/c path explicitly.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid


class HeavyTaskLock:
    def __init__(self, path):
        self.path = Path(path)
        self.token = uuid.uuid4().hex

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {'schema_version': 1, 'token': self.token, 'host': socket.gethostname(),
                    'pid': os.getpid(), 'platform': sys.platform,
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'purpose': 'gamescaling-agent-lab bounded heavy task'}
        with self.path.open('x', encoding='utf-8') as file:
            json.dump(metadata, file)
            file.flush()
            os.fsync(file.fileno())
        return self

    def __exit__(self, *exc):
        try:
            metadata = json.loads(self.path.read_text(encoding='utf-8'))
            if metadata.get('token') == self.token:
                self.path.unlink()
        except (FileNotFoundError, ValueError):
            pass  # Never remove unrecognizable or replaced locks.


class WindowsJob:
    """Job with kill-on-close, attached to a gated launcher before any child starts."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                       ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION is 144 bytes on Win64 (112 on Win32).
        # LimitFlags resides at offset 16; remaining limits/counters are zero.
        limits = ctypes.create_string_buffer(144 if ctypes.sizeof(ctypes.c_void_p) == 8 else 112)
        ctypes.c_uint32.from_buffer(limits, 16).value = 0x2000  # KILL_ON_JOB_CLOSE
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle or not self.kernel.SetInformationJobObject(self.handle, 9, limits, len(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def close(self):
        if getattr(self, 'handle', None):
            self.kernel.TerminateJobObject(self.handle, 124)
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def run_bounded(command, lock_path, seconds):
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not 0 < seconds <= 1200):
        raise ValueError('heavy task must have a finite duration in (0, 1200] seconds')
    with HeavyTaskLock(lock_path) as lock:
        environment = dict(os.environ, GAMESCALING_HEAVY_LOCK=str(Path(lock_path).resolve()),
                           GAMESCALING_HEAVY_TOKEN=lock.token)
        process, job = None, None
        try:
            if os.name == 'nt':
                job = WindowsJob()
                # stdin is the gate; descendant creation starts only after job assignment.
                launcher = [sys.executable, str(Path(__file__).resolve()), '--worker', *command]
                process = subprocess.Popen(launcher, stdin=subprocess.PIPE, env=environment)
                job.assign(process)
                process.stdin.write(b'1')
                process.stdin.close()
            else:
                process = subprocess.Popen(command, start_new_session=True, env=environment)
            return process.wait(timeout=seconds)
        finally:
            # Cleanup also runs after a successful root exit: lingering descendants cannot
            # outlive the exclusive lock. Heavy commands may not detach into new sessions.
            if job:
                job.close()
            if process:
                if os.name != 'nt':
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif process.poll() is None:
                    process.kill()
                process.wait(timeout=10)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        if sys.stdin.buffer.read(1) != b'1':
            return 125
        return subprocess.call(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--latest-start', help='Optional ISO timestamp with timezone')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.latest_start:
        latest = datetime.fromisoformat(args.latest_start)
        if latest.tzinfo is None:
            parser.error('latest-start needs timezone')
        if datetime.now(timezone.utc) >= latest:
            parser.error('heavy task start cutoff has passed')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('command required after --')
    try:
        return run_bounded(command, args.lock, args.seconds)
    except FileExistsError:
        print('BUSY: heavy-task lock already exists; preserved without modification.', file=sys.stderr)
        return 75
    except subprocess.TimeoutExpired:
        print('TIMEOUT: process tree stopped before releasing lock.', file=sys.stderr)
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
