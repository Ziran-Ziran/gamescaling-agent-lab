"""Windows client-area capture and guarded, scan-code keyboard input.

No injection, game memory reads, hooks, drivers, or anti-cheat bypasses.
"""
import ctypes
from ctypes import wintypes as w
import math
import os
import threading
import time


def validate_capture_rect(rect):
    left, top, right, bottom = map(int, rect)
    if right - left < 320 or bottom - top < 180:
        raise ValueError('Game client area is minimized, invalid, or too small')
    return left, top, right, bottom


def normalized_mouse(mouse, size, pixel_cap=20):
    if len(mouse) != 3 or not all(math.isfinite(float(v)) for v in mouse):
        raise ValueError('Mouse coordinates must be finite')
    return tuple(max(-pixel_cap, min(pixel_cap, round(float(v) * n / 1000)))
                 for v, n in zip(mouse[:2], size))


def require_input_access(report):
    current, target = report.get('controller_integrity'), report.get('game_integrity')
    if not isinstance(current, int) or not isinstance(target, int):
        raise RuntimeError('Cannot verify Windows input privileges; input remains disabled')
    if current < target:
        raise RuntimeError('Game runs as administrator but controller has lower privileges. '
                           'Input is disabled. See controller/README.md for manual startup.')


def process_integrity(pid):
    """Read mandatory integrity RID from a process token; never elevate."""
    if os.name != 'nt':
        raise RuntimeError('Windows process tokens are unavailable')
    adv = ctypes.WinDLL('advapi32', use_last_error=True)
    KERNEL.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    KERNEL.OpenProcess.restype = w.HANDLE
    KERNEL.CloseHandle.argtypes = [w.HANDLE]
    adv.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)]
    adv.GetTokenInformation.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD)]
    adv.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
    adv.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
    adv.GetSidSubAuthority.argtypes = [ctypes.c_void_p, w.DWORD]
    adv.GetSidSubAuthority.restype = ctypes.POINTER(w.DWORD)
    process = KERNEL.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    if not process:
        raise ctypes.WinError(ctypes.get_last_error())
    token = w.HANDLE()
    try:
        if not adv.OpenProcessToken(process, 8, ctypes.byref(token)):  # TOKEN_QUERY
            raise ctypes.WinError(ctypes.get_last_error())
        needed = w.DWORD()
        adv.GetTokenInformation(token, 25, None, 0, ctypes.byref(needed))
        if not needed.value:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_string_buffer(needed.value)
        if not adv.GetTokenInformation(token, 25, buffer, needed, ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        count = adv.GetSidSubAuthorityCount(sid)[0]
        return adv.GetSidSubAuthority(sid, count - 1)[0]
    finally:
        if token:
            KERNEL.CloseHandle(token)
        KERNEL.CloseHandle(process)


def input_access_report(game_pid):
    report = {'controller_pid': os.getpid(), 'game_pid': game_pid}
    for key, pid in (('controller_integrity', os.getpid()), ('game_integrity', game_pid)):
        try:
            report[key] = process_integrity(pid)
        except OSError as error:
            report[key] = None
            report[key + '_error'] = str(error)
    current, target = report['controller_integrity'], report['game_integrity']
    report['input_compatible'] = current is not None and target is not None and current >= target
    return report


if os.name == 'nt':
    USER = ctypes.WinDLL('user32', use_last_error=True)
    KERNEL = ctypes.WinDLL('kernel32', use_last_error=True)
    USER.GetForegroundWindow.restype = w.HWND
    USER.IsWindow.argtypes = [w.HWND]
    USER.IsIconic.argtypes = [w.HWND]
    USER.IsWindowVisible.argtypes = [w.HWND]
    USER.ShowWindowAsync.argtypes = [w.HWND, ctypes.c_int]
    USER.ShowWindowAsync.restype = w.BOOL
    USER.SetForegroundWindow.argtypes = [w.HWND]
    USER.SetForegroundWindow.restype = w.BOOL
    USER.GetClientRect.argtypes = [w.HWND, ctypes.POINTER(w.RECT)]
    USER.ClientToScreen.argtypes = [w.HWND, ctypes.POINTER(w.POINT)]
    USER.GetWindowTextLengthW.argtypes = [w.HWND]
    USER.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, ctypes.c_int]
    USER.GetWindowThreadProcessId.argtypes = [w.HWND, ctypes.POINTER(w.DWORD)]
    USER.GetAsyncKeyState.argtypes = [ctypes.c_int]
    USER.GetAsyncKeyState.restype = ctypes.c_short
    try:
        USER.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        USER.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        USER.SetProcessDPIAware()


def list_game_windows(title='原神'):
    if os.name != 'nt':
        raise RuntimeError('Window capture requires Windows')
    result = []
    callback_type = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)

    def visit(hwnd, _):
        if not USER.IsWindowVisible(hwnd):
            return True
        length = USER.GetWindowTextLengthW(hwnd)
        text = ctypes.create_unicode_buffer(length + 1)
        USER.GetWindowTextW(hwnd, text, length + 1)
        if text.value == title:
            pid = w.DWORD()
            USER.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            result.append({'hwnd': int(hwnd), 'pid': pid.value, 'title': text.value,
                           'minimized': bool(USER.IsIconic(hwnd))})
        return True

    callback = callback_type(visit)
    USER.EnumWindows.argtypes = [callback_type, w.LPARAM]
    if not USER.EnumWindows(callback, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return result


class GameWindow:
    def __init__(self, title='原神'):
        matches = list_game_windows(title)
        if len(matches) != 1:
            raise RuntimeError(f'Expected one {title!r} window, found {len(matches)}')
        self.info = matches[0]
        self.hwnd = self.info['hwnd']

    def activate(self, *, timeout=2.0):
        """Restore/activate once at startup; never work around Windows focus policy."""
        if not 0 < timeout <= 5 or not math.isfinite(timeout):
            raise ValueError('Activation timeout must be finite in (0,5]')
        deadline = time.monotonic() + timeout

        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise RuntimeError('Windows could not activate the selected game window')
            return seconds

        def identity():
            matches = list_game_windows('原神')
            if (self.info['title'] != '原神' or len(matches) != 1
                    or any(matches[0][k] != self.info[k] for k in ('hwnd', 'pid', 'title'))
                    or not USER.IsWindow(self.hwnd)):
                raise RuntimeError('Target window identity changed or is not unique')
            pid = w.DWORD()
            USER.GetWindowThreadProcessId(self.hwnd, ctypes.byref(pid))
            if pid.value != self.info['pid']:
                raise RuntimeError('Target window identity changed')

        identity()
        remaining()
        if USER.IsIconic(self.hwnd):
            if not USER.ShowWindowAsync(self.hwnd, 9):  # SW_RESTORE; do not wait on the game's UI thread.
                raise RuntimeError('Windows could not request game window restoration')
            while True:
                identity()
                budget = remaining()
                if not USER.IsIconic(self.hwnd):
                    break
                time.sleep(min(.02, budget))
        identity()
        remaining()
        USER.SetForegroundWindow(self.hwnd)
        while True:
            identity()
            budget = remaining()
            if self.is_foreground():
                self.info['minimized'] = False
                return
            time.sleep(min(.02, budget))

    def is_foreground(self):
        pid = w.DWORD()
        USER.GetWindowThreadProcessId(self.hwnd, ctypes.byref(pid))
        return (bool(USER.IsWindow(self.hwnd)) and not USER.IsIconic(self.hwnd)
                and pid.value == self.info['pid']
                and USER.GetForegroundWindow() == self.hwnd)

    def rect(self):
        if not USER.IsWindow(self.hwnd) or USER.IsIconic(self.hwnd):
            raise RuntimeError('Target game window is closed or minimized')
        pid = w.DWORD()
        USER.GetWindowThreadProcessId(self.hwnd, ctypes.byref(pid))
        if pid.value != self.info['pid']:
            raise RuntimeError('Target window identity changed')
        rect = w.RECT()
        origin = w.POINT(0, 0)
        if not USER.GetClientRect(self.hwnd, ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not USER.ClientToScreen(self.hwnd, ctypes.byref(origin)):
            raise ctypes.WinError(ctypes.get_last_error())
        return validate_capture_rect((origin.x, origin.y,
                                      origin.x + rect.right, origin.y + rect.bottom))

    def capture(self):
        from PIL import ImageGrab, ImageOps, Image
        if not self.is_foreground():
            raise RuntimeError('Game must be foreground before capture')
        rect = self.rect()
        image = ImageGrab.grab(bbox=rect, all_screens=True).convert('RGB')
        if not self.is_foreground() or self.rect() != rect:
            raise RuntimeError('Window changed or lost focus during capture')
        small = ImageOps.pad(image, (640, 360), method=Image.Resampling.LANCZOS)
        return image, small, rect


class MouseInput(ctypes.Structure):
    _fields_ = [('dx', w.LONG), ('dy', w.LONG), ('mouseData', w.DWORD),
                ('dwFlags', w.DWORD), ('time', w.DWORD), ('dwExtraInfo', ctypes.c_size_t)]


class KeyboardInput(ctypes.Structure):
    _fields_ = [('wVk', w.WORD), ('wScan', w.WORD), ('dwFlags', w.DWORD),
                ('time', w.DWORD), ('dwExtraInfo', ctypes.c_size_t)]


class HardwareInput(ctypes.Structure):
    _fields_ = [('uMsg', w.DWORD), ('wParamL', w.WORD), ('wParamH', w.WORD)]


class InputUnion(ctypes.Union):
    _fields_ = [('mi', MouseInput), ('ki', KeyboardInput), ('hi', HardwareInput)]


class Input(ctypes.Structure):
    _anonymous_ = ('u',)
    _fields_ = [('type', w.DWORD), ('u', InputUnion)]


class NativeInput:
    SCANS = {'W': 0x11, 'A': 0x1E, 'S': 0x1F, 'D': 0x20}

    def __init__(self, window, stop_event):
        require_input_access(input_access_report(window.info['pid']))
        self.window = window
        self.stop = stop_event
        self.lock = threading.RLock()
        self.size = None
        USER.SendInput.argtypes = [w.UINT, ctypes.POINTER(Input), ctypes.c_int]
        USER.SendInput.restype = w.UINT

    def _send(self, entry):
        if USER.SendInput(1, ctypes.byref(entry), ctypes.sizeof(Input)) != 1:
            raise RuntimeError('Windows rejected SendInput; check privilege compatibility')

    def _keyboard(self, key, up):
        if key not in self.SCANS:
            raise ValueError('Only WASD input is allowed')
        entry = Input(type=1)
        entry.ki = KeyboardInput(0, self.SCANS[key], 8 | (2 if up else 0), 0, 0)
        self._send(entry)

    def key_down(self, key):
        with self.lock:
            if self.stop.is_set() or not self.window.is_foreground():
                raise RuntimeError('Input stopped or game lost focus')
            self._keyboard(key, False)

    def key_up(self, key):
        with self.lock:
            self._keyboard(key, True)

    def release_all(self):
        with self.lock:
            for key in self.SCANS:
                try:
                    self._keyboard(key, True)
                except RuntimeError:
                    pass

    def move_relative(self, x, y):
        with self.lock:
            if self.stop.is_set() or not self.window.is_foreground():
                raise RuntimeError('Input stopped or game lost focus')
            rect = self.window.rect()
            size = (rect[2] - rect[0], rect[3] - rect[1])
            if self.size != size:
                raise RuntimeError('Game client size changed; recalibrate first')
            dx, dy = normalized_mouse((x, y, 0), size)
            if dx or dy:
                entry = Input(type=0)
                entry.mi = MouseInput(dx, dy, 0, 1, 0, 0)
                self._send(entry)


def emergency_key_pressed():
    # The edge bit also catches a short tap between watchdog polls.
    return bool(USER.GetAsyncKeyState(0x77) & 0x8001)  # F8


class EmergencyHotkey:
    """Reserve F8 (emergency) or F24 (selftest); events remain latched."""
    def __init__(self, *, vk=0x77):
        if type(vk) is not int or vk not in (0x77, 0x87):
            raise ValueError('Only F8 emergency or F24 selftest may be reserved')
        self.vk = vk
        self.label = 'F8' if vk == 0x77 else 'F24'
        self.pressed = threading.Event()
        self.closed = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._messages, daemon=True)
        self.thread.start()
        if not self.ready.wait(2) or self.error:
            self.close()
            raise RuntimeError(self.error or f'{self.label} registration timed out')

    def _messages(self):
        registered = False
        try:
            USER.RegisterHotKey.argtypes = [w.HWND, ctypes.c_int, w.UINT, w.UINT]
            USER.UnregisterHotKey.argtypes = [w.HWND, ctypes.c_int]
            USER.PeekMessageW.argtypes = [ctypes.POINTER(w.MSG), w.HWND, w.UINT, w.UINT, w.UINT]
            message = w.MSG()
            USER.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
            if not USER.RegisterHotKey(None, 0x64A1, 0x4000, self.vk):
                raise RuntimeError(f'Cannot reserve {self.label}; stop before enabling input')
            registered = True
            self.ready.set()
            while not self.closed.is_set():
                while USER.PeekMessageW(ctypes.byref(message), None, 0x312, 0x312, 1):
                    self.pressed.set()
                self.closed.wait(.005)
        except Exception as error:
            self.error = str(error)
            self.pressed.set()
        finally:
            self.ready.set()
            if registered:
                USER.UnregisterHotKey(None, 0x64A1)

    def __call__(self):
        return self.pressed.is_set()

    def close(self):
        self.closed.set()
        self.thread.join(timeout=1)
