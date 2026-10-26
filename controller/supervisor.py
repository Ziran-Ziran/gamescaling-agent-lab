"""Independent, latched stop checks; model requests never own the stop path."""
import threading
import time


class Supervisor:
    def __init__(self, foreground, emergency, release, seconds, *, interval=.01):
        if not 0 < seconds <= 60:
            raise ValueError('Run limit must be in (0, 60] seconds')
        self.foreground, self.emergency, self.release = foreground, emergency, release
        self.seconds, self.interval = seconds, interval
        self.stop = threading.Event()
        self.reason = None
        self._lock = threading.Lock()
        self._thread = None

    def trigger(self, reason):
        with self._lock:
            if not self.stop.is_set():
                self.reason = reason
                self.stop.set()
                self.release()

    def start(self):
        self.deadline = time.monotonic() + self.seconds
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _watch(self):
        try:
            while not self.stop.is_set():
                if self.emergency():
                    self.trigger('F8 emergency stop')
                elif not self.foreground():
                    self.trigger('Game lost foreground')
                elif time.monotonic() >= self.deadline:
                    self.trigger('Time limit')
                self.stop.wait(self.interval)
        except Exception as error:
            self.trigger(f'Watchdog error: {error}')

    def close(self):
        self.trigger('Completed')
        if self._thread:
            self._thread.join(timeout=1)
