"""Low-frequency GPU sampling and UTF-8 experiment artifacts."""
import csv
import shutil
import subprocess
import threading


class GpuSampler:
    def __init__(self, path):
        self.path, self.rows = path, []
        self.stop = threading.Event()
        self.executable = shutil.which('nvidia-smi')

    def start(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.stop.is_set() and self.executable:
            try:
                result = subprocess.run([self.executable,
                    '--query-gpu=timestamp,memory.total,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True,
                    timeout=2, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode == 0:
                    self.rows.extend(csv.reader(result.stdout.splitlines()))
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(1)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        with self.path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'total_MiB', 'used_MiB', 'gpu_percent'])
            writer.writerows(self.rows)
