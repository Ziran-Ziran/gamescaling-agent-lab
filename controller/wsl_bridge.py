"""Persistent JSONL client; timeouts poison the stream and terminate the child."""
import json
import queue
import subprocess
import threading
import time
from pathlib import Path, PureWindowsPath
MAX_LINE = 24 * 1024 * 1024

def windows_to_wsl(path):
 p = PureWindowsPath(path)
 if not p.drive or len(p.drive) != 2: raise ValueError('absolute drive path required')
 return '/mnt/' + p.drive[0].lower() + '/' + '/'.join(p.parts[1:])

class WSLBridge:
 def __init__(self, worker_path=None, *, distro='Ubuntu', repo='/home/ziran/gamescaling-lab/GameScaling', python='/home/ziran/gamescaling-lab/.venv/bin/python', command=None, timeout=10):
  self._lock = threading.Lock(); self._queue = queue.Queue(); self._id = 0; self.closed = False
  self._distro = distro if command is None else None
  self._writer = None
  command = command or ['wsl.exe','-d',distro,'--',python,windows_to_wsl(worker_path or str(Path(__file__).with_name('policy_worker.py'))),'--repo',repo]
  self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
  self._reader = threading.Thread(target=self._read, daemon=True); self._reader.start()
  try:
   self.ready = self._receive(timeout)
   if self.ready.get('type') != 'ready': raise RuntimeError('worker not ready')
  except Exception:
   self.close(); raise
 def _read(self):
  try:
   while True:
    line = self.process.stdout.readline(MAX_LINE + 1)
    if not line: raise EOFError('worker exited')
    if len(line) > MAX_LINE: raise ValueError('worker line too large')
    self._queue.put(json.loads(line))
  except Exception as exc: self._queue.put(exc)
 def _receive(self, timeout):
  try: item = self._queue.get(timeout=timeout)
  except queue.Empty: raise TimeoutError('bridge response timed out') from None
  if isinstance(item, Exception): raise item
  return item
 def request(self, payload, timeout=10):
  with self._lock:
   if self.closed: raise RuntimeError('bridge closed')
   self._id += 1; message = dict(payload, id=self._id)
   try:
    raw = (json.dumps(message)+'\n').encode()
    if len(raw)>MAX_LINE: raise ValueError('request too large')
    deadline = time.monotonic() + timeout
    written = queue.Queue()
    def write():
     try:
      self.process.stdin.write(raw); self.process.stdin.flush(); written.put(None)
     except Exception as exc: written.put(exc)
    self._writer = threading.Thread(target=write, daemon=True); self._writer.start()
    try: failure = written.get(timeout=max(0, deadline-time.monotonic()))
    except queue.Empty: raise TimeoutError('bridge write timed out') from None
    if failure: raise failure
    reply = self._receive(max(0, deadline-time.monotonic()))
    if reply.get('id') != self._id: raise RuntimeError('response id mismatch')
    return reply
   except Exception:
    self.close(); raise
 def close(self):
  if self.closed: return
  self.closed = True
  ready = getattr(self, 'ready', {})
  if self._distro and isinstance(ready.get('pid'), int) and str(ready.get('start_time','')).isdigit():
   script = "import os,sys,signal; p=int(sys.argv[1]); s=sys.argv[2]; f='/proc/%s/stat'%p; data=open(f).read().rsplit(')',1)[1].split(); os.kill(p,signal.SIGTERM) if data[19]==s else None"
   try: subprocess.run(['wsl.exe','-d',self._distro,'--','python3','-c',script,str(ready['pid']),str(ready['start_time'])],timeout=3,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   except (OSError, subprocess.TimeoutExpired): pass
  if self.process.poll() is None:
   self.process.terminate()
   try: self.process.wait(timeout=2)
   except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
  if self._writer: self._writer.join(timeout=2)
  for stream in (self.process.stdin,self.process.stdout):
   if stream: stream.close()
  if threading.current_thread() is not self._reader: self._reader.join(timeout=2)
 def __enter__(self): return self
 def __exit__(self,*args): self.close()
