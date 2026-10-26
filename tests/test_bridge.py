import unittest
from controller.wsl_bridge import windows_to_wsl
class BridgeTests(unittest.TestCase):
 def test_path(self):
  self.assertEqual(windows_to_wsl(r'D:\my folder\worker.py'), '/mnt/d/my folder/worker.py')
if __name__ == '__main__': unittest.main()

class ProtocolTests(unittest.TestCase):
 def test_errors_preserve_ids_without_tracebacks(self):
  import io, json
  from controller.policy_worker import serve
  class Bad:
   def handle(self,q): raise ValueError('bad input')
  out=io.StringIO(); serve(Bad(),io.StringIO('{broken\n{"id":42,"command":"act"}\n'),out)
  rows=[json.loads(x) for x in out.getvalue().splitlines()]
  self.assertEqual(rows[1]['type'],'error'); self.assertEqual(rows[2]['id'],42)
  self.assertNotIn('Traceback',out.getvalue())
 def test_timeout_terminates_child(self):
  import sys
  from controller.wsl_bridge import WSLBridge
  script='import time; print(\'{"type":"ready"}\',flush=True); time.sleep(30)'
  bridge=WSLBridge(command=[sys.executable,'-u','-c',script])
  with self.assertRaises(TimeoutError): bridge.request({'command':'health'},timeout=.05)
  self.assertTrue(bridge.closed); self.assertIsNotNone(bridge.process.poll())
 def test_pending_prevents_phantom_turns(self):
  from collections import deque
  from types import SimpleNamespace
  from controller.policy_worker import Worker
  import base64, io
  from PIL import Image
  buf=io.BytesIO(); Image.new('RGB',(2,2)).save(buf,format='PNG')
  action=SimpleNamespace(mouse=(0,0,0),steps=[[]]*6,parsed=True,truncated=False,to_text=lambda:'canonical')
  class Policy:
   cfg=SimpleNamespace(history_len=0)
   _history=deque()
   def act(self,image):
    self._history.append('phantom')
    return SimpleNamespace(action=action,text='raw',latency_ms=1,error=None)
   def reset(self): self._history.clear()
  p=Policy(); worker=Worker(p,lambda *x:x,lambda *x:'image',lambda text:action)
  result=worker.handle({'command':'act','image_b64':base64.b64encode(buf.getvalue()).decode(),'history_len':2})
  self.assertEqual(list(p._history),[])
  with self.assertRaises(ValueError): worker.handle({'command':'act'})
  with self.assertRaises(ValueError): worker.handle({'command':'commit','pending_id':'wrong','actual_action':'canonical'})
  worker.handle({'command':'commit','pending_id':result['pending_id'],'actual_action':'canonical'})
  self.assertEqual(list(p._history),[('image','','canonical')])
  worker.handle({'command':'reset'}); self.assertIsNone(worker.pending); self.assertEqual(list(p._history),[])

class BlockedWriteTests(unittest.TestCase):
 def test_large_write_deadline(self):
  import sys, time
  from controller.wsl_bridge import WSLBridge
  bridge=WSLBridge(command=[sys.executable,'-u','-c','import time; print(\'{"type":"ready"}\',flush=True); time.sleep(30)'])
  start=time.monotonic()
  with self.assertRaises(TimeoutError): bridge.request({'command':'act','image_b64':'x'*2000000},timeout=.1)
  self.assertLess(time.monotonic()-start,5)
  self.assertFalse(bridge._writer.is_alive())
  self.assertIsNotNone(bridge.process.poll())
