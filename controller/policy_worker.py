"""WSL policy worker. Stdout contains protocol JSON only; no input execution."""
import argparse
import base64
import contextlib
import io
import json
import os
import math
import sys
import uuid
MAX_LINE = 24 * 1024 * 1024
MAX_IMAGE = 16 * 1024 * 1024

class Worker:
 def __init__(self, policy, turn, encode, parse):
  self.policy=policy; self.turn=turn; self.encode=encode; self.parse=parse; self.pending=None
 def handle(self, q):
  command=q.get('command', q.get('type'))
  p=self.policy
  if command=='health':
   r=p._session.get(p.base_url+'/models',timeout=5); r.raise_for_status()
   return {'models':r.json()}
  if command in ('reset','clear'):
   self.pending=None; p.reset(); return {'reset':True}
  if command=='close': return {'closed':True}
  if command=='commit':
   if self.pending is None or q.get('pending_id')!=self.pending[0]: raise ValueError('pending_id mismatch')
   text=q['actual_action']; action=self.parse(text)
   if not action.parsed or action.to_text()!=text: raise ValueError('actual_action must be canonical action text')
   _,url=self.pending
   p._history.append(self.turn(url,'',text))
   while len(p._history)>p.cfg.history_len: p._history.popleft()
   self.pending=None; return {'committed':True}
  if command!='act': raise ValueError('unknown command')
  if self.pending is not None: raise ValueError('commit or reset pending decision first')
  n=q.get('history_len',0)
  if type(n) is not int or not 0<=n<=5: raise ValueError('history_len must be 0..5')
  tokens=q.get('max_tokens',512); timeout=q.get('timeout',5)
  if type(tokens) is not int or not 1<=tokens<=4096 or not math.isfinite(float(timeout)) or not 0<float(timeout)<=120: raise ValueError('invalid inference limits')
  encoded=q['image_b64']
  if len(encoded)>MAX_IMAGE*4//3+4: raise ValueError('image too large')
  data=base64.b64decode(encoded,validate=True)
  if len(data)>MAX_IMAGE: raise ValueError('image too large')
  from PIL import Image
  with Image.open(io.BytesIO(data)) as source:
   if source.format not in ('PNG','JPEG') or source.width*source.height>16000000: raise ValueError('invalid image')
   image=source.convert('RGB')
  p.cfg.history_len=n; p.cfg.max_tokens=tokens; p.cfg.timeout=float(timeout)
  while len(p._history)>n: p._history.popleft()
  old=list(p._history)
  try: decision=p.act(image)
  finally:
   p._history.clear(); p._history.extend(old)
  pending_id=uuid.uuid4().hex
  self.pending=(pending_id,self.encode(image,p.cfg))
  a=decision.action
  return dict(mouse=list(a.mouse),steps=a.steps,parsed=a.parsed,truncated=a.truncated,raw_text=decision.text,latency_ms=decision.latency_ms,error=decision.error,pending_id=pending_id)

def serve(worker, inp=sys.stdin, out=sys.stdout):
 def emit(obj): out.write(json.dumps(obj,ensure_ascii=True)+'\n'); out.flush()
 ready={'type':'ready','protocol':1}
 if sys.platform=='linux':
  from pathlib import Path
  ready.update(pid=os.getpid(),start_time=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
 emit(ready)
 while True:
  line=inp.readline(MAX_LINE+1)
  if not line: break
  q={}
  try:
   if len(line)>MAX_LINE:
    while line and not line.endswith('\n'): line=inp.readline(MAX_LINE+1)
    raise ValueError('line too large')
   q=json.loads(line)
   if not isinstance(q,dict): q={}; raise ValueError('object required')
   if 'id' not in q: raise ValueError('id required')
   with contextlib.redirect_stdout(sys.stderr): result=worker.handle(q)
   emit({'type':'result','id':q['id'],'result':result})
   if q.get('command',q.get('type'))=='close': break
  except Exception as exc: emit({'type':'error','id':q.get('id'),'error':str(exc)})

def parse_args(argv=None):
 parser=argparse.ArgumentParser(); parser.add_argument('--repo',required=True)
 parser.add_argument('--base-url', choices=('http://127.0.0.1:8000/v1','http://127.0.0.1:18080/v1'),default='http://127.0.0.1:8000/v1')
 return parser.parse_args(argv)

def main():
 args=parse_args(); sys.path.insert(0,args.repo)
 from gamescaling.policy import GameScalingPolicy, PolicyConfig, Turn, encode_image
 from gamescaling.action import parse_action
 from gamescaling.prompt import build_system_prompt
 prompt=build_system_prompt('Genshin Impact',rules='Explore a short distance along the visible road. Use W A S D and small camera movements only; no combat, menus or interaction.',key_bindings={'W':'forward','A':'left','S':'back','D':'right'},mouse_aim=True)
 policy=GameScalingPolicy(prompt,PolicyConfig(base_url=args.base_url,history_len=0,max_tokens=512,timeout=5)); policy._session.trust_env=False
 serve(Worker(policy,Turn,encode_image,parse_action))
if __name__=='__main__': main()
