import unittest
from controller.control_core import SafeAction, sanitize_action, execute_chunk, to_action_text

class Clock:
    def __init__(self): self.t = 10
    def now(self): return self.t
    def sleep(self, amount): self.t += amount

class Sink:
    def __init__(self): self.events=[]
    def key_down(self,k): self.events.append(('down',k))
    def key_up(self,k): self.events.append(('up',k))
    def move_relative(self,x,y): self.events.append(('move',x,y))

class CoreTests(unittest.TestCase):
    def test_official_tokens_and_text(self):
        a=sanitize_action([0,0,0],[['Space','W']]+[['W']]*5)
        self.assertEqual(a.steps,(('W',),)*6)
        self.assertEqual(to_action_text(a), '<|action_start|>0 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>')
    def test_overrun_drops_expired_steps(self):
        c=Clock(); s=Sink()
        original=s.key_down
        def slow(k):
            original(k)
            c.t += .3
        s.key_down=slow
        a=sanitize_action([0,0,0],[['W'],['A'],['S'],['D'],['A'],['S']])
        self.assertFalse(execute_chunk(a,s,lambda:False,monotonic=c.now,sleep=c.sleep))
        self.assertEqual([e for e in s.events if e[0]=='down'], [('down','W')])
    def test_validation(self):
        for mouse,steps in [([float('nan'),0,0],[[]]*6),(['1',0,0],[[]]*6),([0,0,0],[[]]*5),([0,0,0],[['W']]*5+[('W',)])]:
            with self.assertRaises(ValueError): sanitize_action(mouse,steps)
        for flags in [dict(parsed=False),dict(truncated=True)]:
            with self.assertRaises(ValueError): sanitize_action([0,0,0],[[]]*6,**flags)
    def test_filter_clamp(self):
        a=sanitize_action([100,-99,3],[['W','1','W','e']]*6,allow_mouse=True)
        self.assertEqual(a.mouse,(15.,-15.,0.))
        self.assertEqual(a.steps,(('W',),)*6)
        self.assertEqual(sanitize_action([2,3,4],[[]]*6).mouse,(0.,0.,0.))
    def test_transitions_and_deadline(self):
        c=Clock(); s=Sink()
        a=sanitize_action([0,0,0],[['W'],['W'],['A'],[],[],[]])
        self.assertTrue(execute_chunk(a,s,lambda:False,monotonic=c.now,sleep=c.sleep))
        self.assertAlmostEqual(c.t,10.2)
        self.assertEqual([e for e in s.events if e[0]=='down'],[('down','W'),('down','A')])
        self.assertTrue(all(('up',k) in s.events for k in 'WASD'))
    def test_stop(self):
        c=Clock(); s=Sink(); a=sanitize_action([0,0,0],[['W']]*6)
        self.assertFalse(execute_chunk(a,s,lambda:c.t>=10.025,monotonic=c.now,sleep=c.sleep))
        self.assertLess(c.t,10.04)
        self.assertTrue(all(('up',k) in s.events for k in 'WASD'))
    def test_exception_release(self):
        s=Sink()
        def fail(k): raise RuntimeError('sink')
        s.key_down=fail
        with self.assertRaises(RuntimeError): execute_chunk(sanitize_action([0,0,0],[['W']]*6),s,lambda:False)
        self.assertTrue(all(('up',k) in s.events for k in 'WASD'))
    def test_direct_construction_rejected(self):
        s=Sink()
        with self.assertRaises(ValueError): execute_chunk(SafeAction((0,0,0),(('e',),)*6),s,lambda:False)

if __name__=='__main__': unittest.main()
