import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

try:
    from controller import backends
except ImportError:
    backends = None


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(backends, 'fixed backend profiles are required')

    def test_default_local_keeps_wsl_bridge_command(self):
        with patch.object(backends, 'WSLBridge') as bridge:
            backends.create_bridge()
        bridge.assert_called_once_with(timeout=15)
        self.assertEqual(backends.backend_target()['base_url'], 'http://127.0.0.1:8000/v1')

    def test_cloud_uses_windows_python_fixed_repo_and_loopback_tunnel(self):
        with patch.object(backends, 'WSLBridge') as bridge:
            backends.create_bridge('cloud')
        args = bridge.call_args.kwargs
        self.assertEqual(args['timeout'], 15)
        self.assertEqual(args['command'], [sys.executable, '-u',
            str(Path(backends.__file__).with_name('policy_worker.py')),
            '--repo', r'\\wsl.localhost\Ubuntu\home\ziran\gamescaling-lab\GameScaling',
            '--base-url', 'http://127.0.0.1:18080/v1'])
        self.assertEqual(backends.backend_target('cloud')['model_root'], 'THUSI-Lab/GameScaling-9B-S3')

    def test_exact_id_and_root_required(self):
        for backend in ('local', 'cloud'):
            target = backends.backend_target(backend)
            good = {'models': {'data': [{'id':'gamescaling', 'root':target['model_root']}]}}
            self.assertEqual(backends.validate_health(good, backend), good)
            for row in [{'id':'other', 'root':target['model_root']},
                        {'id':'gamescaling', 'root':'prefix/' + target['model_root']},
                        {'id':'gamescaling', 'root':target['model_root'] + '-wrong'},
                        {'id':'gamescaling', 'root':backends.backend_target(
                            'cloud' if backend == 'local' else 'local')['model_root']}]:
                with self.subTest(backend=backend, row=row), self.assertRaises(RuntimeError):
                    backends.validate_health({'models': {'data':[row]}}, backend)

    def test_bad_backend_rejected_before_process_creation(self):
        with patch.object(backends, 'WSLBridge') as bridge:
            for value in ('other', '', None, {}, ['local']):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    backends.create_bridge(value)
            bridge.assert_not_called()

    def test_worker_url_allowlist_rejects_arbitrary_hosts(self):
        from controller.policy_worker import parse_args
        for url in ('http://remote/v1', 'http://127.0.0.1:18080/v1/', 'https://127.0.0.1:8000/v1'):
            with self.subTest(url=url), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--repo','unused','--base-url',url])
        self.assertEqual(parse_args(['--repo','unused']).base_url, 'http://127.0.0.1:8000/v1')

    def test_job_backend_injection_refused(self):
        from controller import bounded_session as b
        for mode in b.MODES:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                b.validate_request({'mode':mode, 'backend':'cloud'})

    def test_startup_backend_reaches_bootstrap_and_jobs_and_metadata(self):
        from controller import bounded_session as b
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'request').mkdir()
            (root/'request'/'req0001.json').write_text('{"mode":"check"}')
            window = MagicMock()
            window.info = {'pid':123}
            window.initial_rect = (0,0,800,600)
            calls = []
            def job(req, directory, window, stop, hotkey, *, backend='local'):
                calls.append(backend)
                if len(calls) == 2:
                    stop.trigger('test complete')
                return {'status':'passed'}
            with patch.object(b, 'session_path', return_value=root), \
                    patch.object(b, 'BoundWindow', return_value=window), \
                    patch.object(b.wa, 'GameWindow'), \
                    patch.object(b.wa, 'EmergencyHotkey', return_value=MagicMock(return_value=False)), \
                    patch.object(b.wa, 'input_access_report', return_value={}), \
                    patch.object(b, 'require_worker_access'), \
                    patch.object(b, 'execute_job', side_effect=job):
                self.assertEqual(b.worker('test', backend='cloud'), 0)
            self.assertEqual(calls, ['cloud','cloud'])
            self.assertEqual(json.loads((root/'ready.json').read_text())['target'],
                             backends.backend_target('cloud'))

    def test_execute_job_passes_fixed_backend_to_agent_and_records_target(self):
        from controller import bounded_session as b
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stop=b.SessionStop(root, time.monotonic()+20)
            window=MagicMock()
            window.info={'pid':123}
            with patch.object(b, 'require_worker_access'), \
                    patch.object(b.wa, 'input_access_report', return_value={}), \
                    patch.object(b, 'save_original', return_value=False), \
                    patch.object(b.run_agent, 'main', return_value=0) as main:
                result=b.execute_job(b.validate_request({'mode':'check'}),root/'job',
                                     window,stop,MagicMock(),backend='cloud')
            argv=main.call_args.args[0]
            self.assertEqual(argv[argv.index('--backend')+1], 'cloud')
            self.assertEqual(result['target'], backends.backend_target('cloud'))

    def test_check_and_inference_reject_wrong_model_using_same_factory(self):
        from controller import run_agent as agent
        for mode in ('check', 'shadow'):
            for backend in ('local', 'cloud'):
                with self.subTest(mode=mode, backend=backend), tempfile.TemporaryDirectory() as tmp:
                    bridge=MagicMock()
                    bridge.__enter__.return_value=bridge
                    bridge.request.return_value={'type':'result','result':{
                        'models':{'data':[{'id':'wrong','root':backends.backend_target(backend)['model_root']}]}}}
                    window=MagicMock()
                    window.info={'pid':123}
                    with patch.object(agent,'create_bridge',return_value=bridge) as factory, \
                            patch.object(agent,'GameWindow',return_value=window), \
                            patch.object(agent,'list_game_windows',return_value=[]), \
                            patch.object(agent,'input_access_report',return_value={}), \
                            patch.object(agent,'GpuSampler'), \
                            patch.object(agent,'EmergencyHotkey'), \
                            patch.object(agent,'NativeInput',side_effect=AssertionError('No inputs')), \
                            contextlib.redirect_stdout(io.StringIO()):
                        code=agent.main([mode,'--backend',backend,'--countdown','0','--output',tmp])
                    self.assertEqual(code,1)
                    factory.assert_called_once_with(backend)
                    self.assertEqual(bridge.request.call_count,1)
                    self.assertEqual(bridge.request.call_args.args[0]['command'],'health')
                    summary=json.loads(next(Path(tmp).glob('*/summary.json')).read_text())
                    self.assertIn('Expected',summary['error'])
                    self.assertEqual(summary['target'],backends.backend_target(backend))


if __name__ == '__main__':
    unittest.main()
