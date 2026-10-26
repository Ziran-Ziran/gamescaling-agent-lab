"""Static saved-frame probes only. Never imports an input or game adapter.

Run under WSL with the official GameScaling repository on PYTHONPATH.
"""
import argparse
import hashlib
import json
import math
import platform
import random
import re
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

NOOP = '<|action_start|>0 0 0 ; ; ; ; ; ; <|action_end|>'
INSTRUCTIONS = {
    'W': 'For the next 200ms, move forward by holding only W in all six steps. Keep the mouse still. Do not press any other key.',
    'S': 'For the next 200ms, move backward by holding only S in all six steps. Keep the mouse still. Do not press any other key.',
}


def plan(repeats, seed):
    rows = []
    for temp in (0, .7):
        for i in range(repeats):
            for target in ('W', 'S'):
                rows.append(dict(id=f'probe-{temp}-{target}-{i}', experiment='instruction',
                                 group=f'instruction_temp{temp}_{target}', target=target,
                                 instruction=INSTRUCTIONS[target], history=0, reasoning=False,
                                 temperature=temp, seed=1000+i))
    for history in (0, 1, 5):
        for reasoning in (False, True):
            for i in range(repeats):
                rows.append(dict(id=f'load-{history}-{reasoning}-{i}', experiment='load',
                                 group=f'load_history{history}_reasoning{reasoning}', target='W',
                                 instruction=INSTRUCTIONS['W'], history=history, reasoning=reasoning,
                                 temperature=0, seed=2000+i))
    random.Random(seed).shuffle(rows)
    return rows


def control_plan(repeats, doom_defaults=False):
    if doom_defaults:
        return [dict(id=f'doom-default-{i}', experiment='official_doom_default',
                     group='official_doom_default', target='Doom', instruction='', history=0,
                     reasoning=True, temperature=1, seed=3000+i) for i in range(repeats)]
    return [dict(id=f'canonical-{target}-{i}', experiment='canonical_control',
                 group=f'canonical_{target}', target=target, instruction=INSTRUCTIONS.get(target, ''),
                 history=0, reasoning=False, temperature=0, seed=3000)
            for target in ('W', 'S', 'Doom') for i in range(repeats)]


def make_body(case, image_url, nonce, max_tokens=512):
    from gamescaling.policy import PolicyConfig, build_request, user_turn, assistant_turn
    from gamescaling.prompt import build_system_prompt
    prompt = build_system_prompt('Genshin Impact', rules=case['instruction'],
                                 key_bindings={'W': 'forward', 'S': 'backward'}, mouse_aim=True)
    # Early unique prefix discourages automatic prefix-cache reuse. It changes no task rule.
    prompt = case.get('system_prompt', prompt)
    prefix = f'Evaluation trial identifier: {nonce}. Identifier has no gameplay meaning.\n' if nonce is not None else ''
    if case['history']:
        prefix += 'The prior image/action pairs are synthetic no-op context for a static benchmark; no actions were executed.\n'
    messages = [{'role': 'system', 'content': prefix + prompt}]
    for _ in range(case['history']):
        messages.extend([user_turn(image_url), assistant_turn(NOOP)])
    messages.append(user_turn(image_url))
    cfg = PolicyConfig(reasoning=case['reasoning'], temperature=case['temperature'], max_tokens=max_tokens)
    body = build_request(messages, cfg)
    body['seed'] = case['seed']
    return body


def inspect_action(text, target):
    from gamescaling.action import MOUSE_LIMIT, SCROLL_LIMIT, parse_action, split_reasoning
    action = parse_action(text)
    _, tail, closed = split_reasoning(text)
    match = re.fullmatch(r'\s*<\|action_start\|>(.*?)<\|action_end\|>\s*', tail, re.S)
    parts = match.group(1).split(';') if match else []
    try:
        raw_mouse = [float(token) for token in parts[0].split()] if parts else []
        raw_mouse_valid = (len(raw_mouse) == 3
                           and all(math.isfinite(v) for v in raw_mouse)
                           and all(abs(v) <= limit for v, limit in zip(
                               raw_mouse, (MOUSE_LIMIT, MOUSE_LIMIT, SCROLL_LIMIT))))
    except ValueError:
        raw_mouse_valid = False
    strict = bool(closed and action.parsed and not action.truncated and not action.dropped
                  and len(parts) == 7 and raw_mouse_valid
                  and all(math.isfinite(v) for v in action.mouse)
                  and all(len(part.split()) <= 4 for part in parts[1:]))
    matched = strict and action.mouse == (0., 0., 0.) and action.steps == [[target]] * 6
    return dict(parsed=action.parsed, strict_valid=strict, truncated=action.truncated,
                instruction_match=matched, mouse=list(action.mouse), steps=action.steps,
                dropped=action.dropped)


def summarize(rows):
    out = {}
    for group in sorted({r['group'] for r in rows}):
        rr = [r for r in rows if r['group'] == group]
        latencies = sorted(r['latency_ms'] for r in rr)
        out[group] = dict(requests=len(rr), errors=sum(bool(r['error']) for r in rr),
                          parsed=sum(r['parsed'] for r in rr), strict_valid=sum(r['strict_valid'] for r in rr),
                          instruction_match_rate=sum(r['instruction_match'] for r in rr)/len(rr),
                          latency_median_ms=statistics.median(latencies),
                          latency_p95_ms=latencies[math.ceil(.95*len(rr))-1],
                          completion_tokens=[r['usage'].get('completion_tokens') for r in rr],
                          prompt_tokens=[r['usage'].get('prompt_tokens') for r in rr],
                          error_messages=[r['error'] for r in rr if r['error']])
    return out


def main():
    import requests
    from PIL import Image
    from gamescaling.policy import PolicyConfig, encode_image, finalize
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='inputs/genshin_640.png')
    parser.add_argument('--output', default='output/capability-eval')
    parser.add_argument('--repeats', type=int, default=6, choices=range(1, 11))
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--controls', action='store_true', help='Canonical W/S and official Doom; use --repeats 3 for 9 requests')
    parser.add_argument('--doom-defaults', action='store_true', help='Only Doom, official temperature=1/reasoning=true defaults; use --max-tokens 2048')
    parser.add_argument('--official-repo', default='/home/ziran/gamescaling-lab/GameScaling')
    args = parser.parse_args()
    args.controls = args.controls or args.doom_defaults
    output = Path(args.output) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    base = args.base_url.rstrip('/')
    health = session.get(base+'/models', timeout=10)
    health.raise_for_status()  # Abort immediately if API is unavailable.
    image_path = Path(args.image).resolve()
    with Image.open(image_path) as source:
        image_size = source.size
        image_url = encode_image(source.convert('RGB'), PolicyConfig())
    cases = control_plan(args.repeats, args.doom_defaults) if args.controls else plan(args.repeats, args.seed)
    doom_url = None
    if args.controls:
        doom_image = Path(args.official_repo)/'examples/doom_battle1.jpg'
        doom_prompt = (Path(args.official_repo)/'prompts/example_doom_battle1.txt').read_text(encoding='utf-8')
        with Image.open(doom_image) as source:
            doom_url = encode_image(source.convert('RGB'), PolicyConfig())
        for case in cases:
            if case['target'] == 'Doom':
                case['system_prompt'] = doom_prompt
    config = dict(arguments=vars(args), host=socket.gethostname(), platform=platform.platform(),
                  cwd=str(Path.cwd()), image=str(image_path), image_size=image_size,
                  image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                  encoded_sha256=hashlib.sha256(image_url.encode()).hexdigest(), models=health.json(),
                  plan=cases, history='synthetic repeated saved image plus canonical no-op; never executed',
                  cache='deterministic early system nonce shared within each opposite-instruction pair; unique across pairs/load trials; no service reset; image/encoder caches may remain warm',
                  latency='end-to-end client POST elapsed including body transfer; no warmup excluded; sequential requests',
                  p95='nearest-rank empirical P95; small per-group samples do not reliably estimate tails')
    if args.controls:
        config.update(history='none; identical canonical requests repeated, not independent samples',
                      cache='no nonce or synthetic explanation; existing image caches remain warm',
                      doom_image=str(doom_image), doom_sha256=hashlib.sha256(doom_image.read_bytes()).hexdigest(),
                      doom_encoded_sha256=hashlib.sha256(doom_url.encode()).hexdigest())
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    def telemetry(label):
        try:
            response = session.get(base.removesuffix('/v1')+'/metrics', timeout=10)
            (output/f'metrics-{label}.txt').write_text(f'HTTP {response.status_code}\n'+response.text, encoding='utf-8')
        except requests.RequestException as exc:
            (output/f'metrics-{label}.txt').write_text(str(exc), encoding='utf-8')
        try:
            gpu = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv'], capture_output=True, text=True, timeout=10)
            (output/f'gpu-{label}.txt').write_text(gpu.stdout+gpu.stderr, encoding='utf-8')
        except (OSError, subprocess.TimeoutExpired) as exc:
            (output/f'gpu-{label}.txt').write_text(str(exc), encoding='utf-8')
    telemetry('before')
    rows = []
    print(str(output), flush=True)
    with (output/'results.jsonl').open('w', encoding='utf-8') as log:
        for index, case in enumerate(cases):
            nonce_key = f'{case["temperature"]}:{case["seed"]}' if case['experiment'] == 'instruction' else case['id']
            nonce = hashlib.sha256(f'{args.seed}:{nonce_key}'.encode()).hexdigest()
            body = make_body(case, doom_url if case['target'] == 'Doom' else image_url,
                             None if args.controls else nonce, args.max_tokens)
            row = dict(case, order=index, request=body, timestamp=datetime.now(timezone.utc).isoformat(),
                       error=None, usage={}, response=None, status=None, raw_text='', finalized_text='')
            t0 = time.perf_counter()
            try:
                response = session.post(base+'/chat/completions', json=body, timeout=args.timeout)
                row['status'] = response.status_code
                try:
                    row['response'] = response.json()
                except ValueError:
                    row['response'] = response.text
                response.raise_for_status()
                payload = row['response']
                choice = payload['choices'][0]
                row['usage'] = payload.get('usage', {})
                row['finish_reason'] = choice.get('finish_reason')
                row['raw_text'] = choice['message'].get('content') or ''
                row['finalized_text'] = finalize(row['raw_text'], row['finish_reason'], PolicyConfig(reasoning=case['reasoning']))
            except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
                row['error'] = str(exc)
            row['latency_ms'] = (time.perf_counter()-t0)*1000
            row.update(inspect_action(row['finalized_text'], case['target']))
            if row['error']:
                row['instruction_match'] = False
            rows.append(row)
            log.write(json.dumps(row)+'\n')
            log.flush()
            (output/'summary.json').write_text(json.dumps(summarize(rows), indent=2), encoding='utf-8')
            print(json.dumps({k: row.get(k) for k in ('order','id','status','latency_ms','instruction_match','error')}), flush=True)
    telemetry('after')


if __name__ == '__main__':
    main()
