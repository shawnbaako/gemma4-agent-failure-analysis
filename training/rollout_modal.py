"""Trajectory rollouts with the official harness on Modal (quota-free alternative to Kaggle for data generation).

One H100 serves the declared Gemma 4 QAT model through adk_submission's VllmServer with the same flags the
competition harness uses; swegemma's Evaluator then runs the candidate over a task list with the subprocess
sandbox, K tasks at a time, N samples per task. Results and traces land on the `g4-training` volume and
are downloaded to artifacts/rollouts/<run>/s<i>/ in the same layout as a Kaggle validation run, so
training/filter_trajectories.py and analysis/stratified.py consume them unchanged.

Prerequisites (one-time):
  modal volume put g4-data artifacts/modal-data-stage /data      # tasks.jsonl, snapshots/, wheels/, graphs/, embeddings/
  modal volume put g4-data artifacts/wheelhouse /wheelhouse       # official harness wheels (Kaggle dataset)
  modal secret create huggingface HF_TOKEN=...                   # gated Gemma weights

Usage:
  modal run training/rollout_modal.py --candidate candidates/31b-efficient-v3 --run-name 31b-v3-roll1 \
      --tasks gold-valid --n-samples 4 --concurrency 4 --temperature 0.7

Cost control: ephemeral `modal run` app; the GPU container lives only while rollout() runs. Nothing is
deployed or kept warm.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP_NAME = 'g4-rollout'
MODEL_IDS = {
    'gemma-4-31b-it-qat-w4a16-ct': 'google/gemma-4-31b-it-qat-w4a16-ct',
    'gemma-4-12b-it-qat-w4a16-ct': 'google/gemma-4-12b-it-qat-w4a16-ct',
    'gemma-4-e4b-it-qat-w4a16-ct': 'google/gemma-4-e4b-it-qat-w4a16-ct',
    'gemma-4-e2b-it-qat-w4a16-ct': 'google/gemma-4-e2b-it-qat-w4a16-ct',
}
MODEL_VLLM_EXTRA_ARGS = {'gemma-4-12b-it-qat-w4a16-ct': ('--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}

image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git', 'build-essential', 'curl', 'procps')
    .pip_install('vllm==0.19.1', 'huggingface_hub>=0.34')
    # Transitive deps of the harness wheels (which are force-reinstalled --no-deps at container start, as on Kaggle)
    .pip_install('google-adk==1.36.1', 'litellm==1.83.14', 'docker>=7.0.0', 'cachetools>=5.0.0', 'networkx>=3.0', 'numpy>=2.0.0',
                 'pydantic>=2.0.0', 'python-dotenv>=1.0.0', 'pyyaml>=6.0.0', 'rich>=13.0.0', 'pandas>=2.0.0', 'tokenizers>=0.20.0',
                 'torchvision', 'transformers>=4.40.0', 'safetensors', 'accelerate', 'requests>=2.28.0')
    # Kaggle's image ships pytest and the task repos' test/runtime deps in the host Python, which the subprocess
    # sandbox inherits (system site-packages). Mirror that so tests can run: pytest + plugins, and deps of
    # rich / requests / httpx / fastapi-starlette test suites.
    .pip_install('pytest', 'pytest-asyncio', 'pytest-cov', 'pytest-timeout', 'pytest-mock', 'pytest-xdist', 'hypothesis',
                 'pygments', 'markdown-it-py', 'typing_extensions', 'attrs', 'urllib3', 'certifi', 'charset_normalizer', 'idna',
                 'httpcore', 'anyio', 'sniffio', 'h11', 'trio', 'starlette', 'fastapi', 'python-multipart', 'email-validator',
                 'jinja2', 'orjson', 'ujson', 'sqlmodel', 'sqlalchemy', 'flask', 'dirty-equals', 'inline-snapshot', 'pyyaml',
                 'httpx', 'PySocks', 'trustme', 'chardet', 'brotli', 'zstandard', 'uvicorn', 'itsdangerous', 'werkzeug')
    .env({'HF_HOME': '/vol/hf', 'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
          'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'LITELLM_TELEMETRY': 'False', 'LITELLM_LOG': 'WARNING',
          'OTEL_SDK_DISABLED': 'true', 'OTEL_PYTHON_DISABLED': 'true', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
          'TOKENIZERS_PARALLELISM': 'false', 'OPENAI_API_KEY': 'EMPTY', 'LOCAL_API_KEY': 'EMPTY'})
)
data_vol = modal.Volume.from_name('g4-data', create_if_missing=True)
train_vol = modal.Volume.from_name('g4-training', create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _install_wheelhouse():
    import glob
    import shutil
    import subprocess
    import sys

    # Task wheels must be on local disk: the harness hard-links them into each sandbox, which fails across the volume mount.
    if not Path('/tmp/task_wheels').exists():
        shutil.copytree('/data/data/wheels', '/tmp/task_wheels')

    wheels = sorted(w for w in glob.glob('/data/wheelhouse/*.whl') if 'cutlass' not in w.lower())
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
    print(f'installed {len(wheels)} harness wheels', flush=True)


@app.function(gpu='H100', timeout=6 * 60 * 60, cpu=8, memory=65536, volumes={'/data': data_vol, '/vol': train_vol},
              secrets=[modal.Secret.from_name('huggingface')])
def rollout(run_name: str, cand_files: dict, task_ids: list, n_samples: int, concurrency: int, budgets: dict, adapter_map_in: dict | None = None,
            statement_overrides: dict | None = None) -> dict:
    import asyncio
    import shutil
    import time

    _install_wheelhouse()
    import litellm
    from huggingface_hub import snapshot_download
    litellm.drop_params = True
    from adk_submission import VllmConfig, VllmServer, discover_adapters, validate_directory
    from adk_submission.discovery import discover_declared_models
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks

    t_start = time.time()
    cand = Path('/tmp/candidate')
    if cand.exists():
        shutil.rmtree(cand)
    for rel, text in cand_files.items():
        p = cand / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    adapter_names = [l.split(':', 1)[1].strip() for l in cand_files['agent.yaml'].splitlines() if l.strip().startswith('adapter:')]
    for a in adapter_names:
        src = Path('/vol/adapters') / (adapter_map_in or {}).get(a, a)
        assert src.exists(), f'adapter {a} not on volume'
        shutil.copytree(src, cand / 'adapters' / a, ignore=shutil.ignore_patterns('trainer', 'README.md', 'training_summary.json'))
    declared = discover_declared_models(cand); assert len(declared) == 1, declared
    declared_model = next(iter(declared))
    model_path = snapshot_download(MODEL_IDS[declared_model], token=os.environ['HF_TOKEN'])
    train_vol.commit()
    print(f'model at {model_path} ({time.time() - t_start:.0f}s)', flush=True)

    adapters = discover_adapters(str(cand), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    adapter_map = getattr(adapters, 'adapters', None) or {}
    ranks = [r for i in adapter_map.values() if (r := getattr(i, 'rank', None))]
    cfg = VllmConfig(model=model_path, port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                     max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
                     enable_lora=bool(adapter_map), max_loras=max(1, len(adapter_map)), max_lora_rank=max(ranks) if ranks else 128,
                     scheduler_reserve_full_isl=False, tensor_parallel_size=1, startup_timeout=60 * 20,
                     extra_args=MODEL_VLLM_EXTRA_ARGS.get(declared_model, ()))
    print('adapters registered with vLLM:', {k: getattr(v, 'rank', None) for k, v in adapter_map.items()}, flush=True)
    server = VllmServer(cfg, adapter_manifest=adapters)
    t0 = time.time(); server.start(); t_vllm = time.time() - t0
    print(f'vLLM up in {t_vllm:.0f}s', flush=True)
    models = server.create_model_registry(aliases=[declared_model], model_prefix='openai/', api_key='EMPTY')

    data = Path('/data/data')  # volume g4-data is mounted at /data; the competition files were uploaded under /data
    tasks_path = data / 'tasks.jsonl'
    if statement_overrides:
        # Analysis-only variant (paper, issue-text oracle): same tasks and tests, problem statement replaced.
        tasks_path = Path('/tmp/tasks_override.jsonl')
        with tasks_path.open('w') as f:
            for line in (data / 'tasks.jsonl').read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    if d['instance_id'] in statement_overrides:
                        d['problem_statement'] = statement_overrides[d['instance_id']]
                    f.write(json.dumps(d) + '\n')
        print(f'{len(statement_overrides)} problem statements overridden', flush=True)
    tasks = load_tasks(tasks_path)
    by_id = {t.instance_id: t for t in tasks}
    sample_tasks = [by_id[i] for i in task_ids if i in by_id]
    limits, gen_constraints = build_submission_limits()
    validate_directory(cand, limits=limits)
    out_root = Path('/vol/rollouts') / run_name
    summary = {'run_name': run_name, 'model': declared_model, 'tasks': len(sample_tasks), 'n_samples': n_samples, 'concurrency': concurrency,
               'vllm_startup_seconds': round(t_vllm), 'samples': []}
    try:
        for s in range(n_samples):
            sdir = out_root / f's{s}'
            if sdir.exists():
                shutil.rmtree(sdir)  # a rerun with the same run_name replaces the sample, never appends to it
            sdir.mkdir(parents=True, exist_ok=True)
            eval_config = EvalConfig(
                tasks_path=tasks_path, snapshots_dir=data / 'snapshots', results_dir=sdir / 'results', submission_dir=cand,
                models=models, sandbox='subprocess', timeout_seconds=int(budgets['timeout_seconds']),
                verification_timeout_seconds=int(budgets.get('verification_timeout_seconds', 300)),
                max_time_minutes=float(budgets['max_time_minutes']), max_tool_calls=int(budgets['max_tool_calls']),
                max_turns=int(budgets['max_turns']), limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters,
                context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
                graph_dir=str(data / 'graphs'), embeddings_dir=str(data / 'embeddings'), wheels_dir=Path('/tmp/task_wheels'),
                concurrency=concurrency, verbose=False, display_mode='single')
            evaluator = Evaluator(eval_config)
            log = sdir / 'run_results.jsonl'
            rows: list[dict] = []

            async def run_all():
                sem = asyncio.Semaphore(concurrency)
                lock = asyncio.Lock()

                async def one(idx, task, slot):
                    async with sem:
                        t1 = time.time()
                        try:
                            r = await evaluator.evaluate_task(task=task, task_index=idx, total_tasks=len(sample_tasks), slot_id=slot)
                            patch = (r.agent_patch or '').strip()
                            row = {'id': task.instance_id, 'repo': task.repo, 'sample': s, 'resolved': bool(r.resolved), 'has_patch': bool(patch),
                                   'patch_chars': len(patch), 'task_seconds': round(time.time() - t1, 1), 'duration_seconds': round(r.duration_seconds or 0, 1),
                                   'llm_calls': r.total_llm_calls, 'tool_calls': r.tool_calls, 'exit_code': r.test_exit_code, 'error': (r.error or '')[:300],
                                   'test_output_tail': (getattr(r, 'test_output', '') or '')[-1500:], 'patch': patch[:4000]}
                        except Exception as exc:
                            row = {'id': task.instance_id, 'repo': task.repo, 'sample': s, 'resolved': False, 'has_patch': False,
                                   'task_seconds': round(time.time() - t1, 1), 'exception': repr(exc)[:300]}
                        async with lock:
                            rows.append(row)
                            with log.open('a') as f:
                                f.write(json.dumps(row) + '\n')
                            done = sum(x['resolved'] for x in rows)
                            print(f"[s{s} {len(rows)}/{len(sample_tasks)}] {row['id']} resolved={row['resolved']} patch={row.get('has_patch')} "
                                  f"{row['task_seconds']}s running {done}/{len(rows)} elapsed {(time.time() - t_start) / 60:.0f}m", flush=True)
                            if len(rows) % 5 == 0:
                                train_vol.commit()

                await asyncio.gather(*(one(i + 1, t, i % concurrency) for i, t in enumerate(sample_tasks)))

            asyncio.run(run_all())
            summary['samples'].append({'sample': s, 'n': len(rows), 'resolved': sum(r['resolved'] for r in rows),
                                       'with_patch': sum(bool(r.get('has_patch')) for r in rows),
                                       'mean_task_seconds': round(sum(r['task_seconds'] for r in rows) / max(1, len(rows)), 1)})
            print('SAMPLE SUMMARY', json.dumps(summary['samples'][-1]), flush=True)
            train_vol.commit()
    finally:
        try:
            server.stop()
        except Exception:
            pass
    summary['total_minutes'] = round((time.time() - t_start) / 60, 1)
    (out_root / 'summary.json').write_text(json.dumps(summary, indent=1))
    train_vol.commit()
    print('SUMMARY', json.dumps(summary), flush=True)
    return summary


@app.function(timeout=30 * 60, cpu=4, memory=16384, volumes={'/data': data_vol, '/vol': train_vol})
def goldcheck(task_id: str) -> dict:
    """CPU-only diagnostic: apply the reference patch to one task in the subprocess sandbox and run its tests,
    with harness logging at INFO, to verify the sandbox environment matches Kaggle's."""
    import asyncio
    import io
    import logging
    import time

    _install_wheelhouse()
    from swegemma.config import EvalConfig
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.models import load_tasks
    from swegemma.sandbox.subprocess import SubprocessManager
    import swegemma.harness.verification as official

    buf = io.StringIO()
    h = logging.StreamHandler(buf); h.setLevel(logging.INFO)
    logging.getLogger('swegemma').addHandler(h); logging.getLogger('swegemma').setLevel(logging.INFO)
    data = Path('/data/data')
    task = {t.instance_id: t for t in load_tasks(data / 'tasks.jsonl')}[task_id]
    out = Path('/tmp/goldcheck'); out.mkdir(exist_ok=True)
    config = EvalConfig(tasks_path=data / 'tasks.jsonl', snapshots_dir=data / 'snapshots', results_dir=out / 'verify', submission_dir=out,
                        models=None, sandbox='subprocess', wheels_dir=Path('/tmp/task_wheels'), timeout_seconds=300, verification_timeout_seconds=300)
    manager = SubprocessManager(timeout_seconds=300)
    snapshot, base, patch = resolve_task_snapshot_paths(config.snapshots_dir, task.instance_id, task.repo)
    t0 = time.perf_counter()
    res = asyncio.run(official.verify_task(manager, config, task, snapshot, base_snapshot_path=base, patch_path=patch,
                                           agent_patch=task.patch, start_time=time.perf_counter()))
    # probe the sandbox environment too
    probe = {}
    try:
        sid = manager.start()
        for cmd in ('python3 --version', 'python3 -c "import pytest, sys; print(pytest.__version__, sys.executable)"',
                    'python3 -c "import rich, sys; print(rich.__file__)"', 'ls /wheels | head -3', 'echo PATH=$PATH'):
            r = manager.exec(sid, cmd, timeout=60)
            probe[cmd] = {'exit': r.exit_code, 'out': (r.stdout or '')[-300:], 'err': (r.stderr or '')[-300:]}
    except Exception as e:
        probe['error'] = repr(e)
    manager.cleanup_all()
    return {'task': task_id, 'resolved': bool(res.resolved), 'exit_code': res.test_exit_code, 'error': (res.error or '')[:400],
            'seconds': round(time.perf_counter() - t0), 'test_output_tail': (res.test_output or '')[-1500:], 'log_tail': buf.getvalue()[-6000:], 'probe': probe}


@app.local_entrypoint()
def diag(task_id: str = 'rich_3006'):
    for r in goldcheck.map(task_id.split(',')):
        print(json.dumps({k: v for k, v in r.items() if k not in ('log_tail', 'probe')}), flush=True)
        if not r['resolved']:
            print('LOG TAIL:', r['log_tail'][-2000:]); print('PROBE:', json.dumps(r['probe'])[:1200])


@app.local_entrypoint()
def main(candidate: str = '', run_name: str = '', tasks: str = 'gold-valid', n_samples: int = 1, concurrency: int = 4,
         temperature: float = -1.0, download_only: bool = False, out_root: str = 'artifacts/rollouts', adapter: str = '',
         gold: str = 'artifacts/gold-check-20261109/gold_summary.json', statements: str = ''):
    import subprocess
    import yaml

    assert run_name, '--run-name is required'
    if not download_only:
        cand = Path(candidate)
        assert cand.is_dir(), f'candidate dir {candidate} not found'
        files = {str(p.relative_to(cand)): p.read_text() for p in sorted(cand.rglob('*'))
                 if p.is_file() and 'adapters' not in p.relative_to(cand).parts}
        if temperature >= 0:
            samp = yaml.safe_load(files['configs/sampling.yaml']) or {}
            samp['temperature'] = temperature
            files['configs/sampling.yaml'] = yaml.safe_dump(samp)
        budgets = (yaml.safe_load(files['eval_config.yaml']) or {}).get('evaluation', {})
        task_ids = json.loads(Path(gold).read_text())['valid'] if tasks == 'gold-valid' else json.loads(tasks)
        overrides = json.loads(Path(statements).read_text()) if statements else None
        if overrides:
            overrides = {k: v for k, v in overrides.items() if k in task_ids}
        print(f'{len(task_ids)} tasks x {n_samples} samples, concurrency {concurrency}, temperature {temperature if temperature >= 0 else "candidate default"}', flush=True)
        amap = dict(kv.split('=', 1) for kv in adapter.split(',') if kv)  # e.g. main_lora=e4b-lora-r1 (volume dir under /vol/adapters)
        summary = rollout.remote(run_name, files, task_ids, n_samples, concurrency, budgets, amap, overrides)
        print('rollout done:', json.dumps(summary), flush=True)
    dest = Path(out_root) / run_name
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(['modal', 'volume', 'get', '--force', 'g4-training', f'/rollouts/{run_name}', str(dest.parent)], check=True)
    print(f'downloaded to {dest}')
