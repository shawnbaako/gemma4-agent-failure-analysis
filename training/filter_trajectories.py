#!/usr/bin/env python3
"""Turn validation-run traces into an SFT dataset of passing agent trajectories.

Input: one or more validation run directories as downloaded from Kaggle
(`run_results.jsonl` + `results/traces/trace_<task>.json`). Output: a JSONL file
where each line is {"task_id", "run", "messages", "tools"} in OpenAI chat format,
ready for `train_e4b_lora.py`.

Only trajectories whose task was *resolved* by the harness are kept. Reference
patches and verification tests are never part of a trace, so they cannot leak in.

Usage:
  python3 training/filter_trajectories.py artifacts/val-31b-v1-20261109 artifacts/val-31b-v3-20261109 \
      --out artifacts/sft/trajectories-20261109.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

TOOL_DECLS = Path(__file__).with_name('tool_declarations.json')
BARE_PYTEST = re.compile(r'(^|[;&|]\s*)(python3?\s+-m\s+)?pytest(\s+-\S+)*\s*($|[;&|])')
ARG_LOSS = 'mandatory input parameters are not present'


def load_run(run_dir: Path) -> list[dict]:
    results = {}
    for line in (run_dir / 'run_results.jsonl').read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            results[r['id']] = r
    rows = []
    for p in sorted((run_dir / 'results' / 'traces').glob('trace_*.json')):
        tid = p.name[len('trace_'):-len('.json')]
        r = results.get(tid)
        if r is None:
            continue
        rows.append({'task_id': tid, 'run': run_dir.name, 'resolved': bool(r.get('resolved')), 'trace': json.loads(p.read_text())})
    return rows


def trace_to_messages(trace: dict, *, keep_thoughts: bool, drop_arg_loss: bool, drop_dups: bool = True) -> tuple[list[dict], Counter]:
    """Convert an adk-eval-core trace into OpenAI chat messages. Returns (messages, flags)."""
    steps = trace['steps']
    flags: Counter = Counter()
    msgs: list[dict] = []
    system = next((s for s in steps if s.get('source') == 'system'), None)
    user = next((s for s in steps if s.get('source') == 'user'), None)
    if not system or not user:
        flags['missing_prompt'] += 1
        return [], flags
    msgs.append({'role': 'system', 'content': system['message']})
    msgs.append({'role': 'user', 'content': user['message']})
    n = 0
    seen_calls: set[str] = set()
    for s in steps:
        if s.get('source') != 'agent':
            continue
        calls = s.get('tool_calls') or []
        thought = (s.get('message') or '').strip()
        if not calls:
            # final assistant text
            if thought:
                msgs.append({'role': 'assistant', 'content': thought})
            continue
        obs = (s.get('observation') or {}).get('content') or ''
        if ARG_LOSS in obs:
            flags['arg_loss'] += 1
            if drop_arg_loss:
                continue
        key = json.dumps([(tc['function_name'], tc.get('arguments') or {}) for tc in calls], sort_keys=True)
        if key in seen_calls:
            # an exact repeat of an earlier call returns the same observation: wasted turn, and the 31B r1 adapter
            # learned to repeat calls from such turns. Drop the turn and its result.
            flags['dup_calls'] += 1
            if drop_dups:
                continue
        seen_calls.add(key)
        for tc in calls:
            fn = tc['function_name']
            args = tc.get('arguments') or {}
            if fn == 'run_command' and BARE_PYTEST.search(args.get('command', '')):
                flags['bare_pytest'] += 1
            n += 1
            call_id = tc.get('tool_call_id') or f'call_{n}'
            msgs.append({
                'role': 'assistant',
                'content': thought if keep_thoughts else '',
                'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': fn, 'arguments': args}}],
            })
            msgs.append({'role': 'tool', 'tool_call_id': call_id, 'name': fn, 'content': obs})
    flags['tool_calls'] = n
    return msgs, flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', type=Path, help='validation run directories')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--max-tool-calls', type=int, default=40, help='drop trajectories longer than this')
    ap.add_argument('--per-task', type=int, default=2, help='keep at most N (shortest) trajectories per task')
    ap.add_argument('--keep-thoughts', action='store_true', help='keep model thoughts as assistant content (default: drop)')
    ap.add_argument('--keep-arg-loss', action='store_true', help='keep tool calls the parser rejected (default: drop them)')
    ap.add_argument('--keep-dups', action='store_true', help='keep turns that exactly repeat an earlier tool call (default: drop them)')
    ap.add_argument('--keep-bare-pytest', action='store_true', help='keep trajectories that ran bare pytest (default: drop)')
    ap.add_argument('--include-unresolved', action='store_true', help='for inspection only; never train on these')
    args = ap.parse_args()

    tools = json.loads(TOOL_DECLS.read_text())
    kept: list[dict] = []
    stats = Counter()
    for run_dir in args.runs:
        for row in load_run(run_dir):
            stats['seen'] += 1
            if not row['resolved'] and not args.include_unresolved:
                stats['unresolved'] += 1
                continue
            msgs, flags = trace_to_messages(row['trace'], keep_thoughts=args.keep_thoughts, drop_arg_loss=not args.keep_arg_loss, drop_dups=not args.keep_dups)
            if not msgs:
                stats['no_messages'] += 1
                continue
            if flags['tool_calls'] > args.max_tool_calls:
                stats['too_long'] += 1
                continue
            if flags['bare_pytest'] and not args.keep_bare_pytest:
                stats['bare_pytest'] += 1
                continue
            if not any(m.get('tool_calls') and m['tool_calls'][0]['function']['name'] == 'submit_patch' for m in msgs):
                stats['no_submit'] += 1
                continue
            stats['arg_loss_calls_dropped'] += flags['arg_loss'] if not args.keep_arg_loss else 0
            stats['dup_turns_dropped'] += flags['dup_calls'] if not args.keep_dups else 0
            kept.append({'task_id': row['task_id'], 'run': row['run'], 'resolved': row['resolved'], 'n_tool_calls': flags['tool_calls'], 'messages': msgs, 'tools': tools})

    # per-task cap: keep the shortest trajectories
    by_task: dict[str, list[dict]] = {}
    for k in kept:
        by_task.setdefault(k['task_id'], []).append(k)
    final = []
    for tid, rows in sorted(by_task.items()):
        rows.sort(key=lambda r: r['n_tool_calls'])
        final.extend(rows[: args.per_task])
        stats['per_task_dropped'] += max(0, len(rows) - args.per_task)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    chars = sum(len(json.dumps(r['messages'])) for r in final)
    print(f'kept {len(final)} trajectories over {len(by_task)} tasks -> {args.out}  (~{chars/4/1000:.0f}k tokens est.)')
    print('stats:', dict(stats))
    return 0


if __name__ == '__main__':
    sys.exit(main())
