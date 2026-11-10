#!/usr/bin/env python3
"""Stratified analysis of validation runs against gold-patch features of the public tasks.

Builds, for every public task, a feature row from tasks.jsonl (patch shape, edit-text characters,
problem-statement length, gold validity in the sandbox) and joins per-task outcomes from validation
runs (run_results.jsonl, optionally multi-candidate). Reports solve rates per stratum, budget-aware
solve curves (resolved within T minutes) and trace-derived mechanics counts.

Usage:
  python3 analysis/stratified.py --gold artifacts/gold-check-20261109/gold_summary.json \
      --run 31b-v1=artifacts/val-31b-v1-20261109 --run 31b-v3=artifacts/val-31b-v3-20261109 \
      [--run e4b=artifacts/val-e4b-r1-20261110] --out artifacts/paper
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARG_LOSS = 'mandatory input parameters are not present'


def task_features(tasks_path: Path, gold_valid: set[str]) -> dict[str, dict]:
    feats = {}
    for line in tasks_path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        patch = t.get('patch') or ''
        files = re.findall(r'^\+\+\+ b/(\S+)', patch, re.M)
        changed = [l[1:] for l in patch.splitlines() if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
        added = sum(1 for l in patch.splitlines() if l.startswith('+') and not l.startswith('+++'))
        hunks = len(re.findall(r'^@@', patch, re.M))
        ps = t.get('problem_statement') or ''
        feats[t['instance_id']] = {
            'id': t['instance_id'], 'repo': t['repo'].split('/')[-1], 'gold_valid': t['instance_id'] in gold_valid,
            'n_files': len(files), 'added': added, 'hunks': hunks,
            'single_file': len(files) == 1, 'small': added <= 30,
            'quote_in_edit': any('"' in l for l in changed), 'backtick_in_edit': any('`' in l for l in changed),
            'ps_chars': len(ps), 'ps_terse': len(ps) < 200, 'ps_has_code_span': '`' in ps,
            'test_files': len(re.findall(r'^\+\+\+ b/(\S+)', t.get('test_patch') or '', re.M)),
        }
    return feats


def load_run(name: str, run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / 'run_results.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cand = r.get('candidate')
        r['run'] = f'{name}:{cand}' if cand else name
        rows.append(r)
    # trace-derived mechanics
    trace_dirs = [run_dir / 'results' / 'traces'] + sorted((run_dir / 'results').glob('*/traces'))
    by_key = {}
    for td in trace_dirs:
        cand = td.parent.name if td.parent.name != 'results' else None
        for p in td.glob('trace_*.json'):
            tid = p.name[len('trace_'):-len('.json')]
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            m = Counter()
            first_edit = None
            n = 0
            for s in d['steps']:
                if s.get('source') != 'agent':
                    continue
                obs = (s.get('observation') or {}).get('content') or ''
                for tc in s.get('tool_calls') or []:
                    n += 1
                    fn = tc['function_name']
                    a = tc.get('arguments') or {}
                    if fn == 'edit_file':
                        if 'old_string' not in a or 'new_string' not in a:
                            m['edit_arg_loss'] += 1
                        elif '"status": "error"' in obs:
                            m['edit_failed'] += 1
                        else:
                            m['edit_ok'] += 1
                        first_edit = first_edit or n
                    if fn == 'write_file':
                        m['write_file'] += 1
                        first_edit = first_edit or n
                    if fn == 'read_file':
                        m['reads'] += 1
                        if a.get('start_line') is None:
                            m['reads_unranged'] += 1
                    if fn == 'run_command' and re.search(r'(^|[;&|]\s*)(python3?\s+-m\s+)?pytest(\s+-\S+)*\s*($|[;&|])', a.get('command', '')):
                        m['bare_pytest'] += 1
                    if fn == 'submit_patch':
                        m['submit'] += 1
                met = s.get('metrics') or {}
                if met.get('completion_tokens', 0) > 500:
                    m['long_turns'] += 1
                m['completion_tokens'] += met.get('completion_tokens', 0)
            by_key[(cand, tid)] = dict(m, first_edit_call=first_edit)
    for r in rows:
        r.update(by_key.get((r.get('candidate'), r['id']), {}))
    return rows


def rate(rows, key=lambda r: r['resolved']):
    n = len(rows)
    return f'{sum(1 for r in rows if key(r))}/{n} ({(sum(1 for r in rows if key(r)) / n * 100) if n else 0:.0f}%)'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=Path, default=Path('gemma-4-developer-agent/tasks.jsonl'))
    ap.add_argument('--gold', type=Path, required=True)
    ap.add_argument('--run', action='append', default=[], help='name=run_dir')
    ap.add_argument('--out', type=Path, default=Path('artifacts/paper'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gold_valid = set(json.loads(args.gold.read_text())['valid'])
    feats = task_features(args.tasks, gold_valid)
    with (args.out / 'task_features.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(feats.values())).keys()))
        w.writeheader(); w.writerows(feats.values())

    outcomes = []
    for spec in args.run:
        name, path = spec.split('=', 1)
        for r in load_run(name, Path(path)):
            if r['id'] in feats:
                outcomes.append({**feats[r['id']], **{k: v for k, v in r.items() if k not in ('id', 'repo')}})
    keys = sorted({k for o in outcomes for k in o})
    with (args.out / 'outcomes.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(outcomes)

    md = []
    md.append('# Stratified results\n')
    # 1. task population
    allf = list(feats.values())
    md.append('## Public task population (129 tasks)\n')
    md.append('| feature | all | gold-valid |\n|---|---|---|')
    for label, fn in [('single file', lambda t: t['single_file']), ('<=30 added lines', lambda t: t['small']),
                      ('edit text contains "', lambda t: t['quote_in_edit']), ('edit text contains `', lambda t: t['backtick_in_edit']),
                      ('problem statement < 200 chars', lambda t: t['ps_terse']), ('gold passes in sandbox', lambda t: t['gold_valid'])]:
        md.append(f'| {label} | {rate(allf, fn)} | {rate([t for t in allf if t["gold_valid"]], fn)} |')
    repo_c = Counter((t['repo'], t['gold_valid']) for t in allf)
    md.append('\nGold-valid by repo: ' + ', '.join(f'{r} {repo_c[(r, True)]}/{repo_c[(r, True)] + repo_c[(r, False)]}' for r in sorted({t['repo'] for t in allf})) + '\n')

    runs = sorted({o['run'] for o in outcomes})
    gv = [o for o in outcomes if o['gold_valid']]
    # 2. solve rate by stratum per run
    md.append('## Solve rate by stratum (gold-valid tasks only)\n')
    strata = [('all', lambda t: True), ('single file', lambda t: t['single_file']), ('multi file', lambda t: not t['single_file']),
              ('<=30 added', lambda t: t['small']), ('>30 added', lambda t: not t['small']),
              ('quote in edit', lambda t: t['quote_in_edit']), ('no quote in edit', lambda t: not t['quote_in_edit']),
              ('backtick in edit', lambda t: t['backtick_in_edit']),
              ('terse statement', lambda t: t['ps_terse']), ('long statement', lambda t: not t['ps_terse'])]
    md.append('| stratum | ' + ' | '.join(runs) + ' |\n|---|' + '---|' * len(runs))
    for label, fn in strata:
        md.append(f'| {label} | ' + ' | '.join(rate([o for o in gv if o['run'] == r and fn(o)]) for r in runs) + ' |')
    # 3. budget curves
    md.append('\n## Budget-aware solve rate (gold-valid; resolved with task time <= T)\n')
    md.append('| T | ' + ' | '.join(runs) + ' |\n|---|' + '---|' * len(runs))
    for T in (60, 120, 180, 240, 300, 360, 9999):
        md.append(f'| {"any" if T == 9999 else str(T) + " s"} | ' + ' | '.join(rate([o for o in gv if o['run'] == r], lambda o, T=T: o['resolved'] and o['task_seconds'] <= T) for r in runs) + ' |')
    # 4. mechanics
    md.append('\n## Mechanics per run (all tasks in run)\n')
    md.append('| run | tasks | solved | no patch | timeouts | edit ok | edit arg-loss | edit failed | write_file | unranged reads | bare pytest | long turns (>500 tok) | median first edit call | median task s |\n|---|' + '---|' * 13)
    for r in runs:
        rs = [o for o in outcomes if o['run'] == r]
        fe = [o['first_edit_call'] for o in rs if o.get('first_edit_call')]
        md.append(f"| {r} | {len(rs)} | {sum(o['resolved'] for o in rs)} | {sum(not o.get('has_patch') for o in rs)} | {sum('timeout' in (o.get('error') or '') for o in rs)} | "
                  f"{sum(o.get('edit_ok', 0) for o in rs)} | {sum(o.get('edit_arg_loss', 0) for o in rs)} | {sum(o.get('edit_failed', 0) for o in rs)} | {sum(o.get('write_file', 0) for o in rs)} | "
                  f"{sum(o.get('reads_unranged', 0) for o in rs)}/{sum(o.get('reads', 0) for o in rs)} | {sum(o.get('bare_pytest', 0) for o in rs)} | {sum(o.get('long_turns', 0) for o in rs)} | "
                  f"{statistics.median(fe) if fe else '-'} | {statistics.median(o['task_seconds'] for o in rs):.0f} |")
    # 5. outcome vs mechanics
    md.append('\n## Outcome conditioned on mechanics (gold-valid)\n')
    md.append('| condition | ' + ' | '.join(runs) + ' |\n|---|' + '---|' * len(runs))
    conds = [('first edit by call 15', lambda o: (o.get('first_edit_call') or 99) <= 15), ('first edit after call 15 or never', lambda o: (o.get('first_edit_call') or 99) > 15),
             ('any edit arg-loss', lambda o: o.get('edit_arg_loss', 0) > 0), ('no edit arg-loss', lambda o: o.get('edit_arg_loss', 0) == 0),
             ('called submit_patch', lambda o: o.get('submit', 0) > 0), ('never submitted', lambda o: o.get('submit', 0) == 0)]
    for label, fn in conds:
        md.append(f'| {label} | ' + ' | '.join(rate([o for o in gv if o['run'] == r and fn(o)]) for r in runs) + ' |')
    text = '\n'.join(md) + '\n'
    (args.out / 'stratified.md').write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
