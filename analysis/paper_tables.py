#!/usr/bin/env python3
"""Per-task results table and the paired statistics quoted in the writeup.

Reads run directories (run_results.jsonl + results/traces) and the public tasks.jsonl; writes
results/per_task.csv with one row per (run, task) holding outcomes and trace-derived counts only
(no task text, patches or tool output), and prints the issue-oracle permutation test.

Usage: python3 analysis/paper_tables.py --rollouts artifacts/rollouts --tasks gemma-4-developer-agent/tasks.jsonl --out results
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from fractions import Fraction
from pathlib import Path

RUNS = {  # label -> (directory under --rollouts, sample)
    '31b_v3_rich': ('31b-v3-val42', 's0'), '31b_v3_fastapi': ('31b-v3-fastapi22', 's0'),
    '31b_v3_roll_s0': ('31b-v3-roll1', 's0'), '31b_v3_roll_s1': ('31b-v3-roll1', 's1'),
    '31b_v3_roll_s2': ('31b-v3-roll1', 's2'), '31b_v3_roll_s3': ('31b-v3-roll1', 's3'),
    '31b_v4_reproduce_rule': ('31b-v4-val42', 's0'), '31b_v5_test_rule': ('31b-v5-val42', 's0'),
    '31b_v3_issue_text_s0': ('31b-v3-issue17', 's0'), '31b_v3_issue_text_s1': ('31b-v3-issue17', 's1'),
    '31b_lora_r1': ('31b-lora-r1-val42', 's0'), '31b_lora_r1_maskfix': ('31b-lora-r1-maskfix-val42', 's0'),
    '31b_lora_r2_rich': ('31b-lora-r2-val42', 's0'), '31b_lora_r2_fastapi': ('31b-lora-r2-fastapi22', 's0'),
    'e4b_base': ('e4b-base-val64', 's0'), 'e4b_lora_r1': ('e4b-lora-r1-val64', 's0'),
    'e4b_lora_r2': ('e4b-lora-r2-val64', 's0'), 'e4b_lora_r3': ('e4b-lora-r3-val64', 's0'),
}


def trace_counts(path: Path) -> dict:
    c = {'agent_turns': 0, 'runaway_turns': 0, 'max_calls_in_turn': 0, 'parser_dropped_edits': 0, 'repeated_calls': 0,
         'graph_tool_calls': 0, 'long_turns': 0}
    if not path.exists():
        return c
    seen = set()
    for s in json.loads(path.read_text())['steps']:
        if s.get('source') != 'agent':
            continue
        calls = s.get('tool_calls') or []
        c['agent_turns'] += 1
        c['runaway_turns'] += len(calls) >= 5
        c['max_calls_in_turn'] = max(c['max_calls_in_turn'], len(calls))
        c['long_turns'] += (s.get('metrics') or {}).get('completion_tokens', 0) > 500
        for tc in calls:
            a = tc.get('arguments') or {}
            key = json.dumps([tc['function_name'], a], sort_keys=True)
            c['repeated_calls'] += key in seen
            seen.add(key)
            c['parser_dropped_edits'] += tc['function_name'] == 'edit_file' and ('old_string' not in a or 'new_string' not in a)
            c['graph_tool_calls'] += tc['function_name'] in ('get_code_neighbors', 'search_similar_code', 'get_code_subgraph')
    return c


def permutation_p(with_info: dict, without: dict) -> tuple[float, Fraction]:
    """Exact one-sided test. Statistic: sum over tasks of (solve rate with issue text - solve rate without).
    Null: within a task, which runs carried the issue text is exchangeable among all its runs."""
    dist = {Fraction(0): Fraction(1)}
    obs = Fraction(0)
    for t in with_info:
        a, b = with_info[t], without[t]
        obs += Fraction(sum(a), len(a)) - Fraction(sum(b), len(b))
        allr = a + b
        combos = list(itertools.combinations(range(len(allr)), len(a)))
        new = {}
        for idx in combos:
            x = [allr[i] for i in idx]; y = [allr[i] for i in range(len(allr)) if i not in idx]
            d = Fraction(sum(x), len(x)) - Fraction(sum(y), len(y))
            for k, p in dist.items():
                new[k + d] = new.get(k + d, 0) + p / len(combos)
        dist = new
    return float(obs), sum(p for k, p in dist.items() if k >= obs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollouts', type=Path, default=Path('artifacts/rollouts'))
    ap.add_argument('--tasks', type=Path, default=Path('gemma-4-developer-agent/tasks.jsonl'))
    ap.add_argument('--trained', type=Path, default=None, help='SFT JSONL; marks tasks that contributed training trajectories')
    ap.add_argument('--out', type=Path, default=Path('results'))
    args = ap.parse_args()
    terse = {}
    for line in args.tasks.read_text().splitlines():
        if line.strip():
            t = json.loads(line); terse[t['instance_id']] = len(t.get('problem_statement') or '') < 200
    trained = set()
    if args.trained:
        trained = {json.loads(l)['task_id'] for l in args.trained.read_text().splitlines() if l.strip()}
    rows = []
    for label, (d, s) in RUNS.items():
        rr = args.rollouts / d / s / 'run_results.jsonl'
        if not rr.exists():
            print(f'skip {label}: {rr} missing'); continue
        for line in rr.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({'run': label, 'task': r['id'], 'repo': r['id'].rsplit('_', 1)[0], 'terse_statement': int(terse[r['id']]),
                         'trained_task': int(r['id'] in trained), 'resolved': int(bool(r.get('resolved'))), 'has_patch': int(bool(r.get('has_patch'))),
                         'budget_error': (r.get('error') or '').replace('Agent exceeded ', '')[:40], 'task_seconds': r.get('task_seconds'),
                         'tool_calls': r.get('tool_calls'), **trace_counts(args.rollouts / d / s / 'results' / 'traces' / f"trace_{r['id']}.json")})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'per_task.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} rows -> {args.out / "per_task.csv"}')

    by = {}
    for r in rows:
        by.setdefault(r['run'], {})[r['task']] = r['resolved']
    issue = [k for k in by if 'issue_text' in k]
    if issue:
        tasks = sorted(by[issue[0]])
        plain = [k for k in ('31b_v3_rich', '31b_v3_roll_s0', '31b_v3_roll_s1', '31b_v3_roll_s2', '31b_v3_roll_s3', '31b_v4_reproduce_rule') if k in by]
        w = {t: [by[k][t] for k in issue] for t in tasks}; wo = {t: [by[k][t] for k in plain] for t in tasks}
        obs, p = permutation_p(w, wo)
        print(f'issue text: {[sum(by[k][t] for t in tasks) for k in issue]} of {len(tasks)}; title only: {[sum(by[k][t] for t in tasks) for k in plain]}')
        print(f'  never solved without, solved with: {sum(1 for t in tasks if not any(wo[t]) and any(w[t]))}; in every issue run: {sum(1 for t in tasks if not any(wo[t]) and all(w[t]))}')
        print(f'  summed rate difference {obs:.2f}, exact one-sided permutation p = {float(p):.2g}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
