#!/usr/bin/env python3
"""Build issue-augmented problem statements for the terse-statement tasks (analysis only).

For each task whose statement is under 200 characters and links a public GitHub issue, fetch the issue body
(no comments), strip template boilerplate, and append it to the statement. Tasks whose issue body contains a
line of the reference patch are reported and excluded, so the reference fix never reaches the agent's prompt.

Usage:
  python3 analysis/issue_oracle.py --tasks gemma-4-developer-agent/tasks.jsonl --pool pool42.json --out issue-oracle
  modal run training/rollout_modal.py::main --candidate <dir> --run-name <name> --tasks "$(cat issue-oracle/tasks.json)" \
      --statements issue-oracle/statement_overrides.json --n-samples 2 --temperature 0.2
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


def clean(body: str) -> str:
    b = body.replace('\r', '')
    b = re.sub(r'<!--.*?-->', '', b, flags=re.S)
    b = re.sub(r'<details>.*?</details>', '', b, flags=re.S)   # platform dumps
    b = re.sub(r'!\[[^\]]*\]\([^)]*\)', '[image]', b)
    b = re.sub(r'- \[[xX ]\] I.*\n', '', b)                    # bug-template checklist
    b = re.sub(r'\*\*Platform\*\*.*', '', b, flags=re.S)
    return re.sub(r'\n{3,}', '\n\n', b).strip()[:3000]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=Path, required=True)
    ap.add_argument('--pool', type=Path, required=True, help='JSON list of task ids')
    ap.add_argument('--out', type=Path, default=Path('issue-oracle'))
    args = ap.parse_args()
    tasks = {json.loads(l)['instance_id']: json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()}
    raw = args.out / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    overrides, excluded = {}, []
    for tid in json.loads(args.pool.read_text()):
        ps = tasks[tid]['problem_statement']
        if len(ps) >= 200:
            continue
        parts = [ps.replace('\r', '').strip(), '']
        for owner, repo, n in re.findall(r'https://github.com/([\w-]+)/([\w-]+)/issues/(\d+)', ps):
            p = raw / f'{owner}_{repo}_{n}.json'
            if not p.exists():
                req = urllib.request.Request(f'https://api.github.com/repos/{owner}/{repo}/issues/{n}',
                                             headers={'User-Agent': 'research-script', 'Accept': 'application/vnd.github+json'})
                p.write_bytes(urllib.request.urlopen(req, timeout=30).read()); time.sleep(1)
            j = json.loads(p.read_text())
            parts.append(f"Linked issue {owner}/{repo}#{n}: {j.get('title')}\n\n{clean(j.get('body') or '')}\n")
        if len(parts) == 2:
            continue  # no linked issue
        text = '\n'.join(parts)
        added = [l[1:].strip() for l in tasks[tid]['patch'].splitlines() if l.startswith('+') and not l.startswith('+++')]
        if any(len(a) >= 25 and a in text for a in added):
            excluded.append(tid); continue
        overrides[tid] = text
    (args.out / 'statement_overrides.json').write_text(json.dumps(overrides, indent=1))
    (args.out / 'tasks.json').write_text(json.dumps(sorted(overrides)))
    print(f'{len(overrides)} statements built; excluded for containing reference-patch lines: {excluded}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
