#!/usr/bin/env python3
"""Figures 1 and 2 of the writeup from results/per_task.csv.  Usage: python3 analysis/figures.py results/per_task.csv paper/figures"""
import collections
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

src, out = sys.argv[1], sys.argv[2]
by = collections.defaultdict(dict)
for r in csv.DictReader(open(src)):
    by[r['run']][r['task']] = r
INK, SEC, SURF, BLUE, EMPTY = '#0b0b0b', '#52514e', '#fcfcfb', '#2a78d6', '#e6e5e0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK, 'axes.labelcolor': SEC, 'xtick.color': SEC,
                     'ytick.color': SEC, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})

# Figure 1: every run on the 17 terse tasks, with and without the issue body
plain = [('31b_v3_rich', 'v3\nT 0.2'), ('31b_v3_roll_s0', 'v3\nT 0.7'), ('31b_v3_roll_s1', 'v3\nT 0.7'), ('31b_v3_roll_s2', 'v3\nT 0.7'),
         ('31b_v3_roll_s3', 'v3\nT 0.7'), ('31b_v4_reproduce_rule', 'v4\nT 0.2')]
iss = [('31b_v3_issue_text_s0', 'v3\nT 0.2'), ('31b_v3_issue_text_s1', 'v3\nT 0.2')]
cols = plain + iss
ok = lambda k, t: int(by[k][t]['resolved'])
tasks = sorted(by[iss[0][0]], key=lambda t: (-sum(ok(k, t) for k, _ in iss), -sum(ok(k, t) for k, _ in plain)))
xpos = lambda i: i + (0.6 if i >= len(plain) else 0)
fig, ax = plt.subplots(figsize=(6.8, 6.2))
for y, t in enumerate(tasks):
    for i, (k, _) in enumerate(cols):
        ax.add_patch(FancyBboxPatch((xpos(i) + 0.08, y + 0.12), 0.84, 0.76, boxstyle='round,pad=0,rounding_size=0.12',
                                    fc=BLUE if ok(k, t) else EMPTY, ec='none'))
ax.set_xlim(0, len(cols) + 1.1); ax.set_ylim(len(tasks), -0.1)
ax.set_xticks([xpos(i) + 0.5 for i in range(len(cols))]); ax.set_xticklabels([l for _, l in cols], fontsize=8); ax.xaxis.tick_top()
ax.set_yticks([i + 0.5 for i in range(len(tasks))]); ax.set_yticklabels([t.replace('_', ' ') for t in tasks], fontsize=8)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i, (k, _) in enumerate(cols):
    ax.text(xpos(i) + 0.5, len(tasks) + 0.45, f'{sum(ok(k, t) for t in tasks)}/{len(tasks)}', ha='center', va='center', fontsize=8.5)
ax.text(-0.15, len(tasks) + 0.45, 'solved', ha='right', va='center', fontsize=8.5, color=SEC)
ax.text(3, -1.75, 'PR title only', ha='center', fontsize=10, weight='bold')
ax.text(7.6, -1.75, 'Title + issue body', ha='center', fontsize=10, weight='bold')
for x, c, lab in ((0.2, BLUE, 'hidden tests pass'), (3.4, EMPTY, 'not solved')):
    ax.add_patch(FancyBboxPatch((x, len(tasks) + 1.05), 0.5, 0.45, boxstyle='round,pad=0,rounding_size=0.1', fc=c, ec='none', clip_on=False))
    ax.text(x + 0.65, len(tasks) + 1.28, lab, va='center', fontsize=8.5, color=SEC)
fig.subplots_adjust(left=0.15, right=0.99, top=0.84, bottom=0.1)
fig.savefig(f'{out}/fig1_issue_text.png', dpi=200)

# Figure 2: cumulative share of tasks solved against task time, v3 runs
pool = [r for k in ('31b_v3_rich', '31b_v3_fastapi', '31b_v3_roll_s0', '31b_v3_roll_s1', '31b_v3_roll_s2', '31b_v3_roll_s3') for r in by[k].values()]
sol = sorted(float(r['task_seconds']) for r in pool if r['resolved'] == '1'); n = len(pool)
ys = [100 * (i + 1) / n for i in range(len(sol))]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.step([0] + sol + [420], [0] + ys + [ys[-1]], where='post', color=BLUE, lw=2)
ax.axvline(360, color=SEC, lw=1, ls=(0, (3, 3))); ax.text(356, 3, '6-minute budget', ha='right', fontsize=8.5, color=SEC)
for T in (120, 240):
    y = 100 * sum(s <= T for s in sol) / n
    ax.plot([T], [y], 'o', ms=6, color=BLUE, mec=SURF, mew=1.5); ax.text(T + 8, y - 3.2, f'{y:.0f}% by {T} s', fontsize=8.5)
ax.text(418, ys[-1] + 1.5, f'{ys[-1]:.0f}% solved', ha='right', fontsize=8.5)
ax.set_xlim(0, 420); ax.set_ylim(0, 50); ax.set_xlabel('task time (s), including sandbox setup'); ax.set_ylabel('tasks solved (%)')
ax.grid(axis='y', color=EMPTY, lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#c9c8c2'); ax.tick_params(length=0)
fig.tight_layout(); fig.savefig(f'{out}/fig2_budget_curve.png', dpi=200)
