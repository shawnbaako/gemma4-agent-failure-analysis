# Tool-Call Mechanics, Stop-Token Supervision and Missing Issue Text: A Failure Analysis of Gemma 4 as a Coding Agent

Code, per-task results and the writeup for a study of Gemma 4 (31B int4, E4B) as a bug-fixing agent in the
official harness of the Google Gemma 4 Developer Agent competition.

- `paper/writeup.md`: the paper. Figures in `paper/figures/`.
- `results/per_task.csv`: one row per (run, task) for every run cited: solved, patch produced, budget error, time,
  tool calls and trace-derived counts (runaway turns, parser-dropped edits, repeated calls, graph-tool calls).
  No task text, patches or tool output.
- `results/stop_probe.json`: probability of the `<|tool_response>` stop at 183 held-out stop positions, per adapter.
- `results/stop_audit.json`: tokens supervised by each loss-mask recipe.
- `results/pools.json`, `results/issue_oracle_tasks.json`: task ids of the pools and of the issue-text experiment.
- `prompts/measured_rules.md`: the prompt rules whose effect the paper measures.

## Check your own fine-tuning mask

```
python3 analysis/check_stop_supervision.py --model google/gemma-4-31b-it --data tokenized.jsonl
```

`tokenized.jsonl` holds `input_ids` and `labels` per line as your collator produces them. The script reads the
model's generation-end ids and fails if any position where the model must stop is unlabeled. It is model-agnostic.

## Reproduce

Requirements: the competition dataset (tasks, snapshots, wheels, graphs, embeddings and the harness wheelhouse; accept the
rules on Kaggle to download it), a Hugging Face token with access to Gemma 4, and a Modal account. The competition data is
not redistributed here.

| Result | Command |
|---|---|
| Rollouts and validation runs (Tables 1, 2, 4; Figure 2) | `modal run training/rollout_modal.py::main --candidate <dir> --run-name <name> --tasks "$(cat pool.json)" --temperature 0.2` |
| SFT data from passing runs | `python3 training/filter_trajectories.py <run dirs> --out sft.jsonl` |
| LoRA training (corrected mask) | `modal run training/train_e4b_lora.py --data sft.jsonl --run-name <name> --base-model google/gemma-4-31b-it --gpu H200 --lr 5e-5` |
| Stop probability (Table 2) | `modal run analysis/stop_token_probe.py::probe --data heldout.jsonl --adapters <names>` |
| Mask recipe audit (Table 3) | `modal run analysis/stop_token_probe.py::audit --data heldout.jsonl` |
| Issue-text statements (Figure 1) | `python3 analysis/issue_oracle.py --tasks tasks.jsonl --pool pool.json`, then the rollout command with `--statements` |
| Per-task table and permutation test | `python3 analysis/paper_tables.py --rollouts <dir> --tasks tasks.jsonl --trained sft.jsonl` |
| Figures | `python3 analysis/figures.py results/per_task.csv paper/figures` |
| Mechanics counts (Table 1) | `python3 analysis/stratified.py --gold gold_summary.json --run v1=<dir> --run v3=<dir>` |

The turns-only mask that produced the looping adapter is `_mask_turns_only` in `analysis/stop_token_probe.py`; the
corrected mask is `_assistant_mask` in `training/train_e4b_lora.py`.

Adapter weights are not released: none improved the solve rate. Full candidate directories (complete system prompts and
configs) will be added after the competition's final deadline.

## License

Apache License 2.0; see `LICENSE`. Copyright 2026 Shawn Baako.
