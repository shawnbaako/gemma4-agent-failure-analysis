#!/usr/bin/env python3
"""Check that a fine-tuning loss mask supervises every token the serving stack treats as a stop.

A chat model ends its output by predicting one of the ids in `generation_config.eos_token_id`. For tool-calling
models that list can include a token that, in a rendered training example, is the *first token of the next
segment* (Gemma 4 ends a tool-call turn on `<|tool_response>`). A mask built from turn boundaries then leaves it
unsupervised: training metrics look normal and the adapter never learns to stop after a tool call.

The check is model-agnostic. A stop position is a token whose id is a generation-end id and whose predecessor is
supervised (the model was writing, then had to stop). Every such position should itself be supervised.

Library use:
    from check_stop_supervision import check_example, stop_ids
    ids = stop_ids('google/gemma-4-31b-it')
    report = check_example(input_ids, labels, ids)      # labels use -100 for unsupervised positions
    assert not report['missed'], report

CLI (JSONL with `input_ids` and `labels` per line, as produced by your collator):
    python3 check_stop_supervision.py --model google/gemma-4-31b-it --data tokenized.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter


def stop_ids(model: str, token: str | None = None) -> list[int]:
    from transformers import GenerationConfig
    eos = GenerationConfig.from_pretrained(model, token=token).eos_token_id
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


def check_example(input_ids: list[int], labels: list[int], stops: list[int], ignore_index: int = -100) -> dict:
    """Return counts of stop positions and the ids of those left unsupervised."""
    stops_set = set(stops)
    seen, missed = Counter(), Counter()
    for i in range(1, len(input_ids)):
        if input_ids[i] in stops_set and labels[i - 1] != ignore_index:
            seen[input_ids[i]] += 1
            if labels[i] == ignore_index:
                missed[input_ids[i]] += 1
    return {'stop_positions': dict(seen), 'missed': dict(missed)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='model id whose generation_config lists the stop ids')
    ap.add_argument('--data', required=True, help='JSONL with input_ids and labels')
    ap.add_argument('--hf-token', default=None)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    stops = stop_ids(args.model, args.hf_token)
    tok = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    seen, missed = Counter(), Counter()
    with open(args.data) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                r = check_example(ex['input_ids'], ex['labels'], stops)
                seen.update(r['stop_positions']); missed.update(r['missed'])
    print(f'generation-end ids: {stops} = {tok.convert_ids_to_tokens(stops)}')
    for i in stops:
        name = tok.convert_ids_to_tokens(i)
        print(f'  {name!r}: {seen[i]} stop positions, {missed[i]} unsupervised')
    if sum(missed.values()):
        print('FAIL: the model is never trained to emit these stops; expect runaway generation after fine-tuning.')
        return 1
    print('OK: every stop position is supervised.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
