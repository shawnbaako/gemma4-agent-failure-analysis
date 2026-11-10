"""LoRA fine-tune of Gemma 4 (E4B by default, 31B with --base-model/--gpu) on passing agent trajectories, run on Modal.

Usage (from the repo root):
  modal run training/train_e4b_lora.py --data artifacts/sft/trajectories.jsonl --run-name e4b-lora-v1
  modal run training/train_e4b_lora.py --data artifacts/sft/x.jsonl --run-name 31b-lora-r1 --base-model google/gemma-4-31b-it --gpu H100:2
  modal run training/train_e4b_lora.py --run-name e4b-lora-v1 --download-only

Cost control: this is an ephemeral `modal run` app. The GPU container exists only
while `train()` executes and is torn down when it returns; nothing is deployed and
no `min_containers`/`keep_warm` is set, so idle cost is zero. The only persistent
objects are the `g4-training` volume (data, HF cache, adapters) and the secret.

Setup once:
  modal secret create huggingface HF_TOKEN=<your HF token>   # gated Gemma weights

The adapter is trained on the bf16 instruct checkpoint and served by the harness on
the int4 QAT checkpoint (`gemma-4-e4b-it-qat-w4a16-ct`), which vLLM loads with the
adapter via --lora-modules. Target modules are the standard linear projections so the
module names line up with what vLLM expects.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP_NAME = 'g4-e4b-lora'
VOLUME_NAME = 'g4-training'
BASE_MODEL_DEFAULT = 'google/gemma-4-e4b-it'

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install(
        'torch==2.9.0',
        'transformers>=4.57',
        'peft>=0.17',
        'accelerate>=1.10',
        'datasets>=3.0',
        'sentencepiece',
        'protobuf',
        'huggingface_hub>=0.34',
    )
    .env({'HF_HOME': '/vol/hf', 'TOKENIZERS_PARALLELISM': 'false'})
)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _assistant_mask(tokenizer, ids: list[int]) -> list[bool]:
    """Gemma 4 layout: <|turn>model ... <turn|> is one model turn; inside it, tool results appear as
    <|tool_response>...<tool_response|> blocks that the model did not generate. Supervise model-turn
    tokens (including the closing <turn|> and the <|tool_response> stop token after each call) and nothing else."""
    def tid(tok):
        i = tokenizer.convert_tokens_to_ids(tok)
        if i is None or i == tokenizer.unk_token_id:
            raise RuntimeError(f'special token {tok!r} not in vocab')
        return i
    T_OPEN, T_CLOSE, R_OPEN, R_CLOSE = tid('<|turn>'), tid('<turn|>'), tid('<|tool_response>'), tid('<tool_response|>')
    model_ids = set(tokenizer.encode('model', add_special_tokens=False))
    mask = [False] * len(ids)
    in_model = in_resp = False
    i = 0
    while i < len(ids):
        t = ids[i]
        if t == T_OPEN:
            in_model = i + 1 < len(ids) and ids[i + 1] in model_ids
            i += 2  # skip '<|turn>' and the role token; the newline after the role is not supervised either
            if in_model and i < len(ids):
                i += 1
            continue
        if in_model and t == R_OPEN:
            # '<|tool_response>' is one of Gemma 4's generation end tokens (generation_config eos_token_id =
            # [<eos>, <turn|>, <|tool_response>]): the model ends a tool-call turn by predicting it. Supervise it,
            # or the adapter is never taught to stop after a call and drifts into emitting '<|tool_call>' again
            # (31B r1: hundreds of identical calls per turn). The response body that follows stays unsupervised.
            mask[i] = True
            in_resp = True
        if in_model and not in_resp:
            mask[i] = True
        if t == R_CLOSE:
            in_resp = False
        if t == T_CLOSE:
            in_model = False
        i += 1
    return mask


def _build_examples(tokenizer, rows: list[dict], max_len: int):
    """Tokenize each trajectory with the model's chat template; loss only on model-generated tokens."""
    import torch

    examples, dropped = [], 0
    for row in rows:
        ids = tokenizer.apply_chat_template(row['messages'], tools=row['tools'], tokenize=True)
        if hasattr(ids, 'input_ids'):
            ids = ids['input_ids']
        mask = _assistant_mask(tokenizer, ids)
        if max_len <= 0:  # inspect mode
            sup = [t for t, m in zip(ids, mask) if m]
            examples.append({'n_tokens': len(ids), 'supervised': len(sup), 'first_supervised': tokenizer.decode(sup[:60]), 'last_supervised': tokenizer.decode(sup[-40:])})
            continue
        if len(ids) > max_len or not any(mask):
            dropped += 1
            continue
        labels = [t if m else -100 for t, m in zip(ids, mask)]
        examples.append({'input_ids': torch.tensor(ids), 'labels': torch.tensor(labels)})
    return examples, dropped


def _attn_impl(base_model: str):
    """sdpa for the language model; eager for whichever towers (audio/vision) this checkpoint has."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(base_model, token=os.environ['HF_TOKEN'])
    impl = {}
    for sub in ('text_config', 'audio_config', 'vision_config'):
        if getattr(cfg, sub, None) is not None:
            impl[sub] = 'sdpa' if sub == 'text_config' else 'eager'
    return impl or 'sdpa'


def _train_impl(run_name: str, data_rel: str, base_model: str, epochs: float, lr: float, rank: int, alpha: int, max_len: int, grad_accum: int) -> dict:
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    t0 = time.time()
    out_dir = Path('/vol/adapters') / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path('/vol/data', data_rel).read_text().splitlines() if l.strip()]
    print(f'{len(rows)} trajectories', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.environ['HF_TOKEN'])
    examples, dropped = _build_examples(tokenizer, rows, max_len)
    n_tok = sum(len(e['input_ids']) for e in examples)
    n_lab = sum(int((e['labels'] != -100).sum()) for e in examples)
    print(f'{len(examples)} examples ({dropped} dropped), {n_tok} tokens, {n_lab} supervised', flush=True)

    n_gpu = torch.cuda.device_count()
    device_map = 'cuda' if n_gpu == 1 else 'auto'  # >1 GPU: naive model parallel (layers split across devices)
    attn = _attn_impl(base_model)
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map=device_map, token=os.environ['HF_TOKEN'], attn_implementation=attn)
        print(f'attention: {attn}', flush=True)
    except Exception as e:  # older transformers: fall back to eager everywhere
        print(f'per-submodel attention not accepted ({type(e).__name__}); loading with eager', flush=True)
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map=device_map, token=os.environ['HF_TOKEN'], attn_implementation='eager')
    print(type(model).__name__, f'on {n_gpu} GPU(s), device_map={device_map}', flush=True)
    if n_gpu > 1:
        print('device map:', dict(list(model.hf_device_map.items())[:3]), '...', flush=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # Gemma 4 wraps some projections in Gemma4ClippableLinear(.linear) and shares projection names with the
    # audio/vision towers, so pick the language model's nn.Linear leaves by full name instead of by suffix.
    import torch.nn as nn
    PROJ = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or 'audio' in name or 'vision' in name:
            continue
        parts = name.split('.')
        leaf = parts[-2] if parts[-1] == 'linear' and len(parts) > 1 else parts[-1]
        if leaf in PROJ and 'layers.' in name:
            targets.append(name)
    print(f'{len(targets)} LoRA targets, e.g. {targets[:3]} ... {targets[-1:]}', flush=True)
    if not targets:
        raise RuntimeError('no LoRA target modules found')
    lcfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=targets)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    def collate(batch):
        m = max(len(b['input_ids']) for b in batch)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ids = torch.full((len(batch), m), pad, dtype=torch.long)
        lab = torch.full((len(batch), m), -100, dtype=torch.long)
        att = torch.zeros((len(batch), m), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b['input_ids'])
            ids[i, :n] = b['input_ids']; lab[i, :n] = b['labels']; att[i, :n] = 1
        return {'input_ids': ids, 'labels': lab, 'attention_mask': att}

    targs = TrainingArguments(
        output_dir=str(out_dir / 'trainer'), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=1, gradient_accumulation_steps=grad_accum,
        lr_scheduler_type='cosine', warmup_steps=max(1, int(0.05 * epochs * len(examples) / grad_accum)), logging_steps=1, save_strategy='no',
        bf16=True, gradient_checkpointing=True, report_to=[], remove_unused_columns=False,
    )
    import torch.nn.functional as F

    class SparseLogitsTrainer(Trainer):
        """Compute logits only at supervised positions. Gemma's 262k vocab makes full-sequence logits
        ~26 GB at 25k tokens; supervised positions are ~13% of tokens."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            ids, att, labels = inputs['input_ids'], inputs['attention_mask'], inputs['labels']
            assert ids.shape[0] == 1, 'batch size must be 1'
            tgt = labels[0, 1:]
            pos = (tgt != -100).nonzero().squeeze(-1)  # hidden state at pos predicts token at pos+1
            # pass positions as a python list: accelerate's device hook moves tensor args to GPU 0, which breaks
            # the final-layer indexing when the model is split across GPUs; a list is left alone and indexes anywhere
            out = model(input_ids=ids, attention_mask=att, logits_to_keep=pos.tolist())
            logits = out.logits[0].float()
            loss = F.cross_entropy(logits, tgt[pos].to(logits.device))
            return (loss, out) if return_outputs else loss

    trainer = SparseLogitsTrainer(model=model, args=targs, train_dataset=examples, data_collator=collate)
    result = trainer.train()
    model.save_pretrained(str(out_dir), safe_serialization=True)
    summary = {
        'run_name': run_name, 'base_model': base_model, 'examples': len(examples), 'dropped': dropped,
        'tokens': n_tok, 'supervised_tokens': n_lab, 'epochs': epochs, 'lr': lr, 'rank': rank, 'alpha': alpha,
        'train_loss': result.training_loss, 'log_history': trainer.state.log_history,
        'seconds': round(time.time() - t0), 'gpu': torch.cuda.get_device_name(0),
    }
    (out_dir / 'training_summary.json').write_text(json.dumps(summary, indent=1))
    vol.commit()
    print(json.dumps({k: v for k, v in summary.items() if k != 'log_history'}, indent=1), flush=True)
    return {k: v for k, v in summary.items() if k != 'log_history'}


_FN = dict(timeout=3 * 60 * 60, volumes={'/vol': vol}, secrets=[modal.Secret.from_name('huggingface')])


@app.function(gpu='H100', memory=65536, **_FN)
def train(*a) -> dict:  # E4B: one H100
    return _train_impl(*a)


@app.function(gpu='H100:2', memory=131072, **_FN)
def train_2xh100(*a) -> dict:  # 31B bf16 (62 GB weights): two H100s, layers split across devices
    return _train_impl(*a)


@app.function(gpu='H200', memory=131072, **_FN)
def train_h200(*a) -> dict:  # 31B alternative on one 141 GB GPU
    return _train_impl(*a)


TRAIN_FNS = {'H100': train, 'H100:2': train_2xh100, 'H200': train_h200}


@app.function(timeout=15 * 60, volumes={'/vol': vol}, secrets=[modal.Secret.from_name('huggingface')], memory=16384)
def inspect_data(data_rel: str, base_model: str) -> dict:
    """CPU-only: render trajectories through the chat template and report token counts and assistant spans."""
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in Path('/vol/data', data_rel).read_text().splitlines() if l.strip()]
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.environ['HF_TOKEN'])
    info, _ = _build_examples(tokenizer, rows[:4], max_len=0)
    text = tokenizer.apply_chat_template(rows[0]['messages'][:4], tools=rows[0]['tools'], tokenize=False)
    return {'info': info, 'rendered_tail': text[-700:]}


@app.local_entrypoint()
def main(run_name: str, data: str = '', base_model: str = BASE_MODEL_DEFAULT, epochs: float = 2.0, lr: float = 1e-4,
         rank: int = 16, alpha: int = 32, max_len: int = 32768, grad_accum: int = 8, download_only: bool = False, inspect: bool = False,
         out_root: str = 'artifacts/adapters', gpu: str = 'H100'):
    if gpu not in TRAIN_FNS:
        raise SystemExit(f'--gpu must be one of {list(TRAIN_FNS)}')
    if inspect:
        src = Path(data); data_rel = f'{run_name}/{src.name}'
        with vol.batch_upload(force=True) as b:
            b.put_file(str(src), f'/data/{data_rel}')
        out = inspect_data.remote(data_rel, base_model)
        print(json.dumps(out['info'], indent=1)); print('--- tail ---'); print(out['rendered_tail'])
        return
    if not download_only:
        if not data:
            raise SystemExit('--data <trajectories.jsonl> is required unless --download-only')
        src = Path(data)
        data_rel = f'{run_name}/{src.name}'
        with vol.batch_upload(force=True) as b:
            b.put_file(str(src), f'/data/{data_rel}')
        print(f'uploaded {src} -> volume:/data/{data_rel}', flush=True)
        summary = TRAIN_FNS[gpu].remote(run_name, data_rel, base_model, epochs, lr, rank, alpha, max_len, grad_accum)
        print('train done:', summary, flush=True)
    dest = Path(out_root) / run_name
    dest.mkdir(parents=True, exist_ok=True)
    for entry in vol.listdir(f'/adapters/{run_name}', recursive=False):
        name = entry.path.rsplit('/', 1)[-1]
        if name == 'trainer':
            continue
        with (dest / name).open('wb') as f:
            for chunk in vol.read_file(f'/adapters/{run_name}/{name}'):
                f.write(chunk)
    print(f'adapter downloaded to {dest}:', sorted(p.name for p in dest.iterdir()))
