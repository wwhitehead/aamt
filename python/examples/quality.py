#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""How much does baking to `.aamt` change a model? Original vs `.aamt`, side by side.

    python python/examples/quality.py qwen3-0.6b.aamt Qwen/Qwen3-0.6B

Both models run in Hugging Face transformers on the same device, in float32:
the original from its own checkpoint, the other with every weight from the
`.aamt` file. On GSM8K test text (question + worked answer; MIT licence) it
reports:

- perplexity of each model, and the relative change;
- top-1 agreement: how often both models predict the same next token;
- mean KL divergence, original ‖ .aamt, per token;
- greedy generation on the first N questions: how many outputs are identical,
  and how many tokens match before the first difference.

Nothing is sampled, so a rerun gives the same numbers on the same device.
`--json` writes them for the record.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from aamt.hf import load_causal_lm  # noqa: E402


def gsm8k_test() -> list[dict]:
    path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    return pq.read_table(path).to_pylist()


@torch.no_grad()
def compare_logits(a, b, tok, texts: list[str], max_len: int, device: str) -> dict:
    nll_a = nll_b = kl = agree = n = 0.0
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].to(device)
        la = a(ids).logits[0, :-1].float()
        lb = b(ids).logits[0, :-1].float()
        target = ids[0, 1:]
        pa, pb = F.log_softmax(la, -1), F.log_softmax(lb, -1)
        nll_a += -pa.gather(1, target[:, None]).sum().item()
        nll_b += -pb.gather(1, target[:, None]).sum().item()
        kl += (pa.exp() * (pa - pb)).sum().item()
        agree += (la.argmax(-1) == lb.argmax(-1)).sum().item()
        n += target.numel()
    ppl_a, ppl_b = math.exp(nll_a / n), math.exp(nll_b / n)
    return {
        "tokens": int(n),
        "perplexity_original": ppl_a,
        "perplexity_aamt": ppl_b,
        "perplexity_change_pct": (ppl_b / ppl_a - 1) * 100,
        "top1_agreement_pct": agree / n * 100,
        "mean_kl": kl / n,
    }


@torch.no_grad()
def compare_greedy(a, b, tok, prompts: list[str], new_tokens: int, device: str) -> dict:
    identical = 0
    prefix = []
    examples = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False)["input_ids"].to(device)
        oa = a.generate(ids, max_new_tokens=new_tokens, do_sample=False)[0, ids.shape[1] :].tolist()
        ob = b.generate(ids, max_new_tokens=new_tokens, do_sample=False)[0, ids.shape[1] :].tolist()
        k = next((i for i, (x, y) in enumerate(zip(oa, ob)) if x != y), min(len(oa), len(ob)))
        prefix.append(k)
        identical += oa == ob
        if len(examples) < 2:
            examples.append({"prompt": p, "original": tok.decode(oa, skip_special_tokens=True), "aamt": tok.decode(ob, skip_special_tokens=True)})
    return {
        "prompts": len(prompts),
        "new_tokens": new_tokens,
        "identical_outputs": identical,
        "mean_matching_prefix_tokens": sum(prefix) / len(prefix),
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aamt")
    ap.add_argument("original", help="hub id or local dir of the original checkpoint (config, tokenizer, weights)")
    ap.add_argument("--texts", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.original)
    original = AutoModelForCausalLM.from_pretrained(args.original, dtype=torch.float32).to(args.device).eval()
    baked = load_causal_lm(args.aamt, args.original, device=args.device)
    rows = gsm8k_test()
    texts = [r["question"] + "\n" + r["answer"] for r in rows[: args.texts]]
    prompts = [r["question"] for r in rows[: args.prompts]]

    result = {
        "aamt": str(args.aamt),
        "original": str(args.original),
        "device": args.device,
        "dtype": "float32",
        "data": f"GSM8K test, first {args.texts} items (question + answer), max {args.max_len} tokens each",
        "logits": compare_logits(original, baked, tok, texts, args.max_len, args.device),
        "greedy": compare_greedy(original, baked, tok, prompts, args.new_tokens, args.device),
        "seconds": None,
    }
    result["seconds"] = round(time.perf_counter() - t0, 1)

    L, G = result["logits"], result["greedy"]
    print(f"{args.aamt} vs {args.original} · {args.device} · float32 · {L['tokens']} tokens of GSM8K text\n")
    print(f"perplexity      original {L['perplexity_original']:.3f}   .aamt {L['perplexity_aamt']:.3f}   ({L['perplexity_change_pct']:+.2f}%)")
    print(f"top-1 agreement {L['top1_agreement_pct']:.2f}% of next-token predictions")
    print(f"mean KL         {L['mean_kl']:.4f} nats per token (original ‖ .aamt)")
    print(f"greedy          {G['identical_outputs']}/{G['prompts']} outputs identical over {G['new_tokens']} tokens; "
          f"first difference after {G['mean_matching_prefix_tokens']:.1f} tokens on average")
    for ex in G["examples"]:
        print(f"\n  prompt:   {ex['prompt'][:100]}…\n  original: {ex['original'][:160]!r}\n  .aamt:    {ex['aamt'][:160]!r}")
    print(f"\n{result['seconds']} s")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
