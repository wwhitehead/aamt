#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Generate text from a `.aamt` model in Hugging Face transformers.

    python python/examples/generate.py qwen3-0.6b.aamt Qwen/Qwen3-0.6B "Why does local AI protect privacy?"

The second argument is the model's config and tokenizer: a hub id or a local
directory. Weights come only from the `.aamt` file.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from aamt.hf import load_causal_lm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aamt")
    ap.add_argument("config", help="hub id or local dir with config.json and tokenizer")
    ap.add_argument("prompt")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_causal_lm(args.aamt, args.config, device=args.device)
    tok = AutoTokenizer.from_pretrained(args.config)
    print(f"loaded {args.aamt} in {time.perf_counter() - t0:.1f} s ({args.device})", file=sys.stderr)

    messages = [{"role": "user", "content": args.prompt}]
    kwargs = {"enable_thinking": False} if "qwen3" in getattr(model.config, "model_type", "") else {}
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, **kwargs)["input_ids"].to(args.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
