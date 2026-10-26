# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Open a `.aamt` model in Hugging Face transformers.

The `.aamt` file supplies the weights; the model's own `config.json` (and
tokenizer) supply the architecture. Nothing from the original checkpoint's
weights is read.

    from aamt.hf import load_causal_lm
    model = load_causal_lm("qwen3-0.6b.aamt", "Qwen/Qwen3-0.6B")

This runs on transformers' own PyTorch code, on CPU or GPU — a portable way to
check a `.aamt` file, not a fast one. It does not use the WGSL kernels in
`js/packages/webgpu`.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .model import AamtModel


def load_state_dict(path: str | Path, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Every real tensor in the file, dequantised, as `{name: tensor}`."""
    out: dict[str, torch.Tensor] = {}
    with AamtModel(path) as m:
        for name in m.names():
            out[name] = torch.from_numpy(m.tensor(name)).to(dtype)
    return out


def load_causal_lm(
    aamt_path: str | Path,
    config: str | Path,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
):
    """A transformers causal LM whose weights all come from `aamt_path`.

    `config` is anything `AutoConfig.from_pretrained` accepts: a hub id, or a
    local directory holding `config.json`. The model is built from that
    config, then every parameter is overwritten from the `.aamt` file; a missing
    or unexpected tensor is an error, so no randomly initialised weight survives.
    """
    cfg = AutoConfig.from_pretrained(config)
    model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    state = load_state_dict(aamt_path, dtype)
    tied = getattr(cfg, "tie_word_embeddings", False)
    if tied:
        # The output projection IS the embedding in a tied model; a file may still carry a copy of it.
        state.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if tied:
        missing = [k for k in missing if k != "lm_head.weight"]
        model.tie_weights()
    if missing or unexpected:
        raise ValueError(f"{aamt_path} does not match {config}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.to(device).eval()
