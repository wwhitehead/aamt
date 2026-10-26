# aamt (Python)

Read `.aamt` model files with numpy, and run them in Hugging Face transformers.
This is the portable way to check a `.aamt` file with tools you already trust;
it is not the fast path (the WGSL kernels in `js/packages/webgpu` are).

```bash
pip install -e "python[quality]"
```

```python
from aamt import AamtModel
with AamtModel("qwen3-0.6b.aamt") as m:
    w = m.tensor("model.layers.0.mlp.down_proj.weight")   # float32 (1024, 3072)

from aamt.hf import load_causal_lm
model = load_causal_lm("qwen3-0.6b.aamt", "Qwen/Qwen3-0.6B")   # every weight from the .aamt file
```

```bash
python python/examples/generate.py qwen3-0.6b.aamt Qwen/Qwen3-0.6B "Why does local AI protect privacy?"
python python/examples/quality.py  qwen3-0.6b.aamt Qwen/Qwen3-0.6B --json quality.json
python -m unittest discover -s python/tests        # numpy only
```

`quality.py` runs the original model and the `.aamt` model side by side on GSM8K
text and reports perplexity, next-token agreement, KL divergence and greedy
output agreement. The result for Qwen3-0.6B baked to int4 is in
`examples/quality-qwen3-0.6b-int4.json`, and in `CLAIMS.md`.
