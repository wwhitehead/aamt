# Which shipped models can be rebuilt, and which cannot

The claim worth being able to make is: *bake the same checkpoint with `aamt bake` and you get our file.*
For two of the three curated models you get a file whose **weights are byte-identical**. For the third you
cannot get a file at all, because the checkpoint it came from no longer exists on any machine we have.

This page records which is which, and the evidence for each, so the next release starts from what is true
rather than what we assumed.

---

## 1. What "reproducible" means for an `.aamt`

An `.aamt` file is three things in a row: a 128-byte header, the **tensor blob** (the weights), and a JSON
**manifest** trailer listing each tensor's offset, length and kind, followed by the sentinel entries that
carry model-level facts (`__tera_meta__`, `__rope_meta__`, `__chat_contract__`, `__hybrid_meta__`).

Reproducibility splits along that seam, and the halves are not equally important:

| Part | What it holds | Why it matters |
|---|---|---|
| Tensor blob | Every weight, in its quantised form | This **is** the model. A difference here is a different model |
| Manifest | Offsets, kinds, and the sentinel metadata | Derived from the weights, so it is a *report about* the blob |

The distinction matters because a manifest can differ while the model is the same. The TERA bounds are a
function of the weights — `compute_tera_bounds` returns `t` (from variance), `e` (from mean absolute value),
`r` (from the largest magnitude) and `a` (`(t + e) / 2`). Three of those four are accumulations, and
floating-point addition is not associative, so two correct evaluations of the same weights can land on
different last digits. **Both are right.** Neither describes a different model.

That is a claim about the blob. Whether the blob *works* is a different question with a different check —
[§9](#9-what-proves-a-model-works-and-what-only-proves-a-kernel-works).

---

## 2. The three models, and what each was baked from

| Model | Baked from | That source on disk? | Rebuildable |
|---|---|---|---|
| BeyondSight GaIaN | `.torch-cache/BeyondSight-GaIaN` — bf16, 16.38 GB, 400 tensors | Yes | Yes, verified |
| InnerSight | `prism-ml/Ternary-Bonsai-1.7B-mlx-2bit`, repacked by `bake_bonsai_aamt.py` | Yes, in the Hugging Face cache | Yes, verified |
| BeyondSight NightOwl | — | **No** | **No** |

---

## 3. GaIaN — rebuildable, and verified

`models/.torch-cache/BeyondSight-GaIaN` holds the full-precision checkpoint, so `bake_qwen_aamt.py
--ternary` regenerates it. A bake was run and compared against the unwatermarked original: the 3.23 GB
tensor blob came out **byte-identical, with zero per-tensor differences**. Only the manifest's
`__tera_meta__` digits moved.

---

## 4. InnerSight — rebuildable, but not from a checkpoint of ours

InnerSight is **prism-ml's Ternary-Bonsai-1.7B, repacked**. Two independent pieces of evidence:

| Evidence | Value |
|---|---|
| `Ternary-Bonsai-1.7B-mlx-2bit`, layer-0 `input_layernorm.weight` | `0.068359375` |
| Shipped InnerSight, same tensor | `0.06835938` |
| Vanilla Qwen3-1.7B, same tensor | `0.06591797` |

A norm is stored raw rather than quantised, so that number is source weights, not the quantiser. It matches
Bonsai exactly and vanilla Qwen3 not at all. `bake_bonsai_aamt.py`'s own docstring agrees, listing
"InnerSight, GaIaN" as the plain-Qwen3 checkpoints it handles and describing Bonsai's MLX affine 2-bit
container as a lossless relabeling of `.aamt`'s symmetric ternary scheme.

The catalogue in `apps/aamt-cli` records `base: "Qwen3-1.7B"` for InnerSight. That is true of the ultimate
ancestor — Ternary-Bonsai is itself derived from Qwen3-1.7B — but the *immediate* source is a third-party
checkpoint. Worth knowing before anyone repeats the licence reasoning from scratch.

### The rebake, measured

```
Quantization census: f32=114 int4=1 ternary=196      Verification passed.
```

| Measurement | Value |
|---|---|
| Tensors, shipped vs fresh | 310 vs 310 |
| Tensor blob ends at | byte `723118968`, in **both** files |
| First differing byte | `723159034` — 40,066 bytes into the manifest |
| Conclusion | The 723 MB tensor blob is **byte-identical** |

Layer 0's bounds, the only kind of thing that moved:

```
original: [0.019713498186320066, 0.06963304430246353, 0.01962890625, 0.0446732712443918]
fresh   : [0.01971349804025479,  0.06963304771731298, 0.01962890625, 0.04467327287878388]
                                                 ^ identical — r is a max, a selection, not a sum
```

`r` is the one value with no accumulation behind it, so there is nothing to round, so it does not move. The
other three differ in the 9th significant digit. `a` still equals `(t + e) / 2` to the last bit on both
sides, which is the check that the arithmetic is sound and only the summation order differs.

### One sentinel is added, and it changes nothing

The fresh bake writes `__rope_meta__`; the shipped file predates that sentinel. It is not a behaviour
change, because the value it records is already the engine's fallback:

```ts
const ropeBase = this.ropeThetaOverride ?? (this.hasGatedAttn ? 10_000_000.0 : 1_000_000.0);
```

`__rope_meta__` carries `rope_theta: 1000000.0`, and InnerSight has no gated attention, so the engine
already uses exactly that number. The sentinel makes an implicit default explicit.

---

## 5. NightOwl — not rebuildable, and this is permanent

`models/.mlx-cache/BeyondSight-NightOwl` looks like a checkpoint directory. It is not one. Its
`conversion.json` names its own source:

```json
{ "source": ".../models/BeyondSight-NightOwl.aamt", "quant_bits": 6,
  "quant_group_size": 64, "num_source_tensors": 399, "output_bytes": 6653664770 }
```

That is a **6-bit MLX conversion of the already-baked `.aamt`**, made for local inference. It is downstream
of the release, not upstream of it. A sweep for every `*.safetensors` over 500 MB on this machine finds
only GaIaN's bf16 checkpoint, that derived conversion, and an unrelated vision model.

NightOwl's trained checkpoint is gone. Its shipped `.aamt` is now the only record of what it is, so its
`__tera_meta__` digits will keep the pre-fix values forever, and no future rebake can change that.

---

## 6. Why this was not re-released

Rebuilding GaIaN and InnerSight to close a 1e-9 gap was considered and declined, for three reasons:

1. **It cannot cover all three.** NightOwl's source is gone, so the release would be 2-of-3 reproducible
   either way — a claim that is weaker for being partial and harder to state honestly.
2. **The weights are already identical.** The bounds describe the blob; the blob is byte-for-byte the
   same. There is no correctness argument, only a cosmetic one.
3. **It would invalidate every existing download.** The catalogue pins `size` and `sha384` per model and
   `apps/aamt-cli/src/pull.ts` fetches from `huggingface.co/Mai-iaM/BeyondSight-aamt`. New bytes mean new
   hashes, which means re-uploading to Hugging Face and re-signing `catalog.json` — and until that upload
   lands, every user's verification fails. That is a real cost to pay for a metadata delta that changes
   nothing.

---

## 7. Checking any of this yourself

Whether a given `.aamt` can be rebuilt is a question about its source, so start there:

```bash
# Is the cache a checkpoint, or a conversion of the release?
cat models/.mlx-cache/BeyondSight-NightOwl/conversion.json

# Confirm a rebuild without trusting the manifest: compare the blobs only.
cmp -l gate/orig-InnerSight.aamt staging/InnerSight.aamt | head -1
```

And to confirm a shipped file still carries its watermark, which the catalogue's hash alone does not tell
you:

```bash
python tools/watermark/aamt_watermark.py detect \
  ~/aamt-release/models/InnerSight.aamt ~/aamt-release/gate/orig-InnerSight.aamt
```

```json
{"positions": 537396, "mean_signed_delta": 0.00039997635607792704, "z": 733.0729840882148, "marked": true}
```

The mean signed delta lands on `4e-4` because that is the embedding's `EPS` — the mark lives in the group
scales, at about 1% of positions, which is what `tools/watermark/aamt_watermark.py` documents.

---

## 8. What the code change did buy

The reduction order is now specified rather than inherited. `compute_tera_bounds` is defined **once**, in
`maiiam-alchemist/scripts/bake_qwen_aamt.py`, and imported by every other baker — `bake_bonsai_aamt.py`,
`bake_bonsai2_hadamard_aamt.py`, `bake_awq_bonsai_aamt.py`. So fixing it there fixed the Bonsai path that
produced InnerSight for free, and `computeTeraBounds` in `packages/voxel-runtime/src/baker.ts` mirrors it
operation for operation.

A rebake and an `aamt bake` now agree with **each other** on all three models, verified on GaIaN and on
InnerSight. That was the goal, and it is met. What no longer holds is the stronger claim that either agrees
with the files published before the fix — for NightOwl that is now unachievable, and for the other two it
would need the re-release described in [§6](#6-why-this-was-not-re-released).

To bake a model of your own, see [Serve a model into the tools you already use](SERVE-YOUR-OWN-MODEL.md).

---

## 9. What proves a model works, and what only proves a kernel works

Everything above is about the *blob* — whether a given `.aamt` can be rebuilt, and whether the rebuild lands on
the same weights. That is a different question from the one a user is actually asking, which is whether the
model can talk. The two need different checks, and the difference is not cosmetic: **one of them can pass on a
file that is broken.**

| Check | What it varies | What it can tell you |
|---|---|---|
| `host/generate-smoke.ts` | The checkpoint | Whether this file produces language at all |
| `host/gemm-v4-check.ts`, `host/verify-batch-parity.ts` | Two kernels on the *same* weights | Whether the kernels agree with each other |
| `host/session-bench.ts` | Nothing — it times one path | How fast that path is |

The middle row is the one to be careful with, and the reason is in its own definition. A kernel parity check
compares two implementations reading identical weights, so **when the weights are wrong, both are wrong
together and the check passes.** It is a self-consistency test: it is evidence about the kernel, never about
the checkpoint. The same is true of a tok/s figure — a model whose tensors are permuted decodes at a perfectly
healthy rate, because permuted weights are still weights.

This is not hypothetical here. The first Flow-3 bake passed the kind census, matched the shipped Flow 2's
sentinels byte for byte, and passed `PARITY OK`. Asked a question it produced:

```
)/()/()/()/()/()/()/()/()/(/S/S/S)/()/()/()/(/S)/(/S)/(/S)/(/S)/(/S)/(/S)/(/S)/(/S)…
```

What found it was not a check in this repository. It was a **diff against a working model** — the same
technique §7 describes for weights, applied to the tensor *layout* rather than the bytes. The full account is
[Bonsai 2 §5](BONSAI-2-AND-GGUF.md#the-value-heads-are-interleaved-and-ssm_a-is-not-a_log).

So the ordering is the point:

1. **Ask the model a question first.** `generate-smoke.ts` is the primary gate on a freshly baked checkpoint.
   It varies the thing that matters, and it is the only check here that can fail on a file the others accept.
2. **Then time it.** A tok/s figure is meaningful once the model speaks, and misleading before that.
3. **Treat a parity failure as evidence about the kernel** — and a parity pass as evidence about nothing but
   the kernel.

### The same discipline, arrived at from the other end

This is the shape of a **matched control**, and it is worth naming because it generalises past kernels. A
comparison is only evidence if the two things being compared differ in exactly one respect. Two kernels on one
set of weights differ in nothing that touches correctness of the *weights*; a from-scratch and a fine-tuned run
of the same architecture differ in exactly one thing, which is why
[`leochlon/areals`](https://github.com/leochlon/areals) states every one of its transfer results against a
scratch fit at identical architecture ("Windsor scratch fits use the same architecture as pretrained fits").

The failure mode that discipline exists to prevent is the one above: two measurements sharing an assumption can
agree, and both be wrong. A control that varies the assumption is the only thing that catches it.
