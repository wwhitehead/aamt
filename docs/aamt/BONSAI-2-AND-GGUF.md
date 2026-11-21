# Bonsai 2, and what first-class GGUF support gives us

`prism-ml` publishes the Bonsai 2 family, and this page records what is in it, how it comes into an `.aamt`,
what the GGUF path carries that a safetensors bake does not, and what the quantiser change bought. Every
number here was read off the files or measured — the commands that produce them are in
[§7](#7-doing-it-yourself).

---

## 1. The collection, as it actually is

Read from `huggingface.co/api/models?author=prism-ml` and each repo's own file listing. `image-text-to-text`
means the repo also ships an `mmproj` vision projector; `text-generation` does not.

| Repo                            | Base        | Params | Pipeline               | Quantised file              | Size         |
| ------------------------------- | ----------- | ------ | ---------------------- | --------------------------- | ------------ |
| `Ternary-Bonsai-2-2B-gguf`      | Qwen3.5-2B  | 2B     | **image-text-to-text** | `PTQ1_0` / `PQ2_0` / `Q2_0` | ~0.5 GB      |
| `Ternary-Bonsai-2-2B-mlx-2bit`  | Qwen3.5-2B  | 2B     | image-text-to-text     | MLX affine 2-bit            | —            |
| `Ternary-Bonsai-2-4B-gguf`      | Qwen3.5-4B  | 4B     | **image-text-to-text** | `PTQ1_0` / `PQ2_0` / `Q2_0` | ~1 GB        |
| `Ternary-Bonsai-2-4B-mlx-2bit`  | Qwen3.5-4B  | 5B     | image-text-to-text     | MLX affine 2-bit            | —            |
| `Ternary-Bonsai-2-9B-gguf`      | Qwen3.5-9B  | 9B     | **image-text-to-text** | `PTQ1_0` / `PQ2_0` / `Q2_0` | 1.86–2.38 GB |
| `Ternary-Bonsai-2-9B-mlx-2bit`  | Qwen3.5-9B  | 9B     | image-text-to-text     | MLX affine 2-bit            | —            |
| `Ternary-Bonsai-2-27B-gguf`     | Qwen3.8-27B | 27B    | text-generation        | `PTQ1_0` / `PQ2_0` / `Q2_0` | ~4 GB        |
| `Ternary-Bonsai-2-27B-mlx-2bit` | Qwen3.8-27B | 27B    | text-generation        | MLX affine 2-bit            | —            |
| `Ternary-Bonsai-2-27B-gguf-dev` | Qwen3.8-27B | 27B    | text-generation        | development build           | —            |

The 9B `gguf` repo, as the API lists it:

```
Ternary-Bonsai-2-9B-PTQ1_0.gguf          1997486976   1.75 bpw ternary, group 128
Ternary-Bonsai-2-9B-PQ2_0.gguf           2416851840   2.125 bpw ternary, group 128
Ternary-Bonsai-2-9B-Q2_0.gguf            2556640128   2.25 bpw ternary, group 64
Ternary-Bonsai-2-9B-MTP-Q8_0.gguf        2675410720   multi-token-prediction head
Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf    1574233856   speculative drafter
Ternary-Bonsai-2-9B-mmproj-BF16.gguf      921704864   the vision tower
Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf      624229664   the vision tower, quantised
```

Three things about that list matter for us:

1. **`PTQ1_0` is the one to take.** It is Prism's private ternary codec at group 128, and it is _lossless_
   against `.aamt`'s own symmetric ternary scheme — see [§3](#3-why-a-ternary-gguf-repack-is-exact).
2. **The 2B/4B/9B are vision models.** They carry `vision_config` and a separate `mmproj` file. The 27B does
   not: its own model card describes a vision tower, but the released 27B GGUF repos carry no projector.
3. **`MTP-*` and `DFlash2-*` are separate files**, not tensors inside the main one. They are a speculative
   drafter and a multi-token-prediction head, and `bakeGguf` on the main file neither reads nor needs them.

---

## 2. The model set, v2

`apps/aamt-cli/scripts/make-catalog.ts` is the source of truth for what ships. The v2 rows, and the family
each one succeeds:

| id               | Name            | Params | Baked from             | Succeeds          |
| ---------------- | --------------- | ------ | ---------------------- | ----------------- |
| `innersight2`    | InnerSight 2    | 2B     | `Ternary-Bonsai-2-2B`  | InnerSight (1.7B) |
| `innersight2-4b` | InnerSight 2 4B | 4B     | `Ternary-Bonsai-2-4B`  | —                 |
| `gaian2`         | GaiaN 2         | 9B     | `Ternary-Bonsai-2-9B`  | GaIaN (8B)        |
| `flow3`          | Flow 3          | 27B    | `Ternary-Bonsai-2-27B` | Flow 2            |

Two things to settle before the first release, and both are cheaper now than later:

- **The display names are the owner's call.** The ids are what `aamt pull <id>` takes and what the Ed25519
  signature covers, so an id cannot change after a release without re-signing the catalogue and invalidating
  every existing download — the same cost `MODEL-REPRODUCIBILITY.md` §6 describes for a re-release.
- **The vision models need a schema change, not a row.** `CatalogModel` has one `path` and `pull` fetches one
  file. An image model needs the projector too, as a second path with a role, and a loader that binds both.
  A row pointing at the text half alone would be a model that loads and answers as if it could see, which is
  the worst of the available outcomes.

### InnerSight-2-4B: baked and verified

`Ternary-Bonsai-2-4B-PTQ1_0.gguf` → `InnerSight-2-4B.aamt`, on 2026-11-02. The download was checked against
the publisher's own `SHA256SUMS` before anything was read (`56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60`).

```
InnerSight-2-4B.aamt
  size     1.75 GB
  tensors  431 (f32 225 · int4 1 · ternary 200)
  sha384   d8db374fe0479f6c17d8d63571a8bfb034e600c96e393ae492a4d18bcb809e93a244cfe3d2f5f3612b9c5bc1e2e020c0
  source   GGUF v3 · qwen35 · f32 225 · ptq1_0 201
  wrote    __tera_meta__ · __hadamard_meta__ · __hybrid_meta__ · __rope_meta__ · __chat_contract__
```

`aamt inspect` reads it back as AMDL v1 and says this engine can run it. The kind counts mirror the source's
own precision split exactly — **225 f32 and 201 quantised in, 225 f32 and 201 quantised out** — which is the
result of the rule in [§5](#a-tensor-the-source-kept-unquantised-stays-unquantised).

The file is **1.75 GB against a 0.95 GB source**, and that is not a mistake. The source is `ptq1_0` at 1.75
bits a weight with every GatedDeltaNet parameter in f32; the `.aamt` is 2 bits a weight and keeps those same
parameters in f32, so the quantised part grows while the float part does not shrink. The comparison that
matters is against the slot it takes: the fast slot's v1 model is NightOwl at 2.28 GB, so this is smaller.

`lm_head.weight` is absent on purpose: this checkpoint is tied (`prism.hadamard.tied_output: true`) and ships
no `output.weight`, so `model.embed_tokens.weight` *is* the head and is baked int4 for the reason the recipe
gives. That is also why the Hadamard contract reads `tied_output` — the v2 form of that file, which the
engine's parser tolerates as an extra key.

Its speed is measured in the table below and its agreement with the publisher's runtime under it. It is installed at the OS's `checkpointDir`, so the Model
Manager lists it and its **Tokens/s** gauge will produce a real figure per machine — which is where a figure
belongs. It is **1,753,950,056 bytes** — 1.75 GB decimal,
1.63 GiB as `ls` reports it. (This is the rebake after `gdn_v_grouped` stopped being copied into
`__hadamard_meta__`; the first bake was 38 bytes larger, `41a43d5d…13de4dd2ee`.)

### Flow-3: baked and verified

`Ternary-Bonsai-2-27B-PTQ1_0.gguf` → `Flow-3.aamt`, also 2026-11-02.

```
Flow-3.aamt
  size     10.50 GB (9.78 GiB)
  tensors  856 (int4 1 · f32 449 · ternary 401)
  sha384   92a4a98e53af57d7bdc7be0fa66985231d00aa41528a1b664877499d54ca8c87e6310c615c0d55e09f0539f4837e7bc7
  source   GGUF v3 · qwen35 · ptq1_0 402 · f32 353 · bf16 96
  regroup  336 GatedDeltaNet tensor(s): value heads interleaved → grouped, and ssm_a → A_log
  wrote    __tera_meta__ · __hadamard_meta__ · __hybrid_meta__ · __rope_meta__ · __chat_contract__
```

The `sha384` is the second bake. The first had no `regroup` line and could not speak — see
[§5](#the-value-heads-are-interleaved-and-ssm_a-is-not-a_log).

This repo has **no `SHA256SUMS`** (the 4B one does), so there was no publisher checksum to verify against.
Instead the file was checked structurally, which for a GGUF is stronger than a size: the header and the whole
tensor index were parsed, and **the last tensor ends exactly at the end of the data section** — 5,935,527,936
bytes of data against 5,946,648,928 total less an 11,120,992-byte header and index. A truncated download
cannot produce that. 851 tensors, 26.90 G weights, 64 layers, hidden 5120.

`aamt inspect` reads it back as AMDL v1 with `lm_head.weight` present (this checkpoint is **not** tied) and
says this engine can run it. The hybrid map is right: 64 layers, full attention at every 4th (16 of them), key
heads 16, value heads 48, head dim 128, conv 4 — and `48 × 128 = 6144`, the source's `ssm.inner_size`.

The rule from [§5](#a-tensor-the-source-kept-unquantised-stays-unquantised) fired for **240 tensors** here
against the 4B's 120, and it earned its keep in a way the 4B could not show: this release stores
`ssm_alpha`/`ssm_beta` as **bf16** where the 4B stored them as **f32**, for the same role. A rule written as a
list of names would have been wrong on one of the two releases. The dtype rule covers both.

### The container costs 1.77x the GGUF it came from

| | Weights | Bytes | Bits/weight |
|---|---|---|---|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 26.90 G | 5.95 GB | **1.77** |
| `Flow-3.aamt` | 26.90 G | 10.50 GB | **3.12** |

That is not a mistake and it is not the f32 parameters. It is the container: `.aamt` ternary stores **2 bits of
codes plus a float32 scale per 32 weights**, which is another **1 bit per weight** — `aamt-format.ts`'s own
`bitsPerWeight()` says so (`4 + 32 / groupSize` for int4, the same `+1` for ternary). Prism's `ptq1_0` spends
one fp16 scale on 128 weights, i.e. 1.75 bits total, so the same weights cost 1.77x more here.

The 4B showed the same ratio (0.95 GB → 1.63 GiB) and it was easy to read as "the f32 norms are big". They are
not: the whole float part of Flow-3 is **105 MB of 10.50 GB**. The scale is the cost.

**This is the largest available win in the format and it needs a kernel change, not a bake option.** The
engine asserts group size 32 — the Python baker's own comment says it is "the ONLY value the engine's kernels
support" — so the group cannot be widened from here. The cheaper lever is the scale's *width*: an f16 scale at
group 32 gives `2 + 16/32 = 2.5` bits and about 8.4 GB, and at group 128 `2.125` bits and about 7.1 GB. Both
are decoder changes with a real precision consequence, and this repository's rule is that a precision change
needs a parity run before it ships, so neither is done here.

**On disk it costs nothing to adopt.** `Flow-3.aamt` is 9.78 GiB and the deep slot's current model,
`BeyondSight Flow 2.aamt`, is also 9.78 GiB. The v2 pair is *smaller* overall than the v1 pair, because the
fast slot moves from NightOwl (2.12 GiB) to InnerSight-2-4B (1.63 GiB).

### How MaiiaM OS picks them up

The OS resolves a slot through a list of **tiers**, tried in order, and Bonsai 2 is the first:

| Tier | fast | deep | draft |
|---|---|---|---|
| `PREFERRED_SLOT_FILES` | `InnerSight-2-4B` | `Flow-3` | — |
| `DEFAULT_SLOT_FILES` | `BeyondSight-NightOwl` | `BeyondSight Flow 2` | `BeyondSight-Nomad` |
| `LEGACY_SLOT_FILES` | `BeyondSight-GaIaN` | `BeyondSight-Flow` | `BeyondSight-GaIaN-Draft` |

(`plugins/aamt-sovereign/src/kernel-entry.ts`; `SLOT_TIERS` is one list that both `resolveModel` and
`defaultSlots` walk, so the mapping the Model Manager offers and the mapping a turn uses cannot drift apart.)

Three things about that are deliberate:

- **The preference was declared before the bake existed, and both of its checkpoints are now on disk.** Until
  2026-11-02 nothing was named `InnerSight-2-4B` or `Flow-3`; the tier was empty and the v1 tier below it ran.
  With both baked, **fast resolves to `InnerSight-2-4B` and deep to `Flow-3`**, with no config edit and no code
  change — which is the whole reason to state a preference as a list rather than as a note saying "change this
  later".
- **Slots resolve independently.** That is what let the two land one at a time: between the 4B bake and the 27B
  bake the fast slot had already moved to Bonsai 2 while deep was still Flow 2, and neither had to wait for the
  other.
- **No Bonsai 2 draft.** A speculative draft only works when it shares the target's tokenizer, and this
  repository's rule is that an unverified pairing is never made automatically. Nomad drafts for the v1 deep
  slot and has been measured doing it; no Bonsai 2 drafter has been. The tier declines and the v1 draft
  stands.

`packages/contracts/src/models/aamt-catalog.ts` names the same four files as `BONSAI2_AAMT_IDS`, separately
from `AAMT_MODELS` — which is a manifest of files with their **real** `sizeBytes`, read by the first-run disk
check and by the video's model-law beats. A row there with a guessed size would be a wrong number in two
places at once, so the ids are named and the rows land with the bake. A test in the plugin holds the two
lists together by membership, so a renamed checkpoint cannot be preferred by one and unknown to the other.

### Choosing and benchmarking them

They appear in **Settings → System → Model Manager** with no further work: the panel lists every `.aamt` the
plugin can see, and `listModels` tags the v2 ids `bonsai2` so a 4B is not mistaken for the 1.7B it succeeds.
Each row has a slot picker and a **Tokens/s** button that runs a fixed 96-token prompt
(`router.ts#benchmark`) and stores the measurement.

The figures below are `host/session-bench.ts` on an **M1 Max 32 GB**, with `AAMT_GPU_DEADLINE_MS=60000` (the
plugin's own default for large checkpoints — at the tool's 10 s default the 27B's prefill trips the deadline
and the bench aborts, which is the incident the manifest describes). Every line is MEASURED on this machine.

| | InnerSight-2-4B | Flow-3 (27B) |
|---|---|---|
| decode | **44.9 tok/s** (32-row split) | **12.6 tok/s** (32-row split) |
| load | 2.7 s (3.7 s first, cold file cache) | 18.5 s |
| KV per position | **32 KB** | 64 KB |
| TTFT 512 cold / cached | 2 648 / **35 ms** | 13 773 / **120 ms** |
| TTFT 1024 cold / cached | 5 406 / 43 ms | 28 002 / 140 ms |
| TTFT 2048 cold / cached | 11 815 / 62 ms | 60 114 / 190 ms |
| TTFT 4096 cold / cached | 28 026 / 98 ms | 132 781 / 240 ms |

Against the slots they take: NightOwl is documented at 39.0 tok/s and 147.5 KB a position, Flow 2 at 65.6 KB.
So the 4B is **faster than the fast slot's v1 model and uses a fifth of its KV**, and Flow-3 matches Flow 2's
KV while being a newer base.

**Cold prefill is the weak number and it is worth naming.** 4.1 k tokens costs 28.0 s on the 4B and 133 s on
Flow-3 — 147 and 31 tok/s of prefill. The cached path is 286x and 553x faster because the prefix is reused
from KV. That is the shape of the whole thing: a cold long prompt is slow, everything after it is fast.

**`InnerSight-2-4B` is fixed, verified and installed.** It was withdrawn after the same GGUF conversion bug
that made Flow-3 emit noise; with that fixed it still emitted noise, for two reasons of its own that were in
the engine, not the file — see [§6](#the-4bs-block-size-is-512-and-its-head-is-tied). Both are fixed, it
answers correctly, and it is installed at the OS's model root as `InnerSight-2-4B.aamt` (sha384
`d8db374f…e2e020c0`, 1,753,950,056 bytes), so the fast slot resolves to it. Its column above was re-measured
on 2026-11-02 after the fixes; the figures the column held before were taken on a file that could not speak,
and were within a few percent of these — the bugs cost correctness, not speed.

**Both agree with the publisher's own runtime.** `host/upstream-agreement.ts` feeds the same token ids
through this engine and through Prism ML's MLX pack (`runtime/artifact.py`, the loader the release ships),
teacher-forced over the two passages the watermark gate uses, and compares the next-token argmax at every
position. These are the watermarked release files, not the installed originals:

| | positions | top-1 agreement | disagreements | largest logit gap at the reference's pick |
|---|---|---|---|---|
| InnerSight-2-4B | 146 | **99.32 %** | 1, a near-tie (reference margin 0.017; ours an exact tie) | 0.32 |
| Flow-3 | 146 | **100 %** | 0 | 0.09 |

The reference computes in f16 and this engine mostly in f32, so a gap of a few tenths of a logit and a
disagreement at a near-tie are what two correct runtimes look like; a conversion bug looks like the 5.4-logit
gap in §6. This is agreement, not perplexity or a benchmark score: it says the `.aamt` is the publisher's
model, not how good the publisher's model is.

**The watermark costs nothing measurable.** `tools/watermark/quality-gate.ts`, marked against original on the
engine: 100 % top-1 agreement over 146 positions and 4/4 identical greedy replies, for both (largest logit
difference 0.003 on the 4B, 0.002 on Flow-3). Detection scores z = 1,146 and 2,898, against 0 for the
unmarked file.

---

## 3. Why a ternary GGUF repack is exact

`Ternary-Bonsai-2-*` weights are ternary per 128-wide group: every value is exactly one of `{−d, 0, +d}` with
one f16 `d` for the group. `.aamt`'s ternary blob is the same thing at group 32 — 2 bits a weight, one f32
scale, codes `00 = 0`, `01 = +1`, `11 = −1`, and the scale _is_ the magnitude of the ±1 state.

So a repack is a subdivision, not a re-quantisation:

- `128 % 32 == 0`, so every `.aamt` group lies inside exactly one source group.
- Within a 32-wide subgroup, `maxAbs` is exactly `d` whenever the subgroup has a nonzero — and every nonzero
  is exactly `±d`. So `scale = maxAbs` recovers `d`, and `|w/d|` is exactly `1`, which is above the `0.5`
  threshold, so the code is exactly right.
- A subgroup that is all zeros gets `scale = 1` and all-zero codes, which decodes to zeros. Same answer.

No continuous value is ever rounded. `packages/voxel-runtime/test/gguf.test.ts` asserts this end to end: it
bakes a miniature hybrid Bonsai 2 out of real GGUF bytes and compares the `.aamt`'s decoded weights to the
source trits, position for position.

`PQ2_0` and `Q2_0` carry the same ternary values in a 2-bit field whose fourth code (`+2d`) a ternary-trained
checkpoint never emits, so they repack exactly too. `Q2_0` is group 64, which also divides by 32.

---

## 4. What the GGUF path carries that safetensors does not

This is the part that makes GGUF support more than a convenience. A Bonsai 2 GGUF carries its
**Hadamard contract** in its own metadata, as `prism.hadamard.*` keys:

```
prism.hadamard.version           1
prism.hadamard.block_size        4096          (512 on the 4B, 1024 on the 27B)
prism.hadamard.transform         normalized-sylvester-walsh-hadamard
prism.hadamard.axis              input-last-dimension
prism.hadamard.sign_mode         explicit
prism.hadamard.weight_names      [output.weight, blk.0.attn_qkv.weight, …]
prism.hadamard.inverse_weight_names  [token_embd.weight]
prism.hadamard.sign_widths       [4096, 12288]
prism.hadamard.sign_values       [-1, -1, -1, 1, …]
```

Those are the _same dotted keys_, in the same shape, that `hadamard.ts#parsePrismHadamardManifest` already
parses and that `bake_bonsai2_hadamard_aamt.py` reads out of the MLX snapshot's `hadamard.json`. So the
sentinel `qwen-forward.ts#applyHadamardMeta` consumes is written **from the file itself**.

The one thing that must be _translated_ rather than copied is the tensor names: GGUF says
`blk.0.attn_qkv.weight`, the engine binds `model.layers.0.linear_attn.in_proj_qkv.weight`. A name that cannot
be translated is a **hard error** (`UntranslatableHadamardName`), not a dropped entry. That is deliberate: a
partially-applied rotation leaves some tensors rotated and some not, which is a model that runs, answers
fluently, and is wrong in a way nothing downstream can detect.
The other three sentinels come out of the same metadata block:

| Sentinel            | Source in the GGUF                                      | Relation                                                              |
| ------------------- | ------------------------------------------------------- | --------------------------------------------------------------------- |
| `__hadamard_meta__` | `prism.hadamard.*`                                      | names translated; values verbatim                                     |
| `__rope_meta__`     | `qwen35.rope.*`, `qwen35.context_length`                | `partial_rotary_factor = rope.dimension_count / attention.key_length` |
| `__hybrid_meta__`   | `qwen35.full_attention_interval`, `qwen35.ssm.*`        | layer `i` is full iff `(i+1) % interval == 0`                         |
| `__chat_contract__` | `tokenizer.ggml.tokens`, `tokenizer.ggml.chat_template` | the rule `deriveChatContract` applies to a tokenizer on disk          |

`__hybrid_meta__` is the load-bearing one. Bonsai 2 is a hybrid-attention model — 75% gated-DeltaNet linear
attention, 25% full — and the engine cannot tell which layers are which from the weights. Without the
sentinel, `applyHybridMeta`'s no-op path runs **every** layer as full attention and the model is silently
wrong. So the importer derives the field mapping and then _checks_ it:

```
ssm.group_count     -> linear_num_key_heads      (16)
ssm.time_step_rank  -> linear_num_value_heads    (48 on the 27B, 32 on the 9B)
ssm.state_size      -> linear_key_head_dim       (128, and the value dim is the same)
ssm.conv_kernel     -> linear_conv_kernel_dim    (4)
ssm.inner_size      == time_step_rank * state_size      <- asserted, not assumed
```

That identity holds on both the 27B (48 x 128 = 6144) and the 9B (32 x 128 = 4096). It is checked at bake
time, and a file where it does not hold is refused rather than guessed at, because a wrong layer map is
indistinguishable from a right one downstream.

### The 3-D rotary, and the same idea arrived at independently

`__rope_meta__` carries `mrope_section [11, 11, 10]`: the head is split into three blocks, one per spatial
axis, each with its own slice of a shared log-spaced frequency ladder. That is Qwen3-VL's multimodal rope, and
the reader already has to handle a container-layout detail the source config does not — GGUF pads the sections
out to the head width with a trailing zero (`[11, 11, 10, 0]`), and the unpadded list is what the engine wants
(`gguf.ts#ggufSentinels`).

The same scheme turns up in a different field, arrived at independently. [`leochlon/areals`](https://github.com/leochlon/areals)
(MIT) is a surface-geometry model whose decoder reads *world positions* rather than sequence positions, and its
`Rotary3D` splits a head into three equal blocks over x, y and z — the same three-axes-one-head construction,
with the same log-spaced ladder, for a query that is a point in space rather than a token in a sequence. Its
`Rotary3D` even enforces the arithmetic consequence: `head_dim % 6` must be zero, which is `3 axes × (pairs)`.

Two systems in different domains, no shared lineage, the same positional encoding. Worth recording for two
reasons: a design that two people reach from opposite directions is easier to trust than one that only appears
in the place it was invented, and it makes plain which part of this is *ours*. The three-axis split is not; the
container-layout handling above is, and it is the part that costs a debugging session if it is wrong.

---

## 5. The type table, and what is refused

The reader decodes exactly the types the Bonsai 2 family ships, plus the legacy non-`K` quants. Block sizes
and type sizes are `ggml.c`'s own `type_traits` values, and the _ids_ were confirmed against the real files
rather than remembered:

| id                | Name                             | Group | Bytes/block | Source            |
| ----------------- | -------------------------------- | ----- | ----------- | ----------------- |
| 41                | `q1_0`                           | 128   | 18          | upstream          |
| 42                | `q2_0`                           | 64    | 18          | upstream          |
| 34                | `tq1_0`                          | 256   | 54          | upstream          |
| 35                | `tq2_0`                          | 256   | 66          | upstream          |
| **142**           | **`pq2_0`**                      | 128   | 34          | **Prism-private** |
| **143**           | **`ptq1_0`**                     | 128   | 28          | **Prism-private** |
| 0 / 1 / 30        | f32 / f16 / bf16                 | -     | 4 / 2 / 2   | -                 |
| 2 / 3 / 6 / 7 / 8 | q4_0 / q4_1 / q5_0 / q5_1 / q8_0 | 32    | 18-34       | legacy            |

`Ternary-Bonsai-2-9B-PTQ1_0.gguf` declares `general.file_type = 143` and its 202 quantised tensors carry
ggml type **143**; the Q2_0 file gives 42 and the PQ2_0 file 142. The fork's own `ggml.h` explains why they
sit above upstream's `GGML_TYPE_COUNT` of 43:

```c
// Prism-private Q2_0 at group size 128 (upstream Q2_0 is group 64). High id so it
// slots above upstream types; type_traits is sized to COUNT (143) with 43..141 unused.
GGML_TYPE_PQ2_0 = 142,
GGML_TYPE_PTQ1_0 = 143, // Prism-private ternary, group 128
```

**`K`-quants and the `IQ*` family are refused by name.** Not approximated — `UnsupportedGgmlType` names the
type and says why. A half-ported super-block layout produces a model that loads, passes every structural

---

## 6. The quantiser: what changed, and what it bought

`.aamt` fixes almost everything a reader sees — group 32, 2-bit ternary codes, `value = code * scale`. The one
decision still open at bake time is **which scale**, and the baker had one rule: the group's largest
magnitude. A second rule is now available and is the default for nothing:

`--policy curated-ls` / `--policy bonsai2-ls` start from the group maximum and refine to the scale that
minimises `Σ (w − q·s)²` for the codes that scale implies — `s* = ⟨w, q⟩ / ⟨q, q⟩` — iterated to a fixed
point. It is **never worse** (it only moves to a scale that strictly lowers the error) and it is measured:

| Input (32-wide groups, 4096 weights) | int4 `max` | int4 LS     | ternary `max` | ternary LS  |
| ------------------------------------ | ---------- | ----------- | ------------- | ----------- |
| Gaussian σ=0.02                      | 6.30e-7    | **5.73e-7** | 3.15e-5       | **1.44e-5** |
| Gaussian σ=0.2                       | 6.14e-5    | **5.53e-5** | 3.06e-3       | **1.45e-3** |
| Heavy-tailed                         | 9.86e-4    | **8.37e-4** | 2.99e-3       | **2.88e-3** |
| Exactly ternary                      | 0          | 0           | **0**         | **0**       |

(mean squared error per weight; `packages/voxel-runtime/test/scale-rule.test.ts` asserts both the direction and
the margin, so the table cannot drift from the code)

Three consequences, in order of how much they matter:

1. **It is byte-identical on an exactly-ternary checkpoint under the ternary codec.** Where the values are
   already `{−d, 0, +d}`, the zeros contribute a coefficient of `0`, the fit is `d(p+n)/(p+n)` — exactly `d` —
   and the refinement stops on its first step. So `curated-ls` reproduces the shipped models byte for byte and
   nothing needs re-baking. That is what makes it safe to offer at all.
2. **It halves the error on ternary for a model that was never ternary-trained.** The ternary threshold is a
   _fraction of the scale_, so a single outlier lifts it above the rest of the group and they all quantise to
   zero. Lowering the scale to the least-squares optimum is what brings them back.
3. **The 1-bit codec is the exception, and it is a real one.** A ternary group is mostly zeros and 1 bit has no
   zero to represent — every one of them decodes as `±scale`. So the least-squares scale is the signed mean,
   which shrinks as the zeros multiply and _strictly lowers_ the error. `curated-1bit-ls` therefore produces a
   different file from `curated-1bit` even on an exactly-ternary checkpoint, and a better one. `NightOwl` is
   still on `curated-1bit` and is unaffected either way.

What this is **not**: a comparison against another 4-bit scheme. The table above is `max` against
least-squares _within this format_. `BYOM`'s limits list still stands — no comparison against any other
runtime or 4-bit method has been run.

check, and computes nonsense, which is the same reasoning `safetensors.ts#UnsupportedDtype` already gives for
an already-packed dtype. The Bonsai 2 family uses none of them, so nothing that matters is lost.

### A tensor the source kept unquantised stays unquantised

This rule exists because of a bug, found by baking the 4B and reading the census rather than by reasoning.

The curated recipe says *norms f32, the head int4, everything else ternary*. It was read off **plain Qwen3**
models — `InnerSight`, `GaIaN`, `NightOwl` — which are full attention and have no state-space parameters at
all. Bonsai 2 is **hybrid**, and its GatedDeltaNet parameters match no norm pattern the recipe knows:

| Tensor | Publisher's choice | What a name-based rule does |
|---|---|---|
| `blk.0.attn_qkv.weight` | `PTQ1_0` | ternary ✓ |
| `blk.0.attn_gate.weight` | `PTQ1_0` | ternary ✓ |
| `blk.0.ssm_out.weight` | `PTQ1_0` | ternary ✓ |
| `blk.0.ssm_a` | **F32** | ternary ✗ |
| `blk.0.ssm_dt.bias` | **F32** | ternary ✗ |
| `blk.0.ssm_conv1d.weight` | **F32** | ternary ✗ |
| `blk.0.ssm_alpha.weight` | **F32** | ternary ✗ |
| `blk.0.ssm_beta.weight` | **F32** | ternary ✗ |

`ssm_a` is `A_log`: a **log-space** decay rate, one value per head. Quantising it to ternary is not a rounding
error in a weight, it is a *multiplicative* error in a recurrence — the value is exponentiated — and it buys
96 bytes. `dt_bias` is a time step. `conv1d` is a four-tap causal convolution. The first 4B bake wrote 120 of
these to ternary and the file's census read `f32 105 · ternary 320` against a source of `f32 225 · ptq1_0 201`.

**The fix is not a list of names.** A list would be right for this family and wrong for the next one, which is
exactly how the recipe came to be wrong here. The rule is:

> **A tensor the source container carries as f32, f16 or bf16 is written f32 in the `.aamt`.**

A container that stores a tensor unquantised while its neighbours are `ptq1_0` is telling you that tensor may
not be approximated. That is information only a GGUF has — a safetensors checkpoint is all f32 and says
nothing — so the rule lives in `bakeGguf` and `bakeSafetensors` is untouched.

The re-bake's census is `f32 225 · int4 1 · ternary 200` against the same source's `f32 225 · ptq1_0 201`:
the precision split now mirrors the publisher's exactly. `packages/voxel-runtime/test/gguf.test.ts` pins it,
asserting `A_log` and `dt_bias` are f32 *and* that `in_proj_qkv` beside them is still ternary, so a rule that
swallowed the body would fail the test too.

### The value heads are interleaved, and `ssm_a` is not `A_log`

The bug that mattered most, and the one that nothing else in this repository could have caught.

Both v2 models were baked, installed, and benchmarked at a healthy **44 tok/s and 12 tok/s** before anyone
asked them a question. Asked one, they produced this:

```
InnerSight-2-4B:  spvantchstazz_PRICEçons*".atoriesthessbucksطانayaan avvicessionce rimelyيصordonele从这个…
Flow-3:          )/()/()/()/()/()/()/()/()/(/S/S/S)/()/()/()/(/S)/(/S)/(/S)/(/S)/(/S)/(/S)/(/S)/(/S)…
```

A kind census said `f32 225 · int4 1 · ternary 200`, the tensor count matched the source, all five sentinels
were byte-identical to the shipped Flow 2, and the kernels passed `PARITY OK`. Every check in this repository
passed on a file that could not speak.

**What found it was a diff against a working model.** `BeyondSight Flow 2.aamt` is the same 27B, baked from
the MLX release by the Python baker. Comparing tensor by tensor: every one matched **except** the
GatedDeltaNet tensors — and `in_proj_qkv` was a *pure row permutation* of it (all 10,240 rows present, Q and K
byte-identical, the V block moved). Two separate conversions were missing:

1. **The value heads are interleaved.** The model has 16 key heads and 48 value heads — three per key head.
   HuggingFace writes them key-major (`v = k*3 + j`); llama.cpp interleaves (`p = j*16 + k`). So
   `HF[v] ← GGUF[(v % 3) * 16 + (v // 3)]`, on `in_proj_qkv`'s V block, all of `in_proj_z`, `conv1d`'s V
   block, and the per-head `in_proj_a`, `in_proj_b`, `dt_bias`.
2. **`ssm_a` is the decay, not its logarithm.** llama.cpp stores `A = -exp(A_log)`; HuggingFace stores
   `A_log`. Reconstructing `log(-ssm_a[perm])` matched Flow 2's `A_log` to **4.9e-8** — float32 rounding.
   Copying it across unchanged feeds a decay of −0.04 into a recurrence expecting a log of −3.2: a 100x error
   in every decay rate, which on its own is enough to make the output noise.

The checkpoint's own metadata says so — `prism.hadamard.gdn_v_grouped: true` — and this reader had been copying
that flag into the sentinel without acting on it. The flag describes a *tensor-data* layout, so the conversion
belongs in the baker, and it is now `regroupGdnValueHeads` in `baker.ts`, reported by `aamt bake` as
`regroup N GatedDeltaNet tensor(s)`.

**After the fix Flow-3 answers correctly** — `The capital of France is Paris.`, `stop=eos` — and decodes at
12.6 tok/s against 11.6 on the broken file.

Two things this changes about how the rest of this page should be read:

- **`PARITY OK` is not evidence that a model works.** It compares two kernels on the same weights; when the
  weights are wrong, both are wrong together. `host/generate-smoke.ts` exists because of this and asks the
  model a question instead.
- **The fix was necessary for the 4B and not sufficient.** After it the 4B still emitted noise. The next
  section is what the rest was.

### The 4B's block size is 512, and its head is tied

The technique above starts from a working model to diff against, and for the 4B that is
`prism-ml/Ternary-Bonsai-2-4B-mlx-2bit`. It answered a different question from the one it was fetched for.
Baked from the MLX release instead of the GGUF, the 4B **still** emitted noise in this engine, while the
publisher's own loader (`runtime/artifact.py`, under MLX) answered `The capital of France is Paris.` from
the same weights. A noise that survives a change of source container is not in the conversion, so the place
to look was the engine. It had two bugs, and the 27B can reach neither:

1. **`hadamard_rotate.wgsl` assumed a block of 1024.** The 27B's contract says `block_size: 1024`; the 4B's
   says `512`. The shader had `const BLOCK: u32 = 1024u` and `1/sqrt(1024)` written in, while the host
   dispatched `width / 512` workgroups a row. So every block past the first started 1024 elements in — past
   the end of its own 512 — and every block got a 1024-point transform at the wrong scale. Every rotated
   activation was wrong. The block size and `1/sqrt(block)` now travel in the uniform, and the host refuses a
   block the shader cannot hold rather than transforming it at the wrong width.
2. **A tied output head needs the forward rotation, and the contract never says so.** The 4B has no
   `output.weight`: its head is the embedding used a second time (`prism.hadamard.tied_output: true`). The
   contract lists that tensor once, in `inverse_weight_names`, which is right for its embedding role — the
   stored rows are rotated, so a lookup un-rotates them. The head role is the mirror image: scoring
   `E_true · h` against stored rows means rotating `h` **forward** first. A gate that asked
   `weight_names.has(name)` skipped it, and the head scored a correctly computed hidden state in a basis the
   tokenizer has never seen. `needsForwardRotation` in `hadamard.ts` is now the one gate both dispatch sites
   ask. The 27B is untied, so it never reaches this path.

A third path had the same gap and does not make noise, which is why generation never showed it. The
**selective** projection — `dispatchSelectiveMatvec`, which scores only a few candidate rows of the head for
the router's three-way choice — had no Hadamard gate at all, on any rotated checkpoint, Flow-3's untied
`lm_head` included. `session-bench.ts` caught it: selective and full logits 5.4 apart at the same three
tokens, where NightOwl's agree to 1e-7. It now asks the same gate, and the two agree to 2.4e-7. (The bench
itself had a second, smaller error here: it compared logits from repeated steps at one position, which on a
hybrid model are different recurrent states. It now compares two fresh replays.)

**Each of the first two is enough on its own to make the output noise.** With both fixed, the GGUF bake and the MLX bake
each answer `The capital of France is Paris.`, `stop=eos` — Prism's runtime word for word — and turning
either fix back off returns the noise.

What pins them:

- `host/hadamard-rotate-check.ts` runs the shader on a real GPU against the CPU reference
  (`rotateForward` / `rotateInverse`) at both block sizes, both directions, several rows. It matches to about
  3e-7. Pointed at the old shader it fails every 512 case by about 4 and passes every 1024 one — which is
  exactly why the 27B never showed it.
- `packages/voxel-runtime/test/hadamard-gate.test.ts` pins the tied-head truth table, checks the algebra
  numerically, and asserts the 1024 constant has not come back. It needs no GPU.

The lesson is the same as the section above's, one level down. `generate-smoke.ts` caught both, where every
kernel check had passed. A shader written for one checkpoint carries that checkpoint's constants, and the
comment beside this one said in so many words that "a different block size would need a different shader".
It did, and the 4B shipped without one.

---

## 7. Doing it yourself

```bash
# A Bonsai 2 GGUF, straight to an .aamt. The format is detected from the file's magic, not its name.
npx @asamanthinks/aamt bake ~/Downloads/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
  --out GaiaN-2.aamt --policy bonsai2

# The same, with least-squares group scales. Byte-identical on an exactly-ternary checkpoint.
npx @asamanthinks/aamt bake ~/Downloads/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
  --out GaiaN-2.aamt --policy bonsai2-ls

# Inspect what it wrote, then serve it.
npx @asamanthinks/aamt inspect GaiaN-2.aamt
npx @asamanthinks/aamt serve GaiaN-2.aamt
```

The bake prints the source, the type census and the sentinels:

```
GaiaN-2.aamt
  from    /Users/you/Downloads/Ternary-Bonsai-2-9B-PTQ1_0.gguf
  size    2.11 GB
  tensors 427 (f32 24 · ternary 402 · int4 1)
  policy  bonsai2 (= curated)
  source  GGUF v3 · qwen35 · ptq1_0 202 · f32 225
  wrote   __tera_meta__ · __hadamard_meta__ · __hybrid_meta__ · __rope_meta__ · __chat_contract__
  ✓ the checkpoint's Hadamard contract came out of the file's own metadata, with its
      tensor names translated to the engine's — nothing had to be supplied separately.
```

To read a source file's metadata without baking it, a ranged fetch of the first ~90 MB covers the whole
metadata block and tensor index, and `parseGgufHeader` / `parseGgufMetadata` / `parseGgufTensorInfos` from
`packages/voxel-runtime/src/gguf.ts` give the entire index over that prefix:

```bash
curl -sL -r 0-90000000 \
  https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -o /tmp/probe.gguf
```

That is how every number on this page was read — off the files, rather than from a README.

---

## 8. What this does not do yet

Stated here rather than left to be discovered:

- **The vision models are readable now; the catalogue still has no place to put one.** The Bonsai 2 2B/4B/9B
  releases ship an `mmproj-*.gguf` projector, and `vision-forward.ts` reads it: the tower's shape comes from the
  checkpoint's own `clip.vision.*` metadata rather than from the 27B's hard-coded constants, and the tensor
  names are resolved through `vision-config.ts`'s role map (`v.blk.0.attn_qkv.weight` and
  `vision_tower.blocks.0.attn.qkv.weight` are the same tensor). So a 24-block, 1024-wide, 2048-projecting tower
  runs through the same forward pass as the 27B's 27-block, 1152-wide, 5120-projecting one — which it did not
  before, and which would have been silent rather than loud. What is still missing is the _distribution_ half:
  `CatalogModel` has one `path` and `pull` fetches one file, so an image model needs a second path with a role
  and a loader that binds both. A row pointing at the text half alone would be a model that answers as if it
  could see.
- **The vision token ids are still checkpoint-specific.** `host/server.ts` holds `IMAGE_TOKEN_ID = 248056` and
  its two siblings as constants. They are correct for this `qwen3_5` family and read from its own
  `config.json` originally, but a different family would need them from the model rather than from the server.
- **A 27B bake needs headroom.** Peak memory is one tensor's float32 plus its blob. The embedding is
  248,320 x 4096 weights, so it alone needs ~4 GB of float32 on the way past. `bakeGguf` does not do the
  subprocess-per-chunk dance that `bake_bonsai2_hadamard_aamt.py` does for exactly this reason.
- **`MTP-*` and `DFlash2-*` are not read.** They are a multi-token-prediction head and a speculative drafter,
  in separate files. This repo's speculative path is `speculative-bonsai.ts` and `ngram-spec.ts`, which take a
  separate draft model; a Bonsai 2 drafter is a follow-up.
- **`K`-quants are refused**, by name, on purpose. See §5.
- **The catalogue is not re-signed.** `apps/aamt-cli/scripts/make-catalog.ts` now lists the v2 models, but
  `catalog-data.ts` is generated by running that script against a release folder with the signing key, and
  until it runs the CLI's `models` list and `pull` still see the v1 set. That is the intended workflow: the
  script refuses to sign anything whose SHA-384 it cannot compute from a real file.
