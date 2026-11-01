# Serve a model into the tools you already use

`aamt serve` answers the **OpenAI HTTP API**, so the models this engine runs can drop into the local-AI
setup you already have — Ollama's UI, LM Studio, Open WebUI, Continue, Cursor, LangChain,
`openai-python`, anything with a "base URL" box. No SDK, no adapter, no rewriting your flow.

```bash
aamt serve innersight              # http://127.0.0.1:11434/v1
aamt serve ./my-model.aamt         # a model of your own
aamt serve gaian --port 8080
```

```
  gaian · signed BeyondSight · 3.23 GB

  → http://127.0.0.1:11434/v1
  Open http://127.0.0.1:11434/ in a browser for the setup notes.

  export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
  export OPENAI_API_KEY=unused   # most clients require the field; nothing here reads it

  the model loads on the first request; Ctrl-C stops it
```

The default port is **11434**, which is Ollama's — so on most machines it is already the number in the
right box, and a tool pointed at Ollama needs only its base URL changed.

---

## 1. The three ways to a model

| You have | Command | What it is |
|---|---|---|
| A curated BeyondSight model | `aamt pull gaian` then `aamt serve gaian` | Signed, hash-checked against the catalogue in this package |
| A `.aamt` file of your own | `aamt serve ./my-model.aamt` | Read where it lies; nothing is copied into `~/.aamt` |
| A safetensors checkpoint | `aamt bake ckpt.safetensors --out mine.aamt --policy curated` | Quantised into AMDL by the BeyondSight recipe, recovered from the shipped models — see [§5](#5-baking-your-own) |

`aamt inspect <file>` answers "which of these is this file?" without loading anything:

```bash
aamt inspect ./my-model.aamt
```

```
my-model.aamt
  path     /Users/you/models/my-model.aamt
  size     3.23 GB
  format   AMDL v1, 306 tensor(s) — what this engine runs
  sha384   4f2c…a91e
  signed   no curated model has this hash
  tensors  306 · 3.19 GB of weights
  kinds    ternary 289 (3.11 GB) · int4 17 (84.2 MB)
  covers   model.embed_tokens.weight … lm_head.weight

  ✓ this engine can run it — `aamt run my-model.aamt` or `aamt serve my-model.aamt`
```

That verdict line is the one to read. It comes from the file's own bytes, not from its extension, and it is
why this exists: a `.aamt` can be four different things, and two of them this engine cannot run.

---

## 2. Wiring it into what you already run

Set two variables and point the tool at this machine. `OPENAI_API_KEY` is required by most clients and read
by nothing here — any non-empty value works.

```bash
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=unused
```

### `openai-python`

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="unused")
print(client.chat.completions.create(
    model="gaian",
    messages=[{"role": "user", "content": "hello"}],
).choices[0].message.content)
```

### `curl`

```bash
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gaian","messages":[{"role":"user","content":"hello"}]}'
```

Add `"stream": true` for server-sent events.

### Continue, Cursor, and editors

Both take an OpenAI-compatible provider with a custom base URL:

```json
{
  "models": [
    {
      "title": "BeyondSight GaIaN (local)",
      "provider": "openai",
      "model": "gaian",
      "apiBase": "http://127.0.0.1:11434/v1",
      "apiKey": "unused"
    }
  ]
}
```

### LangChain

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="unused", model="gaian")
```

### Anything that only speaks Ollama's own API

Point it at `http://127.0.0.1:11434` — the port is the same on purpose — but note that this server answers
the **OpenAI** paths, not `/api/chat`. A tool that offers both should be given the OpenAI one.

---

## 3. The endpoints

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | The completion. `stream: true` for SSE. |
| `GET /v1/models` | The one model this server has loaded. |
| `GET /healthz` | `{ ok, model, ready, served }` — whether the weights are on the GPU yet. |
| `GET /` | The setup notes above, in a browser. |

Request fields that are honoured: `model`, `messages`, `stream`, `max_tokens`, `temperature`, `top_p`.

**`model` must be the model this server serves.** Asking for another is a 400 that says so, rather than an
answer from the wrong weights — start a second `aamt serve` on another port for another model.


---

## 4. What is not implemented, and why that is a 400

`n`, `logprobs`, `top_logprobs`, `tools`, `tool_choice`, `response_format`, `stop`, `/v1/completions`, and
the embeddings endpoint.

Each is a real feature. **Each is refused by name** rather than accepted and ignored, because a request that
asks for three choices and gets one, with no indication why, is a wrong answer that looks like a right one.
`aamt serve` is meant to be safe to point a tool at; that is what makes it so.

**Tool-calling is a tier line, not an oversight.** A local model server is infrastructure — like the
compiler, it is free and it stays free. Tool-calling is where an agent loop lives, and an agent loop needs
the state, the journal and the capability model the OS tier exists to provide. That boundary is recorded in
[ADR-044](../decisions/ADR-044-tool-calling-is-the-tier-line.md) so the next person to meet the 400 meets a
decision rather than an omission. It applies in both directions: the OS's own `openai-engine` plugin sends
`tools`, so a tool-using flow through a locally served model is *expected* to fail.

Two limits worth knowing before you build on it:

- **One completion at a time.** The engine holds one KV cache and one conversation, so requests queue in
  arrival order. A client that expects concurrency gets serialization — the honest alternative would be two
  clients silently corrupting each other's context.
- **No authentication.** It binds `127.0.0.1` and nothing else unless you pass `--host`, and a non-loopback
  bind prints a warning. Note that "hosting or serving models to others" is a paid licence tier, so a public
  bind is a licensing question as well as a security one.

---

## 5. Baking your own

Two different things get called "baking", and only one of them produces a file this engine runs. This
section says which is which, because the difference costs an afternoon.

### 5a. Converting weights into an AMDL `.aamt` — **the baker exists, the recipe does not**

The engine reads **AMDL**: group-quantised tensors, a JSON manifest, a 32-byte footer. And this repo can
produce it:

```bash
aamt bake my-checkpoint.safetensors --out my-model.aamt
```

```
my-model.aamt
  from     my-checkpoint.safetensors
  size     3.0 MB
  tensors  112 (int4 112)
  policy   int4
  sha384   4ff4c490…c37f2

  ✓ 112 tensors read back from the footer — `aamt inspect my-model.aamt`
  · every tensor at 4 bits per weight, group-quantised. The floor, not the curated recipe.
  · this is not the curated BeyondSight recipe (ternary and 1-bit with float32 norms).
    It will run; it is not the same model at the same quality.
  · `aamt run my-model.aamt` to chat, or `aamt serve my-model.aamt` to serve it.
```

The pieces are all real and all tested: `bake()` quantises one tensor (int4, ternary or 1-bit), `AmdlWriter`
streams the model out with bounded memory — one tensor at a time, so a 27B checkpoint needs no more RAM than
a 1B one — and `bakeModel()` in `aamt-model.ts` does the same job in a single buffer for small models.
`baker.test.ts` asserts the two writers agree **byte for byte**, so a model baked either way is the same file.

Working on the baker itself, `packages/voxel-runtime/host/bake-aamt.ts` runs without building the CLI:

```bash
tsx packages/voxel-runtime/host/bake-aamt.ts my-checkpoint.safetensors my-model.aamt
```

**The recipe is in the repository, recovered from the models themselves.** The baker takes a `BakePolicy`,
and `--policy` offers three:

| `--policy` | What it does |
|---|---|
| `curated` | The BeyondSight recipe at ternary: norms f32, the output head int4, everything else ternary |
| `curated-1bit` | The same rule at 1 bit, which is NightOwl |
| `int4` | Everything at four bits. The floor, and the default |

The recipe was not guessed and it is not a copy of an external script — it was **read off the three shipped
models**. Each `.aamt` carries its manifest in its last few kilobytes, so every model can be asked what its
own quantisation policy was over an HTTP range request, without downloading three gigabytes:

```
  InnerSight.aamt            0.72 GB  AMDL v1  311 entries  {int4: 1, f32: 113, ternary: 196}
  BeyondSight-NightOwl.aamt  2.28 GB  AMDL v1  401 entries  {int4: 1, 1bit: 253, f32: 145}
  BeyondSight-GaIaN.aamt     3.23 GB  AMDL v1  401 entries  {int4: 1, ternary: 253, f32: 145}
```

All three agree, and the rule is:

| Tensor | Kind |
|---|---|
| every `*norm.weight` — `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `model.norm` | `f32` |
| every projection — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | the body kind |
| the input embedding, when the model has a separate output head | the body kind |
| the output head — `lm_head.weight`, or a tied `model.embed_tokens.weight` | `int4` |
| a tied `lm_head.weight` | **dropped**, not written |

And one rule that is about shape rather than kind: **the embedding is trimmed to the tokenizer's real
vocabulary.** Qwen3 pads its embedding to a multiple of 128 for tensor-core alignment — 151936 rows — while its
tokenizer defines only 151669 (`vocab.json`'s 151643 plus 26 added tokens). The 267 padding rows are
unreachable, and the curated models drop them. That rule is why a bake of Qwen3-1.7B with the recipe matched
InnerSight's tensor count and every kind and still differed from it in *every offset* after the embedding: the
embedding was 267 rows too long, and everything downstream had shifted.

`aamt bake` applies the trim only when the checkpoint carries a tokenizer, and only ever trims. A checkpoint
with no `vocab.json` keeps its table whole, because cutting rows on a hunch would silently remove tokens the
model can emit.

**What a bake cannot reproduce, and why that is not a defect.** Baking Qwen3-1.7B with this recipe gives a file
whose 310 tensors, kinds, shapes, lengths and offsets all match InnerSight exactly — measured, zero per-tensor
differences. It will not match InnerSight's SHA-384, and it cannot, for two reasons:

1. **The curated models are trained derivatives.** The HuggingFace card says *"Derived from Qwen3"*, and the
   weights differ: InnerSight's `input_layernorm[0]` is `0.06835938` where Qwen3-1.7B's is `0.06591797`. An f32
   norm is stored raw, so that is the source weights, not the quantiser. Those weights are the product.
2. **The released files carry a keyed watermark.** `tools/watermark/aamt_watermark.py` — *"PRIVATE: never ship
   this file"* — multiplies about 1% of group scales by `(1 ± 4e-4)` after baking. It is output-preserving by
   design, and removing it breaches the licence.

So the recipe is verifiable up to the weights and no further: the structure is exact, and the values are
theirs. That measurement is also how the second reason was ruled *out* as the cause — 99.99% of one tensor's
scales differed by ~48%, where a watermark moves 1% of them by 0.04%. A 4e-4 perturbation is invisible against
a different model.

Two of the rules are worth understanding rather than copying. **Norms are never quantised**: a norm weight is a
per-channel scale, and two bits collapse 1.5 and 0.5 into the same value, with the error compounding through
every layer it feeds. **The output head is int4 even in the 1-bit model**, because it turns hidden state into
logits and one bit there is a vocabulary of coin-flips — which is also why InnerSight, whose embeddings are
tied, has `embed_tokens` at int4 while GaIaN has it at ternary. Reading the kind off the tensor's *role* and
not its name is what makes one rule cover both.

`packages/voxel-runtime/test/curated-policy.test.ts` holds the policy to those counts, so the recipe is a
fact about the shipped models rather than a preference, and `apps/aamt-cli/test/bake.test.ts` holds the file
the CLI writes to the same rule.

Because the test asserts a table, it cannot notice a model changing. The check that can is a script:

```bash
tsx packages/voxel-runtime/host/verify-curated-recipe.ts
```

It fetches each model's own manifest over byte ranges — a few hundred bytes per model, not 6.2 GB — and
holds `curatedPolicy` to it tensor by tensor, requiring that exactly one body width matches. Run it after a
re-bake, or when adding a model, and it will say which tensors disagree and how.

Both checks were worth writing. The first version of the norm rule matched `q_norm` and `model.norm` and
silently missed `input_layernorm` and `post_attention_layernorm`, which would have quantised 72 of GaIaN's 145
norms to two bits — and only the comparison against the shipped manifest said so.

**Two of the four metadata sentinels are written and two are not.** `__tera_meta__` is recomputed from the
weights, per layer from `gate_proj`, and `__chat_contract__` is copied from the tokenizer — byte for byte
against the shipped models. `__rope_meta__` and `__hybrid_meta__` come from the source `config.json` and from
the Bonsai path's hybrid-attention detection, and neither is written by `aamt bake`.

They carry model-level facts (the chat contract, per-layer state, rope and hybrid-attention configuration)
that come from the source `config.json` and tokenizer rather than from the weights. The engine treats them
as optional and additive, so a model baked without them loads and runs — but it relies on the engine's own
constants matching the source model, which is the assumption `docs/QUALITY_DELTA_2026-09.md` §19.2 records
as unsafe for a new source family. Baking the remaining two is the next piece of work, not a solved one.

Two things it refuses rather than guesses, both because the alternative is a model that loads and is wrong:

| Input | What happens |
|---|---|
| An already-quantised checkpoint (U32/I8/FP8 tensors) | Refused by name. Its bytes are packed bit-fields with separate scales, so re-quantising them produces nonsense — and a *silent* version of this used to bake files that `aamt inspect` called "✓ this engine can run it". |
| A truncated checkpoint | Refused, and no partial file is left behind. `readSync` returns a short count silently, so the missing tail would otherwise bake as zeros. |

### 5b. The wrapper path — **exists**, and is a different thing

[`MODEL-WRAPPER-GUIDE.md`](MODEL-WRAPPER-GUIDE.md) describes wrapping *any* model in a `.aamt` whose model
layer holds a **manifest**: engine facts, file hashes, licences, and either a reference to weights that stay
outside the container or the weights themselves embedded and streamed in. `maiiam model wrap` builds one,
and the OS's `wrapper-engine` plugin verifies it and serves `llm.chat` from it.

That wrapper is a **container**, and it is not an AMDL file. `aamt serve` cannot run one, and it says so
rather than failing obscurely:

```
$ aamt serve ./wrapper.aamt
wrapper.aamt is a v1.3 container (layers, not raw tensors), and this engine runs AMDL.
  A container holds a manifest, and its weights are outside it — docs/aamt/MODEL-WRAPPER-GUIDE.md §3.
  `aamt inspect <file>` prints exactly what a file is.
```

The two paths meet at the container format and not yet at the loader. That is the honest state of it.

### 5c. What is in this repo, end to end

```
safetensors ──▶ baker.ts (BakePolicy) ──▶ AMDL .aamt ──▶ aamt run / serve / bench / mcp
                                              ▲
                                              └── aamt inspect verifies it, and says whether it will load

other weights ──▶ wrapper .aamt ──▶ the OS's wrapper-engine plugin (MaiiaM OS, not aamt)
                     ▲
                     └── maiiam model wrap, and MODEL-WRAPPER-GUIDE.md
```

The top path is real and tested. The one thing standing between it and the curated models is the
`BakePolicy` — the quantisation recipe — which is the piece this repository does not hold.

---

## 6. Measuring what you serve

`aamt bench` takes a path too, so the numbers on your own model come from your own machine:

```bash
aamt bench ./my-model.aamt
```

It reports load time, the GPU kernel tuning, sustained decode tokens per second, time to first token with
and without the prompt already read, branch reuse, and a GPU-equals-CPU check on a real tensor. For a model
of your own the card says `(your own model)` where a curated one says `(signed)`, because the hash on the
card is one computed here rather than one the catalogue knows.

---

## 7. The same models, over MCP

`aamt serve` is for tools that speak HTTP. `aamt mcp` is for tools that speak the Model Context Protocol —
and `aamt_chat` there takes the same model references, so a model of your own is available to an editor as
well as to a Python script. See [`docs/mcp/DEVELOPER-GUIDE.md`](../mcp/DEVELOPER-GUIDE.md).

