# Guide: putting any model in an `.aamt` wrapper

**Status:** proposed design and procedure; the container pieces it relies on exist (marked *exists*), the manifest record, its CLI writer and a provider that verifies before serving exist; starting the engine from a `launch` block is still to build (marked *to build*).
**Why:** to run models you did not bake (GGUF, safetensors, anything a server engine loads) under the same identity, provenance, policy and deployment rules as our own checkpoints, locally or in a cloud, **without converting the weights**.
**First case:** Strata's Qwen3.8-Flash-Next GGUF files. The same steps apply to vLLM, llama.cpp, TGI or any engine that serves an OpenAI-style API.

## 1. The idea in one page

An `.aamt` container is a content-addressed file with up to five layers (`model`, `world`, `mind`, `experience`, `code`). Every record has a SHA-384 identity, and the file has one `fileId` for the whole content (*exists*, ADR-005).

A **wrapper** is an `.aamt` whose `model` layer does not carry the weights. It carries a small, signed-able **manifest** that names the weight files (each with its own SHA-384), the engine that can serve them, and the facts a deployment needs to decide whether to run them: architecture, quantization, licenses, tokenizer, context limit, resources.

```
wrapper.aamt  (KBs)                           weights (GBs, anywhere)
┌──────────────────────────────┐              ┌────────────────────────┐
│ model layer                  │   names +    │ model-00001.gguf       │
│   record: MODEL_MANIFEST ────┼── hashes ──▶ │ model-00002.gguf       │
│   record: tokenizer config   │              │ mmproj.gguf  …         │
│ code layer   (optional Yare) │              └────────────────────────┘
│ experience   (optional)      │                       ▲
└──────────────────────────────┘                       │ engine reads
        ▲   one fileId = the identity of this model      │ (Strata, vLLM, llama.cpp)
        │   as deployed                                  ▼
   OS / cloud control plane ── checks ──▶ engine adapter ──▶ OpenAI-style API
```

What you gain without touching the weights:
- **One identity** for "this model, these exact bytes, this engine config", usable as a pin, a cache key and an audit entry.
- **Verification before load**: the adapter refuses files whose hashes do not match the manifest.
- **Provenance and policy** attached to the model (who made the quantization, which license, what it may be used for) and readable by the OS's policy layer.
- **Sealed deployment**: the manifest, or the weights themselves, can be encrypted per record for cloud storage (*exists*: KEYED layers, ADR-010).
- **Same surface for every model**: the OS asks for `llm.chat`; the wrapper decides which engine answers.

## 2. What exists and what is new

| Piece | State |
|---|---|
| Container, layers, SHA-384 identity, deterministic packing | *exists* (`packages/aamt-container`) |
| A model layer that holds records, with page-aligned payloads | *exists* |
| `RECORD_KIND_AMDL_EMBEDDED` (1) — a whole AMDL file inside | *exists* |
| `RECORD_KIND_AMDL_REF` (2) — "a reference to an AMDL file kept outside the container (UTF-8 JSON)" | *exists as a constant only*: nothing in the repo reads or writes it |
| Streaming writer for multi-GB payloads without holding them (`writeContainer`, `ChunkSource`) | *exists* (ADR-017) |
| Per-record sealing for KEYED layers | *exists* (ADR-010) |
| A manifest record for **non-AMDL** weights, with hashes and engine facts | *exists*: `RECORD_KIND_MODEL_MANIFEST` (6), `packages/model-wrapper`, `maiiam model wrap` |
| Hash-check-before-load in the OS, and a provider that serves from a wrapper | *exists*: `plugins/wrapper-engine` verifies the container, every file and every licence before it registers `llm.chat`; it connects to an engine already running |
| Starting the engine from the manifest (`launch`) | **to build**: the manifest currently has no launch record, and the provider refuses to register a down engine rather than spawn one |

Because `AMDL_REF` is reserved but unused, a new record kind (`MODEL_MANIFEST`, suggested value 6) costs nothing in compatibility: old readers see an unknown kind in a layer they already know how to skip.

## 3. Decide: reference, embed, or both

| Mode | Weights live | Use when |
|---|---|---|
| **Reference** (default) | outside the wrapper: disk, object storage, a registry | big models, shared weights, many wrappers per weight set |
| **Embed** | inside the container, streamed in with `writeContainer` | air-gapped transfer, one file to ship, small models |
| **Embed, sealed** | inside, encrypted per record | untrusted storage (a public bucket), keys held by the deployment |

Rule of thumb: reference for anything over a few GB; embed when you must move exactly one file.

## 4. The manifest (to build)

One JSON record, canonical form (sorted keys, no whitespace), so its identity is stable. Suggested schema (`maiiam.model/1`):

```jsonc
{
  "schema": "maiiam.model/1",
  "name": "Qwen3.8-Flash-Next",
  "variant": "IQ2_XS",
  "family": "qwen3.8-flash-next",
  "architecture": { "kind": "moe", "layers": 48, "experts": 512, "activeExperts": 10, "attention": ["gated-deltanet", "attention"], "vocab": 248320 },
  "format": "gguf",                       // gguf | safetensors | amdl | other
  "quantization": { "scheme": "IQ2_XS", "producer": "ISTA-DASLab (GSQ-RCO)" },
  "files": [
    { "role": "weights", "shard": 1, "of": 2, "path": "Qwen3.8-Flash-Next-IQ2_XS-00001-of-00002.gguf",
      "bytes": 40000000000, "sha384": "…96 hex…",
      "source": "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF" },
    { "role": "weights", "shard": 2, "of": 2, "path": "…-00002-of-00002.gguf", "bytes": 28800000000, "sha384": "…" },
    { "role": "vision-projector", "path": "mmproj-….gguf", "bytes": 0, "sha384": "…" },
    { "role": "draft", "path": "mtp-….gguf", "bytes": 0, "sha384": "…" }
  ],
  "tokenizer": { "kind": "embedded-in-gguf" },
  "context": { "max": 262144, "default": 32768 },
  "engine": {
    "kind": "openai-http",                // how the OS talks to it
    "candidates": ["strata", "llama.cpp"],// engines known to load this format
    "launch": { "strata": ["--model", "{weights[0]}", "--ctx", "{context}"] }
  },
  "resources": { "ramGB": 48, "vramGB": 12, "diskGB": 80 },
  "licenses": [ { "scope": "weights", "id": "Qwen Community License", "url": "…" },
                { "scope": "engine",  "id": "MIT" } ],
  "capabilities": ["llm.chat", "llm.session", "vision"],
  "provenance": { "madeBy": "…", "madeAt": "…", "notes": "…" }
}
```

Rules:
- `sha384` is **our** hash (SHA-384 with the container's `model` context), of the whole file, computed by us. Never copy a hash from a download page and call it verified.
- `bytes` is checked too; a size mismatch is refused before hashing.
- Nothing in `launch` is executed by the OS as a shell string. It is an argument list with named placeholders the adapter fills.
- Licenses are data the policy layer reads; the wrapper does not grant a license, it records one.

## 5. Procedure: wrap a model

### 5.1 Get the files, and decide you trust them

1. Download the weights from the publisher yourself, into a directory used for nothing else. Treat them as untrusted input.
2. Record where they came from (URL, date, publisher's own checksum if given).
3. Read each license; copy its name and URL into the manifest.

### 5.2 Hash and describe

For each file, compute SHA-384 by streaming (never load it whole), and read what you can from the format itself rather than typing it: a GGUF header gives architecture, tensor count and quantization types; a safetensors header gives tensor names and dtypes. Fill `architecture` and `quantization` from the header, and check them against what the publisher says; if they disagree, stop.

### 5.3 Write the wrapper

Reference mode (small file):

```ts
import { packContainer, RECORD_KIND_MODEL_MANIFEST } from '@maiiam/aamt-container';
import { nodeSha384 } from '@maiiam/aamt-container/node';

const manifest = canonicalJson(buildManifest(files));             // §4
const bytes = packContainer(
  { layers: [{ kind: 'model', records: [{ kind: RECORD_KIND_MODEL_MANIFEST, payload: manifest }] }] },
  nodeSha384,
);
// the same content packs to the same bytes, so the wrapper's fileId is reproducible
```

Embed mode (weights inside, streamed, so a 70 GB model is never resident):

```ts
import { writeContainer } from '@maiiam/aamt-container';
import { openFileSink, fileSource, nodeSha384 } from '@maiiam/aamt-container/node';

const sink = await openFileSink('model.aamt');
await writeContainer(
  { layers: [{ kind: 'model', records: [
      { kind: RECORD_KIND_MODEL_MANIFEST, payload: manifest },
      ...files.map((f) => ({ kind: RECORD_KIND_OPAQUE, payload: fileSource(f.path) })),   // a ChunkSource
  ] }] },
  nodeSha384, sink,
);
await sink.close();
```

Embed, sealed: set the layer's `flags` to include `LAYER_FLAG_KEYED` and pass `seal: { ports: nodeCrypto, keys: { model: key } }` to the writer. Records are then sealed per record, and the identities cover what is stored.

### 5.4 Verify what you made

- `verifyContainer(bytes, nodeSha384)` returns `ok`. (`openContainer` trusts the header's identity; `verifyContainer` recomputes it. Always use the second before running anything.)
- Re-pack from the same inputs and confirm the `fileId` is identical (determinism).
- Corrupt one byte of a weight file and confirm the adapter refuses it (§6).

## 6. Procedure: run a wrapped model *(provider done; engine launch to build)*

A provider plugin (`host: "process"`, the kind the OS already supervises) does this, in order, and **fails closed at every step**:

1. **Open and verify the wrapper** (`verifyContainer`), and read the manifest.
2. **Admit by policy.** The manifest's licenses, capabilities and resources go to the OS's policy check (`policy.check`) and to the Sovereign guard. A model whose license the owner has not allowed does not start.
3. **Check the files.** For each `files[]` entry: exists, `bytes` equal, SHA-384 equal. A multi-GB hash takes time, so cache the result keyed by (path, size, mtime) and re-hash when any of the three change. Optionally pin: the owner may store the expected `fileId` and refuse any other.
4. **Choose an engine** from `engine.candidates` that is installed and supports `format`.
5. **Launch it** with the manifest's argument list, bound to `127.0.0.1` only, on a random port, with an API key the plugin generated and keeps in the secret store. Never a shell string; never expose the port beyond loopback.
6. **Wait for health**, with the watchdog rule from `STRATA-IDEAS-PLAN.md` Phase 1 (busy is not dead).
7. **Serve `llm.chat`** by proxying the OpenAI-style streaming API, mapping tools and stop reasons, and reporting engine telemetry to Health.
8. **Record** (activity log): wrapper `fileId`, engine, version, start time. A response can then be traced to the exact model identity.

**Done (2026-10-17), steps 1–3 and 7.** `plugins/wrapper-engine` opens and verifies the container,
checks every file by size and SHA-384 (streamed through `ctx.fs.hashFile`, never loaded whole),
refuses a licence the owner has not named or a wrapper other than the pinned `fileId`, and only then
registers `llm.chat` through `openAiEngine`. It refuses a down engine rather than registering a
provider that cannot answer. **Step 5 remains to build:** the manifest has no `launch` record yet, and
spawning a command from a manifest is its own trust decision. Until then the owner starts the engine
and points `baseUrl` at it; the identity and the file checks are already enforced.
<!-- claim: the wrapper provider verifies container, files and licences before registering an engine | test: plugins/wrapper-engine/test/wrapper-engine.test.ts -->

Unverified, stated plainly: I have not run Strata or any engine against a wrapper. The OS already has the pieces (supervised process plugins, `llm.chat` capability resolution, secret store, Sovereign guard), and Strata documents an OpenAI/Anthropic-compatible localhost API, but the end-to-end path is untested.

## 7. Cloud deployment

The wrapper is small and the weights are large, which fits the usual cloud shape:

```
 control plane                          GPU nodes
 ┌──────────────────────┐              ┌───────────────────────────────┐
 │ registry of wrappers │  fileId →    │ node agent                    │
 │ (small .aamt files,  │──────────────▶│  1 fetch wrapper by fileId    │
 │  signed)             │              │  2 verify, admit (policy)     │
 │ policy + licenses    │              │  3 fetch weights by sha384    │
 └──────────────────────┘              │     from object store / cache │
          ▲                            │  4 start engine on loopback   │
          │ audit: who ran which       │  5 expose llm.chat via the    │
          │ fileId, when, on what node │     node's gateway            │
          │                            └───────────────────────────────┘
```

- **Weights by content hash.** Object storage keyed by SHA-384 gives deduplication (many wrappers, one weight set), safe parallel downloads, and a free integrity check. Nodes keep a local cache keyed the same way.
- **Wrapper as the unit of rollout.** "Deploy model X" means "make node Y run wrapper Z"; rolling back is running the previous `fileId`. Two nodes with the same `fileId` are running the same model by construction.
- **Sealed weights in untrusted storage.** Embed-sealed, or reference plus encrypted objects; the key is released to a node only after it passes admission. Do not put keys in the wrapper.
- **Multi-model hosts.** One node can hold several wrappers; the adapter respects each manifest's `resources` and refuses to over-commit (this is where the OS's `maxResident` and VRAM budget come in).
- **Other engines, same wrapper shape.** For vLLM or TGI, `format` is `safetensors`, `engine.candidates` lists them, and `launch` carries their arguments. The OS and the control plane never need to know the engine.
- **Tenancy.** Each tenant gets its own wrapper identity and API key; the gateway maps key → wrapper → node. Audit lines carry the `fileId`.

## 8. Security checklist

- Weights and manifests are untrusted input until hashed and verified; never execute anything from a manifest except an argument list the adapter itself assembled.
- Loopback only, random port, per-launch key. No `0.0.0.0`.
- Pin by `fileId` in production; "latest" is for development.
- Treat a license mismatch as a hard stop, not a warning.
- Re-hash on any size/mtime change; do not trust a cache after the file moved.
- Keep keys out of the wrapper, out of logs and out of the manifest.
- The engine is a separate process with its own bugs: run it with the fewest permissions the OS offers, and restart it on a stall rather than patching around it.
- Downloaded installers and scripts (Strata's included) run only after you have read them; the OS never runs an engine's installer for you.

## 9. Testing the wrapper path

Write these before the loader:
1. **Determinism:** the same inputs give the same wrapper bytes and `fileId`.
2. **Tamper tests**, each of which must refuse: flip one byte of a weight file; truncate it; swap two shards; change the manifest; change a recorded size; point `path` outside the model directory (`..`, absolute path, symlink).
3. **Policy:** a manifest with a disallowed license does not start; one with a missing capability is refused at admission.
4. **Lifecycle:** engine dies → the provider reports `error` and recovers; port already in use; engine never becomes healthy; API key mismatch.
5. **Large files:** a streamed 10 GB hash does not allocate 10 GB.
6. **Cross-engine:** the same weights wrapped twice for two engines have different wrapper identities and the same weight hashes.

## 10. Suggested build order

1. `RECORD_KIND_MODEL_MANIFEST` and its codec (canonical JSON, validation, size/hash checks), with the tamper tests of §9.
2. A CLI: `maiiam-wrap <dir> --engine strata --out model.aamt` (hash, read headers, write reference wrapper, verify).
3. The provider plugin for one engine (Strata), with the lifecycle tests.
4. Policy hooks (licenses, resources, pins).
5. Embed and sealed modes.
6. The cloud control plane pieces (registry, node agent), reusing 1–4 unchanged.

Steps 1–3 are the minimum for "a Strata model running in the OS under an `.aamt` identity".
