# The MaiiaM MCP surface — developer guide

Two commands serve MaiiaM's tools over the [Model Context Protocol](https://modelcontextprotocol.io):

```bash
npx @asamanthinks/aamt mcp     # no kernel needed: local models, the Vortex math, the Yare compiler
maiiam mcp                     # a running MaiiaM OS: the Yare library, minds, the world, models
```

They speak the same protocol and share **four tool names with identical arguments** — `yare_check`,
`yare_test`, `yare_explain`, `yare_admit` — backed by the same code in `packages/yare`. A model that
learns the admission loop against the free one is not learning a second dialect against the OS.

This guide is for someone wiring either one into an editor or an agent framework. It is written to be
followed: every command, tool name and argument below is what the code does.

- [1. Quick start](#1-quick-start)
- [2. What the two tiers are, and why there are two](#2-what-the-two-tiers-are-and-why-there-are-two)
- [3. `aamt mcp` — the free tier, in full](#3-aamt-mcp--the-free-tier-in-full)
- [4. The Yare admission loop](#4-the-yare-admission-loop)
- [5. `maiiam mcp` — the OS tier](#5-maiiam-mcp--the-os-tier)
- [6. How the transport is implemented, and why not the SDK](#6-how-the-transport-is-implemented-and-why-not-the-sdk)
- [7. Sealing: shipping code and weights without shipping plaintext](#7-sealing-shipping-code-and-weights-without-shipping-plaintext)
- [8. Troubleshooting](#8-troubleshooting)
- [9. What is not here](#9-what-is-not-here)

## 1. Quick start

### Cursor

`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "aamt": { "command": "npx", "args": ["-y", "@asamanthinks/aamt", "mcp"] }
  }
}
```

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "aamt": { "command": "npx", "args": ["-y", "@asamanthinks/aamt", "mcp"] }
  }
}
```

With a running MaiiaM OS, add the second tier:

```json
{
  "mcpServers": {
    "aamt": { "command": "npx", "args": ["-y", "@asamanthinks/aamt", "mcp"] },
    "maiiam": {
      "command": "maiiam",
      "args": ["mcp", "--server", "yare,mind,model,world"]
    }
  }
}
```

### Before you wire anything up

```bash
npx @asamanthinks/aamt tools        # what the server offers, without starting one
npx @asamanthinks/aamt doctor       # what this machine can do
npx @asamanthinks/aamt models       # which models are present
```

`aamt tools --json` prints the same table as JSON — the tool name, its JSON Schema, and a derived
access label. Nothing about the implementation appears in it, which is deliberate and is asserted by
a test.

### Try it without an editor

The protocol is newline-delimited JSON-RPC 2.0, so `printf` is a client:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"shell","version":"1"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"vortex_fr_dist","arguments":{"a":[0,0,0,0],"b":[1,1,1,1]}}}' \
  | npx @asamanthinks/aamt mcp
```

The third message comes back with `6.283185` — the Fisher–Rao distance between two maximally
separated registers, which is 2π exactly, computed as a Euclidean norm in angle coordinates rather
than approximated.

## 2. What the two tiers are, and why there are two

The platform splits along a line that is not arbitrary: **what needs state, and what does not.**

| | `aamt mcp` | `maiiam mcp` |
|---|---|---|
| Needs a running OS | **no** | **yes** (exits `3` without one) |
| Installed as | `npx @asamanthinks/aamt` | part of MaiiaM OS |
| Runtime dependencies | one (`webgpu` — the Dawn binding) | the CLI's own, including a kernel connection |
| Tools | 15 | every tool the OS has registered, filtered by `--server` / `--safe` |
| The Yare compiler | yes — `yare_guide`, `yare_check`, `yare_test`, `yare_explain`, `yare_admit` | the same five, same arguments, same code |
| The Yare *library* | no — there is no library to search or propose into | `yare_list`, `yare_search`, `yare_run`, `yare_propose` |
| The Vortex arithmetic | yes, in full | through `world.*` and the library's own runs |
| Local models | yes — chat, benchmark, download, verify | the OS's model manager (`model_*`) |
| Minds, the world | no | yes |
| Permission gate | the MCP client's own, driven by tool annotations | **the OS's policy host**: allow, deny, or ask in the app |

The line is where the state is. A compiler is a function from source to diagnostics; it needs nothing.
A library is a database of admitted modules with embeddings and a run allowance; it needs a running
kernel. Putting the compiler in both tiers is what makes the free one worth installing, and putting
the library in only the OS tier is what makes the OS worth running.

### Why the same names in both

`yare_check` returns the same shape from both servers, because both call
`packages/yare/src/compile.ts`. `yare_admit` returns the same refusal reasons, because both call
`packages/yare/src/admit.ts`. `yare_guide` returns the same text, because there is one guide, in
`packages/yare/src/guide.ts`, generated from the compiler's own builtin table so it cannot describe a
language the compiler does not accept.

That is a design rule, not an accident, and it is enforced: `apps/aamt-cli/test/tools.test.ts`
asserts the four names and their required arguments literally, and
`apps/kernel/test/yare-tools.test.ts` holds the OS's guide to the same `EXAMPLE` module being
admitted.

### If your tool speaks HTTP rather than MCP

Use [`aamt serve`](../aamt/SERVE-YOUR-OWN-MODEL.md) instead. It answers the OpenAI API —
`POST /v1/chat/completions`, with SSE streaming — so the same models drop into Ollama's UI, LM Studio, Open
WebUI, Continue, Cursor, LangChain and `openai-python` with nothing but a base URL. `11434` is the default
port because it is Ollama's.

The two are siblings, not alternatives: MCP is for an agent host that wants *tools* — the compiler, the
gate, the arithmetic — and `serve` is for a client that wants *a model*. Both take a curated id or a path
to a `.aamt` of your own, and `aamt inspect` answers which a file is.

## 3. `aamt mcp` — the free tier, in full

```bash
aamt mcp                      # serve on stdin/stdout
aamt tools                    # the table, as text
aamt tools --json             # the table, as JSON
```

There are no flags that change what is served. The surface is fixed, and `aamt tools` is the whole of
it. Fifteen tools, in three groups.

### The models

| Tool | Access | What it does |
|---|---|---|
| `aamt_models` | read-only | The curated models with size, kind, and whether each is downloaded |
| `aamt_doctor` | read-only | Node, platform, CPU, memory, the WebGPU adapter and its optional features, free disk |
| `aamt_verify` | read-only | Computes a file's SHA-384 and looks for it in the Ed25519-signed catalogue |
| `aamt_pull` | writes · network | Downloads a model and its tokenizer, resumably, keeping it only if the SHA-384 matches |
| `aamt_chat` | writes | One prompt, one reply, on this machine's GPU |
| `aamt_bench` | read-only | Measures load, tuning, decode tok/s, first-token time at 1k and 4k, branch reuse, and GPU-equals-CPU |

Two things about these are worth knowing before you use them:

**`aamt_chat` will not download a model for you.** If the model is not present it returns an error
naming `aamt_pull` and the exact files that are missing. A 3 GB download that starts because a model
decided to say hello is a bad surprise, and a tool call is not a place to have one. It also keeps the
model warm between calls and keeps the conversation, so a follow-up sees the earlier turns; pass
`reset: true` to start over.

**`aamt_bench` takes about three minutes.** It closes the chat session first, because it opens its
own, and it reports progress on stderr. Cancel it with `notifications/cancelled` and it stops at the
next measurement boundary — the engine's decode loop has no abort, so "cancel" means "stop at the next
seam", and that is what the code does rather than something better-sounding.


### The Vortex arithmetic

| Tool | What it computes |
|---|---|
| `vortex_fr_dist` | The Fisher–Rao distance between two TERA registers, in `[0, 2π]` |
| `vortex_geodesic` | The register at fraction `t` along the geodesic from `a` to `b` |
| `vortex_meji` | The 16-cell Meji distribution, the cell, and the lossless recovery |
| `vortex_kappa_proj` | The projection onto the conservation surface `Σ_κ` |

A **TERA register is four lanes in `[0,1]`**, ordered `thought, emotion, reason, action`. Internally it
is stored as angles — `θ_d = 2·arcsin√v_d` — because the Fisher–Rao metric is flat in `θ`, so a
distance is a Euclidean norm and a geodesic is a straight line. Every tool takes and reports `v`,
because that is what a caller supplies.

Three properties worth relying on:

- **The numbers are exact, not a similarity score.** `vortex_fr_dist` is the same function the WGSL
  kernel is checked against, and the reference machine is the oracle every other backend is proved
  against.
- **A geodesic is constant-speed.** Interpolation is linear in `θ`, which *is* the geodesic under
  Fisher–Rao, so the midpoint is equidistant from both ends.
  `apps/aamt-cli/test/tools.test.ts` asserts exactly that.
- **The Meji distribution is lossless.** It is a rank-1 outer product of four Bernoulli means, so
  `recover(mejiDist(v)) === v` — the 4-bit cell is an encoding of a continuous 4-D state, not a
  summary of it.

One honest caveat, which the tool says in its own description: `vortex_geodesic` re-derives `κ` rather
than carrying it, because a geodesic between two points on `Σ_κ` leaves `Σ_κ`. Pretending otherwise
would make the conservation invariant a lie for exactly one operation.

### The Yare compiler

| Tool | What it does |
|---|---|
| `yare_guide` | The rules, a complete admitted module, and every builtin — generated from the compiler's table |
| `yare_check` | Compile; diagnostics with line and column, or the module's functions |
| `yare_test` | Run every `example` claim the source makes about its own functions |
| `yare_explain` | One function's instructions, every register each one changed, the cost, the feature bits |
| `yare_admit` | The full gate: size, compile, limits, declared cost, capabilities, substrate round trip, parity, examples |

All five are read-only. None writes anything anywhere, and none runs anything from a library, because
in this tier there is no library. [Section 4](#4-the-yare-admission-loop) is the loop.


## 4. The Yare admission loop

This is the part of the surface that is not a wrapper around something else. The compiler is the
oracle: a model does not get a second model's opinion about whether its code is right, it gets
diagnostics with a line and a column, a list of register changes, and a gate that refuses with
reasons.

The order is deliberate, and it is cheapest-first:

```
yare_guide    read this once, before writing anything
    ↓
yare_check    compile → diagnostics (line, col, message) or the module's functions
    ↓
yare_test     run the `example` claims the source makes about its own functions
    ↓
yare_explain  one function: instructions, every register change, cost, features
    ↓
yare_admit    the whole gate, including parity across two backends
```

`yare_admit` is the expensive one, because it runs every function on the reference machine *and* as
lowered code over seeded inputs, and refuses the proposal if a single bit disagrees. Reaching it with
code that already passes `yare_check` and `yare_test` is the intended path, not a shortcut.

### A worked example

Ask for the guide:

```json
{ "name": "yare_guide", "arguments": {} }
```

It ends with a complete module that is admitted — and a test holds that it is admitted, because a
primer whose own example is refused would teach models to fail:

```yare
module mood.examples
lens technical = (thought, energy, resonance, action)
lens wisdom    = (thought, emotion, reason,    action)
frequency cap = capacity(0.5)

// Where this function sits in thought-space (optional): four numbers in [0, 1].
@tera(technical: 0.62, 0.40, 0.81, 0.30)
fn settle(a: tera<technical>, b: tera<technical>) -> tera<technical> weight 8 under cap {
  return geodesic(a, b, 0.25)
}

fn calm(a: tera<technical>) -> f32 weight 2 under cap {
  return coherence(a) * 0.5 + 0.5
}
```

Check it:

```json
{ "name": "yare_check", "arguments": { "source": "…the module above…" } }
```

```json
{
  "ok": true,
  "module": {
    "name": "mood.examples",
    "functions": [
      { "name": "settle", "signature": "…", "tera": { "lens": "technical", "v": [0.62, 0.4, 0.81, 0.3] } },
      { "name": "calm",   "signature": "…" }
    ]
  }
}
```

Read one function's execution:

```json
{ "name": "yare_explain", "arguments": { "source": "…", "fn": "calm", "args": [[0.5, 0.5, 0.5, 0.5]] } }
```

The result carries a `listing`, which is the same text the terminal prints:

```
calm([0.5, 0.5, 0.5, 0.5]) → 0.75
    0  …  r1 ← …
  cost 3 from the cost table · 2 steps executed · features: core, scalar · no weights read · replayable
```

Two numbers appear there on purpose: the cost table charges one for the trailing `HALT`, which is a
real instruction the machine executes and not one a person needs to read, so a program of *n* steps
costs *n* + 1.

Then the gate:

```json
{ "name": "yare_admit", "arguments": { "source": "…" } }
```

```json
{
  "admitted": true,
  "report": { "functions": 2, "instructions": 12, "paritySamples": 64, "examples": 0 },
  "module": { "…": "…" }
}
```

And a refusal, which is the more useful case:

```json
{
  "admitted": false,
  "reasons": ["…fn settle: every function must declare its cost: `weight N under <frequency>`…"]
}
```

### The mistakes a model actually makes

The tool descriptions name them, because they are the ones that cost a round trip:

- **A `tera` argument is one argument, not four.** `calm([0.5, 0.5, 0.5, 0.5])`, not
  `calm(0.5, 0.5, 0.5, 0.5)`. The arity check catches it, and the error message names the three
  accepted shapes.
- **Every function must declare its cost.** `weight 8 under cap` is a promise the checker verifies:
  `weight / frequency ≤ 1`. An action that cannot state its cost does not run.
- **No `page` or `parent_page`.** Those address memory, and a proposal reasons about meaning rather
  than addresses. The type system carries the rule, so it is a compile error rather than a runtime
  one.
- **Lenses are types.** `tera<technical>` and `tera<wisdom>` cannot be mixed without an explicit
  conversion.

### What the model is never told

`yare_check` and `yare_admit` return a module **summary** — signatures, the declared position in
thought-space, the frozen tensors a function reads, and the author's own description — and never the
source. The shape is defined once, in `packages/yare/src/summary.ts`, and used by both tiers.

That is not only about secrecy. A caller that asked whether a module compiles does not need the module
back, and returning it would spend context on bytes the caller already has.


## 5. `maiiam mcp` — the OS tier

```bash
maiiam mcp                          # every tool the OS has registered
maiiam mcp --tools                  # the surface, without starting a server
maiiam mcp --server yare,mind,model # narrow it
maiiam mcp --safe                   # only tools the OS marks read-only
maiiam mcp --actor mcp:cursor       # who a permission grant is keyed by
maiiam mcp --workspace <id>         # what a call is attributed to
```

The OS must be running. If it is not, this command writes to stderr and exits `3` — transient, try
again — rather than starting a server that would answer `tools/list` with nothing. That is the CLI's
rule for every command that needs a kernel (`apps/cli/src/connect.ts`), and an empty surface is a
worse answer than an honest one.

### This is a proxy, and that is the point

Every tool the OS can offer is already in `McpHost`, with a policy gate in front of it. `maiiam mcp`
re-serves that same table over stdio. It adds no policy and takes none away, so a call from Cursor
goes through exactly the gate an in-OS agent's call goes through — **one policy engine, not two.**

What that means in practice:

| The OS decides | What the model is told |
|---|---|
| `allow` | The tool's result. |
| `ask` | `MaiiaM OS is asking whether mcp:cursor may use yare_propose. Approve it in the app, then call again.` |
| `deny` | `MaiiaM OS refused fs_write: <the OS's reason>.` |

The `ask` case is the one worth getting right. The OS has already raised a notification the person
answers in the app; a model that is told "error" instead of "approve this and call again" will
conclude the tool is broken and stop trying. So it is an `isError` result carrying a sentence that
says what to do, and the actor is named in it, because that is what the prompt is about.

### Why the actor matters

A permission grant is keyed by `(actor, tool)`. Handing an external editor the OS's own agent's grants
would make the prompt a lie, so the default actor is `mcp:<client name>`, taken from the client's
`initialize` request. Cursor's first call to anything sensitive asks about `mcp:Cursor`, and the
answer is remembered for that name only.

`--actor` overrides it, and must look like `<kind>:<name>` — the CLI refuses anything else, because an
actor that is not an address is a grant nobody can find again.

### Why the default is everything

`--safe` exists, and `--server` exists, but neither is the default. The reason is that the OS is
already the authority on what may be called: it decides, and it records the decision. Defaulting to a
subset would put a second policy in the CLI, and two policies is one more than can be kept honest.

If you want a narrower surface, narrow it deliberately:

```bash
maiiam mcp --server yare,model      # the compiler and the model manager
maiiam mcp --safe                   # nothing that can ask or be denied
```

### The tools that appear in both tiers

`yare_guide`, `yare_check`, `yare_test`, `yare_explain` and `yare_admit` are served by both servers,
with the same names and the same arguments. In the OS tier they are backed by
`apps/kernel/src/yare/yare.ts`, which calls the same `packages/yare` functions and the same
`summarizeModule`. The OS tier adds the four that need state:

| Tool | What it needs a running OS for |
|---|---|
| `yare_list` | The library's modules, functions, signatures and descriptions |
| `yare_search` | Embeddings — find a function by meaning before writing a new one |
| `yare_run` | The run allowance, charged per call, and the frozen weights a function may read |
| `yare_propose` | Admission, then storage, then indexing. The proposer recorded is the kernel's actor, never a value the model supplies |

`yare_propose` is the one that closes the loop: a model writes a module, checks it, tests it, explains
it, admits it, and proposes it — after which the next model can find it with `yare_search` by
describing what it does rather than by remembering its name.


## 6. How the transport is implemented, and why not the SDK

The transport is `packages/mcp-stdio`: no dependencies, and no `node:*` in its core, so the same code
serves a pipe, a socket, or an in-memory pair. Both CLIs use it, so there is one copy of the wire.

It is implemented from the specification rather than depending on `@modelcontextprotocol/sdk`, for one
reason: `@asamanthinks/aamt` is `npx`-able and its whole pitch is that it installs one dependency —
the Dawn binding for WebGPU. The SDK's stdio server is a thin adapter, but `npm install` brings its
whole transitive tree (the HTTP transports' `express`-family dependencies, `ajv`, a second copy of
`zod`) for a server that speaks four methods over a pipe.

**Interop is proved rather than asserted.** `packages/mcp-stdio/test/sdk-interop.test.ts` drives this
server with the official SDK's client, which parses every response with its own zod schemas —
`InitializeResultSchema`, `ListToolsResultSchema`, `CallToolResultSchema`. A field the server gets
wrong, or a schema it describes in a way the specification does not allow, fails there. And
`apps/aamt-cli/test/mcp.test.ts` spawns the real CLI and speaks to it over an actual pipe, which is the
only way to catch the failures that live in the seams.

### What is implemented

| Method | Notes |
|---|---|
| `initialize` | Version negotiation over `2026-07-28` … `2024-10-07`; echoes the client's revision when it is one of them |
| `notifications/initialized` | Accepted, no response |
| `ping` | Answered before `initialize`, as the specification allows |
| `tools/list` | Every registered tool, with its schema and annotations |
| `tools/call` | Result blocks, `isError` for tool failures, an `AbortSignal` per call |
| `notifications/cancelled` | Aborts the named in-flight call |

Not implemented, because a tool server does not need them: `resources/*`, `prompts/*`, `sampling/*`,
`roots/*`, `elicitation/*`, `logging/*`, and tools-list pagination.

### The two failure channels, kept apart

A JSON-RPC `error` means **the protocol failed** — an unparseable line, an unknown method, a missing
argument. A result carrying `isError: true` means **the tool failed** — the model asked to explain a
function that does not exist, and the answer is a sentence.

Collapsing the second into the first is a real mistake: a host that receives a JSON-RPC error usually
reports a broken server, while an `isError` result is content the model can read and act on. So a
thrown error from a tool becomes `isError`, and only framing and dispatch faults use `error`.

### The one rule that bites

> The server MUST NOT write anything to its stdout that is not a valid MCP message.

A stray `console.log` in a tool does not corrupt one message — it desynchronises the client's parser
for the rest of the session, and the failure looks like a hung editor rather than a bug. So every tool
returns a value instead of printing, the engine's own chatter is muted, and `AAMT_MCP_STRICT=1`
installs a guard that reroutes any other write to stderr and names it. The subprocess test runs with
that on and fails if anything printed.

### Tool annotations are promises

`readOnlyHint: true` says a tool writes nothing, which is what lets a host call it without asking the
person first. In `aamt mcp` the writers are named and there are exactly two of them, `aamt_pull` and
`aamt_chat`; a test asserts that list, because a permission prompt that lies is worse than no prompt.

In `maiiam mcp` the annotation comes from the OS's own `sensitive` flag, so it cannot drift from the
gate that actually decides.


## 7. Sealing: shipping code and weights without shipping plaintext

`LAYER_FLAG_KEYED` is implemented, not reserved. A layer marked KEYED has each of its records sealed
with AES-256-GCM under a key derived from a root key along a path, and both the in-memory writer and
the streaming writer honour it.

```ts
import { LAYER_FLAG_KEYED, deriveKey, openContainer, packContainer, unsealRecord } from '@maiiam/aamt-container';
import { nodeCrypto, nodeKdf, nodeSha384 } from '@maiiam/aamt-container/node';

const key = deriveKey(nodeKdf, root, ['code', 'my-module']);

const bytes = packContainer(
  { layers: [{ kind: 'code', flags: LAYER_FLAG_KEYED, records: [{ payload: bytecode, tera, tier }] }] },
  nodeSha384,
  { ports: nodeCrypto, keys: { code: key } },
);

const layer = openContainer(bytes).layer('code')!;
for (const r of layer.records()) {
  const plaintext = unsealRecord(layer, r, key, nodeCrypto); // null on a wrong key, or a moved record
}
```

The properties that matter, and why they are the ones designed for:

- **Deterministic.** A container packs to the same bytes every time, so a random nonce is out. The
  nonce is synthetic — `HKDF(key, salt = SHA-384(plaintext), info = "nonce")` — so identical plaintexts
  under one key give identical ciphertext, which reveals only equality, something content addressing
  already reveals.
- **Auditable without the key.** A keyed record's identity covers the **stored** bytes, so
  `verifyContainer` recomputes every id and invariant on a sealed file with no key at all. A recipient
  can be told "this file is intact and it is the file that was signed" without being told anything
  about the contents.
- **Bound to where it lives.** The ciphertext is bound by associated data to its layer kind, record
  kind and locator, so a ciphertext moved onto another record fails to open.
- **Refused at the writer, not at the reader.** A KEYED layer with no key supplied is a refusal, so the
  mistake is an error while you are looking at it rather than a file nobody can open.
- **Erase by destroying one key.** A mind sandbox exports its `mind` layer KEYED under its own key:
  destroy that key and the sandbox's thoughts are gone while the world and the promoted experience
  stay readable. The bytes remain; the meaning does not.

**Locators are not encrypted.** A world cell and a TERA search key stay readable, because routing has
to work without the key. Where a position itself is sensitive, store a coarse or zero locator. This is
stated in `packages/aamt-container/src/crypto.ts` and it is a design choice, not an oversight.

### What sealing is not

It is **not** a way to hide the CLI's own JavaScript. `aamt.mjs` is a minified esbuild bundle: that is
obfuscation, not protection, and a determined reader gets the TypeScript back. Sealing protects
**content** — a model layer, a code layer, a sandbox — not the interpreter that reads it.

If protecting the implementation becomes a hard requirement, the path is a WASM VM plus sealed
container layers, so the logic that reads a sealed layer is itself not readable JavaScript. That is not
built today.

For a distribution that must not ship plaintext, the sequence that works now is:

1. Compile the module to Yare bytecode and seal it into the `code` layer with a per-recipient key
   derived from a root key you hold.
2. Ship the container. The recipient can verify it is intact and that it is yours, without learning
   what is in it.
3. Give the key to the recipient you meant to give it to. Destroying that one key revokes exactly that
   recipient.

The weights themselves are a separate case: they are public artifacts on
`huggingface.co/Mai-iaM/BeyondSight-aamt`, verified by SHA-384 against an Ed25519-signed catalogue
(`src/catalog-data.ts`, `aamt verify`). That is provenance, not secrecy, and the licence in
`LICENSE.md` is the boundary.


## 8. Troubleshooting

**The editor says the server failed to start.**
Run the command by hand — `npx @asamanthinks/aamt mcp`, or `maiiam mcp` — and read stderr. For
`maiiam mcp`, exit `3` means no running OS was found; `connect.ts` looks in `<userData>/compat.json`,
so set `MAIIAM_USER_DATA` if the app's data directory is not the default.

**The client connects but shows no tools.**
Check the client is sending `notifications/initialized` after `initialize`; this server refuses
`tools/list` before `initialize` with `-32002` and answers nothing to a notification, both as the
specification requires. `aamt tools` proves the table is there.

**A tool call comes back with `isError` and a sentence.**
That is a tool failure, not a transport failure, and it is meant to be read. For `maiiam mcp`, a
message naming an actor is the OS asking for permission: approve it in the app and call again.

**`aamt_chat` says the model is not downloaded.**
Call `aamt_pull` with the model id first, or run `aamt pull <model>`. The error names the missing
files.

**A tool call never returns.**
`aamt_bench` takes about three minutes; `aamt_chat`'s first call loads gigabytes onto the GPU. Both
report progress on stderr. Send `notifications/cancelled` with the request id to stop at the next
boundary.

**Something wrote to stdout.**
Run with `AAMT_MCP_STRICT=1` and the guard will name it on stderr. This should never happen in a
released build; the test suite fails if it does.

**It is slow on the first call.**
The engine measures the GPU's GEMV kernel split once per model and caches the winner in
`~/.aamt/gemv-tuning.json`, keyed by adapter identity including the driver. A driver update invalidates
it, which is deliberate: the cached number was measured for the old driver.

## 9. What is not here

Read this before quoting anything above.

- **The MCP SDK is not a runtime dependency.** That is a decision with a cost: if the specification
  changes its framing, this server changes with it, and the interop test is what will say so. The
  protocol revisions it will negotiate are listed above, and it answers with its own latest when it
  does not speak the client's.
- **No `resources/*`, `prompts/*`, `sampling/*`, `roots/*`, `elicitation/*`, `logging/*`.** A tool
  server does not need them; a client that requires one will not work against this.
- **No tools-list pagination.** The list is returned whole. 166 tools is a lot of context, which is why
  `--server` exists.
- **No remote transport.** Both servers are stdio and local. There is no HTTP transport, no
  authentication story for one, and therefore no remote-auth guidance to give.
- **`aamt mcp` has no permission gate of its own.** It relies on the client honouring `readOnlyHint`. A
  client that ignores annotations can call `aamt_pull` and download gigabytes. The annotations are
  accurate; whether they are obeyed is the client's business.
- **`aamt_chat` is not a session per client.** It keeps one model and one conversation per process, so
  two clients talking to one server share that conversation.
- **Cancellation is at seams, not mid-token.** The engine's decode loop has no abort, so a cancelled
  `aamt_chat` stops before the next generation rather than inside it.
- **Measured on one machine.** The figures `aamt bench` reports come from whatever hardware you run it
  on. The reference number in the README (BeyondSight NightOwl, 8B 1-bit, ~40 tok/s on a
  `MacBookPro18,2`) is that machine's, not a claim about yours.
- **The CLI's own source is minified, not protected.** See section 7.
- **`apps/aamt-cli`'s `package.json` is `private: true`.** The published name is `@asamanthinks/aamt`,
  produced by `pnpm --filter @maiiam/aamt-cli build` into `dist/package/`, and its only runtime
  dependency is `webgpu`.

## 10. Where the code is

| Path | What |
|---|---|
| `packages/mcp-stdio/src/protocol.ts` | Wire types, protocol revisions, error codes |
| `packages/mcp-stdio/src/framing.ts` | `LineFramer` — newline framing, pure, tested on its own |
| `packages/mcp-stdio/src/server.ts` | `serve()` — dispatch, lifecycle, cancellation. No `node:*` |
| `packages/mcp-stdio/src/node.ts` | `nodeStreams()`, `guardStdout()` |
| `apps/aamt-cli/src/tools.ts` | The fifteen tools, and the model session |
| `apps/aamt-cli/src/mcp.ts` | `aamt mcp` — the instructions, the guard, the flush |
| `apps/cli/src/mcp.ts` | `maiiam mcp` — the proxy, the actor, the gate mapping |
| `packages/yare/src/guide.ts` | The primer, generated from the compiler's builtin table |
| `packages/yare/src/summary.ts` | What `yare_check` and `yare_admit` return, shared by both tiers |
| `packages/aamt-container/src/crypto.ts` | The key tree, the synthetic nonce, seal and open |
| `docs/mcp/TECHNICAL-VISION.md` | Where this goes next, and what is load-bearing about it |

Tests, in the order they will fail if something breaks:

| Suite | What it proves |
|---|---|
| `packages/mcp-stdio/test/framing.test.ts` | Framing, on its own |
| `packages/mcp-stdio/test/server.test.ts` | Lifecycle, the two failure channels, cancellation |
| `packages/mcp-stdio/test/sdk-interop.test.ts` | The official client parses every response |
| `apps/aamt-cli/test/tools.test.ts` | Every tool; the tier-parity contract; the Vortex arithmetic |
| `apps/aamt-cli/test/mcp.test.ts` | The real process, over a real pipe, with the stdout guard on |
| `apps/cli/test/mcp.test.ts` | The gate mapping: allow, ask, deny, and the actor |
| `apps/kernel/test/yare-tools.test.ts` | The OS tier's five `yare_*` tools and the guide's own example |
| `packages/aamt-container/test/crypto.test.ts` | Sealing: round trip, wrong key, moved ciphertext, tampering |

