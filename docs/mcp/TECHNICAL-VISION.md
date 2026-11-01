# The MaiiaM MCP surface — technical vision

**Audience:** engineers doing diligence on MaiiaM's tool surface, and the people who will decide
whether to license it.

**What this document is:** the forward-looking technical case. What is load-bearing, what is
demonstrably built, where it goes next, and — in [section 7](#7-what-would-falsify-this) — what would
prove it wrong. The companion [DEVELOPER-GUIDE.md](DEVELOPER-GUIDE.md) is the how-to; this is the why.

**A note on method.** Every claim below is either marked *built* (there is a test that fails if it stops
being true) or marked *planned* or *absent*. A document that blurs those is worth nothing in a
diligence conversation, so the boundaries are marked explicitly and the omissions are listed rather
than omitted.

---

## 1. The thesis in one paragraph

Local AI on a developer's machine is currently a wrapper around someone else's inference API, or a chat
window with no way to check the answer. MaiiaM's MCP surface is a different kind of thing: a **compiler
with a gate**, served over the same protocol as everything else. A model proposes; the compiler refuses
with a line and a column; the gate refuses with reasons; and every claim the proposer makes about its
own code is executed and compared bit-for-bit before anything runs. The differentiator is not the model
— it is that this platform can *tell a model it is wrong*, in language the model can act on, and can
prove the answer rather than assert it.

That is the whole of the commercial argument. Everything below is the evidence for it.

---

## 2. What is actually load-bearing

### 2.1 The compiler as oracle, not a second model

The usual way to check a model's output is to ask another model. That is a second opinion from the same
class of system, with the same failure modes, and no bound on how wrong it can be. Yare's admission gate
replaces it with a deterministic pipeline in which every step is a refusal point:

| Step | What it refuses | Why a model can act on it |
|---|---|---|
| Size | Source past a byte limit, before anything is parsed | A model can emit megabytes; this is a cost bound, not a style rule |
| Compile | Lenses, coordinate classes, totality, bounded nesting and inlining, the meaning-never-addresses rule | Diagnostics carry line and column |
| Limits | Functions, instructions per function, instructions in total | Named numbers, not a score |
| Yoke | A function that does not declare its cost, or whose `weight / frequency > 1` | An action that cannot state its cost does not run |
| Capabilities | Builtins the proposal was not granted — by default the fabric | "A proposal reasons about meaning; it does not address memory" |
| Substrate round trip | Packed as `YARB` records and lifted back through the verifier, exactly as it would be loaded from a container | The refusal happens before storage, not at load |
| **Parity** | Every function run on the reference Machine *and* as lowered code over seeded inputs; **any bit of disagreement refuses the whole proposal** | Two independent backends agreeing bit-for-bit is a proof, not a plausibility argument |
| Examples | The claims the source makes about itself (`example ease(3.0) == 6.0`), run and compared bit-for-bit | The author stated something checkable; a module whose claims fail is refused |

*Built.* `packages/yare/src/admit.ts`, `compile.ts`, `lower.ts`, `run.ts`; exposed as the `yare_*` MCP
tools in both tiers.

The property worth naming precisely: **an author who states what their function returns has said
something checkable, and the gate checks it.** That is a different relationship between a language and
its users than a type system or a linter. It is closer to a proof obligation attached to a proposal.

### 2.2 Semantics never shape addresses, and the compiler enforces it

The Vortex ISA has **no addressing modes**. `pageIndex` and `Fabric.allocate` accept `SpatialCoord` —
three integers — and no overload takes a register, so a proposal that tries to compute a buffer offset
from a TERA value, a Meji cell, a digital root or a chirality bit does not compile.

This is not a style guideline with a paragraph behind it. Seven proposals that violated it were
implemented and measured, and every one was broken; the most plausible-sounding mapped all of 3-D space
into 729 buckets and called the resulting data loss a memory reduction. Those receipts are in
`packages/vortex-vm/ARCHITECTURE.md`, and the rule lives in the type system rather than in a review
checklist.

*Built.* `packages/vortex-vm/src/fabric.ts`, `packages/vortex-vm/src/isa.ts`.

### 2.3 The register format is chosen so the hard operations are cheap *and* exact

A TERA register stores four lanes as **angles**: `θ_d = 2·arcsin√v_d`. The Fisher information metric is
flat in `θ`, so:

```
d_FR(u, v) = ‖θ(u) − θ(v)‖₂          a Euclidean norm — one subtract, one horizontal add
geodesic   = θ(u) + t·(θ(v) − θ(u))  a straight line — constant speed by construction
```

Two `v128` instructions each, with **no transcendental in the loop**. In `v` coordinates both need four
`arcsin` and four `sqrt`, and the "geodesic" stops being one. Everything else in the machine works in
`v` (readouts, `RI`, entropy, Meji) or in `u = log v` (the `Σ_κ` projection, where the conservation
constraint becomes a hyperplane) — three coordinate systems, each used where it is flat, never mixed.

*Built and measured.* `packages/vortex-vm/src/alu.ts`. Worst-case round-trip error through the register
format: **1.47e-7**, below `f32` resolution and identical on every engine. The conservation constant `κ`
is cached **on the register** so the machine's core invariant never depends on that lossy conversion.

### 2.4 The Meji encoding is lossless, which is why 16 slots are a state and not a label

`mejiDist(v)` is a rank-1 outer product of four Bernoulli means over 16 cells, and `recover(mejiDist(v))`
returns `v` **exactly**. So the 4-bit cell address is a lossless encoding of a continuous 4-D state, not
a summary of it: two actors in the same cell are different states, and the distribution keeps the
difference.

That distinction is the difference between an address that is a labelling scheme and one that has
quietly become a personality. It is also why the saving is `2⁴ − 1 = 15` multiplications rather than
`4 × 2⁴`: the family is rank-1, and a backend gets the saving for free by writing the intermediate into
lane registers.

*Built.* `packages/vortex-vm/src/alu.ts`. The outer-product order is pinned by a parity fixture, because
getting it wrong permutes all sixteen cells.

### 2.5 Determinism is a property of the code, not a promise about it

No clocks, no `Math.random`, no `Date`, no `Math.sin`/`cos`/`asin`. The transcendentals are a declared,
swappable port and the reference port is **tabled** — including `asin`, whose platform-function use the
source corpus's own comments call out as a gap. No allocation-order dependence. The consequence is that
a world hash is stable and a simulation replays: same seed, identical fingerprint.

`packages/arcade-engine/src/nature.ts` implements the same algebra **independently**, and
`packages/vortex-vm/test/parity.test.ts` proves the two agree to ≤1e-6 against 200 fixtures generated by
running the reference implementation. That reimplementation-plus-generated-fixtures pattern is
deliberate: it is the only way a second implementation can be known to be *the same function* rather
than a lookalike that drifts by a few ulps and corrupts a world hash in a way no test on its own output
would catch.

*Built.* `packages/vortex-vm/src/deterministic.ts`, `packages/vortex-vm/test/parity.test.ts`.

### 2.6 Where a weight may come from, and what is absent on purpose

A weight read-out is allowed only from **frozen, content-addressed tensors at static offsets** (class
`0x9`, ADR-015), plus one typed gather whose row is a host-supplied integer (`W_ROW4`, ADR-021).

Two operations that every modern inference stack has are **absent**, and the reasons are recorded rather
than left to be discovered:

- **MoE dispatch** — absent because the corpus says its router "picks a name, not different weights."
- **Simplex routing** — absent because its paper states "no result in §4, §9, or §12 has been
  measured."

Where the corpus has a fitted constant it arrives as a **named parameter** (`floor` on `vetoed`, not a
`0.15` masquerading as a law). Where it has an unmeasured proposal, the operation is absent and the
absence is recorded in `ABSENT_BY_DESIGN` with the reason.

*Built.* `packages/vortex-vm/src/weights.ts`, `packages/vortex-vm/src/alu.ts`.

This is the part of the codebase a diligence team should look at hardest, because it is the opposite of
the usual posture. The industry norm is to ship the plausible thing and measure later. This package
ships the *absence* with a citation.


---

## 3. The container is the distribution format, and that is the latent moat

The `.aamt` container is one immutable, mmap-able file with a layer table: `model`, `world`, `mind`,
`experience`, `code`. Its properties are the ones a distribution channel needs:

| Property | Why it matters commercially |
|---|---|
| **Deterministic packing** | The same content packs to the same bytes, whatever order it was supplied in. `fileId` is a statement about *content*, not about the run that produced it. |
| **384-bit layer-separated identities** | The same bytes can never be mistaken for one another across layers. A model record and a code record are not interchangeable, and that is a hash property rather than a convention. |
| **Zero-copy open, O(layers)** | Opening a 3 GB model does not read 3 GB. |
| **Keyed layers** | A layer can be sealed per record under a key tree. See §4. |
| **Byte-identical streaming** | The streaming writer (`writeContainer`, ADR-017) produces a file byte-identical to the in-memory writer, so a multi-gigabyte model never has to be resident. A test asserts the equality, not the intention. |
| **A code layer that holds Yare bytecode** | `RECORD_KIND_YARE_BYTECODE` already exists. |

The last row is the one that matters for licensing. **Yare bytecode is a distribution format.** The ISA
is 32-bit with no addressing modes — which means a bytecode module cannot compute its way out of a
sandbox, because there is nothing in the instruction set to compute an address *with*. A sealed `code`
layer under a per-recipient key is therefore a unit of licensed software that is:

1. **Verifiable by the recipient without the key** — identities cover the stored bytes, so
   `verifyContainer` proves integrity and provenance on a file nobody can read.
2. **Revocable by destroying one key** — the key tree is `root → layer → scope`, so revocation is
   arithmetic on a label, not a re-issue.
3. **Executable only inside the machine's semantics** — the fabric keeps semantics out of addresses, so
   a module cannot address memory it was not given.

That combination — a verifiable, revocable, sandboxed unit of executable content — is the shape a
commercial licence wants to take, and it is already the container's design rather than something that
would have to be retrofitted.

*Built.* `packages/aamt-container/src/{layout,identity,writer,stream,reader,plan,crypto}.ts`, with
`container.test.ts`, `stream.test.ts`, `crypto.test.ts`, `odu.test.ts`, `tree.test.ts`.

### 3.1 Why this is a moat and not a feature

A compiler is copyable. A container format is copyable. What is not copyable is a **corpus of admitted
modules plus the embeddings that make them findable**, because:

- Every module in the library passed parity across two backends. That is not a filter you can apply to
  scraped code after the fact.
- Every module carries a **declared cost** (`weight N under <frequency>`) that the checker proved, so
  the library can answer "what can I afford to run" without measuring anything.
- `yare_search` finds a function **by meaning** — by the description its author wrote and the position
  it declared in thought-space — which is the difference between a library and a directory.

So the compounding asset is not the language or the format. It is that the gate makes every entry in the
library *true*, and search makes the truth findable. A competitor can clone the compiler in a weekend
and cannot clone a corpus that has been gated.

---

## 4. Sealing, precisely

`LAYER_FLAG_KEYED` is implemented. The mechanism, stated so an engineer can check it:

- **AES-256-GCM**, 96-bit nonce, 128-bit tag. Standard primitives only — no custom cipher.
- **HKDF-SHA-384** key tree: `root ──▶ layer key ("MaiiaM Yare v1 · key · <layer>")`, and a further
  scope below that.
- **Synthetic nonces**, because a container packs to the same bytes every time and a random nonce would
  destroy determinism. `HKDF(key, salt = SHA-384(plaintext), info = "nonce")`, 96 bits. Distinct
  plaintexts get distinct nonces; identical plaintexts under one key give identical ciphertext, which
  reveals only equality — something content addressing already reveals.
- **Associated data** binds a ciphertext to its layer kind, record kind and locator, so a ciphertext
  moved onto another record fails to open.
- **Identities cover the stored bytes**, so a sealed file is fully auditable without the key and an id
  cannot be used to confirm a guessed plaintext.

*Built.* `packages/aamt-container/src/crypto.ts`, `writer.ts`, `stream.ts`, `reader.ts`, `plan.ts`;
`packages/aamt-container/test/crypto.test.ts` covers the round trip, a wrong key, moved ciphertext,
tampered ciphertext and determinism. `packages/mind/src/sandbox.ts` uses it in production: a sealed
sandbox exports its `mind` layer KEYED under its own key, so destroying that one key erases the
sandbox's thoughts while the world and the promoted experience stay readable.

**What sealing is not.** It is not a way to hide the CLI's own JavaScript, which is a minified esbuild
bundle — obfuscation, not protection. Sealing protects *content*. Protecting the implementation requires
a WASM VM so the logic that reads a sealed layer is itself not readable JavaScript. That is **not
built**, and it is the single largest gap between what exists and a fully closed distribution.


---

## 5. The MCP surface as a product

Two servers, one protocol, one shared transport:

| | `aamt mcp` | `maiiam mcp` |
|---|---|---|
| Kernel needed | no | yes |
| Tools | 15 | the OS's whole registered surface |
| Runtime dependencies | one (`webgpu`) | the CLI's own |
| The compiler and the gate | yes | yes, same code |
| The library, minds, the world, models | no | yes |
| Permission gate | the client's, from accurate annotations | the OS's policy host: allow, deny, or **ask in the app** |

**Why this is the right shape for adoption.** `npx @asamanthinks/aamt mcp` is one line in an editor
config and installs one dependency. It is the cheapest possible way to put a *compiler that refuses* in
front of a model. The OS tier is what a user graduates to, and the four tool names are identical in
both — so nothing is relearned.

**Why the gate matters more than the tools.** In the OS tier, a call from an external editor goes
through exactly the gate an in-OS agent's call goes through: one policy engine, per-actor grants keyed
by `(actor, tool)`, and an `ask` that raises a prompt the person answers in the app. The transport adds
no policy. That is what makes the surface auditable: there is one place a decision is made, and one
record of it.

Beyond the tool-level gate, world actions pass a second, arithmetic one: an agent holds authority 8 by
default; a weight-24 action is refused against a limit of 1.0 while a smaller ask from the same agent is
admitted; authority is earned by 31 unreverted good deeds, a revert costs as much as four good deeds,
and nothing lifts an actor past 64. A forbidden action is **absent from the candidates**, so a rogue
plugin cannot pick it.

*Built.* `apps/kernel/src/mcp/mcp.ts` (the host and the gate), `apps/cli/src/mcp.ts` (the outward
serving), `apps/aamt-cli/src/mcp.ts`, `packages/mcp-stdio/` (the transport, with interop proved against
the official SDK's client).

---

## 6. What is genuinely differentiated, ranked

If a diligence conversation has time for three things, these are the three.

**1. A language whose gate runs the author's own claims.** Not a linter, not a type system, not a second
model. `example ease(3.0) == 6.0` is a statement the compiler executes and compares bit-for-bit, and a
module whose claims fail is refused. Parity across two independent backends is the same idea applied to
the whole program. *Built, with the pipeline exposed as MCP tools in two tiers.*

**2. A register format derived from a metric, so the hard operations are exact.** `θ = 2·arcsin√v`
because Fisher–Rao is flat there, which makes distance a norm and a geodesic a straight line with no
transcendental in the loop, and makes 16 Meji cells a lossless encoding rather than a lossy label. The
consequence is that a world hash is stable and a simulation replays. *Built, measured, and proved
against an independent reimplementation.*

**3. A container that is simultaneously a distribution format and a licence mechanism.** Verifiable
without the key, revocable by destroying one key, and executable only inside semantics that cannot
compute an address. *Built; the sealed-distribution workflow around it is the next thing to productise.*

Everything else — the world, the minds, the models — is real and is what makes the above *useful*, but
it is not what is hard to reproduce.

### What this is not a claim about

- **Not model performance.** The measured serving number for the reference machine is **40.8 tok/s**
  (BeyondSight NightOwl, 8B 1-bit, `MacBookPro18,2`). Larger figures that circulate for this platform
  are an ANE *projection* and a *training* throughput, not a serving measurement, and they are not
  comparable to a decode tok/s figure. Conflating them would be the easiest way to make everything else
  in this document suspect, so the distinction is stated here.
- **Not that the transformer runs on the Yare path.** The WebGPU/Yare WGSL lowering **does not cover the
  transformer**: it refuses class `0x9` (weights read-out). Transformer decode runs on the GPU
  `QwenForward` path, not through the Yare WGSL lowering. The two are separate, and the Yare path is the
  one that is proved and sandboxed.
- **Not a safety guarantee.** A prompt-injected agent can still *propose* something harmful. The gate
  decides what runs, not what is proposed, so the damage is bounded rather than eliminated. "Immune" is
  not a claim this document makes.
- **Not a legal opinion.** Nothing here is advice about patentability, infringement, or the scope of any
  claim. It is a technical description of built, tested mechanisms, offered so that a qualified person
  can form their own view.


---

## 7. What would falsify this

A technical vision that cannot be wrong is marketing. These are the specific things that would show the
thesis above is not holding, and how a reader would check.

| If this were true | Then | How to check |
|---|---|---|
| The gate refuses code that is actually correct, often | The compiler is a tax, not an oracle, and models will route around it | Ask for a module, then count the round trips to admission. The `yare_admit` refusal reasons are the evidence. |
| Parity fails intermittently | The two backends are not the same function, and the "proved, not plausible" posture is unfounded | `packages/yare/test/parity.test.ts`, `packages/vortex-vm/test/parity.test.ts` |
| The Meji recovery is lossy in practice | The 16-cell address is a label, and the claim in §2.4 is wrong | `vortex_meji` returns `lossless`; the assertion is in `apps/aamt-cli/test/tools.test.ts` |
| A sealed container can be opened without the key, or a moved ciphertext opens | §4 is wrong | `packages/aamt-container/test/crypto.test.ts` |
| The container's streaming writer differs from the in-memory writer | Determinism is only true for small files | `packages/aamt-container/test/stream.test.ts` |
| The official MCP client cannot parse the server's responses | The transport is not the protocol | `packages/mcp-stdio/test/sdk-interop.test.ts` |
| A tool writes to stdout | Sessions desynchronise in the field, and the subprocess test is the only thing that would have caught it | `apps/aamt-cli/test/mcp.test.ts`, with `AAMT_MCP_STRICT=1` |
| The two tiers' `yare_*` tools drift apart | The "same names, same code" claim is false and users relearn | `apps/aamt-cli/test/tools.test.ts` asserts the names and required arguments literally |

The last row is the one to watch. Two implementations of the same interface is exactly how a "one
platform" story becomes two products that disagree, and the mitigation is that there is **one**
`packages/yare` and the tests bind both tiers to it.

---

## 8. Where this goes next

Ordered by what unblocks the most, with the gate that would prove each phase done.

**Phase 1 — sealed distribution as a product.** The container supports it and nothing ships it. What is
missing is a verb: produce a `.aamt` whose `code` layer holds sealed Yare bytecode under a per-recipient
key, and open one given the key. *Done when* a module can be compiled, sealed, shipped, verified by a
recipient without the key, and run by that recipient with it — with a test for each step.

**Phase 2 — the two-way parity matrix.** The matrix currently names 30 of the OS's registered tools; the
reverse direction (a registered tool with no row) is unchecked, and that asymmetry is how a tool-envelope
bug survived in the world surface until it was found by hand. *Done when* a test fails on a tool that is
registered and unlisted, in both directions.

**Phase 3 — publishable artifacts.** `@maiiam/yare` and `@maiiam/vortex-vm` are `private: true` with
`main: ./src/index.ts`, so they are consumed as TypeScript inside the monorepo and cannot be installed.
*Done when* a real build emits JavaScript and `.d.ts` files under semver, and a consumer outside the repo
can import them.

**Phase 4 — the implementation, not just the content.** A WASM VM so the logic that reads a sealed layer
is not readable JavaScript. This is the largest piece of work here and the only one that changes what
"without exposing source" can mean.

**Phase 5 — a remote transport, and therefore an auth story.** Both servers are stdio and local today,
which is a deliberate choice with a cost: there is no way to serve this to a machine that is not yours,
and no authentication guidance to give because there is nothing to authenticate. *Done when* there is a
transport with a stated threat model, not before.

**Deliberately not on this list:** more tools. The surface is 166 tools across 11 servers and the binding
constraint is context, not capability. `--server` exists for that reason, and adding tools without adding
the ability to *narrow* them makes the product worse.


---

## 9. The licensing surface

Stated as what exists, so a commercial conversation can start from facts rather than intent.

**What a licence would cover today**

| Asset | State | Mechanism |
|---|---|---|
| The curated models | Published, Ed25519-signed catalogue, SHA-384 per file | `LICENSE.md`: free personal and non-commercial use, a 60-day commercial evaluation, and small teams (under US $1M revenue, US $5M raised, 25 people) with free registration. `aamt verify` proves a file is a signed original. |
| The Yare language and its gate | In-repo, `private: true` | Not published as a package; the compiler is what the MCP surface sells. |
| The container format | In-repo, `private: true` | Specified in `docs/decisions/ADR-005` and `ADR-010`. |
| Sealed distribution | Mechanism built, workflow not productised | Phase 1 above. |
| The MCP surface | `@asamanthinks/aamt` is published; `maiiam mcp` ships with the OS | One runtime dependency, so the install story is honest. |

**What is not covered, and should not be quoted as covered:** patent protection of any kind (no filing
is described here), protection of the CLI's own source (minified, not protected), any remote or
multi-tenant deployment (no transport exists), and any performance claim beyond the one measured figure
in §6.

**The honest framing for a licence conversation.** What is licensable today is a *verified* corpus plus a
gate that keeps it verified, delivered in a container that can be sealed per recipient. What is not yet
licensable is the mechanism that would stop a determined licensee from reading the interpreter — which
is Phase 4, and which is the difference between a licence enforced by contract and one enforced by
construction.

---

## 10. How to check any of this yourself

```bash
# the compiler and the gate, with no kernel and no OS
npx @asamanthinks/aamt mcp        # then call yare_guide, yare_check, yare_admit

# the whole admission pipeline, from the outside
pnpm --filter @maiiam/yare test

# the two backends agreeing bit-for-bit
pnpm --filter @maiiam/vortex-vm test

# a sealed container: round trip, wrong key, moved ciphertext, tampering
pnpm --filter @maiiam/aamt-container test

# the transport, proved against the official MCP client
pnpm --filter @maiiam/mcp-stdio test

# the same four yare_* tools, over a real pipe, with the stdout guard on
pnpm --filter @maiiam/aamt-cli test
```

If any of those fail, the corresponding claim in this document is false and should be treated as such
until it passes.

---

## 11. Related reading

| Document | What it gives you |
|---|---|
| [DEVELOPER-GUIDE.md](DEVELOPER-GUIDE.md) | How to wire either server up, and every tool in both tiers |
| [`docs/MAIIAM-MCP-CLI-ARCHITECTURE-BRIEF.md`](../MAIIAM-MCP-CLI-ARCHITECTURE-BRIEF.md) | The OS's MCP host and its policy model, with dated receipts |
| [`docs/yare/DESIGN.md`](../yare/DESIGN.md) | The language: lenses, coordinate classes, the yoke, admission |
| [`docs/decisions/`](../decisions) | ADR-005 (the container), ADR-010 (admission and keyed layers), ADR-017 (the streaming writer), ADR-029 (agents and minds) |
| [`docs/aamt/MODEL-WRAPPER-GUIDE.md`](../aamt/MODEL-WRAPPER-GUIDE.md) | Packing a manifest and weights into a `.aamt`, including sealed deployment |
| [`packages/vortex-vm/ARCHITECTURE.md`](../../packages/vortex-vm/ARCHITECTURE.md) | The seven receipts behind "semantics never shape addresses" |
| [`docs/design/PARITY-MATRIX.md`](../design/PARITY-MATRIX.md) | Every capability against its RPC, MCP tool, CLI command and screen |

