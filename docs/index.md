# .aamt — documentation

`.aamt` is the GPU-native file format for local AI: model weights, agent memory and worlds in one
memory-mappable file. The specification is [`SPEC.md`](https://github.com/wwhitehead/aamt/blob/main/SPEC.md);
these are the guides for reading, baking and serving one.

## The aamt CLI

| Guide | What it answers |
|---|---|
| [Serve a model into the tools you already use](aamt/SERVE-YOUR-OWN-MODEL.md) | `aamt serve` answers the OpenAI HTTP API, so these models drop into Ollama's UI, LM Studio, Open WebUI, Cursor and LangChain. Also what `aamt inspect` refuses, and how `aamt bake` reproduces the curated recipe. |
| [Putting any model in an `.aamt` wrapper](aamt/MODEL-WRAPPER-GUIDE.md) | Running a model you did not bake — GGUF, safetensors, anything a server engine loads — under the same identity, provenance and policy rules, **without converting the weights**. |
| [Which shipped models can be rebuilt, and which cannot](aamt/MODEL-REPRODUCIBILITY.md) | What each curated model was baked from, whether that source still exists, and what a byte-for-byte comparison of a rebake actually shows. |

## The Model Context Protocol server

| Guide | What it answers |
|---|---|
| [Developer guide](mcp/DEVELOPER-GUIDE.md) | The fifteen `aamt mcp` tools: local models, the Vortex arithmetic, and the Yare compiler with its admission gate. |
| [Technical vision](mcp/TECHNICAL-VISION.md) | Where the MCP surface goes next, and why. |

## Getting the CLI

```bash
npx @asamanthinks/aamt                 # chat with InnerSight (0.72 GB download)
npx @asamanthinks/aamt serve gaian     # OpenAI-compatible API on 127.0.0.1:11434
```

The package is [`@asamanthinks/aamt`](https://www.npmjs.com/package/@asamanthinks/aamt).

---

These pages are published from the `aamt` CLI's source repository, so this copy is generated — edits belong
there rather than here.
