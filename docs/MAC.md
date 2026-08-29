# Run on Apple Silicon (Mac, Metal)

Dataroom runs natively on an Apple Silicon Mac with **no Docker and no NVIDIA GPU**. An
OpenAI-compatible server holds `:8080` and the FastAPI app + Pi agent + embedder run in a local
`uv` virtualenv. No application code changes are needed; the model is decoupled behind
`LLAMA_URL`, so the only Mac-specific concerns are which backend serves `:8080` and the model.

Tested on an **M3 Pro / 36 GB**, macOS. At least **32 GB** of unified memory is recommended -
the default model (Qwen3.8-27B, 4-bit) wires ~18 GB plus its context cache.

## The default stack: Qwen3.8-27B + mlx-dspark

`scripts/mac-run.sh` defaults to `BACKEND=dspark`: **[mlx-dspark](https://github.com/ARahim3/mlx-dspark)**
serving **`mlx-community/Qwen3.8-27B-4bit`**. Qwen3.8-27B is a dense hybrid (linear attention on
48 of 64 layers) with a **262K native context** and a built-in MTP head; mlx-dspark adds DSpark /
DFlash speculative decoding (lossless - the target verifies every token), an 8-bit KV cache, and
**prefix caching**.

Why this replaced the previous Qwen3.6-35B-A3B MoE default: the MoE was much faster in raw
tok/s (prefill ~960, decode ~91-145 on an M4 Pro per mlx-dspark's tables vs ~130-184 prefill /
~25-38 decode for the dense 27B), but its fp16/KV-limited context ceiling (~85K on 36 GB) was
what the dataroom's compaction loop was actually bound by. Qwen3.8's linear-attention cache is
~0.086 GB per 1k tokens at bf16 KV, halved by `--kv-bits 8` - a 128K context fits ~25 GB total
on a 36 GB Mac, and 262K is reachable on 64 GB+. Prefix caching also lands directly on the
agent loop: a warm session skips re-prefilling its shared prefix (measured first token ~62 s
cold -> ~1 s on a cached turn, on an ~8k prefix).

mlx-dspark's `--mode auto` (the default) resolves the measured-best drafter for the target -
for Qwen3.8-27B that is the **DFlash 2** head (`incoai/Qwen3.8-27B-DFlash2`), downloaded once
alongside the model. Nothing to configure.

## What's different from the NVIDIA/Docker path

| Area | NVIDIA / Docker | Apple Silicon | Why |
| --- | --- | --- | --- |
| Server | `llama.cpp:server-cuda` container | `mlx-dspark serve` (default) or Homebrew `llama-server` (Metal) | No CUDA / `nvidia-container-toolkit` on macOS. |
| Model | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | `mlx-community/Qwen3.8-27B-4bit` (auto-downloaded), or the Qwen3.8 GGUF for `BACKEND=llamacpp` | See the section above. |
| KV cache | fp16/fp8 in-container | `--kv-bits 8` (dspark) / q4_0 KV (llamacpp) | Unified memory is shared with macOS; quantized KV is what makes 128K+ fit. |
| App + embedder | Docker container | `python -m server.app` in a `uv` venv; embedder on CPU | No GPU passthrough into Docker on macOS; CPU keeps Metal free for the LLM. |
| torch | from the CUDA base image | `uv pip install torch` (MPS/CPU build) | Not pinned in `server/requirements.txt`. |

## Prerequisites

```bash
# Node 22 (Pi agent) + uv (Python env), e.g. via mise:
#   mise use -g node@22
#   curl -LsSf https://astral.sh/uv/install.sh | sh
node --version    # v22.x
uv --version
```

`brew install llama.cpp` is only needed for `BACKEND=llamacpp`. You need a free **Jina API key**
(https://jina.ai/api-dashboard/) and, recommended, a free **Hugging Face read token**
(https://huggingface.co/settings/tokens) for a fast, stable model download.

## 1. Install the agent + Python deps

```bash
npm install -g @earendil-works/pi-coding-agent@0.78.0

# The Pi dataroom-index extension has a runtime dep (typebox); install it so the
# agent can load the extension (otherwise jobs exit with "Cannot find module 'typebox'").
(cd pi/extensions && npm install)

# torch is NOT in server/requirements.txt (upstream got it from a CUDA base image),
# so install it explicitly - it pulls the Apple-Silicon (MPS/CPU) build.
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  torch -r server/requirements.txt jina-cli huggingface-hub

# The default backend lives in its OWN venv - installing it into .venv bumps transformers
# and breaks the embedder.
uv venv .venv-dspark
VIRTUAL_ENV=$PWD/.venv-dspark uv pip install mlx-dspark
```

## 2. The model

With the default `BACKEND=dspark` there is nothing to do: `mlx-dspark` downloads
`mlx-community/Qwen3.8-27B-4bit` (~18 GB) into the Hugging Face cache on first run, and resolves
the DFlash 2 drafter automatically.

For `BACKEND=llamacpp`, download the GGUF (~17.6 GB):

```bash
HF_TOKEN=hf_your_token \
.venv/bin/python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('unsloth/Qwen3.8-27B-GGUF','Qwen3.8-27B-UD-Q4_K_XL.gguf',local_dir='models')"
```

The `HF_TOKEN=` prefix is optional but avoids the unauthenticated rate limit (which can stall the
download). For faster downloads, install `hf_transfer` first:
`uv pip install --python .venv/bin/python hf_transfer`, then set
`HF_HUB_ENABLE_HF_TRANSFER=1` alongside the token.

## 3. Set your key

```bash
cp .env.example .env
sed -i '' 's/^JINA_API_KEY=.*/JINA_API_KEY=jina_your_real_key/' .env
```

Recommended Mac `.env` values (defaults baked into `scripts/mac-run.sh`, override as needed):

```bash
BACKEND=dspark            # mlx-dspark + Qwen3.8-27B-4bit (default); llamacpp is the fallback
CTX_SIZE=65536            # comfortable inside 36 GB; 98304-131072 fits with 8-bit KV
CONTEXT_WINDOW=65536
LLAMA_URL=http://127.0.0.1:8080
JOBS_DIR=./data/jobs
EMBED_DEVICE=cpu          # leave Metal's memory for the LLM
```

## 4. Run

```bash
bash scripts/mac-run.sh
```

Starts the backend on `:8080` (first run downloads the model, then loads ~18 GB), waits for
`/health`, then starts the app on `:8000`. Open **http://localhost:8000/**, or watch a job at
`http://localhost:8000/jobs/{id}/dashboard`.

Stop the app with `Ctrl+C`; stop the model with `pkill -f mlx-dspark` (dspark) or
`pkill -f llama-server` (llamacpp).

## Memory notes

On a 36 GB machine the default model wires ~18 GB weights + ~6 GB cache at a 128K context with
8-bit KV (~25 GB total); the system has real headroom with the compressor active. If a long job
pages heavily, lower `CTX_SIZE` (e.g. `32768`) in `.env`. The v5-nano embedder stays on CPU
(`EMBED_DEVICE=cpu`) precisely so Metal's memory is reserved for the LLM.

### Memory tiers (unified memory)

| Unified memory | Qwen3.8-27B (4-bit, ~18 GB) | Guidance |
| --- | --- | --- |
| 16 GB | ✗ won't fit | Not supported; a 7-14B model is the ceiling on this tier. |
| 24 GB | ⚠ tight | Fits at short context - keep `CTX_SIZE` at `32768` or less, close Chrome/Docker/IDEs. |
| 32-48 GB | ✓ comfortable (**tested tier: 36 GB**) | Recommended. ~18 GB weights + KV + headroom; 131072 ctx fits with 8-bit KV. |
| 64 GB+ | ✓ 262K native context reachable | The full window plus the 8-bit cache fits around ~40 GB. |

## Backends

`scripts/mac-run.sh` honours a `BACKEND` knob:

| | `dspark` *(default)* | `llamacpp` |
| --- | --- | --- |
| Engine | [mlx-dspark](https://github.com/ARahim3/mlx-dspark) (MLX, Apple-native) | llama.cpp (Homebrew, Metal, GGUF) |
| Model | `mlx-community/Qwen3.8-27B-4bit` | `models/Qwen3.8-27B-UD-Q4_K_XL.gguf` |
| Speculative decoding | DSpark/DFlash drafters, auto-resolved (`--mode auto`), lossless | built-in MTP head, `--spec-type draft-mtp --spec-draft-n-max 2` (+33-145% decode in community sweeps) |
| KV cache | `--kv-bits 8` (set `DSPARK_KV_BITS=` for fp16) | q4_0 K/V (required for ctx > ~90K next to 17 GB weights) |
| Prefix caching | yes (on by default) | no |
| Dashboard tok/s + KV gauges | blank (they read llama.cpp's `/metrics` + `/slots`) | live |

Use `dspark` for the agent loop (prefix caching + longer cheap context); use `llamacpp` if you
want the GGUF ecosystem or hit an mlx-dspark issue - the flag stack matches the community-verified
Qwen3.8 MTP recipe exactly. Note llama.cpp needs a **current build** for this arch: older CUDA
builds silently produced garbage output on Qwen3.8's DeltaNet layers
([llama.cpp#27164](https://github.com/ggml-org/llama.cpp/discussions/27164)); on Metal, use a
recent Homebrew bottle.

### Backend env knobs

| Var | Default | Meaning |
| --- | --- | --- |
| `BACKEND` | `dspark` | `dspark` or `llamacpp`. (`mlx` still routes to `dspark`; the old `mlx_lm.server` path is retired.) |
| `DSPARK_MODEL` | `mlx-community/Qwen3.8-27B-4bit` | HF repo id or local MLX model dir; any quant of Qwen3.8-27B resolves the same drafter. |
| `DSPARK_KV_BITS` | `8` | KV cache bits for dspark; set empty for fp16 (the ctx cap then drops to 98304). |
| `DSPARK_CTX_CAP` | `131072` (fp16 KV: `98304`) | Max context for dspark; `CTX_SIZE` is auto-capped to this. |
| `MODEL_FILE` | `Qwen3.8-27B-UD-Q4_K_XL.gguf` | GGUF path under `models/` (llamacpp only). |
| `SPEC_ARGS` | `--spec-type draft-mtp --spec-draft-n-max 2` | llama.cpp MTP flags (llamacpp only). |
| `CHAT_TEMPLATE_FILE` | *(empty = GGUF's embedded template)* | Override for non-Qwen GGUFs (llamacpp only); a wrong template silently corrupts tool-calling. |
| `MODEL_ID` | auto (`Qwen3.8-27B-4bit` / `qwen3.8`) | Pi's model label. dspark resolves it against the loaded model (basename match); llama.cpp ignores it. |

### Caveats

- **Dashboard tok/s + KV gauge are llama.cpp-only.** `server/stats.py` reads llama.cpp's
  `/metrics` and `/slots`; those live widgets stay blank under `dspark` (the job still runs and
  packages normally). mlx-dspark exposes its own `/metrics` + per-response telemetry - wiring the
  dashboard to it is future work.
- **First request on a cold dspark server is the slow one** (model load + prefill); after that
  the prefix cache absorbs most of the compaction loop's prefill.
- **`MODEL_ID` must match under dspark.** `mlx-dspark` resolves the request's `model` field
  against the loaded model; the script pins Pi's label to the basename so they agree.

## Historical: the Qwen3.6-35B-A3B MLX research

The previous default model (Qwen3.6-35B-A3B MoE) and its `mlx_lm.server` path are retired in
favour of the stack above. The measurements that drove the original MLX adoption - ~5.9x prefill
vs llama.cpp, greedy-lossless 4-bit KV, correct recall through ~85K - are journalled in
[`MLX-RESEARCH-FINDINGS.md`](./MLX-RESEARCH-FINDINGS.md) with a
[glossary](./GLOSSARY.md), and reproducible with `scripts/probe_speed.sh`,
`scripts/mlx_ceiling.sh`, `scripts/mlx_parity.sh`, and `scripts/mlx_recall.py` (point `MLX_MODEL`
at a Qwen3.6 MLX build). Also evaluated and ruled out: **Ollama 0.19+ MLX / LM Studio MLX** (no
KV-quant knob, no custom-build import) and **MTPLX for MTP on Qwen3.6** (no verified 35B-A3B MTP
artifact; moot now that Qwen3.8 ships its own MTP head).

## Headless (no web UI)

```bash
JINA_API_KEY=... LLAMA_URL=http://127.0.0.1:8080 MODEL_ID=Qwen3.8-27B-4bit \
  .venv/bin/python -m server.run_dataroom --query "your query" --out ./out
```

## Switching the model

- **dspark:** set `DSPARK_MODEL` to any HF MLX repo id or local model dir and restart. Any
  quant of a registry target auto-resolves its drafter; non-registry targets still get
  drafter-free lookup speculation via `--mode auto`, or pass a drafter via `DSPARK_EXTRA_ARGS`.
- **llamacpp:** set `MODEL_FILE` to a different GGUF in `models/` and restart. The unsloth
  Qwen3.8 GGUFs carry the MTP head and an embedded chat template, so the defaults hold; for a
  non-Qwen GGUF set `CHAT_TEMPLATE_FILE` to that model's own Jinja template (a wrong template
  silently corrupts tool-calling) and check whether it ships an MTP head before keeping
  `SPEC_ARGS`.
