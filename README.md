# RAMGPT Toolchain

**Build. Evaluate. Optimize. Share the results.**

A unified CLI for local LLM engineering. Connect existing model tools, run native evaluations, find configurations that meet your resource limits, and export portable model summaries.

## Install

Linux and Python 3.12 or newer. Download the wheel from [GitHub Releases](https://github.com/bettercallcaleb/ramgpt-toolchain/releases), or install from the source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[all]'
ramgpt doctor --gpu
```

For offline model registration and passport inspection, the core package has no third-party Python dependencies. GPU measurement requires the GPU extra; fidelity analysis requires NumPy. Native tools and models are installed separately.

## Start with a model

```bash
ramgpt register /path/to/model.gguf
ramgpt model-info
ramgpt passport /path/to/model.gguf --out model.passport.json
ramgpt passport-verify model.passport.json --model /path/to/model.gguf
```

A newly registered model has an identity, not evaluation results.

## Connect your tools

```bash
ramgpt init --root /path/to/local/tools --out tools.json
export RAMGPT_TOOLS_CONFIG="$PWD/tools.json"
ramgpt tools
```

Review the paths before use. Resolve ambiguous matches with `--set tool.field=PATH` and a new output file, or edit a copy of [tools.example.json](tools.example.json). Discovery does not download, install or execute the tools. Set `--tools-config` to override the environment.

## Build and run

GGUF with Q4Fit and llama.cpp:

```bash
ramgpt build original-f16.gguf --out output/model.gguf --execute
ramgpt runtime-spec output/model.gguf --gpu-uuid GPU_UUID --out runtime --execute
ramgpt run output/model.gguf --spec runtime/spec.json --seed 931 --budgets 256 --out measurement --max-requests 9 --max-completion-tokens 2112
ramgpt run MEASUREMENT_PLAN_ID --execute
```

Replace GPU_UUID with the selected GPU UUID from `nvidia-smi -L`, and MEASUREMENT_PLAN_ID with the ID returned by preparation. Configure `llama-autotune.runtime` and `budgetfit.evaluator` first. The template capture loads the model but generates no completions; the measurement plan binds the task and smoke request caps. The strict literal smoke must pass before the workload runs.

EXL3 with the official ExLlamaV3 converter:

```bash
ramgpt build /path/to/local-source --out output/exl3 --gpu-uuid GPU_UUID --bits 4 --execute
ramgpt run output/exl3/artifact --dataset measurement/tasks.json --gpu-uuid GPU_UUID --out exl3-eval --budgets 256 --max-requests 8 --max-completion-tokens 2048 --execute
```

The source must be a local unquantized safetensors model directory. Configure First Mover's native manifest script, ExLlamaV3 checkout and a working Python environment. The initial EXL3 generation adapter supports Qwen3 chat templates only. Use the same dataset and native scoring settings for comparisons. This does not convert a safetensors source to GGUF automatically.

## Choose a configuration

```bash
ramgpt recommend --vram 24GiB --min-accuracy 0.85 --objective memory
ramgpt passport MODEL_SHA256 --out optimized.passport.json
```

Recommendations screen compatible existing observations, not arbitrary untested models. Separate workloads and runtimes are not combined into one ranking. With no qualifying evidence, the result is no eligible candidate. GGUF can use measured fixed profiles and bounded independent validation; EXL3 is excluded from recommendations until runtime provenance is frozen. V2 passports preserve the native evidence axes and mark missing measurements. The legacy `passport-export` command retains V1 GGUF output by default.

## Integrated tools

| Tool | Current integration |
|---|---|
| [Q4Fit](https://github.com/bettercallcaleb/q4fit) | Native GGUF quantization, recipe and depth evidence |
| [QuantBench](https://github.com/bettercallcaleb/ramgpt-quantbench) | Native historical fidelity replay and selection |
| [ConceptLadder](https://github.com/bettercallcaleb/conceptladder-bench) | Historical native knowledge and IFEval regrading |
| [TrustMeBro](https://github.com/bettercallcaleb/trustmebro) | Native task scoring through the pinned BudgetFit archive |
| [BudgetFit](https://github.com/bettercallcaleb/budgetfit) | Fixed budget scoring and adaptive router selection/validation |
| [llama-autotune](https://github.com/bettercallcaleb/llama-autotune) | Native runtime candidate planning and comparison |
| GGUF-Healer | Native tensor composition and payload verification |
| [EXL3 First Mover](https://github.com/bettercallcaleb/exl3-first-mover) | Metadata preflight, manifest, official backend conversion and Qwen3 generation |
| [HereticBench](https://github.com/bettercallcaleb/HereticBench) | Paired historical retention, instruction and epistemic regrading |

Use the [BudgetFit v0.6.0 release](https://github.com/bettercallcaleb/budgetfit/releases/tag/v0.6.0) for the pinned evaluator archive. The GGUF-Healer adapter requires an existing compatible local producer script; no standalone public Healer download is included in this release.

All nine have adapters; they are not all independent live execution engines. Native producer code, datasets and model weights are not bundled. See [advanced commands and engineering notes](docs/ENGINEERING.md) for specific native paths, options, evidence semantics and reproducibility details.

## Validation and limits

The development branch has exercised real GGUF quantization and managed inference, EXL3 conversion and eight native requests on Qwen3-0.6B, native historical scoring, and isolated wheel installation. The EXL3 engineering pilot used reduced calibration (16 × 512); its 25% joint accuracy and 25% truncation are retained, not representative quality claims. The local EXL3 backend required a process-level system libstdc++ preload. No automatic model download or publish is performed.

Single-GPU NVIDIA/Linux operation is the live-tested scope. Live standalone IFEval, broad EXL3 architecture coverage, cross-format optimization and automatic end-to-end quantization search remain outside this release. Passports are content-addressed summaries, not signed attestations or deployment certificates. Review [SECURITY.md](SECURITY.md) and [ROADMAP.md](ROADMAP.md) before using the toolchain.

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
