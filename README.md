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

Replace GPU_UUID with the selected GPU UUID from `nvidia-smi -L`, and MEASUREMENT_PLAN_ID with the ID returned by preparation. Configure `llama-autotune.runtime` and `budgetfit.evaluator` first. The template capture loads the model but generates no completions; the measurement plan binds the task and smoke request caps. The runtime smoke must pass before the workload runs. Newly captured specs record literal instruction success separately from transport health; older frozen specs retain their original strict literal requirement.

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


## AutoOptimizer (v0.3.0 development)

The development branch adds a minimal GGUF setup and an automatic bounded calibration/holdout workflow. The commands below are not in the v0.2.0 RC1 release.

```bash
ramgpt setup --profile gguf --runtime /path/to/llama-server --out native-setup --execute
export RAMGPT_TOOLS_CONFIG="$PWD/native-setup/tools.json"
ramgpt runtime-spec model.gguf --gpu-uuid GPU_UUID --out runtime --execute
ramgpt auto-optimize model.gguf --spec runtime/spec.json --out search --vram 24GiB --budgets 256 1024 --kv f16 q8_0 --seed 941 --validation-seed 942 --max-requests 52 --max-completion-tokens 37120
```

Replace the model and GPU UUID with actual local values. Setup obtains the exact pinned BudgetFit archive from its public release, or accepts a matching local archive through --evaluator. It checks a trusted existing llama-server executable and its option support, reports missing NVML dependencies, and never installs a GPU driver or arbitrary runtime. No model is loaded by setup.

Auto-optimize freezes both task manifests, reserves the maximum calibration and paired holdout request/token costs, uses f16 with the largest budget as its anchor, and writes a plan without generation. Repeat the same command with --execute to run it. Candidate selection reuses the native BudgetFit scorer, sampled GPU memory, and existing independent holdout workflow. Only a distinct challenger that passes the holdout receives selected-profile.json; every completed workflow exports model.passport.json. The profile holds local runtime arguments and request-budget settings, not a deployment certificate. The budget parameter is total completion tokens, not a native thinking-only limit. The VRAM constraint screens sampled observations; it is not a hard GPU allocator limit or an OOM guarantee. Changing parameters requires a new plan. Repeating a completed unchanged plan uses existing evidence, not new inference.

The first real local pilot completed both KV configurations on a 0.6B GGUF (16 workload requests plus two smoke requests). Both achieved 12.5% joint accuracy and were rejected under the preselected 85% threshold; no holdout was run. The runtime returned 42. during each smoke, correctly recorded as failed literal instruction following but healthy transport. Separate synthetic-runtime tests cover successful selection, failed holdout, budget rejection and idempotence. No claim of general optimization gains is made.


## Verified native supply bundle (v0.3.0 development)

Install the compatible public tool sources and pinned evaluator together:

```bash
ramgpt supply --llama-root /path/to/llama.cpp --out native --execute
export RAMGPT_TOOLS_CONFIG="$PWD/native"
ramgpt tools
ramgpt supply-verify native
```

The development command downloads five commit-pinned source archives (Q4Fit, QuantBench, llama-autotune, EXL3 First Mover and HereticBench) and the BudgetFit v0.6.0 evaluator used by both BudgetFit and TrustMeBro. Each download is bounded and verified against the catalog's SHA-256 before safe extraction. The bundle retains the archives, records extracted file hashes, and creates a relocatable configuration. When the configuration argument is a bundle directory, RAMGPT verifies the pinned archive and source bytes before loading the paths. This does not certify the behavior of native tools or install their Python dependencies. Use --runtime and --quantizer for separate llama.cpp build directories. An existing --tools-config file or bundle supplies external paths; catalog-provided fields use the pinned copies.

For offline installation, place the five named repository tar.gz archives and budgetfit.pyz in a directory and pass --archive-dir; the same digests must match. Without --execute the command only lists the pinned catalog. The bundle is not a model downloader or a GPU driver installer.

ConceptLadder v0.5 remains an external dependency: the public v0.4 source lacks the required measurement adapter. GGUF-Healer requires a compatible local script; EXL3 needs a separately installed backend and Python environment. Optional components are explicitly marked unconfigured rather than silently substituted. A public-only bundle with a suitable llama.cpp installation configures six adapters by path; supplying existing external paths can configure all nine. Configuration presence is not execution certification.

Integration checks: online and offline bundle installation; exact historical Q4Fit and QuantBench replay; real Qwen3-0.6B floating-point GGUF conversion through the supplied public Q4Fit source, yielding the same Q4_K artifact SHA-256 as the prior verified build. No new GPU inference in this increment.


## Tune a model with one command (v0.3.0 development)

After installing the package and configuring the verified native bundle:

```bash
ramgpt supply --llama-root /path/to/llama.cpp --out native --execute
export RAMGPT_TOOLS_CONFIG="$PWD/native"
ramgpt tune /path/to/model.gguf --vram 24GiB --execute
```

The llama.cpp executable and GPU Python dependency must already work. Tune automatically uses the only idle GPU (or an explicitly selected --gpu-uuid), captures a native runtime specification, freezes the calibration and holdout workload, evaluates the candidate configurations, and exports the model passport under tuning/search. The default pilot searches f16 and q8_0 KV cache with 256 and 1024 total completion-token budgets, reserving at most 52 completion requests and 37,120 completion tokens including smoke and holdout requests. Omit --execute for a read-only workload preview; it runs no GPU model load or generation. Change --out for an independent new experiment. A failed or interrupted capture is retained rather than silently retried. The VRAM target is an observed-memory selection constraint, not an allocator limit. No selected profile is emitted unless a distinct challenger passes independent validation.

## Verified GGUF cache

```bash
ramgpt model-add /path/to/model.gguf --execute
ramgpt tune MODEL_SHA256 --vram 24GiB --execute
```

Model-add copies a local single-file GGUF into state/models/SHA256/model.gguf, verifies the bytes and registers the artifact. Repeated imports reuse the verified file. Tune also accepts ordinary paths; caching is optional. Use --cache to choose another cache root.

For public Hugging Face repositories, list eligible single-file GGUF artifacts and choose the exact filename. RAMGPT can resolve the immutable repository commit, file SHA-256 and size automatically:

```bash
ramgpt model-list Qwen/Qwen3-0.6B-GGUF
ramgpt model-fetch Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --lock-out model.lock.json
ramgpt model-fetch --lock model.lock.json --execute
ramgpt tune model.lock.json --vram 24GiB --execute
```

The first fetch command reads metadata and saves a lock without downloading weights. The second uses that exact immutable revision, with no metadata refresh. Tune resolves the lock to the verified local cache; it does not download missing models. Alternatively, add --execute to the first fetch command to resolve and download in a single invocation. This resolves the requested ref afresh, so keep and reuse the lock when reproducibility matters.

Automatic metadata resolution defaults to an 8 GiB download cap. Use --max-size, for example 20GiB, to approve a larger model. Explicit SHA-256/size mode and saved-lock mode remain supported, bounded by the exact declared size and the 128 GiB implementation limit; --max-size can impose a lower cap in those modes too. Downloads are streamed, checked against the expected size, SHA-256 and supported GGUF header, and published only after verification. Repeated locked fetches verify the existing cache without network requests.

The source lock preserves the repository, immutable commit and expected artifact identity; it is not a signed publisher attestation. Metadata mode trusts public Hugging Face metadata, not an independently obtained checksum. No private/gated repositories, authentication tokens, nested or multi-shard GGUF filenames or EXL3 downloads are supported. A cache hit verifies file content, not historical repository ownership.

The metadata resolver was exercised on the public Qwen3-0.6B GGUF catalog (preview only). The full lock/download/reuse workflow was verified with the 1.19 MB ggml-org/tiny-llamas test artifact, which is not a recommended chat model.

Actual integration check: tune automatically captured the template and executed 16 workload requests plus two transport checks for the existing Qwen3-0.6B GGUF, with no eligible configuration at the fixed accuracy floor; no holdout was executed. Both local 429 MB import and a real 1.19 MB public Hugging Face download were verified, and repeat imports transferred zero new bytes. The GPU returned to baseline. Positive holdout behavior remains covered by the existing synthetic-runtime tests, not a new real positive-control result.


## Install the official llama.cpp runtime without compiling (v0.3.0 development)

For Linux x86_64 with glibc 2.39 or newer, a compatible NVIDIA driver and system CUDA 12 shared libraries:

```bash
ramgpt supply --runtime-pack cuda12.8 --out native
ramgpt supply --runtime-pack cuda12.8 --out native --execute
export RAMGPT_TOOLS_CONFIG="$PWD/native"
ramgpt tune model.gguf --vram 24GiB --execute
```

The first command previews the pinned package. The second installs the upstream llama.cpp b12810 prerelease CUDA 12.8 x86_64 binary archive and extracts the GGUF Python library from the same commit. These two archives total 267,149,553 bytes (about 255 MiB), in addition to the original native tool source packages and evaluator. The package locks the GitHub release asset SHA-256 and the observed commit-source archive SHA-256, verifies sizes, safely handles same-directory shared-library aliases and checks executable permissions. It validates llama-server options and resolves shared-library dependencies before publishing the installation. It retains the original archives for offline integrity verification; passing the bundle directory as the configuration rechecks the pinned installed files. Python bytecode caches are excluded from the source manifests. This verifies content, not a signed upstream attestation.

The official binary pack does not install the CUDA toolkit, NVIDIA driver or Python extras; system CUDA 12 shared libraries remain required. Missing libraries cause setup to fail instead of silently choosing CPU. Install the package GPU extra and NumPy separately where needed. Use the external llama.cpp path workflow on other supported runtime configurations. The pinned archive is an upstream prerelease, not a claim of stable cross-platform support. For offline installation, add llama-source.tar.gz and the named upstream binary archive to the original --archive-dir along with the five native archives and budgetfit.pyz.

Verified on Ubuntu 24.04 x86_64 with RTX 4090 and existing CUDA 12 libraries: official binary and same-commit Python source obtained and hashed; offline combined bundle installation, native template capture, 16 workload requests and two transport checks completed. No candidate passed the 85% accuracy floor in the small-model engineering pilot, so holdout and selected profile were not produced. The GPU returned to baseline. The public-only bundle still does not supply ConceptLadder v0.5, GGUF-Healer or the ExLlamaV3 backend.
