# RAMGPT Toolchain

**Build. Evaluate. Optimize. Share the results.**

RAMGPT brings nine local model-engineering tools into one CLI: Q4Fit, QuantBench, ConceptLadder, TrustMeBro, BudgetFit, llama-autotune, GGUF-Healer, EXL3 First Mover and HereticBench. The current development branch supports GGUF and EXL3 artifacts, native tool adapters, a common model catalog, constrained configuration selection and portable Model Passports.

## Quick start (development branch)

Linux, Python 3.12+. From a checkout of this branch:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[all]'
ramgpt doctor --gpu
ramgpt register /path/to/model.gguf
ramgpt passport /path/to/model.gguf --out model.passport.json
ramgpt passport-verify model.passport.json --model /path/to/model.gguf
```

A newly registered artifact has identity, not evaluation results. The passport lists missing evidence.

For native integrations, point RAMGPT at your existing trusted tool checkouts. It does not download or install them automatically.

```bash
ramgpt init --root /path/to/local/tools --out tools.json
export RAMGPT_TOOLS_CONFIG="$PWD/tools.json"
ramgpt tools
ramgpt model-info
ramgpt recommend --vram 24GiB --min-accuracy 0.85
```

Review the discovered paths before exporting RAMGPT_TOOLS_CONFIG. Resolve duplicate candidates with --set tool.field=PATH and a new output file; see tools.example.json. Use explicit --tools-config to override the environment. Discovery checks paths, not native compatibility.

## Main commands

| Command | What it does |
|---|---|
| build | Quantize a floating-point GGUF using Q4Fit or a local source directory using ExLlamaV3 |
| run | Execute a model-specific GGUF plan, frozen contract or imported Qwen3 EXL3 model |
| recommend | Screen known, compatible observations against your limits |
| passport | Export a portable model summary with separate evidence axes |
| init / tools | Discover trusted local tool paths and inspect configuration |
| runtime-spec | Capture a model-specific GGUF runtime specification |

See **Four common commands** below for complete build and run examples. The GGUF backend can generate its model-specific runtime spec using runtime-spec, and the current EXL3 runner supports Qwen3 templates. The EXL3 adapter has been exercised on a real small model; EXL3 recommendation remains blocked until runtime provenance is frozen. No model or native producer code is bundled.

The toolchain does not pool distinct metrics into an arbitrary overall score. Native scoring remains authoritative. This is a research alpha, not a deployment certificate. Read [SECURITY.md](SECURITY.md) before executing native tools and [ROADMAP.md](ROADMAP.md) for current scope.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## G1/G2: local registry and immutable experiment contracts

See [ROADMAP.md](ROADMAP.md) for the staged development plan.

    python3 ramgpt.py register /path/to/model.gguf
    python3 ramgpt.py artifact SHA256
    python3 ramgpt.py freeze state/spec.json
    python3 ramgpt.py contract CONTRACT_ID
    python3 ramgpt.py verify CONTRACT_ID

The specification has three objects: inputs (role/path entries for model, runtime, dataset, template, evaluator, and optional dependencies/configuration); execution (runtime_args, gpu_uuids, sampling, reasoning_policy); workload (protocol, seed, budgets). Freeze hashes local files and records a content-derived contract ID. Verify rehashes actual inputs; it is not a test of inference behavior. Local state is ignored by Git.

## G3: managed llama-server smoke test

    python3 ramgpt.py runtime-smoke CONTRACT_ID --out state/runtime-smoke-01

Supports the frozen single-GPU NVIDIA llama-server path on Linux. Acquires a same-user, machine-local GPU UUID lease, refuses an occupied GPU or port, verifies frozen file bytes, launches only the declared executable with an allowlisted argument vector, and checks ownership of the loopback listener using Linux procfs. The health probe, frozen rendered template probe and a literal 42 request must pass. It stops only its own server, checks GPU memory and compute processes after cleanup, and rehashes frozen inputs. The new output directory contains raw request/response, server log, template observation and runtime receipt; never reused or overwritten.

The lease is cooperative, not an OS-level GPU reservation. Nonparticipating processes and other OS users can still access the device; hard-killing the manager can leave a server alive. Startup/cleanup receipts and hashes are observations, not cryptographic attestation. The smoke request is not a scored dataset run and does not validate general model quality or a full chat template. The current implementation is intentionally single-GPU and rejects preflight memory above 512 MiB. Runtime shared libraries and driver bytes are not yet fully pinned; the driver version is recorded. Multi-GPU, full runtime dependency inventory and inference schedules are subsequent gates.

## G4: managed inference runs

Bind a request schedule as the single configuration input in an experiment contract. Its schema is ramgpt-request-schedule-v1, containing a requests array. Every request has request_id, task_id, strategy, prompt and max_tokens. The workload uses protocol toolchain-inference-v1 and declares request_count, task_count, strategies, budgets and seed. Prompt generation remains the responsibility of the pinned evaluator; the execution layer does not score or rewrite prompts.

    python3 ramgpt.py run CONTRACT_ID --out state/my-run --max-new 3
    python3 ramgpt.py run CONTRACT_ID --out state/my-run

Each successful batch has its own runtime receipt, raw response records and commit seal. Resume checks the frozen schedule, prior request order, response identity, counts and measurements before running only remaining requests. A completed run performs no new inference. An uncommitted session (including ambiguous HTTP timeout or crash) blocks automatic resume and preserves the pending request/error for inspection. This is checkpointed batch recovery, not exactly-once execution after arbitrary failure; an explicit recovery procedure remains to be implemented. The seals detect inconsistent files, not malicious rewriting of the entire local evidence set. Request timing is measured per HTTP call; startup and verification time are separate. Smoke requests are recorded independently and are not counted in the workload.

Current live validation used eight Qwen3.8-27B Q2_K Core tasks, native BudgetFit prompts, 256-token Aware strategy, seed 608. Three requests completed in the first session and five in the second. The third invocation issued zero new workload requests. The GPU returned to baseline. No evaluation score or cross-tool cache reuse has yet been certified.

## G5: pinned evaluation adapter

The existing BudgetFit 0.6.0 zip application is executed in an isolated Python subprocess. It supplies its own frozen TrustMeBro v0.2, recovery v0.4 and bundled ConceptLadder code. The adapter checks the declared engine versions, rebuilds the native task manifest and prompts, refuses unexpected or duplicate evaluation keys, and reports each track separately.

    python3 ramgpt.py prepare --dataset tasks.json --evaluator budgetfit.pyz --budgets 256 1024 --strategies aware allocated --reserve 64 --out schedule.json
    python3 ramgpt.py score CONTRACT_ID --run state/my-run --out state/evaluation.json

The prepared schedule must then be frozen as the configuration input. For Allocated, include the same reserve in workload.reserve. Managed BudgetFit runs now validate their complete native prompt matrix before inference. The score command requires committed complete responses and matching frozen input hashes, records the evaluator and adapter hashes, and uses native scoring and recovery functions. It does not re-send requests. Blind and native_strict use the original prompt contracts; other strategies are explicitly labeled as BudgetFit-modified. Ladder levels and family reports are not independent benchmarks or an additional composite score.

Recheck historical evaluator compatibility without GPU:

    python3 audit_parity.py --evaluator budgetfit.pyz --reserve 64 /path/to/original/results

The audit compares full original grades, six selected recovery fields, finish reason, token count and elapsed time. It is not a full forensic replay of every diagnostic field or proof of the historical model identity.

Live G4 response replay: 8/8 responses, 6 joint successes, 7 recovered correct, one unknown truncated response, 648 completion tokens. Historical 27B Q2_K, Q4_K_M and Bonsai pilot replay: 1,824/1,824 records matched all checked fields. No GPU inference was performed in G5.

Run native adapter tests with BUDGETFIT_TEST_ARCHIVE set to the trusted local release archive. The current adapter supports the built-in generator and complete fixed budget/strategy matrices with one reserve value. Custom datasets, independent ConceptLadder releases, and additional native tool adapters require separate work. Adaptive router holdouts use the separate G7 adapter below. No deployment policy is certified by the score command.

## G6: content-addressed evidence and historical replay

    python3 ramgpt.py archive-run CONTRACT_ID --run state/my-run --report state/evaluation.json
    python3 ramgpt.py evidence-find --artifact ARTIFACT_SHA256
    python3 ramgpt.py evidence-get SOURCE_ID --out state/export.json
    python3 ramgpt.py replay CONTRACT_ID --out state/replay.json

The evidence database (state/evidence.sqlite by default; override with --evidence-db before the command) separates original execution snapshots from derived evaluations. Each is addressed by its canonical JSON SHA-256. Ingestion verifies the frozen inputs, complete request matrix and sealed sessions, and replays native scoring. An optional supplied report must match the recomputed report apart from its creation time. Identical imports are idempotent. Artifact and contract indexes are checked against the stored content when queried.

Explicit replay requires the exact frozen contract and current evaluator/adapter hashes. It verifies the local frozen inputs and independently recomputes the report from stored responses without invoking a model. If multiple historical sources match, supply --source SOURCE_ID instead of silently choosing one. The replay result clearly labels prior latency, token counts and runtime evidence as historical measurements. It does not count as a new independent replicate or current deployment validation. Normal run commands still execute their own workload; no automatic cross-contract or per-prompt cache is enabled.

Evidence snapshots contain private local paths, prompts, responses, task manifest, runtime receipts and commit seals; they are not automatically published. The database does not copy model weights, full server logs or all runtime dependencies. Canonical content hashes detect corruption and inconsistency, not intentional rewriting of the entire trust store. Inspecting stored evidence works without the original run directory, but verified replay requires the original frozen inputs. Historical external evidence without a bound managed contract remains separate in the existing ledger; it is not eligible for this replay path.

## G7: constrained fixed-configuration optimization

    python3 ramgpt.py select EVALUATION_ID --policy policy.json --validation-seed 610
    python3 ramgpt.py validate-selection DECISION_ID VALIDATION_EVALUATION_ID

The policy declares objective (tokens or latency), min_accuracy, accuracy_tolerance, max_validation_drop, min_savings, max_truncation, min_tasks, and reference [budget, strategy, reserve]. Selection compares profiles in one complete archived evaluation. It reports the three-dimensional accuracy/token/latency Pareto frontier, checks the minimum accuracy and the tolerance relative to both the reference and the best observed accuracy, and rejects insufficient savings or excessive truncation. A decision freezes the candidate, reference, validation seed, policy, and observed environment before held-out inference.

Validation requires the same model, runtime, GPU/driver, evaluator, adapter and execution settings; a different frozen task seed, the same generator configuration, no normalized prompt overlap, and a complete paired matrix containing exactly the candidate and reference. A decision accepts only one validation source; repeating the same validation is idempotent. Results are explicitly pilot evidence, never deployment authorization. No automatic threshold adjustment or alternate holdout selection is performed.

Real pilot (Qwen3.8-27B Q2_K, seed 609, eight Core tasks per arm, Allocated reserve 64, reference 1024): 256 tokens achieved 5/8 joint successes and 46.75 mean completion tokens; 1024 achieved 6/8 and 83.875. The lower cap saved 44.26% observed output tokens but missed the frozen 73% accuracy floor. No candidate was selected, so no holdout inference was started. The 16 requests and decision are archived. This is a small engineering gate test, not evidence of general budget performance. Latency remains order- and runtime-sensitive; shared library and peak VRAM measurements are not yet complete. The current selector handles a fixed model/runtime and fixed budget/strategy matrix. Joint artifact/runtime search remains open; adaptive routing uses the separate adapter below.

## G7-B: native adaptive budget router

Train only from a complete archived BudgetFit calibration evaluation. The adapter pins the BudgetFit 0.6.0 archive SHA-256 and calls its original router training, selection, evaluation and gate functions in an isolated subprocess.

    python3 ramgpt.py router-parity --evaluator budgetfit.pyz --folder /path/to/historical/router
    python3 ramgpt.py router-prepare CALIBRATION_EVALUATION_ID --validation-seed 610 --thresholds thresholds.json --out state/router-pilot
    python3 ramgpt.py run CONTRACT_ID --out state/router-pilot/run
    python3 ramgpt.py archive-run CONTRACT_ID --run state/router-pilot/run
    python3 ramgpt.py evidence-get EVALUATION_ID --out state/router-report.json

Example thresholds (small engineering pilot only):

    {"min_accuracy":0.5,"tolerance":0.02,"max_validation_drop":0.1,"min_bucket":2,"min_savings":0.05}

Preparation freezes the native selection, reference, thresholds, validation seed, generated manifest digest, adapter hash and calibration execution context. It requires disjoint normalized prompts and binds one holdout contract per calibration evaluation. The runtime preflight reconstructs the native sparse schedule: exactly the router and reference requests per task, deduplicated if identical. Fixed-matrix validation remains unchanged. The run path is bound in the evidence database. Archive ingestion regrades the sealed raw responses using the native scorer, then evaluates and gates the router; failed results are retained. Exact-contract replay uses the frozen router adapter identity. No alternate validation seed or threshold is selected after observing the holdout.

The initial scope is native generated datasets and the same reserve for every Allocated configuration. The router feature is the original UTF-8 prompt length, not a new learned difficulty estimator. A pilot pass is not production authorization; shared-library identity, larger independent samples and runtime/weight integration remain open. Local state is private and is not committed. The one-run binding assumes intact local state; it is not tamper-proof against manual deletion or rewriting.

Historical parity: 820 stored Q2/Q4 requests reproduced the original native selection, profiles, validation metrics and gate outcomes without new inference. Q2 passed (84.21% router accuracy, 25.77% token savings); Q4 failed (86.84%, 11.47%). Historical records lack a directly bound artifact identity and are not managed execution evidence.

Managed Q2_K pilot: the existing seed 609 calibration (16 requests) trained a 95-byte threshold, with 1024/256-token short/long caps and 1024-token fallback, all Allocated reserve 64. Calibration accuracy was 75%, with 28.61% observed token savings. Independent seed 610 ran 13 deduplicated requests across eight distinct tasks. Fixed accuracy was 6/8 versus router 7/8; mean completion tokens were 127.125 versus 132.375, respectively. The native gate rejected the router for negative token savings (-4.13%). Both source and derived evidence were archived; no deployment policy was authorized. The GPU returned to its pre-run baseline. No calibration or historical inference was repeated.

## G7-C: measured native runtime candidates

The runtime adapter imports the local llama-autotune v0.1.8 Python source as a private, content-addressed archive. It uses native capability discovery, candidate generation and argument rendering rather than implementing a second tuning algorithm. It does not use the native server lifecycle or modify the upstream project.

    python3 ramgpt.py runtime-prepare --spec state/g7/cal-spec.json --autotune-source /path/to/llama-autotune-v0.1.8 --out state/runtime-pilot --seed 611
    python3 ramgpt.py run F16_CONTRACT_ID --out state/runtime-pilot/f16-run
    python3 ramgpt.py archive-run F16_CONTRACT_ID --run state/runtime-pilot/f16-run
    python3 ramgpt.py run Q8_CONTRACT_ID --out state/runtime-pilot/q8_0-run
    python3 ramgpt.py archive-run Q8_CONTRACT_ID --run state/runtime-pilot/q8_0-run
    python3 ramgpt.py runtime-compare --plan state/runtime-pilot/comparison-plan.json F16_EVALUATION_ID Q8_EVALUATION_ID

The first bounded adapter intentionally selects the native f16 and q8_0 KV candidates only, both on one visible GPU with 4096 context, one slot, explicit batch/ubatch, flash attention and fitting disabled. It freezes a shared generated task manifest, prompts, sampling, runtime binary, model, linked libraries and adapter files. It records shared objects observed in /proc/PID/maps after the workload. An unmapped dependency is not evidence that it was used; an observed but unfrozen library is explicitly labeled.

Optional NVML v2 telemetry samples the selected GPU before launch, throughout startup and inference and after cleanup. It records device-wide used, reserved and free bytes, sample count and maximum observed gap. The sampled high-water mark is a lower bound on instantaneous peak memory, not an exact allocator peak. This path requires pynvml with NVML v2 support. Prefill and decode throughput are computed separately from native response timing counts and durations; HTTP latency and startup-to-health are separate. Startup may benefit from OS file cache and is not a cold-disk benchmark.

Comparison requires matching artifact, task manifest, runtime environment and mapped-library identities, with KV type the only runtime option allowed to differ. The registered plan declares accuracy and minimum observed free-memory constraints; eligible candidates form a Pareto set over accuracy, sampled memory, prefill and decode throughput. This is exploratory paired evidence, not a production validation or a global runtime optimum. No automatic broad search, multi-GPU tuning, context-frontier execution or deployment is enabled by this adapter.

Managed runtime pilot, seed 611: 16 new requests across f16 and q8_0 KV at context 4096. Joint accuracy was 62.5% and 75%, with equal 87.5% semantic accuracy. Sampled device memory high-water marks were 10505.3125 and 10397.3125 MiB (236 and 242 samples); prefill was 758.09 and 751.42 tokens/s, decode 70.40 and 69.85 tokens/s. Both candidates were eligible and Pareto-nondominated. Both runs observed 23 mapped shared libraries, all covered by frozen dependency hashes. These are sequential, small-sample exploratory measurements, not a statistically established quality improvement or deployment validation.

## G7-D: weight, fidelity and recipe evidence

Import completed producer evidence into a separate content-addressed weight report index. The native reports remain independent of managed inference evidence.

    python3 ramgpt.py weight-import quantbench /path/to/complete/run --native /path/to/run-artifact-comparison.py
    python3 ramgpt.py weight-import q4fit /path/to/depth --native /path/to/q4evidence.py
    python3 ramgpt.py weight-import recipe /path/to/model.manifest.json
    python3 ramgpt.py weight-find --reference REFERENCE_SHA256
    python3 ramgpt.py weight-view ARTIFACT_SHA256
    python3 ramgpt.py weight-select REPORT_ID --max-kl 0.015 --max-bytes 19327352832

QuantBench ingestion requires the complete local release package, including its release manifest, compact NPZ files, run specification and native independent validation logs. It checks file hashes, recomputes the published compact statistics and calls the original native bootstrap. NumPy is required. The original full-vocabulary logits and model binaries are not reverified. The producer's historical independent-validation status is retained, not rebranded as a new run. Native scripts must be trusted local source; their hashes are recorded, not a full dependency lock. The compact files and upstream logs remain in the producer directory; the database records the native documents, source file hashes and derived statistics, not a self-contained copy of all binary evidence.

Q4Fit calls its native depth exporter, checks the source JSON hashes and paired counts, and separately imports a quantization manifest only after hashing both actual GGUF files and checking the native recipe metadata. A matching recipe manifest is not proof of the historical quantizer execution. The depth export does not declare reference precision, so it is recorded as unknown rather than inferred from a filename. QuantBench preserves its declared native BF16 reference and open-artifact provenance; it does not establish strict recipe equivalence.

The artifact view joins exact SHA-256 identities only, checks known byte-size consistency, preserves independent metric axes, and explicitly lists missing evidence. The fidelity selector considers one original benchmark cohort at a time, filters by caller-provided maximum KL and file bytes, and returns the Pareto frontier without pooling cohorts or implying runtime fit or deployment readiness. No quantization job or new inference is started. Model file bytes are not a VRAM measurement.

Initial integration replayed two QuantBench releases (six fidelity records, including the same Bartowski artifact in two distinct cohorts), one Q4Fit depth export, and one Q4Fit build manifest. The six unique candidate identities have two reference identities. The existing managed Q2 pilot has no matching fidelity or recipe record, and the imported weight candidates have no matching managed task evidence. This missing intersection is reported rather than repaired using model names. With illustrative limits of mean KL 0.015 and 18 GiB file bytes, the four-artifact Q4_K_M cohort yields Bartowski and Unsloth on its Pareto frontier. No new inference or quantization was performed.

## G7-E: joint constrained selection and independent pilot

The joint selector operates on observed complete fixed configurations, not an inferred Cartesian product of best metrics. Supply an explicit anchor evaluation/profile, candidate evaluation IDs, a JSON policy and an unused validation seed.

    python3 ramgpt.py joint-select --policy policy.json --anchor BASELINE_EVALUATION_ID --budget 1024 --strategy allocated --reserve 64 --validation-seed 612 BASELINE_EVALUATION_ID CHALLENGER_EVALUATION_ID
    python3 ramgpt.py joint-prepare DECISION_ID --out state/joint-holdout
    python3 ramgpt.py run BASELINE_CONTRACT_ID --out state/joint-holdout/baseline-run
    python3 ramgpt.py archive-run BASELINE_CONTRACT_ID --run state/joint-holdout/baseline-run
    python3 ramgpt.py run CANDIDATE_CONTRACT_ID --out state/joint-holdout/candidate-run
    python3 ramgpt.py archive-run CANDIDATE_CONTRACT_ID --run state/joint-holdout/candidate-run
    python3 ramgpt.py joint-validate DECISION_ID BASELINE_HOLDOUT_EVALUATION_ID CANDIDATE_HOLDOUT_EVALUATION_ID

Policy fields: objective (memory, latency or tokens), min_accuracy, max_latency_s, max_sampled_peak_bytes, min_free_bytes, max_mean_tokens, max_truncation, min_tasks, max_validation_drop, accuracy_tolerance, fidelity_report_id, max_kl, max_artifact_bytes, max_sample_gap_s. Set both fidelity fields to null only for an explicitly non-fidelity-constrained, same-artifact experiment. Cross-weight selection requires both artifacts in the same selected fidelity cohort. Missing matching fidelity or continuous memory evidence fails the respective constraint. File bytes do not substitute for measured VRAM.

Candidate evidence must share the exact task manifest, runtime/adapter/evaluator/template identity, sampling, GPU environment and non-tunable execution settings. Runtime tuning flags and fixed budget/prompt strategy may vary. Only measured profiles are eligible; a whole-session high-water mark is conservatively assigned to each observed profile if several profiles share a session. Unknown library identity or missing telemetry is not silently treated as zero. The sampled high-water mark is a lower bound on instantaneous peak memory. Ranking uses hard constraints followed by Pareto dominance and a declared primary objective; there is no weighted composite score.

The frozen decision chooses one challenger before generating new same-family instances, checks prompt overlap against existing managed evidence, and binds exactly two holdout contracts and run paths. The independent pilot checks the original constraints, accuracy drop against calibration and paired reference, and strict improvement on the declared objective. It records paired wins/losses/ties and rejects alternate holdout evidence for the same decision. No pilot authorizes deployment. A new decision under a different policy is a new experiment, not a globally enforced prohibition on post-hoc analysis.

The initial historical database has no exact overlap between imported weight-fidelity evidence and managed runtime/task evidence. With a KL limit of 0.015 on the selected QuantBench cohort, the result is no eligible candidate. Without a fidelity requirement, the same-artifact seed-611 runtime pilot selects the Q8_0 KV configuration. Adaptive router profiles are not accepted by this initial joint selector, even if independently validated; they require a separate schedule-aware adapter. Full cross-weight validation requires actual matched measurements; no automatic model downloads or quantization search is performed.

The seed-612 holdout completed 16 new requests, with no prompt overlap with earlier managed evidence. The F16 reference achieved joint accuracy 87.5%, semantic accuracy 100%, mean 159.875 completion tokens, mean 2.6314 s latency and zero truncations. Q8_0 achieved joint accuracy 75%, semantic accuracy 87.5%, mean 183 completion tokens, mean 2.9210 s latency and 12.5% truncation. Sampled peaks were 10505.3125 and 10397.3125 MiB respectively. The frozen gate rejected Q8_0 for truncation and reference-accuracy drop (0 wins, 1 loss, 7 ties). The decision remains failed; no deployment or replacement holdout is authorized.

## G8: portable Model Passport

Export one model's exact-hash evidence as a deterministic, path-free JSON summary.

    python3 ramgpt.py passport ARTIFACT_SHA256 --out model.passport.json
    python3 ramgpt.py passport-verify model.passport.json --expected-id PASSPORT_SHA256
    python3 ramgpt.py passport-verify model.passport.json --model /path/to/model.gguf
    python3 ramgpt.py passport-verify model.passport.json --local

The passport has a versioned envelope and a canonical SHA-256 identity. It contains independent weight/fidelity/knowledge records with reference identities, managed task profiles and diagnostic breakdowns, dataset and evaluator identities, historical runtime settings and input hashes, GPU model/driver/capacity, sampled memory, native prefill/decode throughput, startup observations, and fixed/router/joint selection or validation outcomes. Missing axes are explicit. Failed holdouts remain visible. No local paths, device UUIDs, raw prompts, model responses or deployment authorization are exported.

Offline verification does not require the original evidence database. A trusted expected ID detects changes, and optional model verification hashes the actual GGUF bytes. Local verification independently rebuilds the summary from the current registry and evidence store and compares it exactly. A newer evidence snapshot can differ legitimately; retain the original passport and digest. None of these operations authenticates a publisher or regenerates the original experiments. Raw evidence remains in the originating stores, and the passport is not a self-contained replay package or production certificate.

The first export covered seven distinct candidate identities: five with QuantBench fidelity, one Q4Fit recipe/knowledge pair and one managed Q2 model with seven archived evaluations. All seven local snapshot comparisons passed. The Q2 passport retains both failed router and joint holdouts; fidelity, knowledge depth and recipe remain missing for that artifact. The Q4Fit candidate's actual local GGUF identity was also verified. No new GPU requests were required.

The managed Q2 GGUF bytes were independently hashed and matched the passport. Sample exports are in private state/g8. Export never overwrites an existing file.

## G9: install the local alpha

RAMGPT Toolchain 0.1.0 is a Linux/Python 3.12+ command-line alpha. This is a GitHub research alpha; no PyPI release is provided. Build and install from a trusted local checkout:

    python3 -m pip wheel --no-build-isolation --no-deps --no-index --wheel-dir dist .
    python3 -m venv .venv
    .venv/bin/python -m pip install --no-index --no-deps dist/ramgpt_toolchain-0.1.0-py3-none-any.whl
    .venv/bin/ramgpt --version
    .venv/bin/ramgpt doctor
    .venv/bin/ramgpt doctor --gpu --evaluator /path/to/budgetfit.pyz --runtime /path/to/llama-server
    .venv/bin/ramgpt passport-verify model.passport.json --expected-id PASSPORT_SHA256

The wheel build requires locally installed setuptools and wheel. The offline core has no third-party Python dependency. QuantBench compact-statistics replay requires NumPy (the fidelity extra); sampled NVIDIA memory telemetry requires nvidia-ml-py (the gpu extra). Install these from a trusted package source or local wheelhouse as needed. No model weights, local state, external evaluator archive, QuantBench runner, Q4Fit runner or llama-autotune source are bundled. The doctor checks only supplied input files and optional GPU visibility; it does not start inference or install dependencies. A detected package is not proof that every native dependency works.

The installed launcher runs the byte-identical engine files from a private directory in an isolated Python subprocess, rather than exposing generic modules such as registry, runtime or evidence at top level. The default state paths are relative to the current working directory, just like the source CLI; use explicit --db and --evidence-db before a command to reuse an existing workspace. Historical contracts retain their original local paths and cannot be relocated by merely installing the wheel. Standalone passport verification and doctor do not create a database.

This is a local research alpha, not an automatic model optimizer, signed evidence system or production-certified deployment tool. The managed runtime remains single-GPU Linux/NVIDIA GGUF. EXL3, GGUF-Healer, HereticBench, automatic cross-weight search and a full new-machine GPU test are not integrated or validated by this package. Third-party tools remain separately distributed and retain their own licenses.

## Unified optimizer (v0.2 development)

    ramgpt optimize model.gguf --vram 24GiB --objective balanced --min-accuracy 0.85 --out optimization.json

Use global --db and --evidence-db options before optimize to point to an existing evidence workspace. The planner registers the actual model bytes and discovers archived evaluations by exact SHA-256. It evaluates only measured, fixed profiles of that artifact. It groups by task manifest, execution context and mapped-library identity; it does not rank across groups. Sealed holdouts and adaptive-router reports are not reused for selection. Previously failed fixed or joint challengers are excluded by their configuration/context identity. Multiple observations of the same candidate in one cohort require explicit replicate resolution rather than picking the best result.

The default constraints are 85% joint accuracy, eight tasks, zero truncations, 1 GiB headroom and maximum 0.25-second memory sample gap. Use --min-accuracy, --min-tasks, --headroom, --max-sample-gap and --context to change the constraints. GB is decimal, GiB binary. Sampled VRAM must fit the requested limit minus headroom, and observed free VRAM must meet headroom. Neither is a guarantee of exact peak memory on a different device. An explicit --fidelity-report plus --max-kl adds a matched historical fidelity gate. Unknown fidelity never passes that gate. Without it the planner is same-artifact only.

Balanced keeps nondominated configurations without combining metrics into a weighted score. Memory, latency and tokens may select an exploratory pick within each eligible cohort. The output always marks deployment unauthorized and reports missing evidence or next steps. New models without measurements return measurement_required; unsuitable historical evidence returns no_eligible_candidate. There is no GPU inference, automatic holdout or quantization in this command. Optional --out writes a new path-free JSON plan without overwriting existing files.

The first integration replay used existing sealed Q2 evidence only. Under the default accuracy floor no calibration profile was eligible. At an explicitly lower floor of 50%, the F16 KV calibration was an exploratory candidate, while Q8_0 was excluded for its previously failed held-out validation. No holdout was rerun, no evidence was reclassified, and no deployment recommendation was issued.


## Bounded measurement planning (v0.2 development)

    ramgpt measurement-prepare --spec base-spec.json --out state/calibration-01 --seed 613 --budgets 256 1024 --kv f16 q8_0 --max-requests 34 --max-completion-tokens 20608
    ramgpt measurement-run PLAN_ID --execute
    ramgpt optimize model.gguf --vram 24GiB --objective balanced --min-accuracy 0.85

Use a trusted base experiment specification with model, runtime, rendered template probe, pinned BudgetFit evaluator and execution settings (see G1/G2). Existing dataset/configuration inputs are replaced by the native Core manifest and schedule. The planner freezes all arms and runtime dependencies before execution. Only the KV cache type is varied across arms; the native budget/strategy matrix is shared. Both smoke completions and workload completions count toward the explicit caps. Prompt tokens, elapsed time and hardware capacity are not guaranteed by these caps.

Preparation does not launch the model or submit inference requests. Execution requires the registered plan ID and explicit --execute. One full session per arm is permitted by this orchestration path. The start marker is persisted before launching; interrupted or missing started runs fail closed rather than silently retrying. Complete sealed runs are archived with the existing native evaluator and repeated invocations reuse the same archived evidence. Direct run/runtime-smoke commands are blocked for measurement calibration contracts. This is an application-level workflow guard, not a security sandbox or machine-wide quota.

A real archived-model preparation test froze two KV arms and 32 workload requests plus two smoke requests, with a total 20,608 completion-token cap; no GPU run was launched. The complete execution/archive/idempotency path was tested using a simulated runtime and the real pinned native evaluator. Actual backend compatibility, KV support, available GPU memory and performance remain subject to live verification. The measurement plan is private local state and does not authorize deployment.


## Bounded optimization workflow (v0.2 development)

After executing the registered measurement plan, hand its complete archived calibration to the existing joint selector:

    ramgpt workflow-select PLAN_ID --policy joint-policy.json --anchor-kv f16 --budget 1024 --strategy allocated --reserve 64 --validation-seed 614 --max-requests 18 --max-completion-tokens 16512
    ramgpt workflow-finish PLAN_ID --execute

The explicit anchor and joint policy use the same fields as joint-select. The workflow requires all calibration arms to have complete sealed runs and matching archives. It previews the existing joint selector, refuses a previously failed challenger with the same configuration/context identity, and creates a holdout only for a distinct candidate with strict calibration objective improvement and a comparable calibration cohort. It never chooses a runner-up after a failed gate. Selection freezes once per measurement plan in this workflow. The lower-level experimental APIs are still available; this is not a global experiment quota or a security boundary.

The holdout budget is separate from the calibration budget and includes both smoke requests. The registered decision, actual native holdout schedules, immutable contract identities, run bindings and cost are rechecked before execution. Each arm gets at most one complete session on this path, with a persistent start marker. Interrupted or missing started runs fail closed. Complete archived holdouts can be reused without new requests. No eligible candidate or no strict improvement produces no new holdout requests.

The final workflow report records the independent pilot outcome and creates a portable Model Passport in the original measurement directory. Failed validation remains visible; a repeated finish returns the same final artifact with zero new requests. Historical raw evidence and local workflow state must remain private. A successful pilot does not authorize deployment.

Integration tests use a simulated runtime with deterministic memory telemetry, real sealed inference/evidence paths and the pinned native evaluator. They cover successful and failed validation, no-candidate completion, repeated finish, budget rejection, and interruption protection. Real end-to-end GPU verification of this new workflow remains a separate step.


## Native tool integration: GGUF-Healer and ConceptLadder

The toolchain can now call the existing GGUF-Healer tensor composer, register an existing repaired artifact after checking every output tensor against its source, and replay historical ConceptLadder v0.5 knowledge scoring without inference. No producer code is bundled.

    ramgpt healer-compose base.gguf donor.gguf --out repaired.gguf --native /path/to/healer.py --gguf-root /path/to/llama.cpp/gguf-py --tensor output.weight --max-growth 0.05 --execute
    ramgpt healer-import repaired.manifest.json --native /path/to/healer.py --gguf-root /path/to/llama.cpp/gguf-py --max-growth 0.05
    ramgpt ladder-import repaired.gguf knowledge.jsonl --native /path/to/conceptladder --seed 211 --variants 1
    ramgpt tool-find --artifact ARTIFACT_SHA256

Use trusted producer code and compatible NumPy/GGUF dependencies. Compose creates a new file only and requires explicit execution. The native compatibility checks and size cap are not relaxed; the caller must establish shared checkpoint lineage. Import verifies all selected and untouched tensor payloads, descriptors, metadata, file size and current input/output hashes. If verification fails after composition, the output is retained for inspection, not registered as successful. A byte-verified repair is not proof of quality improvement.

Ladder import copies raw responses and their manifest to a temporary directory, runs the native score command and requires exact complete-report parity. Original files remain untouched. It requires the declared model path and size to match the supplied model, and records its current SHA-256; this does not independently authenticate historical runtime or weight identity. The two receipts are indexed together by artifact hash and can be retrieved via tool-find. They remain separate from managed benchmark results and from the existing passport schema. IFEval, EXL3 conversion and HereticBench adapters are not yet connected here.

Local integration checks: a synthetic two-tensor GGUF composed by the actual native Healer passed payload re-verification; an existing 27B repaired artifact passed all 866 tensor checks with 32 donor tensors and 4.98% file growth. The native knowledge report replay matched all 60 historical items for that same artifact. No new model inference was launched.


## Native EXL3 First Mover and HereticBench integration

    ramgpt exl3-preflight Qwen/Qwen3-8B --native /path/to/exl3-first-mover/fm.py --backend /path/to/exllamav3 --bits 4
    ramgpt exl3-import /path/to/exl3-artifact /path/to/manifest-4.00bpw.json --native /path/to/exl3-first-mover/fm.py
    ramgpt heretic-import --native /path/to/HereticBench --dataset retention_signal.jsonl --base base.jsonl --variant variant.jsonl --track retention
    ramgpt heretic-import --native /path/to/HereticBench --dataset epistemic_integrity.jsonl --base base-epi.jsonl --variant variant-epi.jsonl --track epistemic
    ramgpt tool-reports --tool exl3-first-mover
    ramgpt tool-reports --tool hereticbench

First Mover preflight calls the native metadata/config workflow in an isolated temporary workspace, with the explicitly selected local backend architecture directory. It uses public repository metadata, performs no weight download or conversion, and requires the native size/disk gate and architecture inspection to pass before reporting readiness. Use a clean trusted fm.py without local config overrides. Readiness is not a GPU or runtime guarantee. The native EXL3 manifest adapter computes a format-specific directory identity from the complete producer file manifest (excluding README.md, as the producer does), requires EXL3 config metadata, and detects changed files. It does not validate tensor semantics or certify a historical conversion. Actual native conversion and runtime execution remain outside the integrated CLI.

HereticBench imports complete paired historical retention/instruction or epistemic runs. It verifies dataset hashes, matching messages and grading rules, unique complete task IDs, and native semantic/format score parity. Each category and track is retained separately, without a cross-axis score. It records dataset and raw-run hashes rather than inventing model artifact hashes for historical runs without authenticated model identities. The aggregate within a dataset is descriptive, not a cross-benchmark optimization target. OpenBehavior and boundary evaluation are not integrated. Native reports are distinct from file-identified tool receipts and are available via tool-reports; they are not yet embedded in the Model Passport.

Local checks: Qwen3-8B metadata preflight returned GREEN with 4 bpw target, 38.65 GiB estimated disk requirement and positive native architecture inspection. Native First Mover manifest creation and import were checked on a synthetic directory fixture, including tamper rejection. The published HereticBench reference retention and epistemic raw files matched their published SHA-256 values; all 480 native semantic/format grade pairs replayed exactly. No new weight conversion or inference was performed.


## One workspace for the native tools

Copy tools.example.json to a private local tools.json and set the paths to your trusted native checkouts. You can configure just the tools you use. Relative paths are resolved against the configuration file directory. Explicit command arguments override configured paths.

    ramgpt --tools-config tools.json tools
    ramgpt --tools-config tools.json build original-f16.gguf --out output/model.gguf --default Q4_K --execute
    ramgpt model-info
    ramgpt model-info ARTIFACT_SHA256
    ramgpt model-info exl3:ARTIFACT_ID

The catalog checks path presence, not compatibility. Keep the same --db and --evidence-db for related operations. Model-info lists known models or shows their producer receipts, managed evaluations, weight evidence, actual recipe/repair derivatives and separate comparison relationships. Unauthenticated historical paired reports are not attached to model hashes. EXL3 identity is explicitly namespaced.

Build delegates to native Q4Fit with llama-quantize, using a floating-point single-file GGUF input. It requires --execute and a fresh output directory, supports native --set REGEX=TYPE overrides, optional --imatrix and --threads, and verifies the native output recipe before registering the model. If a build fails, keep its files for inspection; no automatic overwrite, upload, quality claim or inference. Native dependencies are not bundled. The direct production adapter was exercised on Qwen3-0.6B F16: the native CPU quantizer completed in 7.33 seconds and reproduced the previous Q4_K file SHA-256 exactly.

Existing lower-level commands continue to work. The config supplies native paths to build, healer-compose, healer-import, ladder-import, exl3-preflight, exl3-import, heretic-import, weight-import, runtime-prepare, prepare and router-parity. The standalone TrustMeBro catalog entry uses the existing BudgetFit-integrated evaluator, not a new independent executable.


## Local EXL3 construction and native IFEval replay

    ramgpt --tools-config tools.json exl3-build /path/to/local-unquantized-safetensors --out output/exl3 --gpu-uuid GPU_UUID --bits 4 --execute
    ramgpt --tools-config tools.json ladder-import model.gguf responses.jsonl --track ifeval --limit 120

For EXL3, configure the actual ExLlamaV3 checkout and its working Python environment, including any required native-library environment variables. The new local producer delegates directly to the official backend convert.py instead of the First Mover job wrapper, which currently references an absent exl_convert.py. It accepts a materialized, unquantized local source directory, checks architecture and available disk space, requires an explicit GPU and fresh output directory, and registers the result with the native First Mover manifest reader. Source-directory identity is recorded separately from Hugging Face revision metadata. Files are in output/exl3/artifact; logs and work files are retained alongside. No model download, upload or automatic runtime certification occurs. The actual backend help command worked in the tested environment with a process-level system libstdc++ preload; the conversion/handoff path was tested with a synthetic converter fixture, not a real GPU conversion.

For IFEval, configure the upstream scorer checkout via ifeval_root and optionally an existing dependency directory and NLTK resource directory (or pass the corresponding CLI options). Omit --limit for the official full 541-prompt dataset; pass the exact original subset size for historical subsets. Import requires the native raw JSONL and its adjacent manifest/report, matching current model path and size, exact dataset and upstream source hashes, and native report parity. One explicit legacy schema migration accepts a missing report_version if the replay only adds measurement-report-v0.5.1. Any changed score is rejected. This was exercised on the original 120-prompt Q4 run, with 84.17% strict and 87.50% loose prompt scores, no additional inference. Historical model identity is not independently authenticated by the scorer. Live IFEval execution is not part of this adapter.


## Shared GGUF and EXL3 native evaluation

    ramgpt --tools-config tools.json exl3-run /path/to/imported-exl3 --dataset tasks.json --gpu-uuid GPU_UUID --out output/exl3-eval --budgets 256 1024 --strategies allocated --reserve 64 --max-requests 16 --max-completion-tokens 10240 --execute
    ramgpt cross-compare GGUF_EVALUATION_ID EXL3_REPORT_ID
    ramgpt model-info exl3:ARTIFACT_ID

Use the same native manifest, budget matrix and pinned evaluator as the managed GGUF run. The EXL3 adapter verifies the registered artifact manifest, uses the configured ExLlamaV3 Python environment and native Generator/Job/GreedySampler, checks the context before generating, and sends the original task text to the existing BudgetFit/TrustMeBro scorer. This initial text adapter supports Qwen3 only, rendering the local Qwen3 chat template with a sandboxed Jinja environment and the explicit non-thinking prefix. It does not load Transformers or execute remote tokenizer code. Other architectures/templates are rejected. The worker records native completion count, termination, prefill/decode timings and possible cached tokens. No automatic retry, model download or publish. Use explicit --execute and a new output directory. The request/token caps cover generation jobs, not model load/warmup, prompt tokens or elapsed time.

Cross-compare requires the same task manifest, exact native prompt schedule, evaluator and scoring adapter, and the same budget/strategy/reserve matrix. The two model formats keep distinct identities, and EXL3 results stay in native reports rather than entering GGUF-only optimization contracts. The comparison exposes separate profile values; template, tokenizer, KV caching, runtime and hardware conditions can differ, so it does not establish weight fidelity, causal speed differences or deployment readiness.

Local validation: the actual ExLlamaV3 Python environment imported the native backend and rendered the Qwen3-0.6B chat template successfully using a process-level system libstdc++ preload. Transformers/huggingface-hub versions were incompatible, which is why this initial text adapter uses the local template directly. Actual archived GGUF responses (16 rows, two profiles) scored identically through the new shared adapter; a temporary simulated EXL3 report exercised cross-compare. No real EXL3 GPU inference was performed in this increment, because the prior local EXL3 weights had been removed. The new producer/runner requires real-model validation before being considered production-ready.


## Choose from the whole model catalog

    ramgpt recommend --vram 24GiB --min-accuracy 0.85 --objective memory
    ramgpt recommend --vram 24GiB --max-model-size 18GiB --objective balanced --context 4096
    ramgpt recommend --vram 24GiB --candidate GGUF_SHA256 --candidate exl3:EXL3_ID
    ramgpt tool-sync /path/to/other/evidence.sqlite

Recommend screens the existing model catalog without rehashing model weights or running inference. Use --manifest and --gpu-name to restrict observations to a workload and hardware name. The result lists native producer/evaluation references, missing evidence, candidate rejections and an exploratory Pareto frontier per compatible cohort. It does not rank unrelated workloads or mix backend-specific timing/token measurements into a universal score. Historical failed holdouts remain exclusions through the existing GGUF planner. Fidelity limits require the existing matched QuantBench cohort. Native knowledge/IFEval receipts are shown as separate evidence, not treated as substitutes for managed accuracy or memory.

New EXL3 runs collect device-wide sampled NVML peak, free memory and sampling gaps. This is a lower bound on instantaneous peak and attributed conservatively to the entire session. EXL3 remains ineligible for resource-constrained recommendations until its runtime library provenance is frozen; old runs without telemetry are also excluded. No real EXL3 GPU measurement was performed in this increment.

Tool-sync takes a read-only SQLite snapshot of another evidence store and copies only existing native receipts, native reports and weight reports using their original content identities. It does not copy registry paths, managed execution contracts or optimization decisions. Use the same destination --db and --evidence-db for later commands. Repeated sync is idempotent; if interrupted, rerun to complete the copy. The local integration test combined historical managed evidence and the native-tool workspace: 11 known model identities, four distinct observation cohorts, nine imported report/receipt records and zero new inference.


## Four common commands

The common entry points are build, run, recommend and passport. The original producer, evaluator and contract commands remain available.

    ramgpt --tools-config tools.json build original-f16.gguf --out output/model.gguf --execute
    ramgpt --tools-config tools.json build /path/to/local-source --out output/exl3 --gpu-uuid GPU_UUID --bits 4 --execute
    ramgpt run model.gguf --spec model-specific-spec.json --seed 931 --budgets 256 --out output/measurement --max-requests 9 --max-completion-tokens 2112
    ramgpt run MEASUREMENT_PLAN_ID --execute
    ramgpt --tools-config tools.json run /path/to/imported-exl3 --dataset tasks.json --gpu-uuid GPU_UUID --out output/eval --budgets 256 --max-requests 8 --max-completion-tokens 2048 --execute
    ramgpt recommend --vram 24GiB --min-accuracy 0.85 --objective memory
    ramgpt passport MODEL_SHA256 --out model.passport.json
    ramgpt passport-verify model.passport.json --local

Build recognizes a floating-point GGUF input file or a materialized unquantized source directory. It delegates to Q4Fit or the official ExLlamaV3 converter respectively; it does not implicitly convert a Hugging Face directory to GGUF. Incompatible backend-specific options are rejected. Use --format to require a particular input type.

Run accepts a GGUF path plus its model-specific base spec (including the native template, runtime and GPU settings), a registered measurement plan ID, an imported EXL3 directory plus dataset, or an existing frozen contract ID. GGUF model preparation defaults to a single f16 KV arm; --kv can select more arms. Without --execute, the GGUF model command prepares a bounded plan only. Existing direct contract execution remains backward compatible (requires --out, does not newly require --execute); use --archive for complete native evaluations. EXL3 requires explicit execution, and its adapter remains Qwen3-only. No new benchmark is introduced.

The passport alias now defaults to the consolidated v2 summary. The existing passport-export command retains its v1 GGUF output by default; --consolidated requests v2. V2 accepts a GGUF path/hash or EXL3 namespaced identity, and a registered EXL3 directory with the configured native manifest script. It preserves the existing GGUF passport where available and includes selected native producer/scoring summaries, explicitly missing axes, and the EXL3 directory manifest when applicable. Unbound historical paired reports are not attached to arbitrary model identities. Raw responses, local paths and device UUIDs are excluded. Passport-verify accepts both schemas and supports local snapshot verification and optional model byte verification. These are not signed certificates.

Local checks: common GGUF run created a real native frozen plan with 8 workload requests plus 1 smoke request and a 2112-token completion cap, with zero inference. The consolidated passports for the existing repaired 27B, Q4 IFEval and Q2 managed runtime evidence exported successfully; the repaired and Q2 local snapshots verified. EXL3 dispatch and manifest verification were tested on fixtures; actual EXL3 GPU execution remains pending.


## Real EXL3 engineering smoke and automatic setup

The local Qwen3-0.6B source was converted by the official ExLlamaV3 converter through the common build entry point. The 4 bpw target uses native default head/embedding precision. For this engineering pilot only, explicit --cal-rows 16 --cal-cols 512 were used instead of the normal native calibration defaults; these options are recorded in the conversion report and consolidated passport. The converter completed in about 63 seconds including adapter checks, produced a 666,003,210-byte directory, and returned the GPU to baseline. The same artifact then completed eight existing native tasks at budget 256, with 0.385 s mean request latency, 25% joint accuracy and 25% truncation. The poor score is retained; this is not a representative EXL3 quality benchmark. No extra model download or repeated calibration was performed.

The EXL3 passport verified both the local evidence snapshot and every artifact file. The execution required an already working backend Python environment and a process-level system libstdc++ preload on the test host. No global Python dependencies were modified. Runtime mapped-library provenance and independent validation are still missing, so this pilot remains excluded from resource-constrained recommendations.

Discovery is bounded to explicitly supplied roots, with a default depth of three (maximum five), a directory-count limit and no symlink traversal. It recognizes known tool layouts, checks the pinned BudgetFit archive, skips generated Q4Fit build copies, and reports ambiguous paths instead of silently picking one. It never executes discovered native code, changes global dependencies, overwrites a configuration or downloads anything. On the integration host all nine tool paths were resolved using discovery and explicit overrides for the standalone ConceptLadder checkout and preferred quantizer.


## Generate a GGUF runtime spec

Configure llama-autotune.runtime with your trusted llama-server executable and budgetfit.evaluator with the pinned native archive, or pass --runtime and --evaluator explicitly.

    ramgpt runtime-spec model.gguf --gpu-uuid GPU_UUID --out runtime --execute
    ramgpt run model.gguf --spec runtime/spec.json --seed 931 --budgets 256 --out measurement --max-requests 9 --max-completion-tokens 2112
    ramgpt run MEASUREMENT_PLAN_ID --execute

Runtime-spec explicitly loads the model on the selected idle GPU, checks loopback listener ownership, captures the native apply-template response and stops the server. It issues no completion requests. The generated specification uses explicit single-GPU, non-thinking, greedy defaults (4096 context, eight CPU threads, full GPU offload, f16 KV); configure context, threads, GPU layers and port if needed. It records the input fingerprints and the common run command checks that they still match before preparation. It does not certify the model's literal smoke or evaluation behavior. Existing hand-written specs remain supported.

Fresh-install check: a new Python virtual environment installed the development wheel offline without optional dependencies, registered an existing Qwen3-0.6B GGUF, prepared a bounded 8-task plus 1-smoke plan from the real captured template, correctly returned no eligible recommendation without evidence, and exported and verified a consolidated passport against both the local snapshot and actual model bytes. No completion requests were made. A GPU execution environment still needs the optional GPU dependency and native runtime.
