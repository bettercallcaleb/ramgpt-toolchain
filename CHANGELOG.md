# Changelog

## 0.3.0 (development)

- Minimal verified GGUF dependency setup and executable option checks.
- Versioned transport smoke diagnostics; legacy strict contracts unchanged.
- Bounded AutoOptimizer orchestration for KV precision and generation budgets, with independent holdout and selected profile export.
- Separate Q4Fit recipe-builder and evidence-script configuration.


## 0.2.0rc1

Release candidate for the unified local model toolchain.

- Common build, run, recommend and passport commands for GGUF and EXL3.
- Nine native tool adapters, bounded local discovery and configuration.
- Native Q4Fit and GGUF-Healer production; official ExLlamaV3 local conversion.
- Native historical ConceptLadder knowledge/IFEval and HereticBench replay.
- Shared BudgetFit/TrustMeBro task scoring, model catalog and evidence synchronization.
- GGUF runtime template capture and bounded measurement/validation workflow.
- Consolidated Model Passport V2 with legacy V1 export compatibility.
- Explicitly separate candidate cohorts and preserve failed validation outcomes.
- EXL3 conversion and Qwen3 generation exercised on a small local model; not yet eligible for automated recommendations.

Not included: automatic tool/model downloads, generalized live IFEval, broad EXL3 architecture support, cross-format automatic optimization or deployment certification.

## 0.1.0

Initial research alpha: model registry, managed GGUF inference, native evaluation, evidence store, runtime and budget selection, Model Passport V1.
