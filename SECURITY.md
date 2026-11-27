# Security

Run RAMGPT Toolchain only with trusted local model files, executables, native evaluator archives and producer scripts. Some adapters execute supplied Python or native code with the current user's permissions. Python isolated mode is not a security sandbox. Do not pass untrusted executables to runtime discovery or dependency inspection.

Raw evidence, contracts and runtime receipts may contain private prompts, responses and paths. Keep workspace databases and run folders private. Model passports are portable summaries, not signed attestations; review their contents before publishing.

Report suspected vulnerabilities privately using GitHub private vulnerability reporting if enabled. Otherwise open an issue requesting a private reporting channel without posting sensitive details.
