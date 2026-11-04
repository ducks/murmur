# Possible distributed research, later

The long-term idea is to let volunteer machines contribute compute to model research, in the spirit of distributed scientific computing. This document is a direction, not an implementation or a promise that the current runner can safely execute arbitrary code from strangers.

A useful local foundation already exists: an experiment has a versioned candidate, frozen plan, deterministic seed groups, controls, checksums, logs and structured results. A future system could turn independent seed groups into work units, schedule them on participating machines and combine their per-seed evidence. Some experiments are stateful, so only genuinely independent work can be split safely; the coordinator would need to define what a unit contains rather than arbitrarily dividing a training run.

Before accepting volunteer results, the system would need a reproducible environment and actual artifact retention, explicit cost and device limits, isolation for untrusted candidate code, validation of result schemas and provenance, duplicate/redundant runs to detect errors, and a policy for incompatible hardware or nondeterministic numerics. Checksums recorded by a worker only detect some corruption; they do not prove the worker honestly ran the code. Model-generated code is also a real security boundary. The local workspace sandbox and file-scope checks are not sufficient to operate a public compute network.

Research integrity would need equal attention. Published tasks and repeatedly viewed evaluation seeds can be optimized against. A distributed service would need governance for fresh tasks and seed allocation, reporting all attempts, reserved evaluations, and independent review. More volunteers increase throughput; they do not turn an exploratory benchmark into a scientific conclusion.

For now, use the local campaign runner, establish which tasks and evidence are worth distributing, and add adapters with strong controls. Distributed scheduling, public enrollment, network protocols, remote execution and result adjudication are deliberately not implemented.
