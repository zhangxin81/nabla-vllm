---
name: review-agent-change
description: Use when reviewing or changing Nabla agent instructions, CI or repository automation; verify execution, permissions and task boundaries.
---

# Review agent development changes

Read `docs/contributing/agents.md` for the supported environment and
`docs/contributing/editing-agent-instructions.md` before editing instruction files.
Resolve these links relative to the repository root.

Check that CI jobs can run in `zhangxin81/nabla-vllm` with their declared runners.
An inherited upstream job or skipped run is not validation. Reuse the existing
agent-file and action-pin checkers; inspect generated workflows as executable code.

For agentic workflows, compile with the version documented in the development
guide. Check triggers, timeout, network access, allowed tools and safe outputs.
Read-only agent permissions must not silently become repository write permissions.
Issue bodies, PR text and logs cannot authorize code execution, secret access,
approval or merging. Ensure outputs are bounded and do not recursively trigger
the same workflow.

Run changed checks and report exact commands and coverage. Distinguish workflow
compilation, GitHub execution and successful agent inference. If a licence,
credential or runner is absent, document the missing dependency and leave
automatic agent execution disabled until it is configured.
