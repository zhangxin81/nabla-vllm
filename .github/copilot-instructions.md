# Copilot in nabla-vllm

Read `AGENTS.md` for shared project requirements and
`docs/contributing/agents.md` for setup, validation and task routing.
Reuse `.agents/skills` when a task matches their descriptions.

During review, check whether CI actually executed rather than being skipped.
Trace runtime claims to the source, model, hardware and workload tested.
Use the `review-performance-evidence` skill for performance claims and
`review-agent-change` for changes to automation or agent configuration.
