# Developing nabla-vllm with agents

nabla-vllm uses GitHub issues for bounded implementation tasks, discussions for
architecture and RFCs, and pull requests for reviewable changes. The same shared
instructions and validation expectations apply to local agents and GitHub agents.
The runtime scope remains Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash on NVIDIA
CUDA, including P/D serving. Agent support changes how the repository is developed;
it does not establish new inference features or performance results.

## Give an agent a reproducible task

Use the **Agent development task** issue form. State the observable outcome,
affected area, human maintainer, acceptance commands and available resources.
Search existing issues and PRs before starting overlapping work. Put open design
questions in [Discussions](https://github.com/zhangxin81/nabla-vllm/discussions).

For example, a CI task should identify the failing workflow and the expected
executed checks. A kernel task should identify the target GPU, shape and dtype,
the correctness test and the matched baseline measurement.

Agents may implement authorized tasks and prepare draft PRs. A human maintainer
reviews the changed lines and validation before merging. Include AI assistance,
the duplicate-work search, source SHA, exact commands and results in the PR.
Missing GPU validation stays visible; a skipped job is not a passing test.

## Start from shared instructions

Read [AGENTS.md](../../AGENTS.md) and the applicable directory guides. The
repository keeps reusable workflows under `.agents/skills`, with compatibility
symlinks under `.claude/skills`. Existing skills cover kernel and model
microbenchmarks, Triton kernels and PR review. Additional skills review
performance evidence and agent/CI changes.

GitHub-specific configuration consists of:

| File | Purpose |
| --- | --- |
| `.github/copilot-instructions.md` | Short Copilot-specific guidance pointing to shared rules |
| `.github/instructions/*.instructions.md` | Rules selected by affected paths |
| `.github/agents/implementer.md` | Implement an authorized task |
| `.github/agents/reviewer.md` | Review source and evidence with read-only tools |
| `.github/agents/performance-analyst.md` | Analyze benchmark artifacts and plan validation |

Custom agents are selected in the GitHub agent interface or a compatible CLI/IDE.
They do not grant permissions, provide GPU resources or replace the submitting
human. See GitHub's [customization support](https://docs.github.com/en/copilot/reference/customization-cheat-sheet)
for differences between clients.

## Prepare a lightweight development environment

Docs, CI and agent configuration can be checked without CUDA, model weights or
installing the inference runtime:

```bash
uv venv --python 3.14
uv pip install --python .venv/bin/python -r requirements/lint.txt PyYAML regex
.venv/bin/pre-commit install
.venv/bin/python tools/pre_commit/check_agent_files.py
git ls-files -z '.github/workflows/*.yml' '.github/workflows/*.yaml' \
  | xargs -0 .venv/bin/python tools/pre_commit/check_github_actions_pins.py
.venv/bin/python .github/workflows/scripts/test_run_ci_command.py
.venv/bin/pre-commit run --files <changed-files>
```

Preserve an existing managed `core.hooksPath` rather than replacing system hooks.
If pre-commit installation reports a conflict, run pre-commit explicitly and
configure integration through the owner of those hooks.

The [Copilot setup workflow](../../.github/workflows/copilot-setup-steps.yml)
prepares these tools for Copilot sessions running on GitHub Actions. It must be
on the default branch to be used and does not configure cloud-sandbox sessions.
It deliberately does not install an upstream nightly runtime wheel.

For inference changes, use the [CUDA build instructions](../getting_started/installation/gpu.md)
and [incremental build guide](incremental_build.md), with a source-matched native
build or verified wheel. Reuse authorized checkpoint caches and GPU resources.
Keep kernel correctness, model output/accuracy and serving measurements separate.

## Run and interpret repository checks

`pre-commit` checks the files changed by a PR or push on GitHub-hosted Ubuntu.
`repository-checks` validates shared agent files, action pins and the existing
CI-command unit suite. `agent-workflows` strictly compiles the Markdown workflows
and rejects drift in generated files. Actionlint checks all workflows, with
path-scoped exceptions for three modern syntax diagnostics in generated files;
the gh-aw compiler checks that syntax using its current schema.
These jobs do not require upstream self-hosted runners
or upstream CI labels. Both are suitable branch-rule checks once their first
GitHub runs pass.

Inherited upstream workflows remain scoped to upstream services. Lightweight
checks do not build CUDA kernels or validate model output, P/D continuation or
throughput. Include the required external GPU evidence in runtime PRs rather
than treating these checks as GPU CI.

## Configure GitHub review and access

Repository rules request Copilot review for new PRs and new pushes. The repository
uses Balanced review effort with custom instructions and MCP enabled. Actual
review depends on the requester's Copilot eligibility and available credits.
Human review and required checks control merging; agent approvals do not replace
the maintainer's review.

Copilot cloud access is controlled under the owner's **Copilot settings → Cloud
agent → Repository access**. Claude and Codex partner agents depend on account
eligibility. Repository admins should retain the cloud agent's built-in quality
and security validation and its workflow-run approval requirement.

The default GitHub MCP server provides current-repository context. Configure
external MCP servers only for an identified shared service, with explicitly
selected tools and service credentials. Repository-level MCP settings and local
CLI MCP configuration are separate surfaces; no external GPU or private-document
service is implicitly connected by this setup.

## Enable ongoing maintenance after configuring inference

The repository includes four [GitHub Agentic Workflows](https://github.github.com/gh-aw/):

| Workflow | Trigger | Bounded output |
| --- | --- | --- |
| `agent-issue-triage` | New issue | One comment about missing information or overlapping work |
| `agent-pr-evidence` | PR opened, updated or ready | One validation-evidence comment |
| `agent-ci-analysis` | Failed lightweight check on `main` | One issue for an untracked causal failure |
| `agent-upstream-watch` | Weekly or manual | One issue about new relevant upstream changes |

Agents have read-only repository permissions. Separate safe-output handlers
publish bounded comments or issues. They cannot merge, approve, change source
or run PR code. All workflows have a ten-minute timeout. Issue/PR/log text is
untrusted data, and it cannot authorize additional tools or access.

The compiled files are generated with **gh-aw v1.2.2**. Edit the Markdown source,
compile and commit both the source and generated `.lock.yml` files:

```bash
gh extension install github/gh-aw --pin v1.2.2
gh aw version
gh aw compile --no-check-update --strict --schedule-seed zhangxin81/nabla-vllm
```

Use the documented compiler version when reproducing the generated files. Review
compiler upgrades and changed action/container pins as executable changes.

For this personal-account repository, Copilot inference requires an eligible
Copilot account and a fine-grained PAT with the account **Copilot Requests: Read**
permission, stored as `COPILOT_GITHUB_TOKEN`. The existing GitHub CLI OAuth token
is not suitable. See [gh-aw authentication](https://github.github.com/gh-aw/reference/auth/).
Create credentials through GitHub's account UI; never put them in an issue or PR.

Automatic inference is off until a maintainer supplies the dedicated credential:

```bash
gh secret set COPILOT_GITHUB_TOKEN --repo zhangxin81/nabla-vllm
gh variable set NABLA_AGENT_AUTOMATION --body enabled --repo zhangxin81/nabla-vllm
```

The first command prompts for the secret without placing it in command arguments.
Keep `NABLA_AGENT_AUTOMATION` set to `disabled` until licence and credentials are
available. To stop future invocations, set that variable back to `disabled`;
already-running jobs must be cancelled separately. Monitor the first real runs
and their credit use before expanding triggers or output permissions.

GitHub's separate **Copilot Automations** feature currently supports private and
internal repositories, so this public repository uses versioned Agentic Workflows.

## Contribute upstream separately

Search the current upstream issues and PRs before adapting a Nabla change for
vLLM. Follow the upstream checkout's current contribution instructions and
validation gates. A downstream merge or downstream agent workflow does not grant
permission to bypass upstream policies.
