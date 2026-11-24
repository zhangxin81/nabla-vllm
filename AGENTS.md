# Agent Instructions for nabla-vllm

These instructions apply to `zhangxin81/nabla-vllm`. The runtime targets
Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash on NVIDIA CUDA, including P/D
disaggregation. Python imports remain `vllm`; the primary CLI is `nabla-vllm`.
Start with the [agent development guide](docs/contributing/agents.md).
For contributions to upstream vLLM, read that repository's current instructions.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo zhangxin81/nabla-vllm --comments
gh pr list --repo zhangxin81/nabla-vllm --state open --search "<issue_number> in:body"
gh pr list --repo zhangxin81/nabla-vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Agents may prepare changes and draft PRs for an authorized task. A named human
  maintainer owns the contribution and the merge decision.
- Before merging, the human maintainer reviews every changed line and confirms
  the relevant test results. Agents do not approve their own changes or bypass
  branch rules.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results (for unit tests, reproducible pytest command and exact pytest summary line including time).
    - Model evaluation results when the change affects output, accuracy, or serving.
    - Clear statement that AI assistance was used.
- User-visible changes (features, CLI args, config fields, env vars, models,
  hardware, defaults, deprecations) must update the rendered docs in the same PR;
  follow [Documenting User-Visible Changes](docs/contributing/README.md#documenting-user-visible-changes).
- Before opening a PR (drafts included) or requesting re-review, run the [`pr-checklist`](.agents/skills/pr-checklist/SKILL.md) skill and address its findings.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.14
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# Use a verified native wheel matching this source and Torch/CUDA:
VLLM_PRECOMPILED_WHEEL_LOCATION=/artifacts/matching-native.whl uv pip install -e .
```

For docs, CI and agent configuration, the lightweight setup in the agent
development guide needs no runtime wheel or model weights. For C/C++ or CUDA changes, follow the
[incremental compilation workflow](docs/contributing/incremental_build.md) to
configure and perform incremental builds.

### Tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies (use cuda.in on non-x86_64):
uv pip install -r requirements/test/cuda.in

# Run a specific test file:
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

When adding tests:

- **Design before you write.** Answer four questions first: what is the module
  for, what is its I/O contract, what failure am I guarding against, and what is
  the cheapest level that actually reproduces it? Fixes to user-visible
  behavior need a test through a real entry point (`LLM`, OpenAI server); see
  [`tests/AGENTS.md`](tests/AGENTS.md).
- **Reuse before create.** Extend existing test files, `conftest.py` fixtures, and
  helpers; add a new file only when no nearby suite fits.
- **Test behavior with intent.** Assert observable outcomes through public APIs;
  state why in the name or docstring. Skip trivial wiring; flaky tests are worse
  than no tests.
- **Keep it minimal.** One behavior per test and the smallest setup that
  triggers it; if the test diff dwarfs the code change, cut scope.
- **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in
  `benchmarks/kernels/`; prove correctness in existing pytest suites.
- **Run model evals for model-affecting changes.** Search `tests/evals/` or use
  `vllm bench` and include results in the PR — do not wait for reviewers to ask.

For model-specific requirements, see
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).
Docstrings render as Markdown: write code in single backticks, not double backticks or Sphinx roles like `:class:`, and wrap shapes and indexing like `[block][2][16]` in backticks so mkdocs does not read them as cross-references (the `check-docstring-style` hook enforces this).

### Coding style guidelines

- Match existing code style
- Minimize use of comments. Eliminate comments which are redundant, preferring legible and self-documenting code. When used, keep docstrings and comments brief and direct.
- Assume the reader is familiar with vLLM.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`):

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and
vulnerability process.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.
- **Rust frontend (`rust/`)**: [`rust/AGENTS.md`](rust/AGENTS.md)
  — Crate ownership, design principles, Python parity, and testing conventions.
- **C++/CUDA kernels (`csrc/`)**: [`csrc/AGENTS.md`](csrc/AGENTS.md)
