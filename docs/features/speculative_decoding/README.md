# Speculative Decoding

Nabla retains target-compatible DFlash/DFlash2, DSpine, DSpark and MTP
implementations for its [target models](../../models/supported_models.md).
Select a draft checkpoint compatible with the target architecture, tokenizer
and parallel layout. The fixed H20/H100 serving baselines disable speculation;
retained methods require their own output, acceptance and latency validation.

To train your own draft models for optimized speculative decoding, see [vllm-project/speculators](speculators.md) for seamless training and integration with vLLM.

## vLLM Speculation Methods

The method guides below describe inherited interfaces and algorithms.
Their presence does not imply that every example model remains registered in
Nabla, or that a method improves the selected target workload.

- [EAGLE](eagle.md)
- [LiLiCorr](lilicorr.md)
- [DSpine](#dspine)
- [Multi-Token Prediction (MTP)](mtp.md)
- [DFly](dfly.md)
- [Draft Model](draft_model.md)
- [N-Gram](n_gram.md)
- [Suffix Decoding](suffix.md)
- [Hidden State Extraction](extract_hidden_states.md)
- [Custom Speculators (Experimental)](#custom-speculators-experimental)
- [Dynamic Speculative Decoding](dynamic_speculative_decoding.md)
- [Adaptive Verification](adaptive_verification.md)
- [FLy Verification (Lossy)](fly.md)
- [Per-Request Acceptance Metrics](acceptance_metrics.md)

## Selecting and measuring a method

Measure the selected target/draft pair with the same weights, sampling,
context lengths and parallel layout as the non-speculative baseline. Report
acceptance, failed requests, output checks, TTFT, TPOT and throughput together.
Use the [benchmark CLI guide](../../benchmarking/cli.md) for reproducible commands.
Examples naming other target families below are inherited algorithm references;
check the native registry before adopting one.

## DSpine

DSpine drafts tokens in parallel with adjacent causal injection and
predecessor-conditioned refinement. DSpine checkpoints use the DFlash serving
path. Checkpoint metadata selects the DSpine model.

```bash
vllm serve Qwen/Qwen3.8-27B \
  --language-model-only \
  --speculative-config '{
    "method": "dflash",
    "model": "tencent/Qwen3.8-27B-DSpine-b16",
    "num_speculative_tokens": 15
  }'
```

Set `num_speculative_tokens` to the checkpoint's block size minus one.
Draft sampling defaults to `greedy`; set `draft_sample_method` to
`probabilistic` for probabilistic proposals.
Set `enable_adaptive_verification` to `true` in the speculative config to
enable [adaptive verification](adaptive_verification.md).

## DFlash and DFlash2 context KV updates

On CUDA, Qwen-based DFlash and DFlash2 draft models automatically batch native
FP16 or BF16 context KV cache writes across layers when using FlashAttention or
FlashInfer and the cache dtype matches the draft activations. No additional
configuration is required. Each layer retains its own
slot mapping, including mixed sliding-window and full-attention draft layers.
The optimization applies to eager execution and CUDA graphs.

The shared context write path also serves Qwen DSpark, Gemma4 DSpark, Laguna
DFlash, LiLiCorr, XPress, and Ricursive DFlash2. Muse Spark DSpark uses it for
its native fallback path. Beam DFlash2 and MiniMax DSpark continue to prefer
their dedicated fused context kernels.

Quantized KV caches, shared KV layers, draft decode context parallelism, other
platforms, and unsupported attention backends keep their existing cache update
path. The draft projections, normalization, RoPE, and acceptance algorithm are
unchanged.

Measure end-to-end gains with the [benchmark CLI](../../benchmarking/cli.md) on
your target and draft model pair. Reducing the number of cache write kernels
does not imply the same reduction in total generation latency.

## Custom Speculators (Experimental)

You can plug in your own speculator by registering it as a speculative method
from a [vLLM plugin](../../design/plugin_system.md), and then using it with
`"method": "<name>"`.

A custom speculator consists of:

- The speculator itself: a subclass of
  `vllm.v1.worker.gpu.spec_decode.speculator.BaseSpeculator`, typically of
  `DraftModelSpeculator` (for a speculator that runs its own draft model, which
  the model runner then loads and allocates draft KV cache for) or
  `TargetDependentARSpeculator` (for EAGLE/MTP-style drafting from the target
  model's hidden states). It's constructed with `(vllm_config, device)`.
- Its config: a subclass of
  `vllm.v1.worker.gpu.spec_decode.speculator_config.SpeculatorConfig`, which
  names the speculator class and describes how it behaves (KV lookahead slots,
  whether it needs the target's hidden states, which features it supports,
  ...). It's used in every vLLM process, including those that never build the
  speculator, so its module shouldn't import the speculator. A speculator that
  behaves like a built-in one should subclass that speculator's config (e.g.
  `Eagle3SpeculatorConfig`), which also determines how its draft model is
  configured from `model`.

```python
# my_package/config.py
from vllm.v1.worker.gpu.spec_decode.eagle.config import Eagle3SpeculatorConfig


class MySpeculatorConfig(Eagle3SpeculatorConfig):
    speculator_class = "my_package.speculator.MySpeculator"
    # Accept options in `speculator_options` (validated in `apply_options`).
    supported_options = frozenset({"speculator_options"})

    @classmethod
    def apply_options(cls, spec):
        options = spec.speculator_options or {}
        if unknown := options.keys() - {"my_option"}:
            raise ValueError(f"Unknown speculator_options: {sorted(unknown)}")


# my_package/__init__.py, registered as a `vllm.general_plugins` entry point
def register():
    from vllm.v1.worker.gpu.spec_decode.speculator_config import (
        register_speculative_method,
    )

    register_speculative_method("my_spec", "my_package.config.MySpeculatorConfig")
```

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "my_spec",
    "model": "<draft-model>",
    "num_speculative_tokens": 3,
    "speculator_options": {"my_option": 1}
  }'
```

The speculator reads its options from
`vllm_config.speculative_config.speculator_options`. Once set up,
`speculative_config.custom_method` is the registered name (`"my_spec"`), and
`speculative_config.method` is the built-in method its config is based on
(`"eagle3"`), so vLLM treats it like that built-in speculator.

A speculator that keeps per-request state can override `BaseSpeculator`'s
no-op request hooks, `add_request` and `on_requests_finished`, and `shutdown`.
An override of `shutdown` must call `super().shutdown()`, which releases the
draft model.
A preempted request's slot is freed without a hook, and the request is added
again when it resumes, so keep such state per slot and reset it in
`add_request`.

## `--speculative-config` schema

Use `--speculative-config` to pass speculative decoding settings as a JSON
object on the CLI:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "draft_model",
    "model": "<draft-model>",
    "num_speculative_tokens": 5
  }'
```

The same keys are accepted from Python via `LLM(..., speculative_config={...})`.
The tables below highlight common user-facing keys accepted in this JSON
object; they are not an exhaustive schema reference.
For more details, see the generated [engine arguments reference](../../configuration/engine_args.md)
and the API docs for [vllm.config.SpeculativeConfig][].

### Common keys

These keys are commonly used across speculative decoding setups, though some
only apply to model-based methods such as `draft_model`, `mtp`, `eagle3`, and
`dflash`.

| Key | Type | Default | Allowed values / meaning |
| --- | --- | --- | --- |
| `method` | `string` | `None` | Speculation method. Common values include `draft_model`, `ngram`, `suffix`, `mtp`, `eagle3`, and `dflash`. If omitted, vLLM infers the method from the provided configuration when possible. |
| `model` | `string` | `None` | Draft model, EAGLE head, or auxiliary model identifier. For `ngram`, `ngram_gpu`, `suffix`, and `mtp`, this can often be omitted. |
| `num_speculative_tokens` | `integer > 0` | `None` | Number of speculative tokens to propose per step. Required for methods that do not infer it from model metadata. |
| `draft_tensor_parallel_size` | `integer >= 1` | `None` | Draft tensor parallel size. For DSpine, this must match the target. |
| `draft_decode_context_parallel_size` | `integer >= 1` | `None` | Decode context parallel size for the draft model: `1` or the target's `decode_context_parallel_size`. Defaults to the target's, except `dflash`/`dspark` drafts, which default to `1` unless they are full-attention MLA. |
| `draft_enable_eplb` | `boolean` | `true` | Enable EPLB for the draft when target EPLB is enabled. Set to `false` to disable EPLB for the draft. |
| `max_model_len` | `integer >= 1` | `None` | Maximum context length for the draft model. A draft limit below the target's may also lower the engine's `max_model_len`. |
| `parallel_drafting` | `boolean` | `null` | Deprecated and has no effect. `dflash` and `dspark` always draft in parallel, and no other method supports it. |
| `rejection_sample_method` | `string` | `standard` | `standard`, `synthetic`, `block`, or the lossy `fly` verifier. |
| `draft_sample_method` | `string` | `greedy` | `greedy` uses target-only acceptance for random requests; `probabilistic` retains the full draft distribution for probability-ratio acceptance. |
| `synthetic_acceptance_rates` | `list[float]` | `None` | Per-position unconditional acceptance rates for `synthetic` rejection sampling. Each entry in `[0, 1]`; length must equal `num_speculative_tokens`; must be non-increasing. |
| `synthetic_acceptance_length` | `float` | `None` | Target mean acceptance length for `synthetic`; in `[1, num_speculative_tokens + 1]`. Mutually exclusive with `synthetic_acceptance_rates`. |
| `fly_window_size` | `integer >= 1` | `min(6, num_speculative_tokens - 1)` | Number of subsequent native acceptance decisions checked by FLy. FLy requires at least two speculative tokens, and the window must be smaller than `num_speculative_tokens`. With `synthetic`, runs FLy's entropy computation without deferring rejections (see [FLy](fly.md#measuring-fly-cost-with-synthetic-acceptance)). |
| `fly_entropy_threshold` | `float >= 0` | `0.3` | Minimum normalized top-k entropy, in `[0, 1]`, required for FLy to defer a native rejection. |
| `fly_entropy_top_k` | `integer >= 1` | `3` | Number of largest processed target scores renormalized for FLy's entropy gate, capped at the vocabulary size. |

For an EPLB-backed MoE draft, set its redundant expert count with
`eplb_config.draft_num_redundant_experts` in addition to configuring
speculative decoding; the count setting is ignored without a speculative MoE
draft. See the
[EPLB deployment guide](../../serving/expert_parallel_deployment.md#multimodel-eplb-with-speculative-decoding).

!!! note
    Gemma 4 assistant checkpoints are handled as Gemma 4 MTP speculators, not
    as generic draft models. Use `"method": "mtp"` with the assistant
    checkpoint in `model`, as shown in the [MTP guide](mtp.md#gemma-4-assistant-models).

    If startup logs show `SpeculativeConfig(method='draft_model', ...)` for a
    Gemma 4 assistant checkpoint, the installed vLLM version does not include
    Gemma 4 MTP support for that path. Upgrade to a version that includes
    Gemma 4 MTP support instead of forcing the assistant checkpoint through
    generic draft-model speculative decoding.

### Method-specific keys

#### N-gram

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt_lookup_max` | `integer >= 1` | `5` if both lookup bounds are omitted; otherwise mirrors `prompt_lookup_min` when omitted | Maximum n-gram window size. |
| `prompt_lookup_min` | `integer >= 1` | `5` if both lookup bounds are omitted; otherwise mirrors `prompt_lookup_max` when omitted | Minimum n-gram window size. |

Example:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 4,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5
  }'
```

#### Suffix decoding

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `suffix_decoding_max_tree_depth` | `integer` | `24` | Maximum suffix match length. |
| `suffix_decoding_max_cached_requests` | `integer` | `10000` | Set `0` to disable the cross-request corpus of finished responses. |
| `suffix_decoding_corpus_tokens` | `integer` | `1048576` | Token capacity of the GPU corpus of finished responses. Set `0` to match only within each request. |
| `suffix_decoding_max_spec_factor` | `float` | `1.0` | Deprecated and has no effect. |
| `suffix_decoding_min_token_prob` | `float` | `0.1` | Deprecated and has no effect. |

Example:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "suffix",
    "num_speculative_tokens": 8,
    "suffix_decoding_max_tree_depth": 24,
    "suffix_decoding_corpus_tokens": 1048576
  }'
```

### Notes

- `--speculative-config` expects a JSON object on the CLI. In YAML config
  files, use a nested mapping instead of an escaped JSON string.
- `tensor_parallel_size` is not a valid key in `speculative_config`. Use
  `draft_tensor_parallel_size` instead.
- Keys such as `temperature` and `top_p` are sampling parameters, not
  `--speculative-config` fields.
- Internal fields such as `target_model_config`, `draft_model_config`,
  `target_parallel_config`, `draft_parallel_config`, and `draft_load_config`
  are populated by vLLM and are not intended to be set by users.

## Speculative Decoding with Pipeline Parallelism

With pipeline parallelism (`pipeline_parallel_size > 1`), the draft model runs as a whole model on the last pipeline stage, as it would at `pipeline_parallel_size=1`. A custom `VLLM_PP_LAYER_PARTITION` applies only to the target model.

## Lossless guarantees of Speculative Decoding

In vLLM, speculative decoding aims to enhance inference efficiency while maintaining accuracy. This section addresses the lossless guarantees of
speculative decoding, breaking down the guarantees into three key areas:

1. **Theoretical Losslessness**
   \- Speculative decoding sampling is theoretically lossless up to the precision limits of hardware numerics. Floating-point errors might
   cause slight variations in output distributions, as discussed
   in [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318)

    !!! warning
        These guarantees apply to the standard rejection sampler. FLy
        deliberately relaxes verification and is not distribution preserving.

2. **Algorithmic Losslessness**
   \- vLLM’s implementation of speculative decoding is algorithmically validated to be lossless. Key validation tests include:

    > - **Rejection Sampler Convergence**: Ensures that samples from vLLM’s rejection sampler align with the target
    >   distribution. [View Test Code](https://github.com/vllm-project/vllm/blob/47b65a550866c7ffbd076ecb74106714838ce7da/tests/samplers/test_rejection_sampler.py#L252)
    > - **Greedy Sampling Equality**: Confirms that greedy sampling with speculative decoding matches greedy sampling
    >   without it. This verifies that vLLM's speculative decoding framework, when integrated with the vLLM forward pass and the vLLM rejection sampler,
    >   provides a lossless guarantee. Almost all of the tests in [tests/spec_decode/e2e](../../../tests/v1/spec_decode)
    >   verify this property using [this assertion implementation](https://github.com/vllm-project/vllm/blob/b67ae00cdbbe1a58ffc8ff170f0c8d79044a684a/tests/spec_decode/e2e/conftest.py#L291)

3. **vLLM Logprob Stability**
   \- vLLM does not currently guarantee stable token log probabilities (logprobs). This can result in different outputs for the
   same request across runs. For more details, see the FAQ section
   titled *Can the output of a prompt vary across runs in vLLM?* in the [FAQs](../../usage/faq.md).

While vLLM strives to ensure losslessness in speculative decoding, variations in generated outputs with and without speculative decoding
can occur due to following factors:

- **Floating-Point Precision**: Differences in hardware numerical precision may lead to slight discrepancies in the output distribution.
- **Batch Size and Numerical Stability**: Changes in batch size may cause variations in logprobs and output probabilities, potentially
  due to non-deterministic behavior in batched operations or numerical instability.

For mitigation strategies, please refer to the FAQ entry *Can the output of a prompt vary across runs in vLLM?* in the [FAQs](../../usage/faq.md).

## Known Feature Incompatibility

1. Pipeline parallelism is not composable with speculative decoding as of `vllm<=0.15.0`
2. Speculative decoding with draft models is not supported in `vllm<=0.10.0`

## Resources for vLLM contributors

- [[vLLM Office Hours #40] Intro to Speculators](https://www.youtube.com/watch?v=2ISAr_JVGLs)
- [A Hacker's Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)
- [What is Lookahead Scheduling in vLLM?](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)
- [Information on batch expansion](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)
- [Dynamic speculative decoding](https://github.com/vllm-project/vllm/issues/4565)
