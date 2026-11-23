# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.states import RequestState

from ...utils import spawn_new_process_for_each_test


def _text_config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        # `Qwen4ExpTextConfig` requires an EOS token whenever PLE is enabled.
        "eos_token_id": 1,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True},
    }
    values.update(kwargs)
    return Qwen4ExpTextConfig(**values)


def _ple_state(tmp_path, depth):
    config = _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        vocab_size=128,
        eos_token_id=99,
        ngram_size=depth + 1,
        heads_per_ngram=1,
        ple_embed_dim=depth * 8,
    )
    config.save_pretrained(tmp_path)
    vllm_config = EngineArgs(
        model=str(tmp_path),
        max_model_len=8,
        max_num_seqs=8,
        max_num_batched_tokens=32,
        skip_tokenizer_init=True,
    ).create_engine_config()
    device = torch.device("cuda")
    state = Qwen4ExpModelState(vllm_config, torch.nn.Module(), None, device)
    requests = RequestState(8, 8, 32, 0, 128, device, use_dense_all_token_ids=True)
    return state, requests, InputBuffers(8, 32, device)


def _ple_batch(buffers, num_reqs, padded, offsets, mapping):
    batch = InputBatch.make_dummy(num_reqs, offsets[-1], buffers)
    batch.num_reqs_after_padding = padded
    batch.idx_mapping = torch.tensor(mapping, dtype=torch.int64, device=buffers.device)
    batch.query_start_loc = torch.tensor(
        offsets, dtype=torch.int32, device=buffers.device
    )
    return batch


def test_sm90_prefill_graph_coverage_preserves_explicit_limits(tmp_path):
    config = Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=_text_config(ple_layer_ids=[]),
    )
    config.save_pretrained(tmp_path)
    args = dict(
        model=str(tmp_path),
        max_model_len=2048,
        max_num_seqs=128,
        max_num_batched_tokens=1024,
        language_model_only=True,
    )
    resolved = EngineArgs(**args).create_engine_config().compilation_config
    if current_platform.is_cuda() and current_platform.is_device_capability_family(90):
        assert resolved.max_prefill_cudagraph_capture_tokens == 1024
        assert resolved.max_decode_cudagraph_capture_tokens == 512
        assert resolved.prefill_cudagraph_capture_token_sizes is not None
        assert resolved.decode_cudagraph_capture_request_sizes is not None
        assert 1024 in resolved.prefill_cudagraph_capture_token_sizes
        assert 40 in resolved.decode_cudagraph_capture_request_sizes
    explicit = (
        EngineArgs(
            **args,
            compilation_config={
                "max_prefill_cudagraph_capture_tokens": 512,
                "decode_cudagraph_capture_request_sizes": [1, 8, 32, 40, 64, 128],
            },
        )
        .create_engine_config()
        .compilation_config
    )
    assert explicit.max_prefill_cudagraph_capture_tokens == 512
    assert max(explicit.prefill_cudagraph_capture_token_sizes) == 512
    assert explicit.decode_cudagraph_capture_request_sizes == [1, 8, 32, 40, 64, 128]


@pytest.mark.parametrize("workers, expected", [(None, 4), (1, 1), (8, 8)])
def test_renderer_workers_resolve_for_target_generation(tmp_path, workers, expected):
    """Qwen generation parallelizes rendering while respecting an explicit size."""
    config = Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=_text_config(ple_layer_ids=[]),
    )
    config.save_pretrained(tmp_path)
    worker_args = {} if workers is None else {"renderer_num_workers": workers}
    args = EngineArgs(
        model=str(tmp_path),
        tokenizer_mode="skip",
        max_model_len=128,
        language_model_only=True,
        **worker_args,
    )
    model_config = args.create_model_config()
    assert model_config.renderer_num_workers == expected


def test_moe_sp_rejects_pipeline_parallel(tmp_path) -> None:
    """Automatic MoE SP must validate HC topology even without --enable-hc-sp."""
    Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=_text_config(num_experts=4, ple_layer_ids=[]),
    ).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="MoE SP requires PP=1"):
        EngineArgs(
            model=str(tmp_path),
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            enable_expert_parallel=True,
            nnodes=4,
            all2all_backend="allgather_reducescatter",
            language_model_only=True,
        ).create_engine_config()


@spawn_new_process_for_each_test
@pytest.mark.parametrize("backend", ["nvidia"])
@pytest.mark.parametrize("use_sp", [False, True])
def test_qwen4_exp_mtp_returns_sample_and_multi_streams(
    backend: str, use_sp: bool
) -> None:
    """MTP returns complete, contiguous outputs after gathering padded SP shards."""
    mtp_module = f"vllm.models.qwen4_exp.{backend}.mtp"
    model = object.__new__(import_module(mtp_module).Qwen4ExpMultiTokenPredictor)
    model.use_hc_sequence_parallel = use_sp
    model.use_sequence_parallel = False
    torch.nn.Module.__init__(model)
    model.hc_count = 2
    model.hidden_size = 4
    model.num_mtp_layers = 1
    if backend == "nvidia":
        # The nvidia forward passes the norm weights to the fused Triton
        # kernel (bypassed below): the norms only carry weights here.
        class NormStub(torch.nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(width))
                self.variance_epsilon = 1e-6

        model.pre_fc_norm_embedding = NormStub(model.hidden_size)
        model.pre_fc_norm_hidden = NormStub(model.hc_count * model.hidden_size)
    else:
        model.pre_fc_norm_embedding = torch.nn.Identity()
        model.pre_fc_norm_hidden = torch.nn.Identity()
    model.fc_embedding = torch.nn.Identity()
    model.fc_hidden = torch.nn.Identity()

    def decoder(**kwargs):
        """Check that the embedding residual shares the hidden state's token slice."""
        hidden = kwargs["hidden_states"]
        if backend == "nvidia":
            prev_block_output = kwargs["prev_block_output"]
            assert prev_block_output.shape == (hidden.shape[0], model.hidden_size)
        return hidden, hidden, torch.zeros(hidden.shape[0], model.hc_count)

    model.layers = [decoder]
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda hidden_states, block_output, injection: (
            hidden_states,
            hidden_states.unflatten(-1, (2, 4)).mean(-2),
            None,
        ),
    )
    multi_hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    expected_sample = multi_hidden.unflatten(-1, (2, 4)).mean(-2)

    def gather(local):
        """Supply rank 1's last token and padding for a simulated two-rank gather."""
        assert local.shape[0] == 2
        remote = torch.cat([expected_sample[2:], multi_hidden[2:]], dim=-1)
        return torch.cat([local, remote, torch.full_like(remote, float("nan"))])

    with (
        patch(f"{mtp_module}.sp_shard", side_effect=lambda x: x[:2]),
        patch(f"{mtp_module}.sp_all_gather", side_effect=gather),
        # The fused norm kernel is CUDA-only; it has dedicated kernel tests.
        (
            patch(
                f"{mtp_module}.fused_pre_fc_norms_split",
                side_effect=lambda embeds, hidden, *args: (embeds, hidden),
            )
            if backend == "nvidia"
            else nullcontext()
        ),
    ):
        sample_hidden, returned_multi_hidden = model.forward(
            input_ids=None,
            positions=torch.arange(3),
            hidden_states=multi_hidden,
            inputs_embeds=torch.zeros(3, 4),
        )

    torch.testing.assert_close(sample_hidden, expected_sample)
    torch.testing.assert_close(returned_multi_hidden, multi_hidden)
    assert sample_hidden.is_contiguous()
    assert returned_multi_hidden.is_contiguous()


@spawn_new_process_for_each_test
@pytest.mark.parametrize("backend", ["nvidia"])
def test_qwen4_exp_mtp_remaps_mixed_precision_layer_indices(backend: str) -> None:
    mtp_module = import_module(f"vllm.models.qwen4_exp.{backend}.mtp")

    quantized_layers = {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        "mtp.layers.0.mlp.experts": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "group_size": 128,
        },
    }

    assert mtp_module._remap_quantized_layers(quantized_layers, 48) == {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        "mtp.layers.48.mlp.experts": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "group_size": 128,
        },
    }


@spawn_new_process_for_each_test
@pytest.mark.parametrize("backend", ["nvidia"])
def test_qwen4_exp_mtp_remaps_quark_layer_names(backend: str) -> None:
    from vllm.model_executor.layers.fused_moe import (
        RoutedExperts,
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    mtp_module = import_module(f"vllm.models.qwen4_exp.{backend}.mtp")

    # `re:mtp\.fc_hidden`, not a catch-all `re:mtp\..*`: a broad regex would
    # already exclude layer 48's experts on its own (`re:` entries are never
    # remapped), masking the bug this test is meant to catch.
    exclude = ["lm_head", "mtp.layers.0.mlp.experts.0.gate_proj", r"re:mtp\.fc_hidden"]
    checkpoint_config = {
        "exclude": exclude,
        "layer_quant_config": {"mtp.layers.0.self_attn.q_proj": {}},
    }
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=object())
    )
    with (
        patch.object(
            mtp_module,
            "get_draft_quant_config",
            return_value=QuarkConfig(checkpoint_config),
        ),
        patch.object(mtp_module, "replace", return_value=SimpleNamespace()),
    ):
        draft_config = mtp_module._make_draft_vllm_config(vllm_config, 48, 1)

    remapped = draft_config.quant_config.quant_config
    assert remapped["exclude"] == [
        "lm_head",
        "mtp.layers.48.mlp.experts.0.gate_proj",
        r"re:mtp\.fc_hidden",
    ]
    assert remapped["layer_quant_config"] == {"mtp.layers.48.self_attn.q_proj": {}}
    assert exclude[1] == "mtp.layers.0.mlp.experts.0.gate_proj"

    # Without the remap, layer 48's experts fall through to the global MXFP4
    # scheme instead of being treated as unquantized bf16 -- assert the
    # decision `get_quant_method` actually makes, not just the exclude list.
    _, _, method_cls = draft_config.quant_config.get_quant_method_target(
        "mtp.layers.48.mlp.experts", RoutedExperts
    )
    assert method_cls is UnquantizedFusedMoEMethod


@pytest.mark.parametrize("wrapped_config", [False, True])
def test_qwen4_exp_mtp_override_sets_draft_config(
    wrapped_config: bool,
) -> None:
    text_config = _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        index_share_for_mtp_iteration=True,
    )
    config = (
        Qwen4ExpConfig(
            architectures=["Qwen4ExpForConditionalGeneration"],
            text_config=text_config,
        )
        if wrapped_config
        else text_config
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.index_share_for_mtp_iteration is True
    assert draft_config.model_type == "qwen4_exp_mtp"
    assert draft_config.architectures == ["Qwen4ExpMTP"]
    assert draft_config.hc_mult == 2
    assert draft_config.n_predict == 1


def test_qwen4_exp_model_state_prepares_ngram_context(tmp_path) -> None:
    model_state, req_states, buffers = _ple_state(tmp_path, 3)
    for req, tokens, computed in ((0, [1, 2, 3, 4], 3), (1, [20, 21, 22, 23], 1)):
        req_states.all_token_ids.stage_write(req, 0, tokens)
        req_states.num_computed_tokens.stage_write_elem(req, computed)
    req_states.apply_staged_writes()
    input_batch = _ple_batch(buffers, 2, 3, [0, 2, 3, 3], [1, 0])
    model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc = torch.full((9,), 3, dtype=torch.int32)
    expected_query_start_loc[0] = 0
    expected_query_start_loc[1] = 2
    torch.testing.assert_close(
        model_inputs["query_start_loc"].cpu(), expected_query_start_loc
    )
    expected_context = torch.full((8, 3), 99, dtype=torch.int32)
    expected_context[:2] = torch.tensor([[99, 99, 20], [1, 2, 3]])
    torch.testing.assert_close(model_inputs["ngram_context"].cpu(), expected_context)

    # Retain the views to detect reallocations as the request layout changes.
    query_start_loc = model_inputs["query_start_loc"]
    ngram_context = model_inputs["ngram_context"]
    input_batch = _ple_batch(buffers, 1, 1, [0, 3], [0])
    model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc.fill_(3)
    expected_query_start_loc[0] = 0
    torch.testing.assert_close(
        model_inputs["query_start_loc"].cpu(), expected_query_start_loc
    )
    expected_context.fill_(99)
    expected_context[0] = torch.tensor([1, 2, 3])
    torch.testing.assert_close(model_inputs["ngram_context"].cpu(), expected_context)
    assert model_inputs["query_start_loc"].data_ptr() == query_start_loc.data_ptr()
    assert model_inputs["ngram_context"].data_ptr() == ngram_context.data_ptr()


def test_qwen4_exp_model_state_prepares_stable_dummy_ngram_inputs(tmp_path) -> None:
    model_state, _, _ = _ple_state(tmp_path, 3)
    first = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)
    query_start_loc_ptr = first["query_start_loc"].data_ptr()
    ngram_context_ptr = first["ngram_context"].data_ptr()
    second = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)

    expected_query_start_loc = torch.full((9,), 4, dtype=torch.int32)
    expected_query_start_loc[:4] = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    torch.testing.assert_close(
        second["query_start_loc"].cpu(), expected_query_start_loc
    )
    torch.testing.assert_close(
        second["ngram_context"].cpu(), torch.full((8, 3), 99, dtype=torch.int32)
    )
    assert second["query_start_loc"].data_ptr() == query_start_loc_ptr
    assert second["ngram_context"].data_ptr() == ngram_context_ptr


@pytest.mark.parametrize(
    "model_cls", [Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration]
)
def test_qkv_proj_mapping_allows_bf16_indexer(model_cls) -> None:
    """Regression test for https://github.com/vllm-project/vllm/issues/62308.

    Layout of e.g. ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ: q/k/v are
    quantized and only the indexer projection is in `ignore`.
    """
    assert not should_ignore_layer(
        "model.layers.3.self_attn.qkv_proj",
        ignore=["model.layers.3.self_attn.indexer.index_qk_proj"],
        fused_mapping=model_cls.packed_modules_mapping,
    )


@pytest.mark.parametrize(
    "state_cls",
    [Qwen4ExpModelState],
    ids=["nvidia"],
)
def test_qwen4_exp_model_state_pads_replayed_request_range(state_cls, tmp_path) -> None:
    """The PLE buffers must be valid for the capture-time request range.

    A PIECEWISE graph replays with the request count of the piece it was captured
    for, so its request lookup scans every row of `ple_query_start_loc` up to that
    count and reads the matching `ngram_context` rows.
    """
    model_state, req_states, buffers = _ple_state(tmp_path, 2)
    assert isinstance(model_state, state_cls)
    for req in range(8):
        req_states.all_token_ids.stage_write(req, 0, list(range(req * 8, req * 8 + 8)))
        req_states.num_computed_tokens.stage_write_elem(req, 1 if req < 2 else 0)
    req_states.apply_staged_writes()

    # A wider earlier batch leaves a non-monotonic tail behind.
    wide_batch = _ple_batch(buffers, 3, 3, [0, 1, 2, 3], [0, 1, 2])
    model_state.prepare_inputs(wide_batch, req_states)

    # One request, 8 tokens: the 8-token piece was captured with R_cap = 8, but
    # PIECEWISE does not pad requests, so num_reqs_after_padding == num_reqs == 1.
    batch = _ple_batch(buffers, 1, 1, [0, 8], [0])
    model_state.prepare_inputs(batch, req_states)

    decoded = model_state.ple_query_start_loc
    assert decoded[0] == 0
    assert torch.all(decoded[1:] >= decoded[:-1]), decoded
    # Unused capacity must not capture real tokens.
    assert torch.all(decoded[2:] >= 8), decoded
    # A search over the captured 8-request range must resolve every real token
    # to request 0, not to the rows left over from the wider batch.
    for token in range(8):
        assert max(i for i in range(9) if int(decoded[i]) <= token) == 0
    # Row 0 keeps its live context; rows the batch does not own must be sentinels
    # (they are still read for padding tokens).
    expected_context = torch.full((8, 2), 99, dtype=torch.int32)
    expected_context[0] = torch.tensor([99, 0], dtype=torch.int32)
    torch.testing.assert_close(model_state.ngram_context.cpu(), expected_context)

    # A request-padded batch (FULL graphs): the runner already pads its own copy,
    # so the buffer tail past it must be padded with the same value.
    padded_batch = _ple_batch(buffers, 2, 4, [0, 2, 4, 6, 6], [0, 1])
    model_state.prepare_inputs(padded_batch, req_states)
    torch.testing.assert_close(
        model_state.ple_query_start_loc.cpu(),
        torch.tensor([0, 2, 4, 6, 6, 6, 6, 6, 6], dtype=torch.int32),
    )
    assert torch.all(model_state.ngram_context[2:] == 99), model_state.ngram_context

    # A full batch leaves no tail to pad.
    full_batch = _ple_batch(buffers, 8, 8, list(range(9)), list(range(8)))
    model_state.prepare_inputs(full_batch, req_states)
    torch.testing.assert_close(
        model_state.ple_query_start_loc.cpu(), torch.arange(9, dtype=torch.int32)
    )

    # Capture-time dummy inputs follow the same convention.
    model_state.prepare_dummy_inputs(num_reqs=2, num_tokens=5)
    torch.testing.assert_close(
        model_state.ple_query_start_loc[:3].cpu(),
        torch.tensor([0, 2, 5], dtype=torch.int32),
    )
    assert torch.all(model_state.ple_query_start_loc[3:] >= 5)
