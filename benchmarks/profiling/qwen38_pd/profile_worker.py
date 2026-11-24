# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Outer NVTX annotations for graph-preserving Qwen P/D profiles."""

import json
import os
from pathlib import Path

import torch

from vllm.v1.worker.gpu_worker import Worker


class ProfileWorker(Worker):
    """Label real scheduler iterations without changing model execution."""

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self._profile_case = None
        self._profile_step = 0
        self._profile_records = []
        self._profile_label = ""
        model = self.model_runner.model
        config = self.vllm_config
        receipt = {
            "rank": self.rank,
            "role": os.environ["NABLA_PROFILE_ROLE"],
            "pid": os.getpid(),
            "pid_start_ticks": Path("/proc/self/stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()[19],
            "gpu": torch.cuda.get_device_name(self.local_rank),
            "config": repr(config),
            "modules": [
                {"name": name, "type": type(module).__qualname__}
                for name, module in model.named_modules()
            ],
        }
        directory = Path(os.environ["NABLA_PROFILE_RECEIPTS"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"worker-{receipt['role']}-{self.rank}.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        return result

    def profile(self, is_start=True, profile_prefix=None, **kwargs):
        if is_start:
            self._profile_case = profile_prefix or "unnamed"
            self._profile_step = 0
            self._profile_records = []
        else:
            # Flush outstanding kernels before ending the CUPTI collection.
            torch.accelerator.synchronize()
            torch.cuda.nvtx.range_end(self._profile_session)
        result = super().profile(is_start, profile_prefix, **kwargs)
        if is_start:
            role = os.environ["NABLA_PROFILE_ROLE"]
            self._profile_session = torch.cuda.nvtx.range_start(
                f"nabla_session|case={self._profile_case}|role={role}|rank={self.rank}"
            )
        if not is_start:
            directory = Path(os.environ["NABLA_PROFILE_RECEIPTS"])
            role = os.environ["NABLA_PROFILE_ROLE"]
            (
                directory / f"steps-{self._profile_case}-{role}-{self.rank}.json"
            ).write_text(json.dumps(self._profile_records, indent=2) + "\n")
            self._profile_case = None
        return result

    def execute_model(self, scheduler_output):
        if self._profile_case is None:
            return super().execute_model(scheduler_output)
        self._profile_step += 1
        details = scheduler_output.iteration_details
        record = {
            "case": self._profile_case,
            "role": os.environ["NABLA_PROFILE_ROLE"],
            "rank": self.rank,
            "step": self._profile_step,
            "ctx": details.num_ctx_requests if details else 0,
            "gen": details.num_generation_requests if details else 0,
            "tok": scheduler_output.total_num_scheduled_tokens,
        }
        self._profile_records.append(record)
        self._profile_label = "nabla_profile|" + "|".join(
            f"{key}={value}" for key, value in record.items()
        )
        with torch.cuda.nvtx.range(self._profile_label + "|stage=execute"):
            result = super().execute_model(scheduler_output)
        state = self.model_runner.execute_model_state
        if record["tok"] and state is not None:
            batch = state.input_batch
            record.update(
                active_tokens=int(batch.num_tokens),
                padded_tokens=int(batch.num_tokens_after_padding),
                actual_reqs=int(batch.num_reqs),
                padded_reqs=int(batch.num_reqs_after_padding),
                decode_graph_eligible=bool(batch.decode_graph_eligible),
            )
        return result

    def sample_tokens(self, grammar_output):
        if self._profile_case is None:
            return super().sample_tokens(grammar_output)
        with torch.cuda.nvtx.range(self._profile_label + "|stage=sample"):
            return super().sample_tokens(grammar_output)
