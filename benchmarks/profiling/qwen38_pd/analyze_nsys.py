# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute Nsight SQLite GPU activities to the outer worker NVTX ranges."""

import argparse
import bisect
import csv
import json
import sqlite3
from collections import defaultdict
from functools import cache
from pathlib import Path

import regex as re


def parse_label(text):
    return dict(part.split("=", 1) for part in text.split("|")[1:])


def phase(label):
    ctx, gen, tokens = (int(label.get(key, 0)) for key in ("ctx", "gen", "tok"))
    if not tokens:
        return "transfer_only"
    if ctx and gen:
        return "mixed"
    if ctx:
        return "prefill" if label["role"] == "P" else "handoff"
    if gen:
        return "decode" if label["role"] == "D" else "P_generation"
    return "unknown"


def union_ns(intervals):
    stop, total = None, 0
    for start, end in sorted(intervals):
        if stop is None or start >= stop:
            total += end - start
        elif end > stop:
            total += end - stop
        stop = max(stop if stop is not None else end, end)
    return total


def overlap_ns(first, second):
    events = [
        event
        for group, intervals in enumerate((first, second))
        for start, end in intervals
        for event in ((start, group, 1), (end, group, -1))
    ]
    active, total, last = [0, 0], 0, 0
    for time, group, delta in sorted(events):
        if active[0] and active[1]:
            total += time - last
        active[group] += delta
        last = time
    return total


@cache
def category(name):
    rules = [
        (
            r"nccl|all_reduce|allreduce|all_gather|reduce_scatter|cross_device",
            "collective",
        ),
        (r"ple_|ngram|offloaded_gather", "PLE"),
        (
            r"HcDownSilu|hc_down|hc_up|hc_combine|hc_gate|hc_silu|grouped_gemma",
            "HC_fused_glue",
        ),
        (r"qsa_mqa", "QSA_index_logits"),
        (
            r"expand_qsa|compress_qsa|store_qsa|qsa_prepare|qsa_physical|build_qsa_metadata",
            "QSA_prepare_cache",
        ),
        (r"qsa_sparse|qsa_merge|Qsa", "QSA_attention"),
        (r"cooperative_topk|radix_topk|topk.*(2048|index)", "index_topk"),
        (r"fused_moe_kernel|grouped.*gemm|GroupedGemm", "MoE_expert_GEMM"),
        (r"moe_|::moe::|fused_topk|topkGating|grouped_topk", "MoE_routing_combine"),
        (
            r"Gdn|GDN|gdn|gated_delta|causal_conv|fused_post_conv|l2norm",
            "GDN_core_prepare",
        ),
        (r"fp8|quantize|quantization|group_quant", "activation_quant"),
        (r"silu|act_and_mul|sigmoid_gate", "MLP_activation_gate"),
        (r"rms_norm|rmsnorm|layer_norm|layernorm|RMSNorm", "other_norm"),
        (
            r"topk|top_k|argmax|sampling|sample|gumbel|log_softmax|softmax",
            "selection_sampling",
        ),
        (r"skinny|gemm|gemv|Gemm|Gemv|cublas|matmul|nvjet", "dense_GEMM"),
        (r"embedding", "token_embedding"),
        (
            (
                r"prepare_prefill_inputs|prepare_pos_seq_lens|compute_slot_mappings|"
                r"gather_block_tables|gather_initial_states|post_update|apply_write|"
                r"scatter_num_accepted"
            ),
            "input_cache_state",
        ),
        (r"elementwise|index_select|direct_copy", "tensor_glue"),
    ]
    for pattern, result in rules:
        if re.search(pattern, name):
            return result
    return "other_kernel"


def read_database(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges, sessions, cpu = defaultdict(list), defaultdict(list), []
    for row in db.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        text = row["text"] or strings.get(row["textId"], "")
        if not text.startswith(("nabla_profile|", "nabla_session|")):
            continue
        label = parse_label(text)
        item = (row["start"], row["end"], label)
        if text.startswith("nabla_session|"):
            sessions[row["globalTid"] >> 24].append(item)
        else:
            ranges[row["globalTid"]].append(item)
            cpu.append(item)
    assert sessions, f"No profiling session ranges in {path}"
    for items in ranges.values():
        items.sort(key=lambda item: item[0])
    starts = {tid: [item[0] for item in items] for tid, items in ranges.items()}
    launches = {}
    for table in ("CUPTI_ACTIVITY_KIND_DRIVER", "CUPTI_ACTIVITY_KIND_RUNTIME"):
        if table not in tables:
            continue
        for row in db.execute(f"SELECT start, globalTid, correlationId FROM {table}"):
            tid = row["globalTid"]
            index = bisect.bisect_right(starts.get(tid, []), row["start"]) - 1
            if index >= 0:
                item = ranges[tid][index]
                if row["start"] <= item[1]:
                    launches[(tid >> 24, row["correlationId"])] = item[2]
    activities = []
    for table, kind in (
        ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
    ):
        if table not in tables:
            continue
        for row in db.execute(f"SELECT * FROM {table}"):
            columns = set(row.keys())
            pid = row["globalPid"] >> 24
            session = next(
                (
                    item[2]
                    for item in sessions.get(pid, [])
                    if item[0] <= row["start"] <= item[1]
                ),
                None,
            )
            if session is None:
                continue
            label = launches.get((pid, row["correlationId"]))
            if label is None or label["case"] != session["case"]:
                label = {
                    **session,
                    "phase": "outside_step",
                    "step": 0,
                    "stage": "background",
                }
            else:
                label = {**label, "phase": phase(label)}
            name = strings[row["demangledName"]] if kind == "kernel" else kind
            activities.append(
                {
                    **label,
                    "start": row["start"],
                    "end": row["end"],
                    "kind": kind,
                    "name": name,
                    "category": category(name) if kind == "kernel" else kind,
                    "graph_node": row["graphNodeId"] if "graphNodeId" in columns else 0,
                    "bytes": row["bytes"] if "bytes" in columns else 0,
                }
            )
    db.close()
    return activities, cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("databases", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    activities, cpu = [], []
    for path in args.databases:
        gpu_rows, cpu_rows = read_database(path)
        activities.extend(gpu_rows)
        cpu.extend(cpu_rows)
    assert activities, "No GPU activities inside recorded profiling sessions"
    grouped, kernels = defaultdict(list), defaultdict(lambda: [0, 0, 0])
    for row in activities:
        key = tuple(row[k] for k in ("case", "role", "rank", "phase"))
        grouped[key].append(row)
        if row["kind"] == "kernel":
            kernel = kernels[(*key, row["stage"], row["category"], row["name"])]
            kernel[0] += row["end"] - row["start"]
            kernel[1] += 1
            kernel[2] += bool(row["graph_node"])
    cpu_groups = defaultdict(list)
    for start, end, label in cpu:
        key = (label["case"], label["role"], label["rank"], phase(label))
        cpu_groups[key].append((start, end))
    summaries = []
    for key, rows in sorted(grouped.items()):
        kernel_rows = [row for row in rows if row["kind"] == "kernel"]
        sums = defaultdict(int)
        for row in kernel_rows:
            sums[row["category"]] += row["end"] - row["start"]
        total = sum(sums.values())
        assert total == sum(row["end"] - row["start"] for row in kernel_rows)
        summary = dict(zip(("case", "role", "rank", "phase"), key))
        summary.update(
            {
                "kernel_ns": total,
                "kernel_count": len(kernel_rows),
                "kernel_union_ns": union_ns(
                    [(r["start"], r["end"]) for r in kernel_rows]
                ),
                "kernel_span_ns": (
                    max(r["end"] for r in kernel_rows)
                    - min(r["start"] for r in kernel_rows)
                    if kernel_rows
                    else 0
                ),
                "graph_kernel_count": sum(
                    bool(row["graph_node"]) for row in kernel_rows
                ),
                "stage_kernel_ns": {
                    stage: sum(
                        r["end"] - r["start"]
                        for r in kernel_rows
                        if r["stage"] == stage
                    )
                    for stage in sorted({r["stage"] for r in kernel_rows})
                },
                "activity_union_ns": union_ns([(r["start"], r["end"]) for r in rows]),
                "activity_span_ns": max(r["end"] for r in rows)
                - min(r["start"] for r in rows),
                "cpu_range_union_ns": union_ns(cpu_groups[key]),
                "collective_hc_norm_overlap_ns": overlap_ns(
                    [
                        (r["start"], r["end"])
                        for r in kernel_rows
                        if r["category"] == "collective"
                    ],
                    [
                        (r["start"], r["end"])
                        for r in kernel_rows
                        if "_hc_combine_norm_kernel" in r["name"]
                    ],
                ),
                "memcpy_ns": sum(
                    r["end"] - r["start"] for r in rows if r["kind"] == "memcpy"
                ),
                "memcpy_bytes": sum(r["bytes"] for r in rows if r["kind"] == "memcpy"),
                "memset_ns": sum(
                    r["end"] - r["start"] for r in rows if r["kind"] == "memset"
                ),
                "memset_bytes": sum(r["bytes"] for r in rows if r["kind"] == "memset"),
                "categories": {
                    name: {"ns": ns, "pct": 100 * ns / total}
                    for name, ns in sorted(sums.items(), key=lambda x: -x[1])
                },
            }
        )
        summaries.append(summary)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    totals = {
        tuple(row[k] for k in ("case", "role", "rank", "phase")): row["kernel_ns"]
        for row in summaries
    }
    with (args.output / "kernels.csv").open("w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "case",
                "role",
                "rank",
                "phase",
                "stage",
                "category",
                "kernel",
                "total_ns",
                "calls",
                "graph_calls",
                "pct",
            ]
        )
        for key, (ns, calls, graph_calls) in sorted(kernels.items()):
            writer.writerow([*key, ns, calls, graph_calls, 100 * ns / totals[key[:4]]])
    batches = defaultdict(list)
    for row in activities:
        if row["phase"] == "decode" and row["kind"] == "kernel":
            key = (row["case"], row["rank"], int(row["tok"]))
            batches[key].append(row)
    with (args.output / "decode-batches.csv").open("w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "case",
                "rank",
                "active_tokens",
                "steps",
                "category",
                "total_ns",
                "calls",
                "pct",
                "kernel_union_ns",
            ]
        )
        for key, rows in sorted(batches.items()):
            sums, calls = defaultdict(int), defaultdict(int)
            for row in rows:
                sums[row["category"]] += row["end"] - row["start"]
                calls[row["category"]] += 1
            total = sum(sums.values())
            steps = len({row["step"] for row in rows})
            active = union_ns([(row["start"], row["end"]) for row in rows])
            for name, ns in sorted(sums.items()):
                writer.writerow(
                    [*key, steps, name, ns, calls[name], 100 * ns / total, active]
                )
    print(
        json.dumps(
            {
                "databases": len(args.databases),
                "GPU_activities": len(activities),
                "phase_rank_groups": len(summaries),
            }
        )
    )


if __name__ == "__main__":
    assert union_ns([(0, 10), (5, 15), (20, 25)]) == 20
    assert overlap_ns([(0, 10), (5, 15)], [(8, 12), (11, 20)]) == 7
    assert phase({"role": "D", "ctx": "1", "gen": "0", "tok": "1"}) == "handoff"
    assert phase({"role": "D", "ctx": "0", "gen": "8", "tok": "8"}) == "decode"
    main()
