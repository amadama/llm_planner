import argparse
import json
from pathlib import Path

from .catalog import (
    BITS_PER_WEIGHT,
    DEFAULT_RUNTIME_BY_VENDOR,
    GPUS,
    MODELS,
    RUNTIMES,
    SCENARIOS,
)
from .model import estimate, memory_capacity
from .report import print_report, write_json
from .types import Model


def posint(v):
    n = int(v)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return n


def posfloat(v):
    n = float(v)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return n


def parser():
    p = argparse.ArgumentParser(
        description="Plan LLM inference capacity with hardware/runtime/model separation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gpu", choices=sorted(GPUS))
    p.add_argument("--gpus", type=posint, default=1)
    p.add_argument("--runtime", choices=sorted(RUNTIMES), help="Inference runtime profile; defaults by GPU vendor")
    p.add_argument("--model", choices=sorted(MODELS) + ["custom"])
    p.add_argument("--users", "--concurrency", dest="users", type=posint, default=1)
    p.add_argument("--quant", choices=sorted(BITS_PER_WEIGHT), default="q4")
    p.add_argument("--input-tokens", type=posint, default=4096)
    p.add_argument("--output-tokens", type=posint, default=512)
    p.add_argument("--context", type=posint)
    p.add_argument("--tensor-parallel", type=posint, default=1)
    p.add_argument("--model-name")
    p.add_argument("--params-b", type=posfloat)
    p.add_argument("--active-params-b", type=posfloat)
    p.add_argument("--layers", type=posint, default=32)
    p.add_argument("--hidden-size", type=posint, default=4096)
    p.add_argument("--gqa-ratio", type=posfloat)
    p.add_argument("--kv-bits", type=posfloat, default=16)
    p.add_argument("--vram-utilization", type=float, default=0.90)
    p.add_argument("--runtime-overhead", type=float, default=0.08)
    p.add_argument("--bandwidth-utilization", type=float, help="Override final sustained bandwidth fraction")
    p.add_argument("--compute-utilization", type=float, help="Override final sustained compute fraction")
    p.add_argument("--compute-tflops", type=posfloat)
    p.add_argument("--calibration-file")
    p.add_argument("--json-report")
    p.add_argument("--list", action="store_true")
    return p


def resolve_model(args):
    if args.model in MODELS:
        m = MODELS[args.model]
        if args.gqa_ratio is None:
            return m
        values = vars(m).copy()
        values["gqa_ratio"] = args.gqa_ratio
        return Model(**values)
    if args.params_b is None:
        raise ValueError("--params-b is required with --model custom")
    return Model(
        args.model_name or "Custom model",
        args.params_b,
        args.active_params_b or args.params_b,
        args.layers,
        args.hidden_size,
        args.gqa_ratio or 0.125,
    )


def print_catalogs():
    print("GPU presets:")
    for key, value in GPUS.items():
        print(
            f"  {key:<27} {value.vendor:<7} {value.architecture:<13} "
            f"{value.vram_gb:g} GB, {value.bandwidth_gbps:g} GB/s, {value.topology}"
        )
    print("\nRuntime presets:")
    for key, value in RUNTIMES.items():
        vendors = "/".join(value.compatible_vendors)
        print(f"  {key:<27} {value.platform:<5} {value.engine:<14} [{vendors}]")
    print("\nModel presets:")
    for key, value in MODELS.items():
        print(f"  {key:<27} {value.total_params_b:g}B total, {value.active_params_b:g}B active")


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.list:
        print_catalogs()
        return 0
    if not a.gpu or not a.model:
        p.error("--gpu and --model are required unless --list is used")
    if a.context is not None:
        a.input_tokens = a.context
    if a.tensor_parallel > a.gpus or a.gpus % a.tensor_parallel:
        p.error("--tensor-parallel must divide --gpus and cannot exceed it")
    if not 0.25 <= a.vram_utilization <= 0.99:
        p.error("--vram-utilization must be between 0.25 and 0.99")
    for name in ("bandwidth_utilization", "compute_utilization"):
        value = getattr(a, name)
        if value is not None and not 0.05 <= value <= 0.95:
            p.error(f"--{name.replace('_', '-')} must be between 0.05 and 0.95")

    try:
        gpu = GPUS[a.gpu]
        runtime_key = a.runtime or DEFAULT_RUNTIME_BY_VENDOR[gpu.vendor]
        runtime = RUNTIMES[runtime_key]
        if gpu.vendor not in runtime.compatible_vendors:
            raise ValueError(f"runtime '{runtime_key}' is incompatible with {gpu.vendor} GPU '{a.gpu}'")
        if a.quant not in runtime.supported_quantizations:
            raise ValueError(f"runtime '{runtime_key}' does not support quantization '{a.quant}'")
        model = resolve_model(a)
        calibration = None
        if a.calibration_file:
            calibration = json.loads(Path(a.calibration_file).read_text())
        cap = memory_capacity(
            gpu, a.gpus, model, BITS_PER_WEIGHT[a.quant], a.kv_bits,
            a.input_tokens, a.output_tokens, a.runtime_overhead,
            a.vram_utilization,
        )
        estimates = [
            estimate(
                scenario, runtime, gpu, a.gpus, model, a.quant,
                a.input_tokens, a.output_tokens, a.users, a.kv_bits,
                a.tensor_parallel, a.compute_tflops,
                a.bandwidth_utilization, a.compute_utilization, calibration,
            )
            for scenario in SCENARIOS.values()
        ]
        print_report(a, gpu, runtime, model, cap, estimates)
        if a.json_report:
            write_json(a.json_report, a, gpu, runtime, model, cap, estimates)
            print(f"\nJSON report written to {a.json_report}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    return 0
