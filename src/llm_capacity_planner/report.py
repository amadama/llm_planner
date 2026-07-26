import json
from dataclasses import asdict
from pathlib import Path


def human(value):
    for divisor, name in [
        (1e15, "quadrillion"), (1e12, "trillion"), (1e9, "billion"),
        (1e6, "million"), (1e3, "thousand"),
    ]:
        if value >= divisor:
            return f"{value / divisor:,.3f} {name}"
    return f"{value:,.0f}"


def print_report(args, gpu, runtime, model, capacity, estimates):
    nominal = next(e for e in estimates if e.scenario == "nominal")
    required = capacity["fixed_gb"] + capacity["kv_per_request_gb"] * args.users
    fits = required <= capacity["usable_gb"]
    print("\nLLM INFERENCE CAPACITY PLAN\n" + "=" * 84)
    print(f"GPU configuration       : {args.gpus} x {gpu.name}")
    print(f"Hardware vendor/arch    : {gpu.vendor} / {gpu.architecture}")
    print(f"Runtime                 : {runtime.name}")
    print(f"Topology / TP           : {gpu.topology} / {args.tensor_parallel}")
    print(f"Model                   : {model.name}")
    print(f"Quantization            : {args.quant}")
    print(f"Input / output tokens   : {args.input_tokens:,} / {args.output_tokens:,}")
    print(f"Concurrent requests     : {args.users}")
    print("-" * 84)
    print(f"Model weights           : {capacity['weights_gb']:,.2f} GB")
    print(f"KV cache/request        : {capacity['kv_per_request_gb']:,.3f} GB")
    print(f"Estimated VRAM required : {required:,.2f} GB")
    print(f"Usable GPU VRAM         : {capacity['usable_gb']:,.2f} GB")
    print(f"Memory-limited max users: {capacity['max_concurrency']:,}")
    print(f"VRAM status             : {'FITS' if fits else 'DOES NOT FIT'}")
    print("-" * 84)
    print(f"{'Scenario':<14}{'Prompt tok/s':>15}{'Output tok/s':>15}{'Per-user tok/s':>17}{'TTFT':>10}{'ITL':>11}")
    for item in estimates:
        print(f"{item.scenario:<14}{item.prompt_tps:>15,.1f}{item.output_tps:>15,.1f}{item.per_user_tps:>17,.2f}{item.ttft_s:>9.2f}s{item.itl_ms:>9.1f}ms")
    print("-" * 84)
    print(f"Nominal request latency : {nominal.request_latency_s:,.2f} seconds")
    print(f"Nominal request rate    : {nominal.requests_per_s:,.3f} requests/second")
    print(f"Nominal prefill bound   : {'memory bandwidth' if nominal.prefill_bandwidth_bound_tps < nominal.prefill_compute_bound_tps else 'compute'}")
    print(f"Nominal decode bound    : {'memory bandwidth' if nominal.decode_bandwidth_bound_tps < nominal.decode_compute_bound_tps else 'compute'}")
    for label, seconds in [("hour", 3600), ("day", 86400), ("week", 604800), ("30-day month", 2592000), ("365-day year", 31536000)]:
        value = nominal.output_tps * seconds
        print(f"Output tokens/{label:<12}: {value:>20,.0f}  ({human(value)})")
    if not fits:
        print("\nWARNING: requested workload does not fit in usable VRAM; throughput assumes all-GPU execution.")


def write_json(path, args, gpu, runtime, model, capacity, estimates):
    payload = {
        "inputs": vars(args),
        "gpu": asdict(gpu),
        "runtime": asdict(runtime),
        "model": asdict(model),
        "capacity": capacity,
        "estimates": [asdict(item) for item in estimates],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
