import math
from typing import Optional

from .catalog import BITS_PER_WEIGHT, TOPOLOGY_EFFICIENCY
from .types import Estimate, GPU, Model, RuntimeStack, Scenario


def model_weight_gb(params_b: float, bits: float) -> float:
    return params_b * bits / 8.0


def kv_bytes_per_token(model: Model, kv_bits: float, gqa_ratio: float) -> float:
    return 2.0 * model.layers * model.hidden_size * gqa_ratio * (kv_bits / 8.0)


def model_flops_per_token(model: Model) -> float:
    return 2.0 * model.active_params_b * 1e9


def attention_flops_per_token(model: Model, seq: int) -> float:
    return 4.0 * model.layers * model.hidden_size * seq


def precision_peak_tflops(gpu: GPU, quant: str) -> float:
    """Return a dense peak appropriate to the requested weight format.

    Sparse advertised peaks are intentionally not used. Quantized formats fall
    back to the nearest defensible dense integer or floating-point capability.
    """
    if quant == "fp16":
        return gpu.fp16_dense_tflops
    if quant == "bf16":
        return gpu.bf16_dense_tflops
    if quant == "fp8":
        if gpu.fp8_dense_tflops is None:
            raise ValueError(f"{gpu.name} has no native FP8 capability in the catalog")
        return gpu.fp8_dense_tflops
    if quant == "int8":
        if gpu.int8_dense_tops is None:
            raise ValueError(f"{gpu.name} has no INT8 capability in the catalog")
        return gpu.int8_dense_tops
    if quant in {"q6", "q5", "q4", "q3"}:
        base = gpu.int8_dense_tops or gpu.fp16_dense_tflops
        return base * min(8.0 / BITS_PER_WEIGHT[quant], 2.0)
    raise ValueError(f"unsupported quantization: {quant}")


def resource_scale(gpus: int, topology: str, comm_eff: float, tp: int) -> float:
    if gpus == 1:
        return 1.0
    base = TOPOLOGY_EFFICIENCY.get(topology, 0.72)
    collective = (base * comm_eff) ** math.log2(max(1, tp))
    return tp * collective * (gpus / tp)


def memory_capacity(
    gpu: GPU,
    gpus: int,
    model: Model,
    weight_bits: float,
    kv_bits: float,
    input_tokens: int,
    output_tokens: int,
    runtime_overhead: float,
    vram_utilization: float,
) -> dict:
    weights = model_weight_gb(model.total_params_b, weight_bits)
    overhead = weights * runtime_overhead
    kv_req = (
        kv_bytes_per_token(model, kv_bits, model.gqa_ratio)
        * (input_tokens + output_tokens)
        / 1e9
    )
    usable = gpu.vram_gb * gpus * vram_utilization
    fixed = weights + overhead
    max_users = max(0, math.floor((usable - fixed) / max(kv_req, 1e-12)))
    return {
        "weights_gb": weights,
        "runtime_overhead_gb": overhead,
        "kv_per_request_gb": kv_req,
        "usable_gb": usable,
        "fixed_gb": fixed,
        "max_concurrency": max_users,
    }


def estimate(
    scenario: Scenario,
    runtime: RuntimeStack,
    gpu: GPU,
    gpus: int,
    model: Model,
    quant: str,
    input_tokens: int,
    output_tokens: int,
    users: int,
    kv_bits: float,
    tp: int,
    compute_tflops: Optional[float] = None,
    bw_util: Optional[float] = None,
    compute_util: Optional[float] = None,
    calibration: Optional[dict] = None,
) -> Estimate:
    if gpu.vendor not in runtime.compatible_vendors:
        raise ValueError(f"{runtime.name} is not compatible with {gpu.vendor} hardware")
    if quant not in runtime.supported_quantizations:
        raise ValueError(f"{runtime.name} does not support catalog format '{quant}'")

    bits = BITS_PER_WEIGHT[quant]
    total_w = model_weight_gb(model.total_params_b, bits) * 1e9
    active_w = model_weight_gb(model.active_params_b, bits) * 1e9
    kv_bpt = kv_bytes_per_token(model, kv_bits, model.gqa_ratio)

    runtime_comm = min(0.99, runtime.communication_efficiency * scenario.communication_multiplier)
    scale = resource_scale(gpus, gpu.topology, runtime_comm, tp)

    bandwidth_fraction = (
        bw_util
        if bw_util is not None
        else runtime.bandwidth_efficiency * scenario.bandwidth_multiplier
    )
    compute_fraction = (
        compute_util
        if compute_util is not None
        else runtime.compute_efficiency * scenario.compute_multiplier
    )
    bandwidth_fraction = min(0.95, bandwidth_fraction)
    compute_fraction = min(0.95, compute_fraction)

    effective_bw = (
        gpu.bandwidth_gbps
        * 1e9
        * bandwidth_fraction
        * gpu.hardware_bandwidth_calibration
        * gpu.benchmark_multiplier
        * scale
    )
    peak = compute_tflops if compute_tflops is not None else precision_peak_tflops(gpu, quant)
    effective_flops = (
        peak
        * 1e12
        * compute_fraction
        * gpu.hardware_compute_calibration
        * gpu.benchmark_multiplier
        * scale
    )

    overhead = min(0.40, runtime.kernel_overhead * scenario.overhead_multiplier)
    useful_fraction = 1.0 - overhead
    activation = model.layers * model.hidden_size * 4.0

    prefill_flops = model_flops_per_token(model) + attention_flops_per_token(
        model, max(1, input_tokens // 2)
    )
    prefill_compute = effective_flops / max(prefill_flops, 1)
    batch_prompt = max(1, input_tokens * users)
    prefill_bytes = total_w / batch_prompt + activation + kv_bpt
    prefill_bw = effective_bw / max(prefill_bytes, 1)
    prompt = min(prefill_compute, prefill_bw) * useful_fraction

    avg_ctx = input_tokens + max(0, output_tokens - 1) / 2
    decode_bytes = active_w / max(1, users) + kv_bpt * avg_ctx + activation
    decode_bw = effective_bw / max(decode_bytes, 1)
    decode_flops = model_flops_per_token(model) + attention_flops_per_token(
        model, int(avg_ctx)
    )
    decode_compute = effective_flops / max(decode_flops, 1)
    output = min(decode_bw, decode_compute) * useful_fraction

    if calibration:
        if calibration.get("prompt_tps"):
            prompt *= float(calibration["prompt_tps"]) / max(
                float(calibration.get("predicted_prompt_tps", prompt)), 1e-9
            )
        if calibration.get("output_tps"):
            output *= float(calibration["output_tps"]) / max(
                float(calibration.get("predicted_output_tps", output)), 1e-9
            )

    per_user = output / max(1, users)
    prefill_s = input_tokens * users / max(prompt, 1e-9)
    ttft = prefill_s + users / max(output, 1e-9)
    itl = 1000 / max(per_user, 1e-9)
    latency = prefill_s + output_tokens / max(per_user, 1e-9)
    rps = output / max(output_tokens, 1)
    return Estimate(
        scenario.name,
        prompt,
        output,
        per_user,
        ttft,
        itl,
        latency,
        rps,
        decode_compute,
        decode_bw,
        prefill_compute,
        prefill_bw,
    )
