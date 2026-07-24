"""Built-in hardware, model, precision, and planning-scenario catalogs.

This module contains the default numerical inputs used by the analytical
capacity model.  The entries are deliberately kept as data rather than being
embedded in the estimator equations, so that a user can add or revise presets
without changing the model itself.

Important interpretation notes
------------------------------
* GPU compute figures are peak/theoretical capabilities expressed in the units
  named by :class:`~llm_capacity_planner.types.GPU`.  The planner applies a
  scenario-specific ``compute_utilization`` before using them.
* GPU memory bandwidth is the vendor peak bandwidth in decimal GB/s.  The
  planner applies ``bandwidth_utilization`` before using it.
* Model parameter counts are in billions of parameters.  ``active_params_b``
  may be lower than ``total_params_b`` for mixture-of-experts models.
* Quantized formats include an allowance for scales, zero-points, or other
  metadata; consequently, labels such as ``q4`` are represented by an effective
  bit width rather than exactly four stored bits per parameter.
* Scenario and topology efficiencies are planning assumptions, not immutable
  hardware specifications.  Measurements from the intended inference stack
  should be used to calibrate them whenever possible.
"""

from .types import GPU, Model, Scenario


# GPU catalog
# -----------
# GPU entries use keyword arguments so published specifications remain clearly
# separated from planner assumptions.  Core fields used by the current model:
#
#   name, architecture, vram_gb, bandwidth_gbps, fp16_tflops, fp8_tflops,
#   int8_tops, topology
#
# Optional published metadata includes fp32_tflops, int4_tops, power_w, the
# human-readable interconnect name, and host_link_bandwidth_gbps.  The fields
# sustained_bandwidth_efficiency, sustained_compute_efficiency, and
# benchmark_multiplier are planner calibration assumptions, not NVIDIA specs.
# They are recorded now for transparent calibration and future model revisions;
# the current estimator continues to use SCENARIOS for its active utilization.
#
# ``topology`` must be a key in TOPOLOGY_EFFICIENCY.  It selects the default
# communication-scaling assumption for multi-GPU work partitioning.
GPUS = {
    "rtx-pro-6000-blackwell": GPU(
        name="NVIDIA RTX PRO 6000 Blackwell",
        architecture="Blackwell",
        vram_gb=96,
        bandwidth_gbps=1792,
        fp16_tflops=125,
        fp8_tflops=250,
        int8_tops=250,
        topology="pcie",
        interconnect="PCIe",
        sustained_bandwidth_efficiency=0.52,
        sustained_compute_efficiency=0.45,
    ),
    "h200-sxm": GPU(
        name="NVIDIA H200 SXM",
        architecture="Hopper",
        vram_gb=141,
        bandwidth_gbps=4800,
        fp16_tflops=989,
        fp8_tflops=1979,
        int8_tops=1979,
        topology="nvswitch",
        interconnect="NVLink/NVSwitch",
        sustained_bandwidth_efficiency=0.58,
        sustained_compute_efficiency=0.50,
    ),
    "h100-sxm": GPU(
        name="NVIDIA H100 SXM",
        architecture="Hopper",
        vram_gb=80,
        bandwidth_gbps=3350,
        fp16_tflops=989,
        fp8_tflops=1979,
        int8_tops=1979,
        topology="nvswitch",
        interconnect="NVLink/NVSwitch",
        sustained_bandwidth_efficiency=0.57,
        sustained_compute_efficiency=0.50,
    ),
    "a100-80gb": GPU(
        name="NVIDIA A100 80GB SXM",
        architecture="Ampere",
        vram_gb=80,
        bandwidth_gbps=2039,
        fp16_tflops=312,
        fp8_tflops=624,  # Planner-side equivalent; A100 has no native FP8.
        int8_tops=624,
        topology="nvlink",
        interconnect="NVLink",
        sustained_bandwidth_efficiency=0.50,
        sustained_compute_efficiency=0.42,
    ),
    "rtx-5090": GPU(
        name="NVIDIA GeForce RTX 5090",
        architecture="Blackwell",
        vram_gb=32,
        bandwidth_gbps=1792,
        fp16_tflops=105,
        fp8_tflops=210,
        int8_tops=210,
        topology="pcie",
        interconnect="PCIe",
        sustained_bandwidth_efficiency=0.48,
        sustained_compute_efficiency=0.42,
    ),
    "rtx-4090": GPU(
        name="NVIDIA GeForce RTX 4090",
        architecture="Ada Lovelace",
        vram_gb=24,
        bandwidth_gbps=1008,
        fp16_tflops=82.6,
        fp8_tflops=165.2,
        int8_tops=165.2,
        topology="pcie",
        interconnect="PCIe",
        sustained_bandwidth_efficiency=0.46,
        sustained_compute_efficiency=0.40,
    ),
    "t4": GPU(
        name="NVIDIA T4",
        architecture="Turing",
        vram_gb=16,
        bandwidth_gbps=320,
        fp16_tflops=65,
        fp8_tflops=65,  # Compatibility fallback; T4 has no native FP8.
        int8_tops=130,
        topology="pcie",
        fp32_tflops=8.1,
        int4_tops=260,
        power_w=70,
        interconnect="PCIe Gen3 x16",
        host_link_bandwidth_gbps=32,
        sustained_bandwidth_efficiency=0.48,
        sustained_compute_efficiency=0.60,
        benchmark_multiplier=1.0,
    ),
}


# Model catalog
# -------------
# Each Model(...) entry uses positional arguments in this order:
#
#   Model(
#       name,              # Human-readable model/family name.
#       total_params_b,    # All stored parameters, in billions.
#       active_params_b,   # Parameters exercised for one token, in billions.
#       layers,            # Number of transformer layers.
#       hidden_size,       # Residual/embedding width of each token vector.
#       gqa_ratio,         # KV heads / query-attention heads.
#   )
#
# ``total_params_b`` determines weight-memory capacity.  For a dense model,
# ``active_params_b`` is normally equal to ``total_params_b``.  For an MoE
# model, only the routed experts and shared components are active for a token,
# so ``active_params_b`` is lower and is used to estimate token-level compute
# and weight traffic.
#
# ``gqa_ratio`` controls KV-cache size and KV-read traffic.  A value of 1.0
# represents ordinary multi-head attention with as many KV heads as query
# heads.  Values below 1.0 represent grouped-query or multi-query attention and
# reduce KV storage approximately in proportion to the ratio.
#
# Model records can also include optional MoE and architecture metadata such as
# expert counts, attention-head counts, maximum context, DSA, and MTP support.
# The current estimator consumes the six core fields above; the extra fields
# are retained for reporting and future expert-parallel/sparse-attention models.
MODELS = {
    "llama-3.1-8b": Model(
        "Llama 3.1 8B",
        8,       # total_params_b
        8,       # active_params_b: dense model
        32,      # layers
        4096,    # hidden_size
        0.25,    # gqa_ratio
    ),
    "llama-3.1-70b": Model(
        "Llama 3.1/3.3 70B",
        70,      # total_params_b
        70,      # active_params_b: dense model
        80,      # layers
        8192,    # hidden_size
        0.125,   # gqa_ratio
    ),
    "qwen2.5-14b": Model(
        "Qwen 2.5 14B",
        14.7,    # total_params_b
        14.7,    # active_params_b: dense model
        48,      # layers
        5120,    # hidden_size
        0.125,   # gqa_ratio
    ),
    "qwen2.5-32b": Model(
        "Qwen 2.5 32B",
        32.5,    # total_params_b
        32.5,    # active_params_b: dense model
        64,      # layers
        5120,    # hidden_size
        0.125,   # gqa_ratio
    ),
    "qwen3-30b-a3b": Model(
        "Qwen3 30B-A3B",
        30.5,    # total_params_b: all experts and shared weights
        3.3,     # active_params_b: approximate parameters active per token
        48,      # layers
        2048,    # hidden_size
        0.125,   # gqa_ratio
    ),
    "deepseek-r1-671b": Model(
        "DeepSeek-R1 671B-A37B",
        671,     # total_params_b: all experts and shared weights
        37,      # active_params_b: approximate parameters active per token
        61,      # layers
        7168,    # hidden_size
        0.125,   # gqa_ratio
    ),
    "gpt-oss-120b": Model(
        "GPT-OSS 120B",
        117,     # total_params_b
        5.1,     # active_params_b: approximate parameters active per token
        36,      # layers
        2880,    # hidden_size
        0.125,   # gqa_ratio
    ),
    "glm-5.2": Model(
        name="GLM-5.2",
        total_params_b=744.0,
        active_params_b=40.0,
        layers=78,
        hidden_size=6144,
        gqa_ratio=1.0,
        model_family="GLM",
        architecture="moe-dsa",
        # The official release reports 744B total and 40B activated
        # parameters.  A precise dense/routed parameter decomposition is not
        # published in config.json, so those optional fields are intentionally
        # left unset rather than fabricated.
        intermediate_size=12288,
        attention_heads=64,
        kv_heads=64,
        vocab_size=154880,
        max_context=1_048_576,
        experts=256,
        experts_per_token=8,
        shared_experts=1,
        moe_layer_frequency=1,
        default_weight_precision="fp8",
        default_kv_precision="fp16",
        supports_speculative=True,
        supports_mtp=True,
        supports_dsa=True,
        recommended_batch_size=64,
    ),
}


# Effective storage width for each model-weight format, in bits per parameter.
# The planner converts this value to model-weight memory with:
#
#     weight_GB = parameter_count_billions * bits_per_weight / 8
#
# FP16/BF16 and FP8/INT8 map directly to their nominal widths.  The q6/q5/q4/q3
# values intentionally include approximate quantization metadata and packing
# overhead, so they are slightly larger than the nominal data-bit count.
BITS_PER_WEIGHT = {
    "fp16": 16.0,
    "bf16": 16.0,
    "fp8": 8.0,
    "int8": 8.0,
    "q6": 6.5,
    "q5": 5.5,
    "q4": 4.5,
    "q3": 3.5,
}


# Planning scenarios
# ------------------
# Each Scenario(...) entry uses positional arguments in this order:
#
#   Scenario(
#       name,                       # Display/key name.
#       bandwidth_utilization,      # Sustained fraction of peak memory BW.
#       compute_utilization,        # Sustained fraction of peak compute.
#       communication_efficiency,   # Useful fraction after distributed comms.
#       kernel_overhead,             # Fractional runtime/kernel overhead.
#   )
#
# These values form a sensitivity range; they are not confidence intervals.
# ``conservative`` represents a less optimized or latency-sensitive stack,
# ``nominal`` is the default planning case, and ``optimistic`` represents a
# well-tuned engine and favorable workload.  Calibrate these assumptions with
# measurements from the exact model, quantization, engine, and batch shape.
SCENARIOS = {
    "conservative": Scenario(
        "conservative",
        0.35,  # bandwidth_utilization
        0.25,  # compute_utilization
        0.65,  # communication_efficiency
        0.15,  # kernel_overhead
    ),
    "nominal": Scenario(
        "nominal",
        0.55,  # bandwidth_utilization
        0.40,  # compute_utilization
        0.82,  # communication_efficiency
        0.08,  # kernel_overhead
    ),
    "optimistic": Scenario(
        "optimistic",
        0.72,  # bandwidth_utilization
        0.58,  # compute_utilization
        0.92,  # communication_efficiency
        0.04,  # kernel_overhead
    ),
}


# Default scaling efficiency associated with each multi-GPU interconnect class.
# A value of 1.0 would imply perfect scaling.  Lower values account, in one
# aggregate planning coefficient, for collective communication, synchronization,
# topology contention, and imperfect overlap of communication with computation.
# The estimator combines this topology factor with the scenario's
# ``communication_efficiency``; therefore neither value alone is the final
# multi-GPU efficiency.  Real scaling also depends on tensor/pipeline/expert
# parallel strategy, message sizes, host topology, and the inference engine.
TOPOLOGY_EFFICIENCY = {
    "pcie": 0.72,
    "nvlink": 0.88,
    "nvswitch": 0.95,
}
