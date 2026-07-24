from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GPU:
    """Hardware and planner metadata for one accelerator preset.

    Published hardware fields describe the physical card.  Planner calibration
    fields are explicit assumptions and must not be mistaken for vendor specs.
    The current roofline model still uses the scenario-level utilization values;
    the per-GPU calibration fields are retained for reporting and future
    benchmark-derived tuning.
    """

    name: str
    architecture: str
    vram_gb: float
    bandwidth_gbps: float
    fp16_tflops: float
    fp8_tflops: float
    int8_tops: float
    topology: str

    # Additional published specifications.  ``None`` means the precision or
    # feature is not natively supported or no defensible catalog value is set.
    fp32_tflops: Optional[float] = None
    int4_tops: Optional[float] = None
    power_w: Optional[float] = None
    interconnect: Optional[str] = None
    host_link_bandwidth_gbps: Optional[float] = None

    # Planner calibration assumptions, not vendor specifications.
    sustained_bandwidth_efficiency: float = 0.50
    sustained_compute_efficiency: float = 0.40
    benchmark_multiplier: float = 1.0


@dataclass(frozen=True)
class Model:
    """Architecture and capacity-planning metadata for one model preset.

    The first six fields are the stable core consumed by the current roofline
    planner.  The optional fields describe dense/MoE structure and model
    capabilities so future versions can model expert parallelism, sparse
    attention, and multi-token prediction without another catalog redesign.

    ``total_params_b`` determines weight-memory capacity.  ``active_params_b``
    determines the first-order compute and active-weight traffic per token.
    For dense models they are normally equal; for MoE models active parameters
    are substantially lower than total stored parameters.
    """

    name: str
    total_params_b: float
    active_params_b: float
    layers: int
    hidden_size: int
    gqa_ratio: float

    # Descriptive architecture metadata.
    model_family: Optional[str] = None
    architecture: str = "dense"

    # Optional parameter decomposition, in billions.  These fields are
    # informational today; total_params_b and active_params_b remain the source
    # of truth for capacity and roofline calculations.
    dense_params_b: Optional[float] = None
    routed_params_b: Optional[float] = None

    # Transformer and attention details.
    intermediate_size: Optional[int] = None
    attention_heads: Optional[int] = None
    kv_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    max_context: Optional[int] = None

    # Mixture-of-experts details.
    experts: Optional[int] = None
    experts_per_token: Optional[int] = None
    shared_experts: Optional[int] = None
    moe_layer_frequency: Optional[int] = None

    # Default deployment assumptions and architecture features.
    default_weight_precision: Optional[str] = None
    default_kv_precision: Optional[str] = None
    supports_speculative: bool = False
    supports_mtp: bool = False
    supports_dsa: bool = False
    recommended_batch_size: Optional[int] = None


@dataclass(frozen=True)
class Scenario:
    name: str
    bandwidth_utilization: float
    compute_utilization: float
    communication_efficiency: float
    kernel_overhead: float


@dataclass(frozen=True)
class Estimate:
    scenario: str
    prompt_tps: float
    output_tps: float
    per_user_tps: float
    ttft_s: float
    itl_ms: float
    request_latency_s: float
    requests_per_s: float
    decode_compute_bound_tps: float
    decode_bandwidth_bound_tps: float
    prefill_compute_bound_tps: float
    prefill_bandwidth_bound_tps: float
