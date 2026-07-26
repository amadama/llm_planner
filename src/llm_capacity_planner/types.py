from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GPU:
    """Vendor-neutral accelerator hardware record.

    Peak compute values are *dense* matrix rates unless a field explicitly says
    otherwise. Software-stack efficiency does not belong here; it is represented
    by :class:`RuntimeStack`. The optional hardware calibration multiplier is a
    benchmark-derived correction for a specific card implementation.
    """

    name: str
    vendor: str
    architecture: str
    vram_gb: float
    memory_type: str
    bandwidth_gbps: float
    fp16_dense_tflops: float
    bf16_dense_tflops: float
    fp8_dense_tflops: Optional[float]
    int8_dense_tops: Optional[float]
    topology: str
    fp32_tflops: Optional[float] = None
    int4_dense_tops: Optional[float] = None
    power_w: Optional[float] = None
    interconnect: Optional[str] = None
    host_link: Optional[str] = None
    host_link_bandwidth_gbps: Optional[float] = None
    hardware_bandwidth_calibration: float = 1.0
    hardware_compute_calibration: float = 1.0
    benchmark_multiplier: float = 1.0

    # Compatibility aliases for older callers and reports.
    @property
    def fp16_tflops(self) -> float:
        return self.fp16_dense_tflops

    @property
    def fp8_tflops(self) -> float:
        return self.fp8_dense_tflops or self.fp16_dense_tflops

    @property
    def int8_tops(self) -> float:
        return self.int8_dense_tops or self.fp16_dense_tflops

    @property
    def int4_tops(self) -> Optional[float]:
        return self.int4_dense_tops


@dataclass(frozen=True)
class RuntimeStack:
    """Software platform and inference-engine planning profile.

    Efficiencies are baseline sustained fractions or multipliers for the named
    stack. They are planning assumptions, not vendor specifications. Calibrate
    them with benchmark data from the exact engine version and workload.
    """

    name: str
    platform: str
    engine: str
    compatible_vendors: Tuple[str, ...]
    bandwidth_efficiency: float
    compute_efficiency: float
    communication_efficiency: float
    kernel_overhead: float
    supported_quantizations: Tuple[str, ...]
    supports_paged_attention: bool = True
    supports_continuous_batching: bool = True
    supports_speculative: bool = False
    supports_graph_capture: bool = False
    notes: str = ""


@dataclass(frozen=True)
class Model:
    """Architecture and capacity-planning metadata for one model preset."""

    name: str
    total_params_b: float
    active_params_b: float
    layers: int
    hidden_size: int
    gqa_ratio: float
    model_family: Optional[str] = None
    architecture: str = "dense"
    dense_params_b: Optional[float] = None
    routed_params_b: Optional[float] = None
    intermediate_size: Optional[int] = None
    attention_heads: Optional[int] = None
    kv_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    max_context: Optional[int] = None
    experts: Optional[int] = None
    experts_per_token: Optional[int] = None
    shared_experts: Optional[int] = None
    moe_layer_frequency: Optional[int] = None
    default_weight_precision: Optional[str] = None
    default_kv_precision: Optional[str] = None
    supports_speculative: bool = False
    supports_mtp: bool = False
    supports_dsa: bool = False
    recommended_batch_size: Optional[int] = None


@dataclass(frozen=True)
class Scenario:
    """Sensitivity multipliers around a runtime profile's baseline assumptions."""

    name: str
    bandwidth_multiplier: float
    compute_multiplier: float
    communication_multiplier: float
    overhead_multiplier: float


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
