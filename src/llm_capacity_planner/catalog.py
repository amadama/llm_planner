"""Built-in hardware, runtime, model, precision, and scenario catalogs.

The catalogs deliberately separate three concerns:

* ``GPUS`` contains vendor-neutral, mostly immutable hardware capabilities.
* ``RUNTIMES`` contains CUDA/ROCm platform and inference-engine assumptions.
* ``MODELS`` contains model architecture and serving characteristics.

Runtime efficiencies and scenario multipliers are assumptions. They should be
calibrated using measurements from the exact software version and workload.
"""

from .types import GPU, Model, RuntimeStack, Scenario


GPUS = {
    "rtx-pro-6000-blackwell": GPU(
        "NVIDIA RTX PRO 6000 Blackwell", "NVIDIA", "Blackwell", 96, "GDDR7",
        1792, 125, 125, 250, 250, "pcie", interconnect="PCIe",
    ),
    "h200-sxm": GPU(
        "NVIDIA H200 SXM", "NVIDIA", "Hopper", 141, "HBM3e",
        4800, 989, 989, 1979, 1979, "nvswitch",
        interconnect="NVLink/NVSwitch",
    ),
    "h100-sxm": GPU(
        "NVIDIA H100 SXM", "NVIDIA", "Hopper", 80, "HBM3",
        3350, 989, 989, 1979, 1979, "nvswitch",
        interconnect="NVLink/NVSwitch",
    ),
    "a100-80gb": GPU(
        "NVIDIA A100 80GB SXM", "NVIDIA", "Ampere", 80, "HBM2e",
        2039, 312, 312, None, 624, "nvlink", interconnect="NVLink",
    ),
    "rtx-5090": GPU(
        "NVIDIA GeForce RTX 5090", "NVIDIA", "Blackwell", 32, "GDDR7",
        1792, 105, 105, 210, 210, "pcie", interconnect="PCIe",
    ),
    "rtx-4090": GPU(
        "NVIDIA GeForce RTX 4090", "NVIDIA", "Ada Lovelace", 24, "GDDR6X",
        1008, 82.6, 82.6, None, 165.2, "pcie", interconnect="PCIe",
    ),
    "t4": GPU(
        "NVIDIA T4", "NVIDIA", "Turing", 16, "GDDR6", 320,
        65, 65, None, 130, "pcie", fp32_tflops=8.1,
        int4_dense_tops=260, power_w=70, interconnect="PCIe Gen3 x16",
        host_link="PCIe Gen3 x16", host_link_bandwidth_gbps=32,
    ),
    "mi300x": GPU(
        "AMD Instinct MI300X", "AMD", "CDNA 3", 192, "HBM3", 5325,
        653.7, 653.7, 1307.4, 1307.4, "infinity-fabric",
        fp32_tflops=163.4, power_w=750, interconnect="AMD Infinity Fabric",
        host_link="PCIe 5.0 x16", host_link_bandwidth_gbps=64,
    ),
    "mi325x": GPU(
        "AMD Instinct MI325X", "AMD", "CDNA 3", 256, "HBM3e", 6000,
        653.7, 653.7, 1307.4, 1307.4, "infinity-fabric",
        fp32_tflops=163.4, power_w=1000, interconnect="AMD Infinity Fabric",
        host_link="PCIe 5.0 x16", host_link_bandwidth_gbps=64,
    ),
    "radeon-pro-w7900": GPU(
        "AMD Radeon PRO W7900", "AMD", "RDNA 3", 48, "GDDR6", 864,
        61.0, 61.0, None, 123.0, "pcie", fp32_tflops=61.0,
        int4_dense_tops=245.0, power_w=295, interconnect="PCIe 4.0 x16",
        host_link="PCIe 4.0 x16", host_link_bandwidth_gbps=32,
    ),
}


# Runtime efficiencies are initial planning defaults. They are intentionally
# separate from GPU records because the same hardware can perform differently
# under TensorRT-LLM, vLLM, SGLang, or llama.cpp and across CUDA/ROCm versions.
RUNTIMES = {
    "tensorrt-llm-cuda": RuntimeStack(
        "TensorRT-LLM on CUDA", "CUDA", "TensorRT-LLM", ("NVIDIA",),
        0.64, 0.53, 0.92, 0.055,
        ("fp16", "bf16", "fp8", "int8", "q4"),
        supports_speculative=True, supports_graph_capture=True,
        notes="NVIDIA-specific, throughput-oriented baseline.",
    ),
    "vllm-cuda": RuntimeStack(
        "vLLM on CUDA", "CUDA", "vLLM", ("NVIDIA",),
        0.57, 0.44, 0.86, 0.075,
        ("fp16", "bf16", "fp8", "int8", "q6", "q5", "q4", "q3"),
        supports_speculative=True, supports_graph_capture=True,
    ),
    "vllm-rocm": RuntimeStack(
        "vLLM on ROCm", "ROCm", "vLLM", ("AMD",),
        0.53, 0.40, 0.82, 0.085,
        ("fp16", "bf16", "fp8", "int8", "q6", "q5", "q4", "q3"),
        supports_speculative=True, supports_graph_capture=True,
    ),
    "sglang-cuda": RuntimeStack(
        "SGLang on CUDA", "CUDA", "SGLang", ("NVIDIA",),
        0.60, 0.48, 0.88, 0.065,
        ("fp16", "bf16", "fp8", "int8", "q4"),
        supports_speculative=True, supports_graph_capture=True,
    ),
    "sglang-rocm": RuntimeStack(
        "SGLang on ROCm", "ROCm", "SGLang", ("AMD",),
        0.51, 0.39, 0.81, 0.09,
        ("fp16", "bf16", "fp8", "int8", "q4"),
        supports_speculative=True,
    ),
    "llama-cpp-cuda": RuntimeStack(
        "llama.cpp on CUDA", "CUDA", "llama.cpp", ("NVIDIA",),
        0.48, 0.35, 0.72, 0.11,
        ("fp16", "bf16", "q6", "q5", "q4", "q3"),
    ),
    "llama-cpp-rocm": RuntimeStack(
        "llama.cpp on ROCm", "ROCm", "llama.cpp", ("AMD",),
        0.44, 0.32, 0.69, 0.12,
        ("fp16", "bf16", "q6", "q5", "q4", "q3"),
    ),
}

DEFAULT_RUNTIME_BY_VENDOR = {"NVIDIA": "vllm-cuda", "AMD": "vllm-rocm"}


MODELS = {
    "llama-3.1-8b": Model("Llama 3.1 8B", 8, 8, 32, 4096, 0.25),
    "llama-3.1-70b": Model("Llama 3.1/3.3 70B", 70, 70, 80, 8192, 0.125),
    "qwen2.5-14b": Model("Qwen 2.5 14B", 14.7, 14.7, 48, 5120, 0.125),
    "qwen2.5-32b": Model("Qwen 2.5 32B", 32.5, 32.5, 64, 5120, 0.125),
    "qwen3-30b-a3b": Model("Qwen3 30B-A3B", 30.5, 3.3, 48, 2048, 0.125, architecture="moe"),
    "deepseek-r1-671b": Model("DeepSeek-R1 671B-A37B", 671, 37, 61, 7168, 0.125, architecture="moe"),
    "gpt-oss-120b": Model("GPT-OSS 120B", 117, 5.1, 36, 2880, 0.125, architecture="moe"),
    "glm-5.2": Model(
        name="GLM-5.2", total_params_b=744.0, active_params_b=40.0,
        layers=78, hidden_size=6144, gqa_ratio=1.0, model_family="GLM",
        architecture="moe-dsa", intermediate_size=12288,
        attention_heads=64, kv_heads=64, vocab_size=154880,
        max_context=1_048_576, experts=256, experts_per_token=8,
        shared_experts=1, moe_layer_frequency=1,
        default_weight_precision="fp8", default_kv_precision="fp16",
        supports_speculative=True, supports_mtp=True, supports_dsa=True,
        recommended_batch_size=64,
    ),
}

BITS_PER_WEIGHT = {
    "fp16": 16.0, "bf16": 16.0, "fp8": 8.0, "int8": 8.0,
    "q6": 6.5, "q5": 5.5, "q4": 4.5, "q3": 3.5,
}

# Sensitivity factors around each runtime profile. These are not independent
# absolute utilization assumptions and therefore avoid double-counting stack
# efficiency.
SCENARIOS = {
    "conservative": Scenario("conservative", 0.78, 0.72, 0.84, 1.35),
    "nominal": Scenario("nominal", 1.00, 1.00, 1.00, 1.00),
    "optimistic": Scenario("optimistic", 1.16, 1.22, 1.08, 0.72),
}

TOPOLOGY_EFFICIENCY = {
    "pcie": 0.72,
    "nvlink": 0.88,
    "nvswitch": 0.95,
    "infinity-fabric": 0.91,
}
