# LLM Capacity Planner

A transparent analytical capacity planner for large-language-model inference on NVIDIA and AMD GPUs.

The planner estimates:

- prompt/prefill throughput in input tokens per second;
- aggregate decode throughput in output tokens per second;
- per-user decode rate;
- time to first token (TTFT);
- inter-token latency (ITL);
- end-to-end request latency;
- request throughput;
- model-weight, runtime-overhead, and KV-cache memory use;
- memory-limited maximum concurrency; and
- conservative, nominal, and optimistic planning ranges.

It separates **hardware**, **runtime**, and **model** assumptions so that a GPU's published capabilities are not confused with CUDA/ROCm engine efficiency.

> This is an analytical planner, not a kernel simulator or benchmark replacement. Use it to compare configurations, identify bottlenecks, and form capacity hypotheses. Calibrate against the exact hardware, runtime version, model build, quantization, context distribution, batch policy, and parallel topology before procurement or production SLO commitments.

## Project layout

```text
llm-capacity-planner/
├── examples/
│   ├── calibration.json
│   └── sample-report.json
├── src/llm_capacity_planner/
│   ├── __init__.py
│   ├── __main__.py
│   ├── catalog.py       # GPU, runtime, model, precision, and scenario presets
│   ├── cli.py           # command-line parsing and validation
│   ├── model.py         # analytical capacity and roofline calculations
│   ├── report.py        # terminal and JSON reporting
│   └── types.py         # catalog and result dataclasses
├── tests/test_model.py
└── pyproject.toml
```

## Catalog architecture

The built-in catalog is deliberately split into independent layers:

### Hardware catalog: `GPUS`

Contains vendor-neutral accelerator specifications and topology metadata:

- vendor and architecture;
- VRAM capacity and memory type;
- peak memory bandwidth;
- dense FP16, BF16, FP8, INT8, and optional INT4 capability;
- interconnect and host-link information; and
- optional benchmark-derived hardware correction factors.

CUDA or ROCm efficiency does **not** belong in a GPU record.

Built-in NVIDIA presets include RTX PRO 6000 Blackwell, H200, H100, A100 80 GB, RTX 5090, RTX 4090, and T4. Built-in AMD presets include MI300X, MI325X, and Radeon PRO W7900.

### Runtime catalog: `RUNTIMES`

Contains platform- and engine-specific planning assumptions:

- CUDA or ROCm platform;
- inference engine;
- compatible GPU vendors;
- sustained bandwidth and compute efficiency;
- collective-communication efficiency;
- kernel overhead;
- supported quantization formats; and
- features such as paged attention, continuous batching, speculative decoding, and graph capture.

Built-in profiles include:

```text
tensorrt-llm-cuda
vllm-cuda
vllm-rocm
sglang-cuda
sglang-rocm
llama-cpp-cuda
llama-cpp-rocm
```

When `--runtime` is omitted, the planner selects:

```text
NVIDIA → vllm-cuda
AMD    → vllm-rocm
```

Incompatible combinations are rejected. For example, `vllm-rocm` cannot be paired with an H200, and TensorRT-LLM cannot be paired with an AMD GPU.

### Model catalog: `MODELS`

Contains architecture and serving metadata:

- total parameters, used for model-memory sizing;
- active parameters per token, used for decode work and active-weight traffic;
- transformer layers and hidden size;
- grouped-query-attention ratio;
- optional attention-head, KV-head, vocabulary, and context metadata;
- MoE expert counts and routing metadata; and
- feature flags for speculative decoding, MTP, and DSA.

Built-in models include Llama 3.1/3.3, Qwen 2.5, Qwen3 MoE, DeepSeek-R1, GPT-OSS 120B, and GLM-5.2.

### Scenario catalog: `SCENARIOS`

The planner evaluates every workload using conservative, nominal, and optimistic multipliers around the selected runtime profile. The scenarios vary sustained bandwidth, sustained compute, communication efficiency, and runtime overhead. They are sensitivity ranges, not three separate hardware specifications.

## Requirements

- Python 3.9 or newer
- No third-party runtime dependencies

## Installation

From the project directory:

```bash
python -m pip install -e .
```

This installs the `llm-capacity-planner` command.

Run directly without installing:

```bash
PYTHONPATH=src python -m llm_capacity_planner --list
```

## Quick start

List all GPU, runtime, and model presets:

```bash
llm-capacity-planner --list
```

NVIDIA H200 with TensorRT-LLM:

```bash
llm-capacity-planner \
  --gpu h200-sxm \
  --runtime tensorrt-llm-cuda \
  --gpus 1 \
  --model llama-3.1-70b \
  --quant q4 \
  --users 16 \
  --input-tokens 4096 \
  --output-tokens 512
```

AMD MI300X with vLLM on ROCm:

```bash
llm-capacity-planner \
  --gpu mi300x \
  --runtime vllm-rocm \
  --gpus 1 \
  --model llama-3.1-70b \
  --quant q4 \
  --users 16 \
  --input-tokens 4096 \
  --output-tokens 512
```

GLM-5.2 on four MI325X GPUs:

```bash
llm-capacity-planner \
  --gpu mi325x \
  --gpus 4 \
  --tensor-parallel 4 \
  --runtime vllm-rocm \
  --model glm-5.2 \
  --quant fp8 \
  --users 32 \
  --input-tokens 4096 \
  --output-tokens 512
```

## Command-line reference

### Required workload selection

#### `--gpu PRESET`

Selects a GPU from `GPUS`, such as `h200-sxm`, `mi300x`, or `t4`.

Required unless `--list` is used.

#### `--model PRESET|custom`

Selects a built-in model or enables custom-model fields.

Required unless `--list` is used.

### Hardware and topology

#### `--gpus N`

Number of identical GPUs in the configuration. Default: `1`.

Total usable VRAM is calculated from this count. Compute and bandwidth scale through the topology and tensor-parallel model rather than by assuming perfect linear scaling.

#### `--tensor-parallel N`

Tensor-parallel degree. Default: `1`.

It must divide `--gpus` exactly and cannot exceed the GPU count. Examples:

```text
--gpus 8 --tensor-parallel 8   valid
--gpus 8 --tensor-parallel 4   valid
--gpus 8 --tensor-parallel 3   invalid
```

The remaining GPU factor is treated as replicated capacity outside the tensor-parallel group.

#### `--runtime PRESET`

Selects the software platform and inference-engine profile. When omitted, the runtime defaults by GPU vendor.

The runtime controls sustained bandwidth/compute assumptions, communication efficiency, kernel overhead, quantization support, and compatibility validation.

### Workload shape

#### `--users N` / `--concurrency N`

Number of concurrent active requests. Default: `1`.

This is an active batch, not a request-arrival rate. Increasing it can improve aggregate output throughput by amortizing model-weight traffic, while usually reducing per-user output rate and increasing latency.

#### `--input-tokens N`

Average input/prompt length per request. Default: `4096`.

It affects:

- prompt work and prompt throughput;
- KV-cache allocation;
- time to first token; and
- decode attention work over the existing context.

#### `--output-tokens N`

Average number of generated tokens per request. Default: `512`.

It affects:

- KV-cache allocation;
- average decode context;
- request latency; and
- conversion from output tokens/second to requests/second.

#### `--context N`

Backward-compatible alias for `--input-tokens`.

When both are present, `--context` replaces the input-token value. New scripts should use `--input-tokens`.

### Precision and memory

#### `--quant FORMAT`

Weight format. Default: `q4`.

Built-in formats:

| Format | Effective bits/weight |
|---|---:|
| `fp16` | 16.0 |
| `bf16` | 16.0 |
| `fp8` | 8.0 |
| `int8` | 8.0 |
| `q6` | 6.5 |
| `q5` | 5.5 |
| `q4` | 4.5 |
| `q3` | 3.5 |

The non-integer values include approximate quantization metadata overhead. The selected runtime must advertise support for the format, and the GPU must expose a defensible compute capability. For example, native FP8 is rejected on the NVIDIA T4.

#### `--kv-bits BITS`

KV-cache precision in bits per element. Default: `16`.

Lower values reduce KV memory and KV traffic, but this option does not independently validate whether a selected runtime/model combination implements that KV precision.

#### `--vram-utilization FRACTION`

Fraction of physical VRAM considered usable. Default: `0.90`; allowed range: `0.25` to `0.99`.

Reserve headroom for the driver, runtime allocations, memory fragmentation, temporary workspaces, and operational safety.

#### `--runtime-overhead FRACTION`

Additional fixed VRAM as a fraction of model-weight memory. Default: `0.08`.

For example, `0.08` adds 8% of the model-weight footprint for runtime and workspace allocations. This affects capacity/fitting only; runtime kernel overhead is separately defined by the runtime profile.

### Custom models

Use `--model custom` with the following fields.

#### `--model-name TEXT`

Display name for the custom model. Default: `Custom model`.

#### `--params-b BILLIONS`

Total parameter count in billions. Required for custom models.

This determines the full model-weight footprint.

#### `--active-params-b BILLIONS`

Parameters active per generated token in billions. Defaults to `--params-b`.

For a dense model, this normally equals total parameters. For MoE models, use the routed/active amount to improve decode compute and active-weight-traffic estimates while retaining total parameters for VRAM sizing.

#### `--layers N`

Transformer layer count. Default: `32`.

Used in KV-cache and attention calculations.

#### `--hidden-size N`

Transformer hidden dimension. Default: `4096`.

Used in KV-cache, activation traffic, and attention calculations.

#### `--gqa-ratio RATIO`

KV heads divided by attention heads. Built-in models use catalog values; custom models default to `0.125`.

Examples:

```text
MHA: ratio = 1.0
8 KV heads / 64 attention heads: ratio = 0.125
```

Smaller ratios reduce KV-cache memory and KV traffic.

Custom MoE example:

```bash
llm-capacity-planner \
  --gpu mi325x --gpus 4 --tensor-parallel 4 \
  --runtime vllm-rocm \
  --model custom --model-name "Example 400B-A40B" \
  --params-b 400 --active-params-b 40 \
  --layers 80 --hidden-size 8192 --gqa-ratio 0.125 \
  --quant fp8 --users 32 \
  --input-tokens 4096 --output-tokens 512
```

### Analytical overrides

These options are useful for calibration, sensitivity analysis, or hardware not fully represented by the catalog.

#### `--bandwidth-utilization FRACTION`

Overrides the final sustained memory-bandwidth fraction for all scenarios. Allowed range: `0.05` to `0.95`.

When supplied, it replaces `runtime.bandwidth_efficiency × scenario.bandwidth_multiplier`. Hardware calibration, topology scaling, and benchmark multiplier still apply.

#### `--compute-utilization FRACTION`

Overrides the final sustained compute fraction for all scenarios. Allowed range: `0.05` to `0.95`.

When supplied, it replaces `runtime.compute_efficiency × scenario.compute_multiplier`.

#### `--compute-tflops VALUE`

Overrides the per-GPU peak compute selected from GPU precision fields.

Use a **dense**, precision-appropriate value. Do not pass a structured-sparsity peak unless the actual workload and runtime use that sparsity.

### Calibration and output

#### `--calibration-file PATH`

Loads a JSON calibration record. The built-in example is:

```json
{
  "prompt_tps": 9500,
  "predicted_prompt_tps": 8906.4,
  "output_tps": 700,
  "predicted_output_tps": 624.5
}
```

The planner applies multiplicative corrections:

```text
prompt multiplier = measured prompt_tps / predicted_prompt_tps
output multiplier = measured output_tps / predicted_output_tps
```

If a `predicted_*` field is omitted, the current uncalibrated prediction is used as the denominator. For reproducible calibration, include all four fields and record the exact:

- GPU model and count;
- topology and tensor-parallel degree;
- runtime and version;
- model build and quantization;
- KV precision;
- input and output lengths;
- concurrency; and
- benchmark methodology.

Calibration is global within a run; it is not an interpolation database.

#### `--json-report PATH`

Writes the complete plan as JSON, including resolved CLI inputs, GPU/runtime/model records, capacity calculations, and all scenario estimates.

Example:

```bash
llm-capacity-planner \
  --gpu h200-sxm --model llama-3.1-70b \
  --users 16 --input-tokens 4096 --output-tokens 512 \
  --json-report capacity-report.json
```

#### `--list`

Prints GPU, runtime, and model preset identifiers and exits. `--gpu` and `--model` are not required with this option.

## How the planner calculates capacity

### Model-weight memory

```text
weight GB = total parameters (billions) × bits per weight ÷ 8
```

Total parameters are used even for MoE models because every resident expert contributes to the model footprint.

### KV-cache memory

Approximate KV bytes per sequence token:

```text
2 × layers × hidden size × GQA ratio × KV bytes/element
```

The factor of two represents key and value caches. Per-request KV allocation covers input plus generated output tokens.

### Maximum memory-limited concurrency

```text
usable VRAM = physical VRAM × GPU count × VRAM utilization
fixed memory = model weights + configured runtime memory overhead
max users = floor((usable VRAM - fixed memory) / KV GB per request)
```

A workload that does not fit is reported as `DOES NOT FIT`. The throughput numbers still assume all-GPU execution and therefore should not be interpreted as CPU-offload performance.

### Prefill model

The prefill phase computes all input tokens. The planner calculates both:

- a compute roof from effective precision compute divided by estimated model and attention FLOPs; and
- a memory roof from effective bandwidth divided by amortized weight, activation, and KV traffic.

Prompt throughput is the lower roof after runtime kernel overhead.

### Decode model

For autoregressive generation, the planner estimates:

- active model-weight traffic amortized over concurrent requests;
- KV-cache reads over the average sequence length;
- activation traffic;
- model and attention FLOPs per output token; and
- topology/runtime scaling.

Output throughput is the lower of the compute and bandwidth roofs after runtime overhead.

### Multi-GPU scaling

Scaling uses:

- GPU topology (`pcie`, `nvlink`, `nvswitch`, or `infinity-fabric`);
- runtime communication efficiency; and
- tensor-parallel degree.

It intentionally assumes less than perfect collective scaling. It does not explicitly simulate PCIe switch layout, NUMA placement, expert-parallel dispatch, pipeline bubbles, or networked multi-node communication.

## Understanding the report

The terminal report shows three scenarios.

### Prompt tok/s

Aggregate prompt tokens processed per second across the active request batch.

### Output tok/s

Aggregate generated output tokens per second across all active requests.

### Per-user tok/s

```text
aggregate output tok/s ÷ concurrent requests
```

This is the approximate streaming generation rate experienced by each active user under a symmetric workload.

### TTFT

Estimated time to first token, including batch prefill and an approximate first decode step.

### ITL

Inter-token latency in milliseconds, calculated from per-user output throughput.

### Request latency

Approximate time to process the prompt and generate the configured output length for one request under the concurrent workload.

### Requests/s

```text
aggregate output tok/s ÷ output tokens per request
```

This assumes a steady-state homogeneous workload.

### Prefill/decode bound

Indicates whether the nominal roofline result is limited by estimated compute or memory bandwidth.

### Tokens per hour/day/week/month/year

These are aggregate nominal output tokens extrapolated under continuous 100% utilization. They do not account for idle time, maintenance, queueing, failures, traffic variation, or SLO-driven throttling.

## Adding a GPU

Add a `GPU` record in `src/llm_capacity_planner/catalog.py` using dense published capabilities:

```python
"example-gpu": GPU(
    name="Example Accelerator",
    vendor="AMD",
    architecture="Example Architecture",
    vram_gb=192,
    memory_type="HBM3",
    bandwidth_gbps=5000,
    fp16_dense_tflops=600,
    bf16_dense_tflops=600,
    fp8_dense_tflops=1200,
    int8_dense_tops=1200,
    topology="infinity-fabric",
    power_w=750,
    interconnect="Example Fabric",
    host_link="PCIe 5.0 x16",
    host_link_bandwidth_gbps=64,
)
```

Use `None` when a native precision is unavailable. Avoid substituting sparse peak values for dense inference capability.

Optional calibration fields:

- `hardware_bandwidth_calibration` adjusts sustained bandwidth independently of runtime;
- `hardware_compute_calibration` adjusts sustained compute independently of runtime; and
- `benchmark_multiplier` applies to both and should normally remain `1.0`.

A new vendor also requires a compatible runtime profile and an entry in `DEFAULT_RUNTIME_BY_VENDOR` if automatic runtime selection is desired.

## Adding a runtime

Add a `RuntimeStack` record:

```python
"example-runtime": RuntimeStack(
    name="Example Engine on ROCm",
    platform="ROCm",
    engine="Example Engine",
    compatible_vendors=("AMD",),
    bandwidth_efficiency=0.52,
    compute_efficiency=0.40,
    communication_efficiency=0.82,
    kernel_overhead=0.08,
    supported_quantizations=("fp16", "bf16", "fp8", "int8", "q4"),
    supports_paged_attention=True,
    supports_continuous_batching=True,
    supports_speculative=False,
    supports_graph_capture=True,
    notes="Calibrate against the exact engine release.",
)
```

Runtime efficiency values are planning assumptions, not vendor specifications. Derive them from exact-stack benchmarks when possible.

## Adding a model

Dense model example:

```python
"example-70b": Model(
    name="Example 70B",
    total_params_b=70,
    active_params_b=70,
    layers=80,
    hidden_size=8192,
    gqa_ratio=0.125,
    architecture="dense",
)
```

MoE model example:

```python
"example-moe": Model(
    name="Example 400B-A40B",
    total_params_b=400,
    active_params_b=40,
    layers=80,
    hidden_size=8192,
    gqa_ratio=0.125,
    architecture="moe",
    experts=128,
    experts_per_token=8,
)
```

The current analytical model directly uses total parameters, active parameters, layer count, hidden size, and GQA ratio. Other metadata is retained for reporting and future model refinements.

## Testing and verification

Run the unit tests from the project root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Compile all package files:

```bash
python -m compileall -q src tests
```

Recommended smoke tests:

```bash
# Catalog listing
PYTHONPATH=src python -m llm_capacity_planner --list

# NVIDIA/CUDA
PYTHONPATH=src python -m llm_capacity_planner \
  --gpu h200-sxm --runtime vllm-cuda \
  --model llama-3.1-70b --quant q4 \
  --users 16 --input-tokens 4096 --output-tokens 512

# AMD/ROCm
PYTHONPATH=src python -m llm_capacity_planner \
  --gpu mi300x --runtime vllm-rocm \
  --model llama-3.1-70b --quant q4 \
  --users 16 --input-tokens 4096 --output-tokens 512
```

## Common errors

### `--gpu and --model are required`

Supply both options, or use `--list` by itself.

### `runtime ... is incompatible`

Select a runtime whose `compatible_vendors` includes the selected GPU vendor.

### `runtime ... does not support quantization`

Choose a format listed in the runtime profile or add a validated runtime profile that supports it.

### `has no native FP8 capability`

The GPU record has no defensible dense FP8 capability. Use another precision or accelerator.

### `--tensor-parallel must divide --gpus`

Choose a tensor-parallel degree that divides the total GPU count.

### `VRAM status: DOES NOT FIT`

Reduce concurrency, input/output length, KV precision, model precision, or model size; increase GPU count/capacity; or choose a larger-memory accelerator. The planner does not currently predict CPU-offload throughput.

## Important limitations

The current model does not explicitly simulate:

- request-arrival distributions and queueing;
- prefix caching or shared prompts;
- chunked prefill and prefill/decode disaggregation;
- speculative-decoding acceptance rates;
- multi-token prediction speedups;
- DSA-specific sparse-attention kernels;
- expert-routing imbalance and expert-parallel communication;
- pipeline-parallel bubbles;
- multi-node network fabric;
- model-specific quantization kernel quality;
- power, thermal, or clock throttling;
- SLO-constrained scheduler behavior; or
- heterogeneous request lengths.

Treat uncalibrated outputs as comparative engineering estimates. A defensible production plan combines this analytical model with benchmark records from the exact deployment stack.

## License

See [LICENSE](LICENSE).
