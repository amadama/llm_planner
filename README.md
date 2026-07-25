# LLM Capacity Planner

A transparent analytical planner for LLM inference. It separates prompt prefill from autoregressive decode and reports input throughput, aggregate and per-user output throughput, TTFT, ITL, request latency, request rate, VRAM use, and long-period token capacity.

The planner uses a phase-specific roofline model rather than a single opaque tokens/s multiplier. It reports conservative, nominal, and optimistic scenarios and can be calibrated from measured benchmark results.

## Install

```bash
python -m pip install -e .
```

Or run without installation:

```bash
PYTHONPATH=src python -m llm_capacity_planner --list
```

## Example

```bash
PYTHONPATH=src python -m llm_capacity_planner \
  --gpu h200-sxm \
  --gpus 1 \
  --model llama-3.1-70b \
  --quant q4 \
  --users 16 \
  --input-tokens 4096 \
  --output-tokens 512
```

Generate JSON:

```bash
PYTHONPATH=src python -m llm_capacity_planner ... --json-report report.json
```

Calibrate with `--calibration-file examples/calibration.json`. Calibration should come from the exact GPU, model, quantization, serving engine, input/output lengths, and concurrency.

## Model boundaries

This is an analytical capacity planner, not a kernel simulator. Uncalibrated results are engineering bounds. Benchmark the exact serving stack before procurement or production SLO commitments.

## GLM-5.2 example

```bash
PYTHONPATH=src llm_capacity_planner --gpu h200-sxm --gpus 8 --tensor-parallel 8 \
  --model glm-5.2 --quant fp8 --users 32 \
  --input-tokens 4096 --output-tokens 512
```

The catalog models GLM-5.2 as a 744B-total, 40B-active MoE model.
