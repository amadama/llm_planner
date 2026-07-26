import unittest

from llm_capacity_planner.catalog import (
    BITS_PER_WEIGHT,
    DEFAULT_RUNTIME_BY_VENDOR,
    GPUS,
    MODELS,
    RUNTIMES,
    SCENARIOS,
)
from llm_capacity_planner.model import estimate, memory_capacity, precision_peak_tflops


class PlannerTests(unittest.TestCase):
    def test_memory_capacity_positive(self):
        c = memory_capacity(
            GPUS["h200-sxm"], 1, MODELS["llama-3.1-70b"],
            BITS_PER_WEIGHT["q4"], 16, 4096, 512, 0.08, 0.9,
        )
        self.assertGreater(c["max_concurrency"], 0)

    def test_batching_increases_aggregate(self):
        kw = dict(
            scenario=SCENARIOS["nominal"], runtime=RUNTIMES["vllm-cuda"],
            gpu=GPUS["h200-sxm"], gpus=1, model=MODELS["llama-3.1-70b"],
            quant="q4", input_tokens=4096, output_tokens=512, kv_bits=16, tp=1,
        )
        one = estimate(users=1, **kw)
        sixteen = estimate(users=16, **kw)
        self.assertGreater(sixteen.output_tps, one.output_tps)
        self.assertLess(sixteen.per_user_tps, one.per_user_tps)

    def test_t4_catalog_entry(self):
        t4 = GPUS["t4"]
        self.assertEqual(t4.vendor, "NVIDIA")
        self.assertEqual(t4.vram_gb, 16)
        self.assertEqual(t4.bandwidth_gbps, 320)
        self.assertEqual(t4.int8_dense_tops, 130)
        self.assertEqual(t4.int4_dense_tops, 260)

    def test_amd_catalog_entries(self):
        mi300x = GPUS["mi300x"]
        self.assertEqual(mi300x.vendor, "AMD")
        self.assertEqual(mi300x.vram_gb, 192)
        self.assertEqual(mi300x.bandwidth_gbps, 5325)
        self.assertEqual(mi300x.topology, "infinity-fabric")
        self.assertEqual(GPUS["mi325x"].vram_gb, 256)

    def test_runtime_vendor_compatibility(self):
        with self.assertRaises(ValueError):
            estimate(
                SCENARIOS["nominal"], RUNTIMES["vllm-rocm"],
                GPUS["h200-sxm"], 1, MODELS["llama-3.1-8b"],
                "q4", 1024, 128, 1, 16, 1,
            )

    def test_vendor_defaults(self):
        self.assertEqual(DEFAULT_RUNTIME_BY_VENDOR["NVIDIA"], "vllm-cuda")
        self.assertEqual(DEFAULT_RUNTIME_BY_VENDOR["AMD"], "vllm-rocm")

    def test_no_fake_fp8_on_t4(self):
        with self.assertRaises(ValueError):
            precision_peak_tflops(GPUS["t4"], "fp8")

    def test_longer_context_uses_more_vram(self):
        gpu = GPUS["h200-sxm"]
        model = MODELS["llama-3.1-70b"]
        short = memory_capacity(gpu, 1, model, BITS_PER_WEIGHT["q4"], 16, 2048, 512, 0.08, 0.9)
        long = memory_capacity(gpu, 1, model, BITS_PER_WEIGHT["q4"], 16, 32768, 512, 0.08, 0.9)
        self.assertGreater(long["kv_per_request_gb"], short["kv_per_request_gb"])

    def test_glm_52_catalog_metadata(self):
        model = MODELS["glm-5.2"]
        self.assertEqual(model.total_params_b, 744.0)
        self.assertEqual(model.active_params_b, 40.0)
        self.assertEqual(model.experts, 256)
        self.assertTrue(model.supports_dsa)
        self.assertTrue(model.supports_mtp)

    def test_glm_52_capacity_estimate_runs_on_amd(self):
        result = estimate(
            SCENARIOS["nominal"], RUNTIMES["vllm-rocm"],
            GPUS["mi325x"], 4, MODELS["glm-5.2"], "fp8",
            4096, 512, 32, 16.0, 4,
        )
        self.assertGreater(result.output_tps, 0)
        self.assertGreater(result.prompt_tps, 0)


if __name__ == "__main__":
    unittest.main()
