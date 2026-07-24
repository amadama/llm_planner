import unittest
from llm_capacity_planner.catalog import GPUS,MODELS,BITS_PER_WEIGHT,SCENARIOS
from llm_capacity_planner.model import memory_capacity,estimate

class PlannerTests(unittest.TestCase):
    def test_memory_capacity_positive(self):
        c=memory_capacity(GPUS["h200-sxm"],1,MODELS["llama-3.1-70b"],BITS_PER_WEIGHT["q4"],16,4096,512,.08,.9)
        self.assertGreater(c["max_concurrency"],0)
    def test_batching_increases_aggregate(self):
        kw=dict(scenario=SCENARIOS["nominal"],gpu=GPUS["h200-sxm"],gpus=1,model=MODELS["llama-3.1-70b"],quant="q4",input_tokens=4096,output_tokens=512,kv_bits=16,tp=1)
        one=estimate(users=1,**kw); sixteen=estimate(users=16,**kw)
        self.assertGreater(sixteen.output_tps,one.output_tps)
        self.assertLess(sixteen.per_user_tps,one.per_user_tps)

    def test_t4_catalog_entry(self):
        t4 = GPUS["t4"]
        self.assertEqual(t4.vram_gb, 16)
        self.assertEqual(t4.bandwidth_gbps, 320)
        self.assertEqual(t4.int8_tops, 130)
        self.assertEqual(t4.int4_tops, 260)
        self.assertEqual(t4.topology, "pcie")

    def test_longer_context_uses_more_vram(self):
        g=GPUS["h200-sxm"];m=MODELS["llama-3.1-70b"]
        a=memory_capacity(g,1,m,BITS_PER_WEIGHT["q4"],16,2048,512,.08,.9)
        b=memory_capacity(g,1,m,BITS_PER_WEIGHT["q4"],16,32768,512,.08,.9)
        self.assertGreater(b["kv_per_request_gb"],a["kv_per_request_gb"])

if __name__=="__main__":unittest.main()


def test_glm_52_catalog_metadata():
    model = MODELS["glm-5.2"]
    assert model.total_params_b == 744.0
    assert model.active_params_b == 40.0
    assert model.layers == 78
    assert model.hidden_size == 6144
    assert model.experts == 256
    assert model.experts_per_token == 8
    assert model.max_context == 1_048_576
    assert model.supports_dsa is True
    assert model.supports_mtp is True


def test_glm_52_capacity_estimate_runs():
    gpu = GPUS["h200-sxm"]
    model = MODELS["glm-5.2"]
    scenario = SCENARIOS["nominal"]
    result = estimate(
        scenario, gpu, 8, model, "fp8", 4096, 512, 32, 16.0, 8
    )
    assert result.output_tps > 0
    assert result.prompt_tps > 0
