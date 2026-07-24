import math
from typing import Optional
from .types import GPU, Model, Scenario, Estimate
from .catalog import BITS_PER_WEIGHT, TOPOLOGY_EFFICIENCY

def model_weight_gb(params_b: float,bits: float)->float: return params_b*bits/8.0

def kv_bytes_per_token(model:Model,kv_bits:float,gqa_ratio:float)->float:
    return 2.0*model.layers*model.hidden_size*gqa_ratio*(kv_bits/8.0)

def model_flops_per_token(model:Model)->float: return 2.0*model.active_params_b*1e9

def attention_flops_per_token(model:Model,seq:int)->float: return 4.0*model.layers*model.hidden_size*seq

def precision_peak_tflops(gpu:GPU,quant:str)->float:
    if quant in {"fp16","bf16"}: return gpu.fp16_tflops
    if quant=="fp8": return gpu.fp8_tflops
    if quant=="int8": return gpu.int8_tops
    return gpu.int8_tops*min(8.0/BITS_PER_WEIGHT[quant],2.0)

def resource_scale(gpus:int,topology:str,comm_eff:float,tp:int)->float:
    if gpus==1: return 1.0
    base=TOPOLOGY_EFFICIENCY.get(topology,.72)
    collective=(base*comm_eff)**math.log2(max(1,tp))
    return tp*collective*(gpus/tp)

def memory_capacity(gpu:GPU,gpus:int,model:Model,weight_bits:float,kv_bits:float,input_tokens:int,output_tokens:int,runtime_overhead:float,vram_utilization:float)->dict:
    weights=model_weight_gb(model.total_params_b,weight_bits)
    overhead=weights*runtime_overhead
    kv_req=kv_bytes_per_token(model,kv_bits,model.gqa_ratio)*(input_tokens+output_tokens)/1e9
    usable=gpu.vram_gb*gpus*vram_utilization
    fixed=weights+overhead
    max_users=max(0,math.floor((usable-fixed)/max(kv_req,1e-12)))
    return {"weights_gb":weights,"runtime_overhead_gb":overhead,"kv_per_request_gb":kv_req,"usable_gb":usable,"fixed_gb":fixed,"max_concurrency":max_users}

def estimate(scenario:Scenario,gpu:GPU,gpus:int,model:Model,quant:str,input_tokens:int,output_tokens:int,users:int,kv_bits:float,tp:int,compute_tflops:Optional[float]=None,bw_util:Optional[float]=None,compute_util:Optional[float]=None,calibration:Optional[dict]=None)->Estimate:
    bits=BITS_PER_WEIGHT[quant]
    total_w=model_weight_gb(model.total_params_b,bits)*1e9
    active_w=model_weight_gb(model.active_params_b,bits)*1e9
    kv_bpt=kv_bytes_per_token(model,kv_bits,model.gqa_ratio)
    scale=resource_scale(gpus,gpu.topology,scenario.communication_efficiency,tp)
    effective_bw=gpu.bandwidth_gbps*1e9*(bw_util if bw_util is not None else scenario.bandwidth_utilization)*scale
    peak=compute_tflops if compute_tflops is not None else precision_peak_tflops(gpu,quant)
    effective_flops=peak*1e12*(compute_util if compute_util is not None else scenario.compute_utilization)*scale
    activation=model.layers*model.hidden_size*4.0
    prefill_flops=model_flops_per_token(model)+attention_flops_per_token(model,max(1,input_tokens//2))
    prefill_compute=effective_flops/max(prefill_flops,1)
    batch_prompt=max(1,input_tokens*users)
    prefill_bytes=total_w/batch_prompt+activation+kv_bpt
    prefill_bw=effective_bw/max(prefill_bytes,1)
    prompt=min(prefill_compute,prefill_bw)*(1-scenario.kernel_overhead)
    avg_ctx=input_tokens+max(0,output_tokens-1)/2
    decode_bytes=active_w/max(1,users)+kv_bpt*avg_ctx+activation
    decode_bw=effective_bw/max(decode_bytes,1)
    decode_flops=model_flops_per_token(model)+attention_flops_per_token(model,int(avg_ctx))
    decode_compute=effective_flops/max(decode_flops,1)
    output=min(decode_bw,decode_compute)*(1-scenario.kernel_overhead)
    if calibration:
        if calibration.get("prompt_tps"): prompt*=float(calibration["prompt_tps"])/max(float(calibration.get("predicted_prompt_tps",prompt)),1e-9)
        if calibration.get("output_tps"): output*=float(calibration["output_tps"])/max(float(calibration.get("predicted_output_tps",output)),1e-9)
    per_user=output/max(1,users)
    prefill_s=input_tokens*users/max(prompt,1e-9)
    ttft=prefill_s+users/max(output,1e-9)
    itl=1000/max(per_user,1e-9)
    latency=prefill_s+output_tokens/max(per_user,1e-9)
    rps=output/max(output_tokens,1)
    return Estimate(scenario.name,prompt,output,per_user,ttft,itl,latency,rps,decode_compute,decode_bw,prefill_compute,prefill_bw)
