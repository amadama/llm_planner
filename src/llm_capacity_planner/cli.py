import argparse,json
from pathlib import Path
from .catalog import GPUS,MODELS,BITS_PER_WEIGHT,SCENARIOS
from .types import Model
from .model import memory_capacity,estimate
from .report import print_report,write_json

def posint(v):
    n=int(v)
    if n<=0: raise argparse.ArgumentTypeError("must be greater than zero")
    return n
def posfloat(v):
    n=float(v)
    if n<=0: raise argparse.ArgumentTypeError("must be greater than zero")
    return n
def parser():
    p=argparse.ArgumentParser(description="Plan LLM inference capacity with prefill/decode roofline bounds.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--gpu",choices=sorted(GPUS));p.add_argument("--gpus",type=posint,default=1)
    p.add_argument("--model",choices=sorted(MODELS)+["custom"]);p.add_argument("--users","--concurrency",dest="users",type=posint,default=1)
    p.add_argument("--quant",choices=sorted(BITS_PER_WEIGHT),default="q4")
    p.add_argument("--input-tokens",type=posint,default=4096);p.add_argument("--output-tokens",type=posint,default=512);p.add_argument("--context",type=posint)
    p.add_argument("--tensor-parallel",type=posint,default=1)
    p.add_argument("--model-name");p.add_argument("--params-b",type=posfloat);p.add_argument("--active-params-b",type=posfloat);p.add_argument("--layers",type=posint,default=32);p.add_argument("--hidden-size",type=posint,default=4096);p.add_argument("--gqa-ratio",type=posfloat)
    p.add_argument("--kv-bits",type=posfloat,default=16);p.add_argument("--vram-utilization",type=float,default=.90);p.add_argument("--runtime-overhead",type=float,default=.08)
    p.add_argument("--bandwidth-utilization",type=float);p.add_argument("--compute-utilization",type=float);p.add_argument("--compute-tflops",type=posfloat)
    p.add_argument("--calibration-file");p.add_argument("--json-report");p.add_argument("--list",action="store_true")
    return p
def resolve(args):
    if args.model in MODELS:
        m=MODELS[args.model]
        return m if args.gqa_ratio is None else Model(m.name,m.total_params_b,m.active_params_b,m.layers,m.hidden_size,args.gqa_ratio)
    if args.params_b is None: raise ValueError("--params-b is required with --model custom")
    return Model(args.model_name or "Custom model",args.params_b,args.active_params_b or args.params_b,args.layers,args.hidden_size,args.gqa_ratio or .125)
def main(argv=None):
    p=parser();a=p.parse_args(argv)
    if a.list:
        print("GPU presets:");[print(f"  {k:<27} {v.architecture:<13} {v.vram_gb:g} GB, {v.bandwidth_gbps:g} GB/s, {v.topology}") for k,v in GPUS.items()]
        print("\nModel presets:");[print(f"  {k:<27} {v.total_params_b:g}B total, {v.active_params_b:g}B active") for k,v in MODELS.items()];return 0
    if not a.gpu or not a.model:p.error("--gpu and --model are required unless --list is used")
    if a.context is not None:a.input_tokens=a.context
    if a.tensor_parallel>a.gpus or a.gpus%a.tensor_parallel:p.error("--tensor-parallel must divide --gpus and cannot exceed it")
    if not .25<=a.vram_utilization<=.99:p.error("--vram-utilization must be between 0.25 and 0.99")
    for n in ("bandwidth_utilization","compute_utilization"):
        v=getattr(a,n)
        if v is not None and not .05<=v<=.95:p.error(f"--{n.replace('_','-')} must be between 0.05 and 0.95")
    try:
        gpu=GPUS[a.gpu];model=resolve(a);cal=None
        if a.calibration_file: cal=json.loads(Path(a.calibration_file).read_text())
        cap=memory_capacity(gpu,a.gpus,model,BITS_PER_WEIGHT[a.quant],a.kv_bits,a.input_tokens,a.output_tokens,a.runtime_overhead,a.vram_utilization)
        est=[estimate(s,gpu,a.gpus,model,a.quant,a.input_tokens,a.output_tokens,a.users,a.kv_bits,a.tensor_parallel,a.compute_tflops,a.bandwidth_utilization,a.compute_utilization,cal) for s in SCENARIOS.values()]
        print_report(a,gpu,model,cap,est)
        if a.json_report:write_json(a.json_report,a,gpu,model,cap,est);print(f"\nJSON report written to {a.json_report}")
    except (OSError,ValueError,json.JSONDecodeError) as e:p.error(str(e))
    return 0
