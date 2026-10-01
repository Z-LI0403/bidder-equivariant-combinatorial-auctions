"""Repeat fixed E1 seed-11 MLP/DeepSet runs; exclude repeats from statistics."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import csv
import json
import torch
from .train import ROOT, run, save_json
from .audit import audit


def equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return isinstance(b,type(a)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def repeat(key,record):
    reference=ROOT/record["run"]
    config=json.loads((reference/"config.json").read_text())
    log=ROOT/"results/verification"/("reproduce_"+key+".log")
    with log.open("x",encoding="utf-8") as stream,redirect_stdout(stream),redirect_stderr(stream):
        directory=run(config,config["seed"])
        audit(directory)
    files=("datasets.pt","initial_model.pt","initial_validation_raw.pt","checkpoint.pt","train_raw.pt","validation_raw.pt","test_raw.pt","attack_audit_raw.pt","independent_audit_raw.pt")
    checks={name:equal(torch.load(reference/name,weights_only=True),torch.load(directory/name,weights_only=True)) for name in files}
    first=json.loads((reference/"metrics.json").read_text())
    second=json.loads((directory/"metrics.json").read_text())
    for value in (first,second): value.pop("elapsed_seconds")
    checks["metrics_without_runtime"]=equal(first,second)
    def trajectory(path):
        with path.open() as stream: return [{k:v for k,v in row.items() if k!="elapsed_seconds"} for row in csv.DictReader(stream)]
    checks["trajectory_without_runtime"]=equal(trajectory(reference/"trajectory.csv"),trajectory(directory/"trajectory.csv"))
    return {"job":key,"reference":str(reference.relative_to(ROOT)),"repeat":str(directory.relative_to(ROOT)),"bitwise_equal":checks,"passed":all(checks.values()),"excluded_from_primary_statistics":True}


def main():
    target=ROOT/"results/verification/final_reproducibility.json"
    if target.exists(): raise FileExistsError("Reproduction evidence already exists; do not overwrite")
    progress=json.loads((ROOT/"results/study_v1/progress.json").read_text())
    if progress["status"]!="TRAINING_COMPLETE": raise RuntimeError("Required training matrix must finish first")
    keys=[f"n2_m3_N20000_b0p5_d0p0_{model}_s11" for model in ("mlp","deepset")]
    plan={"jobs":keys,"excluded_from_primary_statistics":True,"comparison":"Bitwise tensor and trajectory equality; runtime excluded; same-environment check."}
    plan_path=ROOT/"configs/final_reproducibility.json"
    if plan_path.exists(): raise FileExistsError("Reproduction plan already exists")
    save_json(plan_path,plan)
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(repeat,key,progress["completed"][key]) for key in keys]
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            print("REPRODUCTION_DONE "+json.dumps(result),flush=True)
    save_json(target,{"status":"PASSED" if all(r["passed"] for r in results) else "FAILED","results":results,"scope":"Two full final-setting repeats in this environment; not cross-platform reproducibility."})
    if not all(r["passed"] for r in results): raise AssertionError("Final-setting reproducibility failed")


if __name__=="__main__":main()
