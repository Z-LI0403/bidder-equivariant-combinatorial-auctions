"""Prespecified E5 distribution shifts and A2 bidder-count transfer."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import hashlib
import torch
from .models import build_model
from .valuations import generate
from .evaluation import evaluate
from .train import ROOT,save_json


def evaluate_job(relative,kind):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    directory=ROOT/relative
    target=directory/("post_"+kind+".json")
    if target.exists(): return json.loads(target.read_text())
    config=json.loads((directory/"config.json").read_text())
    model=build_model(config)
    model.load_state_dict(torch.load(directory/"checkpoint.pt",weights_only=True)["model"])
    records=[]
    specs=[("beta_shift",1.,[1.,1.],2,420000),("scale_shift",.5,[1.,2.],2,430000)] if kind == "E5" else [("bidder_transfer",.5,[1.,1.,1.],3,440000)]
    for label,beta,scales,n,offset in specs:
        if n != config["n"]:
            transferred=build_model({**config,"n":n})
            state={k:v for k,v in model.state_dict().items() if not k.startswith("decoder.")}
            missing,unexpected=transferred.load_state_dict(state,strict=False)
            if set(missing) != {"decoder.owners","decoder.incidence"} or unexpected:
                raise AssertionError("unexpected transfer state mismatch")
            used=transferred
        else: used=model
        data=generate(1024,n,config["m"],beta,config["seed"]+offset,scales)
        metrics,raw=evaluate(used,data["x"],config["eval_attack"],config["seed"]+offset+10000)
        torch.save({"data":data,"raw":raw},directory/("post_"+label+"_raw.pt"))
        records.append({"evaluation":label,"training_model":config["model"],"seed":config["seed"],"n":n,"m":config["m"],"beta":beta,"scales":scales,**metrics})
    result={"kind":kind,"run":relative,"records":records,"policy":"No retraining or test-driven checkpoint selection; prespecified evaluation shift."}
    source=ROOT/"src/post_evaluate.py"
    (directory/("post_"+kind+"_source.py")).write_bytes(source.read_bytes())
    result["source_sha256"]=hashlib.sha256(source.read_bytes()).hexdigest()
    save_json(target,result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    manifest=json.loads((ROOT/"configs"/"study_v1"/"manifest.json").read_text())
    progress=json.loads((ROOT/"results"/"study_v1"/"progress.json").read_text())
    if progress["status"] != "TRAINING_COMPLETE": raise RuntimeError("Final training incomplete")
    tasks=[(progress["completed"][key]["run"],"E5") for key in manifest["experiments"]["E5"]]
    tasks += [(progress["completed"][key]["run"],"A2") for key in manifest["experiments"]["A2"]]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(evaluate_job,*task) for task in tasks]
        for f in as_completed(futures):
            result=f.result()
            results.append(result)
            print(f"POST_EVAL_DONE {len(results)}/{len(tasks)} {result['kind']} {result['run']}",flush=True)
    save_json(ROOT/"results"/"study_v1"/"post_evaluation.json",{"results":results,"completed_tasks":len(results)})


if __name__=="__main__": main()
