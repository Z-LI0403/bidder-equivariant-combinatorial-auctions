"""Independent saved-artifact consistency checks after the whole matrix."""
from collections import defaultdict
import hashlib
import json
import torch
from .train import ROOT,save_json


def main():
    torch.set_num_threads(1)
    manifest=json.loads((ROOT/"configs/study_v1/manifest.json").read_text())
    progress=json.loads((ROOT/"results/study_v1/progress.json").read_text())
    if set(progress["completed"]) != set(manifest["jobs"]): raise AssertionError("Incomplete job set")
    paired=defaultdict(list)
    nested=defaultdict(dict)
    errors=[]
    e0_files=list((ROOT/"results/E0").glob("metrics*.json"))
    e0=json.loads(max(e0_files,key=lambda p:p.stat().st_mtime).read_text())
    if not e0["all_structural_variants_passed"]: errors.append({"error":"E0 structural gate failed"})
    if any(r["equivariance_passed"] for r in e0["records"] if r["model"]=="mlp"):
        errors.append({"error":"E0 negative-control outcome unexpected"})
    count=0
    for key,record in progress["completed"].items():
        directory=ROOT/record["run"]
        config=json.loads((directory/"config.json").read_text())
        frozen=json.loads((ROOT/manifest["jobs"][key]).read_text())
        if config != frozen: errors.append({"job":key,"error":"config differs from frozen spec"})
        metadata=json.loads((directory/"metadata.json").read_text())
        actual=hashlib.sha256((directory/"datasets.pt").read_bytes()).hexdigest()
        if actual != metadata["datasets_sha256"]: errors.append({"job":key,"error":"dataset file hash mismatch"})
        setting=(config["n"],config["m"],config["train_size"],config["beta"],config["delta"],config["seed"])
        paired[setting].append(actual)
        if config["model"]=="mlp" and config["n"]==2 and config["m"]==3 and config["beta"]==.5 and config["delta"]==0:
            nested[config["seed"]][config["train_size"]]=directory
        metrics=json.loads((directory/"metrics.json").read_text())
        for split in ("train","validation","test"):
            raw=torch.load(directory/(split+"_raw.pt"),weights_only=True)
            if not all(torch.isfinite(value).all() for value in raw.values()): errors.append({"job":key,"error":"nonfinite raw result"})
            if (raw["regret"]<0).any(): errors.append({"job":key,"error":"negative recorded regret"})
            if (raw["responses"]<0).any() or (raw["responses"]>config["report_upper"]+1e-6).any():
                errors.append({"job":key,"error":"response outside declared report domain"})
            calculated={"revenue":float(raw["revenue"].mean()),"regret_mean_bidder":float(raw["regret"].mean()),"regret_mean_total":float(raw["regret"].sum(-1).mean())}
            for metric,value in calculated.items():
                if abs(value-metrics[split][metric])>1e-7: errors.append({"job":key,"split":split,"error":"raw metric mismatch","metric":metric})
        count+=1
    for setting,hashes in paired.items():
        if len(set(hashes))!=1: errors.append({"setting":setting,"error":"model datasets differ"})
    for seed,paths in nested.items():
        if set(paths)!={1000,5000,20000,50000}: errors.append({"seed":seed,"error":"nested dataset sizes missing"});continue
        full=torch.load(paths[50000]/"datasets.pt",weights_only=True)
        for size,path in paths.items():
            data=torch.load(path/"datasets.pt",weights_only=True)
            for split in full:
                for name in full[split]:
                    expected=full[split][name][:size] if split=="train" else full[split][name]
                    if not torch.equal(data[split][name],expected): errors.append({"seed":seed,"size":size,"split":split,"error":"not a nested identical-data prefix"})
    post=json.loads((ROOT/"results/study_v1/post_evaluation.json").read_text())
    if post["completed_tasks"]!=25: errors.append({"error":"missing post-evaluation tasks"})
    post_keys=[]
    for result in post["results"]:
        directory=ROOT/result["run"]
        config=json.loads((directory/"config.json").read_text())
        source=directory/("post_"+result["kind"]+"_source.py")
        if hashlib.sha256(source.read_bytes()).hexdigest()!=result["source_sha256"]:
            errors.append({"run":result["run"],"error":"post-evaluation source hash mismatch"})
        for row in result["records"]:
            post_keys.append((row["evaluation"],row["training_model"],row["seed"]))
            saved=torch.load(directory/("post_"+row["evaluation"]+"_raw.pt"),weights_only=True)
            raw=saved["raw"]
            x=saved["data"]["x"]
            if tuple(x.shape)!=(1024,row["n"],2**row["m"]-1):
                errors.append({"run":result["run"],"error":"post-evaluation data shape mismatch"})
            if not all(torch.isfinite(value).all() for value in raw.values()):
                errors.append({"run":result["run"],"error":"nonfinite post-evaluation raw result"})
            if (raw["regret"]<0).any() or (raw["responses"]<0).any() or (raw["responses"]>config["report_upper"]+1e-6).any():
                errors.append({"run":result["run"],"error":"invalid post-evaluation regret or response"})
            calculated={"revenue":float(raw["revenue"].mean()),"regret_mean_bidder":float(raw["regret"].mean()),"regret_mean_total":float(raw["regret"].sum(-1).mean()),"regret_p95_bidder":float(torch.quantile(raw["regret"].flatten(),.95)),"regret_max":float(raw["regret"].max()),"truthful_utility_min":float(raw["truthful_utility"].min()),"optimal_welfare":float(raw["welfare"].mean())}
            for metric,value in calculated.items():
                if abs(value-row[metric])>1e-7:
                    errors.append({"run":result["run"],"error":"post-evaluation raw metric mismatch","metric":metric})
    expected_post={(label,model,seed) for label in ("beta_shift","scale_shift") for model in ("mlp","mlp_aug","deepset") for seed in (11,22,33,44,55)}
    expected_post|={("bidder_transfer",model,seed) for model in ("deepset","deepset_sum") for seed in (11,22,33,44,55)}
    if len(post_keys)!=40 or set(post_keys)!=expected_post:
        errors.append({"error":"missing or duplicate post-evaluation records"})
    for relative in manifest["jobs"].values():
        if not (ROOT/relative).exists(): errors.append({"error":"missing config","path":relative})
    result={"status":"PASSED" if not errors else "FAILED","training_runs_checked":count,"paired_dataset_groups":len(paired),"nested_seed_pools_checked":len(nested),"post_evaluation_tasks":post["completed_tasks"],"post_evaluation_records_checked":len(post_keys),"errors":errors,"scope":"Frozen-config equality, data file integrity and model pairing, raw training/post-evaluation metric recalculation, finite/nonnegative outputs and actual nested prefixes; not a proof of global IC."}
    save_json(ROOT/"results/verification/final_artifact_validation.json",result)
    print(json.dumps(result,indent=2))
    if errors: raise AssertionError("Final artifact validation failed")


if __name__=="__main__":main()
