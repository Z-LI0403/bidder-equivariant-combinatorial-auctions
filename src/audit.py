"""Independent derivative-free post-training attack, outside the gradient solver."""
import argparse
import hashlib
import json
import torch
from .train import ROOT,save_json
from .models import build_model


def attack_bidder(model,truth,i,start,seed):
    """Random search + coordinate moves. Utilities constructed without regret.py."""
    rng = torch.Generator().manual_seed(seed)
    upper = model.report_upper
    b,n,k = truth.shape
    def utility_at(candidates):
        r = len(candidates)
        profiles = truth[None].expand(r,b,n,k).clone()
        profiles[:,:,i] = candidates
        out = model(profiles.flatten(0,1))
        return ((out["g"][:,i]*truth[:,i].repeat(r,1)).sum(-1)-out["p"][:,i]).reshape(r,b)
    with torch.no_grad():
        current = start.clone()
        best = utility_at(current[None]).squeeze(0)
        for chunk in range(8):
            # 512 cube reports, 512 local perturbations and 512 scaled truths.
            for mode in ("box","local","scaled"):
                if mode == "box":
                    candidates = torch.rand(64,b,k,generator=rng)*upper
                elif mode == "local":
                    candidates = (truth[None,:,i]+torch.randn(64,b,k,generator=rng)*(.25*upper)).clamp(0,upper)
                else:
                    candidates = truth[None,:,i]*torch.rand(64,b,1,generator=rng)
                values = utility_at(candidates)
                maxima,indices = values.max(0)
                selected = candidates[indices,torch.arange(b)]
                current = torch.where((maxima>best)[:,None],selected,current)
                best = torch.maximum(best,maxima)
        for sweep in range(3):
            for j in range(k):
                # Fixed absolute coordinate moves, coarse to fine and boundaries.
                candidates = current[None].repeat(12,1,1)
                offsets = torch.tensor([-2.,-1.,-.5,-.2,-.05,.05,.2,.5,1.,2.])
                candidates[:10,:,j] = (current[:,j][None]+offsets[:,None]).clamp(0,upper)
                candidates[10,:,j], candidates[11,:,j] = 0.,upper
                values = utility_at(candidates)
                maxima,indices = values.max(0)
                selected = candidates[indices,torch.arange(b)]
                current = torch.where((maxima>best)[:,None],selected,current)
                best = torch.maximum(best,maxima)
    return current,best


def audit(directory):
    if (directory/"independent_audit.json").exists():
        raise FileExistsError("Preserve the existing audit; use a fresh run for another attempt")
    config = json.loads((directory/"config.json").read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = build_model(config)
    model.load_state_dict(torch.load(directory/"checkpoint.pt",weights_only=True)["model"])
    data = torch.load(directory/"datasets.pt",weights_only=True)["test"]["x"][:config["audit_size"]]
    original = torch.load(directory/"attack_audit_raw.pt",weights_only=True)
    responses = original["strong_responses"].clone()
    with torch.no_grad():
        out = model(data)
        truthful = (data*out["g"]).sum(-1)-out["p"]
    values = []
    for i in range(config["n"]):
        responses[:,i], best = attack_bidder(model,data,i,responses[:,i],config["seed"]+110000+i)
        values.append((best-truthful[:,i]).clamp_min(0))
    regret = torch.stack(values,-1)
    baseline = original["strong_regret"]
    metrics = {"profiles":len(data),"independent_mean_regret":float(regret.mean()),
               "strong_gradient_mean_regret":float(baseline.mean()),
               "mean_increase":float((regret-baseline).mean()),"max_profile_bidder_increase":float((regret-baseline).max()),
               "budget":"Per bidder: 512 full-box + 512 local + 512 scaled-truth candidates, then 3 coordinate sweeps with 12 candidates per coordinate; initialized at strong gradient attack.",
               "status":"NUMERICALLY_VERIFIED only; neither search certifies global regret."}
    torch.save({"responses":responses,"regret":regret},directory/"independent_audit_raw.pt")
    sources = {}
    for name in ("audit.py", "models.py", "feasibility.py"):
        source = ROOT/"src"/name
        target = directory/"independent_audit_source"/name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    save_json(directory/"independent_audit_metadata.json", {"code_sha256":sources,"torch":torch.__version__,"seed":config["seed"]+110000})
    save_json(directory/"independent_audit.json",metrics)
    print(directory.name,json.dumps(metrics),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection",default="results/p0_run_selection.json")
    args = parser.parse_args()
    selection = json.loads((ROOT/args.selection).read_text())
    for path in selection["primary_runs"]:
        audit(ROOT/path)


if __name__ == "__main__":
    main()
