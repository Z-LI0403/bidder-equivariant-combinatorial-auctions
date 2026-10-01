"""E0: saved multi-seed numerical structural check before any comparisons."""
import json
from itertools import permutations,product
import torch
from .models import DeepSetMechanism,MLPMechanism
from .valuations import generate
from .evaluation import permutation_errors
from .train import ROOT,save_json


def main():
    torch.set_num_threads(1)
    directory=ROOT/"results"/"E0"
    directory.mkdir(parents=True,exist_ok=True)
    records=[]
    for n,m,seed in product((2,3),(3,4),(11,22,33,44,55)):
        upper=3*m+m*(m-1)//2
        x=generate(64,n,m,1,seed)["x"]
        for dtype in (torch.float32,torch.float64):
            for variant in ("deepset","local","deepset_sum","mlp"):
                torch.manual_seed(seed)
                model=MLPMechanism(n,m,report_upper=upper) if variant == "mlp" else DeepSetMechanism(n,m,width=20,report_upper=upper,context=variant != "local",aggregation="sum" if variant == "deepset_sum" else "mean")
                model=model.to(dtype)
                values=x.to(dtype)
                errors=permutation_errors(model,values)
                absolute={"g":0.,"p":0.}
                with torch.no_grad():
                    reference=model(values)
                    for perm in permutations(range(n)):
                        altered=model(values[:,perm])
                        for key in absolute:
                            absolute[key]=max(absolute[key],float((altered[key]-reference[key][:,perm]).abs().max()))
                row={"n":n,"m":m,"seed":seed,"dtype":str(dtype),"model":variant,**errors,"max_absolute_allocation":absolute["g"],"max_absolute_payment":absolute["p"]}
                tolerance=3e-6 if dtype == torch.float32 else 1e-12
                row["structural_check_applicable"]=variant != "mlp"
                row["equivariance_passed"]=max(errors.values()) <= tolerance
                records.append(row)
    passed=all(r["equivariance_passed"] for r in records if r["structural_check_applicable"])
    target=directory/"metrics.json"
    version=2
    while target.exists():
        target=directory/f"metrics_v{version}.json"
        version+=1
    save_json(target,{"records":records,"all_structural_variants_passed":passed,"tolerances":{"float32":3e-6,"float64":1e-12},"negative_control":"MLP is measured against the same tolerance but is not a structural prerequisite; its failed check is expected."})
    if not passed:
        raise AssertionError("E0 failed; downstream training prohibited")
    print(json.dumps({"checks":len(records),"max_structural_float32_error":max(max(r["permutation_allocation"],r["permutation_payment"]) for r in records if r["model"] != "mlp" and r["dtype"] == "torch.float32"),"passed":True}))


if __name__ == "__main__":
    main()
