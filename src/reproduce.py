"""Run fresh numerical reproduction in an isolated directory, preserving evidence."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def portable(value):
    if isinstance(value,str): return value.replace("\\","/")
    if isinstance(value,list): return [portable(v) for v in value]
    if isinstance(value,dict): return {k:portable(v) for k,v in value.items()}
    return value


def staged(mode,workers):
    from .train import ROOT,run,save_json
    from .audit import audit
    def command(module,*args):
        subprocess.run([sys.executable,"-m",module,*args],cwd=ROOT,check=True)
    (ROOT/"results/verification").mkdir(parents=True,exist_ok=True)
    command("pytest","-q","--junitxml=results/verification/reproduction_tests.xml")
    command("src.equivariance")
    if mode=="smoke":
        config=json.loads((ROOT/"configs/p1_pilot_mlp.json").read_text())
        config.update(stage="SMOKE",experiment_id="publication_smoke",seed=11,seeds=[11],iterations=5,train_size=64,train_pool_size=64,train_eval_size=64,validation_size=32,test_size=32,audit_size=16,log_every=5)
        for name in ("train_attack","eval_attack","audit_attack"):
            config[name].update(steps=2,random_starts=1)
        save_json(ROOT/"configs/publication_smoke.json",config)
        directory=run(config,11)
        save_json(ROOT/"results/reproduction_status.json",{"status":"SMOKE_COMPLETE","run":str(directory.relative_to(ROOT)),"scientific_use":"Software pipeline check only; five updates are not a trained research comparison and need not pass low-regret gates."})
        return
    pilots=[]
    for filename in ("p1_pilot_mlp.json","p1_pilot_mlp_aug.json","p1_pilot_deepset.json","p1_large_setting_pilot.json"):
        config=json.loads((ROOT/"configs"/filename).read_text())
        directory=run(config,config["seed"])
        audit(directory)
        if not json.loads((directory/"metrics.json").read_text())["gate"]["passed"]:
            raise RuntimeError("Pilot failed; numerical reproduction must stop before scaling")
        pilots.append(str(directory.relative_to(ROOT)))
    save_json(ROOT/"results/p1_pilot_selection.json",{"runs":pilots,"policy":"Fresh fixed pilots; every gate required before the final matrix."})
    command("src.study","--workers",str(workers))
    command("src.post_evaluate","--workers",str(workers))
    command("src.analyze_study")
    command("src.validate_study")
    command("src.reproduce_final")
    save_json(ROOT/"results/reproduction_status.json",{"status":"NUMERICAL_REPRODUCTION_COMPLETE","unique_training_runs":205,"full_repeat_runs":2,"pilot_runs":4,"scope":"Fixed benchmark numerical reproduction; use the pinned environment."})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--mode",choices=("smoke","full"),default="smoke")
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--run-staged",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.run_staged:
        staged(args.mode,args.workers)
        return
    if args.output is None: parser.error("--output is required; it must be a new directory")
    root=Path(__file__).resolve().parents[1]
    target=args.output.resolve()
    if target.exists(): raise FileExistsError("Use a new directory; existing results are never overwritten")
    if target==root or target in root.parents: raise ValueError("Output must not replace the source repository")
    target.mkdir(parents=True)
    for name in ("src","tests","configs"):
        shutil.copytree(root/name,target/name,ignore=shutil.ignore_patterns("__pycache__","*.pyc","final_reproducibility.json"))
    for path in (target/"configs").rglob("*.json"):
        value=json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for name in ("pytest.ini","requirements.txt"):
        shutil.copy2(root/name,target/name)
    print(f"Isolated {args.mode} reproduction: {target}",flush=True)
    subprocess.run([sys.executable,"-m","src.reproduce","--run-staged","--mode",args.mode,"--workers",str(args.workers)],cwd=target,check=True)


if __name__=="__main__":main()
