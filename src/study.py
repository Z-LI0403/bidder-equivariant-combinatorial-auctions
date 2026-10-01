"""Dependency-ordered, resumable execution of the frozen final study."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from contextlib import redirect_stdout,redirect_stderr
import json
import time
from datetime import datetime,timezone
from pathlib import Path
from .train import ROOT,run,save_json
from .audit import audit


def worker(config_path,log_path):
    config=json.loads((ROOT/config_path).read_text())
    with (ROOT/log_path).open("w",encoding="utf-8",buffering=1) as stream,redirect_stdout(stream),redirect_stderr(stream):
        directory=run(config,config["seed"])
        audit(directory)
    metrics=json.loads((directory/"metrics.json").read_text())
    independent=json.loads((directory/"independent_audit.json").read_text())
    return {"run":str(directory.relative_to(ROOT)),"gate_passed":metrics["gate"]["passed"],"regret":metrics["test"]["regret_mean_bidder"],"revenue":metrics["test"]["revenue"],"audit_increase":independent["mean_increase"],"log":log_path}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    manifest_path=ROOT/"configs"/"study_v1"/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    e0=json.loads((ROOT/"results"/"E0"/"metrics.json").read_text())
    if not e0["all_structural_variants_passed"]:
        raise RuntimeError("E0 prerequisite failed")
    pilots=json.loads((ROOT/"results"/"p1_pilot_selection.json").read_text())
    for relative in pilots["runs"]:
        if not json.loads((ROOT/relative/"metrics.json").read_text())["gate"]["passed"]:
            raise RuntimeError("P1 pilot prerequisite failed")
    directory=ROOT/"results"/"study_v1"
    directory.mkdir(parents=True,exist_ok=True)
    logs=directory/"logs"
    logs.mkdir(exist_ok=True)
    progress_path=directory/"progress.json"
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {"completed":{},"stage_history":[],"failures":[]}
    progress.update({"status":"RUNNING","total_unique_jobs":manifest["unique_training_runs"],"workers":args.workers})
    def persist():
        progress["updated_utc"]=datetime.now(timezone.utc).isoformat()
        save_json(progress_path,progress)
    started=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for stage in manifest["stage_order"]:
                progress["stage"]=stage
                pending=[key for key in manifest["experiments"][stage] if key not in progress["completed"]]
                progress["stage_pending"]=len(pending)
                persist()
                print(f"STAGE_START {stage} new_jobs={len(pending)} completed_total={len(progress['completed'])}",flush=True)
                futures={}
                for key in pending:
                    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                    log=str((logs/(key+"_"+stamp+".log")).relative_to(ROOT))
                    futures[executor.submit(worker,manifest["jobs"][key],log)]=key
                while futures:
                    finished,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
                    for future in finished:
                        key=futures.pop(future)
                        try:
                            outcome=future.result()
                        except Exception as error:
                            progress["failures"].append({"job":key,"error":repr(error)})
                            progress["status"]="FAILED"
                            persist()
                            for rest in futures: rest.cancel()
                            raise
                        progress["completed"][key]=outcome
                        print(f"JOB_DONE {len(progress['completed'])}/{manifest['unique_training_runs']} {key} revenue={outcome['revenue']:.4f} regret={outcome['regret']:.6f} gate={outcome['gate_passed']}",flush=True)
                    progress["stage_pending"]=len(futures)
                    progress["elapsed_controller_seconds"]=time.perf_counter()-started
                    persist()
                # A model comparison can be negative without a software failure.
                # Record gate diagnostics instead of discarding poor-performing models.
                stage_runs=[progress["completed"][key] for key in manifest["experiments"][stage]]
                record={"stage":stage,"jobs":len(stage_runs),"all_operational_gates_passed":all(r["gate_passed"] for r in stage_runs),"max_independent_audit_increase":max(r["audit_increase"] for r in stage_runs)}
                if stage not in [r["stage"] for r in progress["stage_history"]]:
                    progress["stage_history"].append(record)
                persist()
                print("STAGE_COMPLETE "+json.dumps(record),flush=True)
                # A failed baseline mean-regret gate triggers debugging before scaling.
                baseline_failures=[key for key in manifest["experiments"][stage] if json.loads((ROOT/manifest["jobs"][key]).read_text())["model"] == "mlp" and progress["completed"][key]["regret"]>.02]
                if baseline_failures:
                    raise RuntimeError("Baseline regret gate failed; debug before downstream work: "+str(baseline_failures))
                if record["max_independent_audit_increase"]>.005:
                    raise RuntimeError("Independent measurement audit failed; debug before downstream work")
        progress["status"]="TRAINING_COMPLETE"
        persist()
        print(f"TRAINING_COMPLETE jobs={len(progress['completed'])}",flush=True)
    except BaseException:
        if progress["status"] != "FAILED": progress["status"]="NEEDS_DEBUGGING"
        persist()
        raise


if __name__=="__main__": main()
