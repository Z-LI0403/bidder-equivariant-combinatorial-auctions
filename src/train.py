"""Reproducible training runner. Every attempt creates a new immutable run directory."""
import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
from .valuations import generate
from .models import build_model
from .regret import best_responses, regrets
from .evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def code_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("src", "tests", "configs") for p in sorted((ROOT/folder).glob("*")) if p.is_file()}


def run(config, seed):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    label = config.get("experiment_id", config["stage"])
    directory = ROOT/"results"/"runs"/f"{label}_{config['model']}_seed{seed}_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    effective = {**config, "seed": seed}
    save_json(directory/"config.json", effective)
    save_json(directory/"status.json", {"status": "RUNNING"})
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model = build_model(config)
        metadata = {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                    "device": "cpu", "threads": 1, "started_utc": stamp,
                    "parameter_count": sum(p.numel() for p in model.parameters()), "code_sha256": code_hashes()}
        # Snapshot source: hashes alone would not preserve previous failed attempts.
        for folder in ("src", "configs", "tests"):
            for source in (ROOT/folder).glob("*"):
                if source.is_file():
                    target = directory/"source"/folder/source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
        data = {}
        for split, offset in [("train", 10000), ("validation", 20000), ("test", 30000)]:
            size = config.get("train_pool_size",config["train_size"]) if split == "train" else config[split+"_size"]
            full = generate(size, config["n"], config["m"], config["beta"], seed+offset+config.get("data_seed_offset",0), config["scales"])
            data[split] = {key:value[:config[split+"_size"]] for key,value in full.items()}
            if data[split]["x"].max() > config["report_upper"]:
                raise ValueError("data outside configured report domain")
        torch.save(data, directory/"datasets.pt")
        metadata["datasets_sha256"] = hashlib.sha256((directory/"datasets.pt").read_bytes()).hexdigest()
        save_json(directory/"metadata.json", metadata)
        torch.save(model.state_dict(), directory/"initial_model.pt")
        initial, initial_raw = evaluate(model, data["validation"]["x"], config["eval_attack"], seed+40000)
        save_json(directory/"initial_validation.json", initial)
        torch.save(initial_raw, directory/"initial_validation_raw.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        dual = torch.zeros(config["n"])
        rho = config["rho"]
        rng = torch.Generator().manual_seed(seed+50000)
        augmentation_rng = torch.Generator().manual_seed(seed+55000)
        attack_cache = data["train"]["x"].clone()
        fieldnames = ["step", "loss", "batch_revenue", "batch_regret", "rho", "dual_mean", "validation_revenue", "validation_regret", "elapsed_seconds"]
        with (directory/"trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for step in range(1, config["iterations"]+1):
                indices = torch.randint(config["train_size"], (config["batch_size"],), generator=rng)
                x = data["train"]["x"][indices]
                cached = attack_cache[indices]
                if config["model"] == "mlp_aug":
                    # One independently drawn uniform permutation per profile.
                    perm = torch.rand(len(x),config["n"],generator=augmentation_rng).argsort(-1)
                    gather = perm[...,None].expand_as(x)
                    x, cached = x.gather(1,gather), cached.gather(1,gather)
                response = best_responses(model, x, **config["train_attack"], seed=seed*100000+step, initial=cached)
                if config["model"] == "mlp_aug":
                    attack_cache[indices] = response.gather(1,perm.argsort(-1)[...,None].expand_as(response))
                else:
                    attack_cache[indices] = response
                outcome = model(x)
                r = regrets(model, x, response).mean(0)
                revenue = outcome["p"].sum(-1).mean()
                # Explicit choice: sum of squared per-bidder mean regrets.
                loss = -revenue + (dual*r).sum() + rho/2 * r.square().sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite parameter gradient")
                optimizer.step()
                if step % config["dual_update_every"] == 0:
                    with torch.no_grad():
                        dual += rho*regrets(model, x, response).mean(0)
                if step % config["rho_update_every"] == 0:
                    rho += config["rho_increment"]
                if step % config["log_every"] == 0 or step == 1:
                    validation, _ = evaluate(model, data["validation"]["x"][:64], config["train_attack"], seed+60000)
                    row = {"step": step, "loss": float(loss.detach()), "batch_revenue": float(revenue.detach()),
                           "batch_regret": float(r.detach().mean()), "rho": rho, "dual_mean": float(dual.mean()),
                           "validation_revenue": validation["revenue"], "validation_regret": validation["regret_mean_bidder"],
                           "elapsed_seconds": time.perf_counter()-started}
                    writer.writerow(row)
                    stream.flush()
                    print(f"seed={seed} step={step} revenue={row['validation_revenue']:.4f} regret={row['validation_regret']:.5f}", flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "dual": dual, "rho": rho}, directory/"checkpoint.pt")
        metrics = {"initial_validation": initial}
        for split in ("train", "validation", "test"):
            evaluated = data[split]["x"]
            if split == "train":
                evaluated = evaluated[:config.get("train_eval_size",len(evaluated))]
            metrics[split], raw = evaluate(model, evaluated, config["eval_attack"], seed+70000)
            metrics[split]["evaluated_profiles"] = len(evaluated)
            torch.save(raw, directory/(split+"_raw.pt"))
        # Nested audit candidates guarantee the strong attack contains the standard results.
        subset = data["test"]["x"][:config["audit_size"]]
        standard, raw = evaluate(model, subset, config["eval_attack"], seed+80000)
        strong_responses = []
        for start in range(0, len(subset), 64):
            strong_responses.append(best_responses(model, subset[start:start+64], **config["audit_attack"], seed=seed+90000+start, initial=raw["responses"][start:start+64]))
        with torch.no_grad():
            strong = regrets(model, subset, torch.cat(strong_responses))
        torch.save({"standard": raw, "strong_responses": torch.cat(strong_responses), "strong_regret": strong}, directory/"attack_audit_raw.pt")
        metrics["attack_audit"] = {"standard_regret": standard["regret_mean_bidder"], "strong_regret": float(strong.mean()),
                                   "increase": float(strong.mean())-standard["regret_mean_bidder"]}
        metrics["gaps"] = {key: metrics["test"][key]-metrics["train"][key] for key in ("revenue", "regret_mean_bidder")}
        gate = config["gate"]
        test = metrics["test"]
        checks = {"low_estimated_regret": test["regret_mean_bidder"] <= gate["max_mean_bidder_regret"],
                  "regret_revenue_ratio": test["regret_mean_total"]/max(test["revenue"], 1e-12) <= gate["max_mean_regret_over_revenue"],
                  "noncollapsed_revenue": test["revenue"]/test["optimal_welfare"] >= gate["min_revenue_over_welfare"],
                  "attack_budget_sensitivity": metrics["attack_audit"]["increase"] <= gate["max_audit_regret_increase"],
                  "regret_reduction": metrics["validation"]["regret_mean_bidder"] <= initial["regret_mean_bidder"]*(1-gate["min_regret_reduction_fraction"]),
                  "ir_sanity": test["truthful_utility_min"] >= -1e-6}
        metrics["gate"] = {"checks": checks, "passed": all(checks.values())}
        metrics["elapsed_seconds"] = time.perf_counter()-started
        save_json(directory/"metrics.json", metrics)
        save_json(directory/"status.json", {"status": "COMPLETED", "pilot_gate_passed": metrics["gate"]["passed"]})
        print(f"RUN_COMPLETE {directory} gate={metrics['gate']['passed']}", flush=True)
        return directory
    except BaseException as error:
        save_json(directory/"status.json", {"status": "FAILED_OR_INTERRUPTED", "error": repr(error), "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p1_pilot_mlp.json")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for seed in ([args.seed] if args.seed is not None else config["seeds"]):
        run(config, seed)


if __name__ == "__main__":
    main()
