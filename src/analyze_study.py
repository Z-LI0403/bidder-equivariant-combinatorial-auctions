"""Full study tables, prespecified paired comparisons and scientific figures."""
import csv
import json
import math
from itertools import product
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .train import ROOT,save_json

METRICS=("revenue","regret_mean_bidder","regret_mean_total","regret_p95_bidder","regret_max","permutation_allocation","permutation_payment","optimal_welfare")
LABELS={"mlp":"MLP","mlp_aug":"MLP + augmentation","deepset":"DeepSet mean","local":"Shared local only","deepset_sum":"DeepSet sum"}
COLORS={"mlp":"#4466aa","mlp_aug":"#dd8844","deepset":"#339977","local":"#9966aa","deepset_sum":"#cc5577"}


def t4_critical():
    # For T~t_4, u=t/sqrt(4+t^2), F(t)=1/2+3u/4-u^3/4.
    low,high=0.,1.
    for _ in range(60):
        u=(low+high)/2
        if .5+.75*u-.25*u**3 < .975: low=u
        else: high=u
    u=(low+high)/2
    return 2*u/math.sqrt(1-u*u)


def stats(values):
    values=np.asarray(values,dtype=float)
    return {"mean":float(values.mean()),"std":float(values.std(ddof=1)),"n":len(values),"values":values.tolist()}


def contrast(first,second):
    if len(first)!=5 or len(second)!=5: raise ValueError("Five paired seeds required")
    result=stats(np.asarray(first)-np.asarray(second))
    half=t4_critical()*result["std"]/math.sqrt(5)
    result["ci95"]=[result["mean"]-half,result["mean"]+half]
    return result


def group_rows(rows):
    groups={}
    for row in rows:
        key=(row["n"],row["m"],row["train_size"],row["beta"],row["delta"],row["model"])
        groups.setdefault(key,[]).append(row)
    result=[]
    for key,entries in sorted(groups.items()):
        entries=sorted(entries,key=lambda r:r["seed"])
        if [r["seed"] for r in entries] != [11,22,33,44,55]: raise AssertionError("Missing or duplicate seed")
        result.append({"n":key[0],"m":key[1],"train_size":key[2],"beta":key[3],"delta":key[4],"model":key[5],"parameter_count":entries[0]["parameter_count"],"gate_passed_count":sum(r["gate_passed"] for r in entries),"metrics":{k:stats([r[k] for r in entries]) for k in METRICS},"gaps":{k:stats([r[k] for r in entries]) for k in ("revenue_gap","regret_gap")},"runs":[r["run"] for r in entries]})
    return result


def paired_comparisons(groups,rule):
    by_setting={}
    for group in groups:
        key=(group["n"],group["m"],group["train_size"],group["beta"],group["delta"])
        by_setting.setdefault(key,{})[group["model"]]=group
    comparisons=[]
    for key,models in by_setting.items():
        pairs=[("deepset","mlp"),("deepset","mlp_aug"),("mlp_aug","mlp"),("deepset","local"),("deepset_sum","deepset")]
        for first,second in pairs:
            if first not in models or second not in models: continue
            a,b=models[first],models[second]
            differences={metric:contrast(a["metrics"][metric]["values"],b["metrics"][metric]["values"]) for metric in METRICS}
            comparable=(a["metrics"]["regret_mean_bidder"]["mean"]<=rule["absolute_regret_gate"] and b["metrics"]["regret_mean_bidder"]["mean"]<=rule["absolute_regret_gate"] and abs(differences["regret_mean_bidder"]["mean"])<=rule["matched_regret_tolerance"])
            positive=differences["revenue"]["ci95"][0]>0
            negative=differences["revenue"]["ci95"][1]<0
            dr, dg=differences["revenue"]["mean"],differences["regret_mean_bidder"]["mean"]
            joint="higher_revenue_lower_estimated_regret" if dr>0 and dg<0 else "lower_revenue_higher_estimated_regret" if dr<0 and dg>0 else "mixed_or_equal"
            comparisons.append({"n":key[0],"m":key[1],"train_size":key[2],"beta":key[3],"delta":key[4],"first":first,"second":second,"differences":differences,"observed_regret_comparable":comparable,"observed_mean_tradeoff":joint,"revenue_interval_direction":"positive" if positive else "negative" if negative else "includes_zero","interpretation":"descriptive only; finite estimated-regret matching and unadjusted paired t interval, not a discovery test; joint point labels do not override the preregistered revenue-comparison rule"})
    return comparisons


def save_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig,path):
    fig.tight_layout(rect=(0,.085,1,1) if getattr(fig,"_study_legend",False) else (0,0,1,1))
    fig.savefig(path.with_suffix(".png"),dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def shared_legend(fig,axis):
    handles,labels=axis.get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9,frameon=False)
    fig._study_legend=True


def sweep_figure(groups,variable,values,title,path):
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for model in ("mlp","mlp_aug","deepset"):
        selected=sorted([g for g in groups if g["n"]==2 and g["m"]==3 and g["model"]==model and (g["train_size"]==20000 or variable=="train_size") and (g["beta"]==.5 or variable=="beta") and (g["delta"]==0 or variable=="delta")],key=lambda g:g[variable])
        if [g[variable] for g in selected]!=values: raise AssertionError("Invalid sweep grouping")
        for axis,metric in zip(axes,("revenue","regret_mean_bidder")):
            axis.errorbar(values,[g["metrics"][metric]["mean"] for g in selected],yerr=[g["metrics"][metric]["std"] for g in selected],label=LABELS[model],color=COLORS[model],marker="o",capsize=3)
    for axis in axes:
        axis.set_xlabel({"train_size":"Training profiles (fixed 1500 updates)","beta":"Complementarity beta","delta":"Heterogeneity delta; scales [1, 1+delta]"}[variable])
        axis.grid(alpha=.2)
        if variable=="train_size": axis.set_xscale("log");axis.set_xticks(values,labels=["1k","5k","20k","50k"])
    shared_legend(fig,axes[0])
    axes[0].set_ylabel("Truthful test revenue per auction")
    axes[1].set_ylabel("Estimated test regret per bidder")
    axes[1].set_yscale("log")
    fig.suptitle(title+" — five-seed mean ± SD; n=2, m=3")
    save_figure(fig,path)


def build_figures(groups,comparisons,post,folder):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    sweep_figure(groups,"train_size",[1000,5000,20000,50000],"E2 sample efficiency",folder/"sample_efficiency")
    sweep_figure(groups,"beta",[0.,.25,.5,1.],"E3 complementarity strength",folder/"beta_sweep")
    sweep_figure(groups,"delta",[0.,.5,1.,2.],"E4 symmetry breaking",folder/"symmetry_breaking")
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for second in ("mlp","mlp_aug"):
        selected=sorted([c for c in comparisons if c["n"]==2 and c["m"]==3 and c["train_size"]==20000 and c["beta"]==.5 and c["first"]=="deepset" and c["second"]==second],key=lambda c:c["delta"])
        for ax,metric in zip(axes,("revenue","regret_mean_bidder")):
            ax.errorbar([c["delta"] for c in selected],[c["differences"][metric]["mean"] for c in selected],yerr=[(c["differences"][metric]["ci95"][1]-c["differences"][metric]["ci95"][0])/2 for c in selected],marker="o",color=COLORS[second],capsize=4,label="DeepSet minus ("+LABELS[second]+")")
    for ax in axes:
        ax.axhline(0,color="gray",linestyle="--")
        ax.set_xlabel("Heterogeneity delta")
        ax.grid(alpha=.2)
    shared_legend(fig,axes[0])
    axes[0].set_ylabel("Paired truthful revenue difference")
    axes[1].set_ylabel("Paired estimated-regret difference")
    fig.suptitle("E4 paired differences — descriptive 95% t intervals, five seeds")
    save_figure(fig,folder/"symmetry_breaking_differences")
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,(n,m) in zip(axes.flatten(),product((2,3),(3,4))):
        selected=[g for g in groups if g["n"]==n and g["m"]==m and g["train_size"]==20000 and g["beta"]==.5 and g["delta"]==0 and g["model"] in ("mlp","mlp_aug","deepset")]
        for g in selected:
            x,y=g["metrics"]["regret_mean_bidder"],g["metrics"]["revenue"]
            ax.errorbar(x["mean"],y["mean"],xerr=x["std"],yerr=y["std"],fmt="o",color=COLORS[g["model"]],capsize=4,label=LABELS[g["model"]])
        ax.set_title(f"n={n}, m={m}")
        ax.set_xlabel("Estimated regret per bidder")
        ax.set_ylabel("Truthful revenue per auction")
        ax.grid(alpha=.2)
    shared_legend(fig,axes.flatten()[0])
    fig.suptitle("E1 Revenue–Regret trade-off — final checkpoints, mean ± SD")
    save_figure(fig,folder/"revenue_regret_tradeoff")
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    main=[g for g in groups if g["n"]==2 and g["m"]==3 and g["train_size"]==20000 and g["beta"]==.5 and g["delta"]==0]
    for ax,metric in zip(axes,("permutation_allocation","permutation_payment")):
        for index,g in enumerate(main):
            value=g["metrics"][metric]
            # A mean-minus-SD interval can cross zero and cannot be shown on log axes.
            # Show the five actual positive seed measurements and arithmetic mean.
            ax.scatter(index+np.linspace(-.12,.12,5),value["values"],color=COLORS[g["model"]],alpha=.5,s=25)
            ax.scatter(index,value["mean"],marker="D",color=COLORS[g["model"]],s=50)
        ax.set_xticks(range(len(main)),labels=[LABELS[g["model"]] for g in main],rotation=20,ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Normalized "+metric.split("_")[1]+" permutation error")
        ax.grid(alpha=.2)
    fig.suptitle("E5 and ablations — permutation error: five seeds (circles), mean (diamonds)")
    save_figure(fig,folder/"permutation_robustness")
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    post_groups={}
    for result in post["results"]:
        if result["kind"]!="E5": continue
        for row in result["records"]:
            post_groups.setdefault((row["evaluation"],row["training_model"]),[]).append(row)
    for model in ("mlp","mlp_aug","deepset"):
        iid=next(g for g in groups if g["n"]==2 and g["m"]==3 and g["train_size"]==20000 and g["beta"]==.5 and g["delta"]==0 and g["model"]==model)
        for ax,metric in zip(axes,("revenue","regret_mean_bidder")):
            sets=[iid["metrics"][metric],stats([r[metric] for r in post_groups[("beta_shift",model)]]),stats([r[metric] for r in post_groups[("scale_shift",model)]])]
            ax.errorbar(range(3),[s["mean"] for s in sets],yerr=[s["std"] for s in sets],marker="o",capsize=3,label=LABELS[model],color=COLORS[model])
    for ax in axes:
        ax.set_xticks(range(3),labels=["IID test","beta 0.5 → 1","scales [1,1] → [1,2]"])
        ax.grid(alpha=.2)
    shared_legend(fig,axes[0])
    axes[0].set_ylabel("Truthful revenue per auction")
    axes[1].set_ylabel("Estimated regret per bidder")
    axes[1].set_yscale("log")
    fig.suptitle("E5 held-out distribution shifts — no retraining; mean ± SD")
    save_figure(fig,folder/"distribution_shift")


def main():
    manifest=json.loads((ROOT/"configs/study_v1/manifest.json").read_text())
    progress=json.loads((ROOT/"results/study_v1/progress.json").read_text())
    if progress["status"]!="TRAINING_COMPLETE" or len(progress["completed"])!=manifest["unique_training_runs"]:
        raise RuntimeError("Cannot analyze an incomplete final matrix")
    rows=[]
    for key,record in progress["completed"].items():
        directory=ROOT/record["run"]
        c=json.loads((directory/"config.json").read_text())
        metrics=json.loads((directory/"metrics.json").read_text())
        independent=json.loads((directory/"independent_audit.json").read_text())
        row={"job":key,"run":record["run"],"model":c["model"],"seed":c["seed"],"n":c["n"],"m":c["m"],"train_size":c["train_size"],"beta":c["beta"],"delta":c["delta"],"parameter_count":c["parameter_count"],"gate_passed":metrics["gate"]["passed"],**{k:metrics["test"][k] for k in METRICS},"revenue_gap":metrics["gaps"]["revenue"],"regret_gap":metrics["gaps"]["regret_mean_bidder"],"gradient_audit_increase":metrics["attack_audit"]["increase"],"independent_audit_increase":independent["mean_increase"],"elapsed_seconds":metrics["elapsed_seconds"]}
        rows.append(row)
    groups=group_rows(rows)
    comparisons=paired_comparisons(groups,manifest["regret_comparison_rule"])
    post=json.loads((ROOT/"results/study_v1/post_evaluation.json").read_text())
    output=ROOT/"results/study_v1/analysis"
    output.mkdir(exist_ok=True)
    save_csv(output/"per_seed.csv",sorted(rows,key=lambda r:r["job"]))
    # Portable profile-level raw metrics, with an empty third bidder field for n=2.
    profile_fields=["job","model","seed","n","m","train_size","beta","delta","profile_index","revenue","regret_bidder_0","regret_bidder_1","regret_bidder_2","regret_mean","regret_total","truthful_utility_min","optimal_welfare"]
    with (output/"test_profiles.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=profile_fields)
        writer.writeheader()
        for row in sorted(rows,key=lambda r:r["job"]):
            raw=torch.load(ROOT/row["run"]/"test_raw.pt",weights_only=True)
            revenues,regrets,utilities,welfare=(raw[k].tolist() for k in ("revenue","regret","truthful_utility","welfare"))
            identity={k:row[k] for k in ("job","model","seed","n","m","train_size","beta","delta")}
            for index,(rev,r,u,w) in enumerate(zip(revenues,regrets,utilities,welfare)):
                writer.writerow({**identity,"profile_index":index,"revenue":rev,"regret_bidder_0":r[0],"regret_bidder_1":r[1],"regret_bidder_2":r[2] if len(r)==3 else "","regret_mean":sum(r)/len(r),"regret_total":sum(r),"truthful_utility_min":min(u),"optimal_welfare":w})
    post_rows=[]
    for result in post["results"]:
        for row in result["records"]:
            post_rows.append({k:row[k] for k in ("evaluation","training_model","seed","n","m","beta","revenue","regret_mean_bidder","regret_mean_total","regret_p95_bidder","regret_max","permutation_allocation","permutation_payment")})
    save_csv(output/"post_evaluation_per_seed.csv",post_rows)
    flat=[]
    for g in groups:
        row={k:g[k] for k in ("n","m","train_size","beta","delta","model","parameter_count","gate_passed_count")}
        for metric in METRICS:
            row[metric+"_mean"]=g["metrics"][metric]["mean"]
            row[metric+"_std"]=g["metrics"][metric]["std"]
        flat.append(row)
    save_csv(output/"aggregate.csv",flat)
    summary={"unique_training_runs":len(rows),"groups":groups,"paired_comparisons":comparisons,"t4_critical":t4_critical(),"operational_gates_passed":sum(r["gate_passed"] for r in rows),"max_gradient_audit_increase":max(r["gradient_audit_increase"] for r in rows),"max_independent_audit_increase":max(r["independent_audit_increase"] for r in rows),"failed_attempts":progress["failures"],"post_evaluation":post,"claim_policy":"Finite attack lower bounds; descriptive unadjusted t intervals; no novelty claims; no higher-revenue ranking when regret is materially worse."}
    save_json(output/"summary.json",summary)
    folder=ROOT/"figures/study_v1"
    folder.mkdir(parents=True,exist_ok=True)
    build_figures(groups,comparisons,post,folder)
    print(json.dumps({k:summary[k] for k in ("unique_training_runs","operational_gates_passed","max_gradient_audit_increase","max_independent_audit_increase","t4_critical")},indent=2))


if __name__=="__main__": main()
