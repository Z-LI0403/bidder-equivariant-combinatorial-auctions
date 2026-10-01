"""Metrics and exact bidder permutation enumeration for n<=3 pilots."""
from itertools import permutations
import torch
from .models import utility
from .regret import best_responses, regrets


def permutation_errors(model, x):
    with torch.no_grad():
        reference = model(x)
        errors = {"allocation": [], "payment": []}
        for perm in permutations(range(model.n)):
            if perm == tuple(range(model.n)):
                continue
            altered = model(x[:, perm])
            for key, name in [("g", "allocation"), ("p", "payment")]:
                expected = reference[key][:, perm].flatten(1)
                actual = altered[key].flatten(1)
                ratio = torch.linalg.vector_norm(actual-expected, dim=-1) / torch.linalg.vector_norm(expected, dim=-1).clamp_min(1e-12)
                errors[name].append(ratio)
        return {"permutation_"+name: float(torch.stack(values).mean()) if values else 0.0 for name, values in errors.items()}


def evaluate(model, x, budget, seed, batch_size=64):
    rows = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start+batch_size]
        responses = best_responses(model, batch, **budget, seed=seed+start)
        with torch.no_grad():
            outcome = model(batch)
            r = regrets(model, batch, responses)
            rows.append({"revenue": outcome["p"].sum(-1), "regret": r,
                         "utility": utility(batch, outcome), "responses": responses})
    revenue = torch.cat([row["revenue"] for row in rows])
    regret = torch.cat([row["regret"] for row in rows])
    utils = torch.cat([row["utility"] for row in rows])
    responses = torch.cat([row["responses"] for row in rows])
    with torch.no_grad():
        # Exact welfare oracle over all feasible assignments; not an auction baseline.
        welfare = (x.flatten(1) @ model.decoder.incidence.flatten(1).T).max(-1).values
    metrics = {"revenue": float(revenue.mean()), "revenue_mc_se": float(revenue.std()/len(x)**.5),
               "regret_mean_bidder": float(regret.mean()), "regret_mean_total": float(regret.sum(-1).mean()),
               "regret_p95_bidder": float(torch.quantile(regret.flatten(), .95)),
               "regret_max": float(regret.max()), "regret_per_bidder": regret.mean(0).tolist(),
               "truthful_utility_min": float(utils.min()), "optimal_welfare": float(welfare.mean())}
    metrics.update(permutation_errors(model, x))
    return metrics, {"revenue": revenue, "regret": regret, "truthful_utility": utils, "responses": responses, "welfare": welfare}
