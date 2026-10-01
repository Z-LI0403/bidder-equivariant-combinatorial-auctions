"""Projected multistart attack. Numerical regret is a lower bound, not a proof."""
import torch
from .models import utility


def deviation_utilities(model, truth, deviations):
    """truth [B,n,K], deviations [R,B,n,K] -> utilities [R,B,n].

    Each of the n deviations is unilateral: all other bidders remain truthful.
    Utilities always use TRUE values, while payments use submitted reports.
    """
    r, b, n, k = deviations.shape
    diagonal = torch.eye(n, device=truth.device, dtype=truth.dtype)[None, None, :, :, None]
    profiles = truth[None, :, None, :, :].expand(r, b, n, n, k)
    profiles = profiles * (1-diagonal) + deviations[:, :, :, None, :] * diagonal
    out = model(profiles.reshape(r*b*n, n, k))
    g = out["g"].reshape(r, b, n, n, k)
    p = out["p"].reshape(r, b, n, n)
    idx = torch.arange(n, device=truth.device)
    selected_g, selected_p = g[:, :, idx, idx, :], p[:, :, idx, idx]
    return (selected_g * truth[None]).sum(-1) - selected_p


def best_responses(model, truth, *, steps, random_starts, lr, seed, initial=None):
    """Adam on normalized box reports z in [0,1]^K, retaining best iterates.

    Truth, zero, upper corner, random starts, and optional cached candidates
    are included. Parameter gradients are not accumulated in the inner loop.
    """
    if steps < 0 or random_starts < 0 or lr <= 0:
        raise ValueError("invalid attack budget")
    rng = torch.Generator(device=truth.device).manual_seed(seed)
    upper = model.report_upper
    starts = [truth.detach()/upper, torch.zeros_like(truth), torch.ones_like(truth)]
    starts += [torch.rand(truth.shape, generator=rng, device=truth.device, dtype=truth.dtype) for _ in range(random_starts)]
    if initial is not None:
        starts.append(initial.detach()/upper)
    z = torch.stack(starts).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    with torch.no_grad():
        best_u = utility(truth, model(truth))
    best = truth.detach().clone()
    for step in range(steps+1):
        with torch.enable_grad():
            u = deviation_utilities(model, truth, z*upper)
            values, indices = u.detach().max(0)
            candidates = (z.detach()*upper).gather(0, indices[None, :, :, None].expand(1, *truth.shape)).squeeze(0)
            improved = values > best_u
            best = torch.where(improved[..., None], candidates, best)
            best_u = torch.maximum(best_u, values)
            if step == steps or not u.requires_grad:
                break
            gradient, = torch.autograd.grad(-u.sum(), z, allow_unused=True)
            if gradient is None:
                break
            optimizer.zero_grad(set_to_none=True)
            z.grad = gradient
            optimizer.step()
            with torch.no_grad():
                z.clamp_(0, 1)
    return best.detach()


def regrets(model, truth, responses):
    truthful = utility(truth, model(truth))
    deviating = deviation_utilities(model, truth, responses[None]).squeeze(0)
    return (deviating-truthful).clamp_min(0)
