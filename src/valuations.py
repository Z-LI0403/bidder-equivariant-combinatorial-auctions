"""Bundle bitmask order is 1, ..., 2**m-1; the empty bundle has value zero."""
from itertools import combinations
import torch


def bundle_features(m: int, *, dtype=torch.float32):
    if m < 1:
        raise ValueError("m must be positive")
    masks = torch.arange(1, 2**m)
    items = ((masks[:, None] >> torch.arange(m)) & 1).to(dtype)
    pairs = list(combinations(range(m), 2))
    interactions = torch.stack([items[:, j] * items[:, k] for j, k in pairs], -1) if pairs else torch.empty(len(masks), 0, dtype=dtype)
    return items, interactions


def from_coefficients(a: torch.Tensor, c: torch.Tensor, beta: float):
    """a: [B,n,m], c: [B,n,m(m-1)/2], returns [B,n,2**m-1]."""
    if a.ndim != 3 or c.shape != (*a.shape[:2], a.shape[-1] * (a.shape[-1]-1)//2):
        raise ValueError("invalid coefficient dimensions")
    if beta < 0 or not torch.isfinite(a).all() or not torch.isfinite(c).all() or (a < 0).any() or (c < 0).any():
        raise ValueError("finite nonnegative coefficients and beta required")
    items, interactions = bundle_features(a.shape[-1], dtype=a.dtype)
    return a @ items.to(a.device).T + beta * (c @ interactions.to(c.device).T)


def generate(size: int, n: int, m: int, beta: float, seed: int, scales=None):
    """Independent U(0,scale_i) a coefficients; independent U(0,1) c.

    Heterogeneity scales item coefficients a only; c remains U(0,1).
    Coefficients and values are retained so generation can be audited.
    """
    if size < 1 or n < 1 or m < 1:
        raise ValueError("positive size, n and m required")
    scales = torch.ones(n) if scales is None else torch.tensor(scales, dtype=torch.float32)
    if scales.shape != (n,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("one finite positive scale per bidder required")
    rng = torch.Generator().manual_seed(seed)
    a = torch.rand(size, n, m, generator=rng) * scales[None, :, None]
    c = torch.rand(size, n, m*(m-1)//2, generator=rng)
    return {"x": from_coefficients(a, c, beta), "a": a, "c": c}
