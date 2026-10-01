"""A lottery over exhaustive feasible deterministic item assignments."""
from itertools import product
import torch
from torch import nn


class FeasibleLottery(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        if n < 1 or m < 1:
            raise ValueError("positive n and m required")
        self.n, self.m, self.k = n, m, 2**m-1
        # owner 0 means unallocated; owner i+1 means bidder i.
        owners = torch.tensor(list(product(range(n+1), repeat=m)), dtype=torch.long)
        masks = torch.stack([((owners == i+1).long() * (2**torch.arange(m))).sum(-1) for i in range(n)], -1)
        incidence = (masks[..., None] == torch.arange(1, 2**m)).float()
        self.register_buffer("owners", owners)
        self.register_buffer("incidence", incidence)  # [Q,n,K], Q=(n+1)**m

    def forward(self, scores):
        """[B,n,K] scores -> ([B,n,K] bundle marginals, [B,Q] lottery).

        The score of an outcome is the sum of its allocated bundle scores.
        Empty bundles have score zero. This is a restricted Gibbs family;
        all later backbones must use the same decoder for comparability.
        """
        if scores.shape[1:] != (self.n, self.k):
            raise ValueError("invalid score dimensions")
        logits = scores.flatten(1) @ self.incidence.flatten(1).T
        lottery = logits.softmax(-1)
        marginals = (lottery @ self.incidence.flatten(1)).reshape(-1, self.n, self.k)
        return marginals, lottery
