"""Common feasible decoder with MLP, DeepSet, and local-only backbones."""
import torch
from torch import nn
from .feasibility import FeasibleLottery


def network(inputs, hidden, outputs):
    layers, previous = [], inputs
    for width in hidden:
        layers.extend([nn.Linear(previous, width), nn.Tanh()])
        previous = width
    layers.append(nn.Linear(previous, outputs))
    result = nn.Sequential(*layers)
    for layer in result:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return result


class MLPMechanism(nn.Module):
    def __init__(self, n, m, hidden=(64, 64), report_upper=6.0):
        super().__init__()
        if report_upper <= 0:
            raise ValueError("positive report bound required")
        self.n, self.m, self.k, self.report_upper = n, m, 2**m-1, float(report_upper)
        self.allocation = network(n*self.k, hidden, n*self.k)
        self.payment = network(n*self.k, hidden, n)
        self.decoder = FeasibleLottery(n, m)

    def forward(self, reports):
        features = reports.flatten(1) / self.report_upper
        scores = self.allocation(features).reshape(-1, self.n, self.k)
        g, lottery = self.decoder(scores)
        fraction = self.payment(features).sigmoid()
        p = fraction * (g * reports).sum(-1)
        return {"g": g, "p": p, "lottery": lottery, "fraction": fraction}


def utility(true_values, outcome):
    return (true_values * outcome["g"]).sum(-1) - outcome["p"]


class SharedBackbone(nn.Module):
    def __init__(self, k, width, output, context=True, aggregation="mean"):
        super().__init__()
        self.context, self.aggregation = context, aggregation
        self.phi = network(k, (), width)
        self.psi = network(width*(2 if context else 1), (), width)
        self.head = network(width, (), output)

    def forward(self, x):
        e = self.phi(x).tanh()
        if self.context:
            pooled = e.mean(1, keepdim=True) if self.aggregation == "mean" else e.sum(1, keepdim=True)
            e = torch.cat((e, pooled.expand_as(e)), -1)
        return self.head(self.psi(e).tanh())


class DeepSetMechanism(nn.Module):
    def __init__(self, n, m, width=50, report_upper=12., context=True, aggregation="mean"):
        super().__init__()
        if report_upper <= 0 or aggregation not in ("mean", "sum"):
            raise ValueError("invalid report bound or aggregation")
        self.n, self.m, self.k, self.report_upper = n, m, 2**m-1, float(report_upper)
        self.allocation = SharedBackbone(self.k, width, self.k, context, aggregation)
        self.payment = SharedBackbone(self.k, width, 1, context, aggregation)
        self.decoder = FeasibleLottery(n, m)

    def forward(self, reports):
        x = reports/self.report_upper
        g, lottery = self.decoder(self.allocation(x))
        fraction = self.payment(x).squeeze(-1).sigmoid()
        p = fraction*(g*reports).sum(-1)
        return {"g":g, "p":p, "lottery":lottery, "fraction":fraction}


def build_model(config):
    name = config["model"]
    if name in ("mlp", "mlp_aug"):
        return MLPMechanism(config["n"],config["m"],config["hidden"],config["report_upper"])
    if name in ("deepset", "local", "deepset_sum"):
        return DeepSetMechanism(config["n"],config["m"],config["width"],config["report_upper"],
                                context=name != "local",aggregation="sum" if name == "deepset_sum" else "mean")
    raise ValueError("unknown model: "+name)
