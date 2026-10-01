from itertools import permutations,product
import unittest
import torch
from src.models import DeepSetMechanism, build_model
from src.valuations import generate
from src.regret import best_responses,regrets


class DeepSetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def test_equivariance_all_sizes_and_variants(self):
        for n,m,context,aggregation in product((2,3),(3,4),(True,False),("mean","sum")):
            model = DeepSetMechanism(n,m,width=12,report_upper=18,context=context,aggregation=aggregation).double()
            for x in (generate(7,n,m,.5,99)["x"].double(),torch.zeros(7,n,2**m-1,dtype=torch.float64),torch.full((7,n,2**m-1),18.,dtype=torch.float64)):
                reference = model(x)
                for perm in permutations(range(n)):
                    actual = model(x[:,perm])
                    for key in ("g","p"):
                        torch.testing.assert_close(actual[key],reference[key][:,perm],rtol=1e-12,atol=1e-12)

    def test_mean_sum_reparameterization_at_fixed_n(self):
        n,m = 3,3
        mean = DeepSetMechanism(n,m,width=10,report_upper=12).double()
        total = DeepSetMechanism(n,m,width=10,report_upper=12,aggregation="sum").double()
        total.load_state_dict(mean.state_dict())
        with torch.no_grad():
            for branch in (total.allocation,total.payment):
                branch.psi[0].weight[:,10:] /= n
        x = generate(5,n,m,.5,2)["x"].double()
        for key in ("g","p"):
            torch.testing.assert_close(total(x)[key],mean(x)[key],rtol=1e-12,atol=1e-12)

    def test_shared_parameters_and_attack_backward(self):
        config={"n":2,"m":3,"model":"deepset","width":50,"report_upper":12}
        model=build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()),11308)
        truth=generate(5,2,3,.5,2)["x"]
        response=best_responses(model,truth,steps=5,random_starts=1,lr=.05,seed=1)
        regret=regrets(model,truth,response)
        (-model(truth)["p"].sum(-1).mean()+100*regret.mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_augmentation_cache_inverse(self):
        x=torch.arange(4*3*7).reshape(4,3,7)
        perm=torch.tensor([[2,0,1],[0,2,1],[1,2,0],[1,0,2]])
        shuffled=x.gather(1,perm[...,None].expand_as(x))
        restored=shuffled.gather(1,perm.argsort(-1)[...,None].expand_as(x))
        torch.testing.assert_close(restored,x,rtol=0,atol=0)
