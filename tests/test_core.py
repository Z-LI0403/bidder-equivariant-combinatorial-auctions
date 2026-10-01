from itertools import permutations, product
import unittest
import torch
from src.valuations import generate, from_coefficients
from src.feasibility import FeasibleLottery
from src.models import MLPMechanism, utility
from src.regret import best_responses, regrets, deviation_utilities
from src.evaluation import permutation_errors


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_manual_nonadditive_values(self):
        a = torch.tensor([[[1., 2., 3.]]])
        c = torch.tensor([[[4., 5., 6.]]])  # pairs (0,1), (0,2), (1,2)
        actual = from_coefficients(a, c, .5)
        torch.testing.assert_close(actual, torch.tensor([[[1., 2., 5., 3., 6.5, 8., 13.5]]]))

    def test_independent_generation_enumeration(self):
        data = generate(10, 3, 4, .25, 7)
        for b in range(10):
            for i in range(3):
                pairs = [(j,k) for j in range(4) for k in range(j+1,4)]
                for mask in range(1,16):
                    v = sum(float(data["a"][b,i,j]) for j in range(4) if mask & (1<<j))
                    v += .25*sum(float(data["c"][b,i,l]) for l,(j,k) in enumerate(pairs) if mask & (1<<j) and mask & (1<<k))
                    self.assertAlmostEqual(v, float(data["x"][b,i,mask-1]), places=5)
        torch.testing.assert_close(data["x"], generate(10,3,4,.25,7)["x"], rtol=0, atol=0)
        self.assertFalse(torch.equal(data["x"], generate(10,3,4,.25,8)["x"]))

    def test_additive_monotonic_and_degenerate(self):
        for beta in (0., .25, .5, 1.):
            x = generate(5,2,3,beta,8)["x"]
            for s in range(1,8):
                for t in range(1,8):
                    if s & t == s:
                        self.assertTrue((x[...,s-1] <= x[...,t-1]+1e-6).all())
            if beta == 0:
                torch.testing.assert_close(x[...,6], x[...,0]+x[...,1]+x[...,3])
        torch.testing.assert_close(from_coefficients(torch.tensor([[[2.]]]), torch.empty(1,1,0), 1.), torch.tensor([[[2.]]]))
        with self.assertRaises(ValueError):
            generate(1,2,3,-1,2)

    def test_uniform_feasible_lottery_manual(self):
        decoder = FeasibleLottery(2,2)
        g, lottery = decoder(torch.zeros(1,2,3))
        torch.testing.assert_close(lottery, torch.full((1,9),1/9))
        # P(exact singleton)=1/3*2/3; P(both)=1/9.
        torch.testing.assert_close(g, torch.tensor([[[2/9,2/9,1/9],[2/9,2/9,1/9]]]))

    def test_feasibility_all_requested_dimensions(self):
        for n,m in product((2,3),(3,4)):
            decoder = FeasibleLottery(n,m)
            g,q = decoder(torch.randn(8,n,2**m-1)*20)
            self.assertEqual(len(q[0]), (n+1)**m)
            torch.testing.assert_close(q.sum(-1),torch.ones(8))
            self.assertTrue((g.sum(-1) <= 1+1e-6).all())
            # Independent incidence reconstruction directly from owner rows.
            manual = torch.zeros_like(g)
            for l, owners in enumerate(decoder.owners.tolist()):
                for i in range(n):
                    mask = sum(1<<j for j, owner in enumerate(owners) if owner == i+1)
                    if mask:
                        manual[:,i,mask-1] += q[:,l]
            torch.testing.assert_close(g, manual, atol=2e-6, rtol=2e-6)
            for j in range(m):
                masks = [mask-1 for mask in range(1,2**m) if mask & (1<<j)]
                self.assertTrue((g[:,:,masks].sum((1,2)) <= 1+2e-6).all())

    def test_fractional_feasibility_counterexample(self):
        decoder = FeasibleLottery(3,3)
        z = torch.zeros(3,7)
        z[0,2], z[1,4], z[2,5] = .5,.5,.5  # bundles 3,5,6
        self.assertTrue((z.sum(-1) <= 1).all())
        for j in range(3):
            self.assertEqual(float(z[:,[s-1 for s in range(1,8) if s & (1<<j)]].sum()),1.)
        # Each feasible outcome satisfies this valid inequality; z violates it.
        occupancy = decoder.incidence[:,0,2]+decoder.incidence[:,1,4]+decoder.incidence[:,2,5]
        self.assertLessEqual(float(occupancy.max()),1.)
        self.assertEqual(float(z[0,2]+z[1,4]+z[2,5]),1.5)

    def test_decoder_bidder_equivariance(self):
        torch.manual_seed(20)
        for n in (2,3):
            decoder = FeasibleLottery(n,3).double()
            scores = torch.randn(5,n,7,dtype=torch.float64)
            reference,_ = decoder(scores)
            for perm in permutations(range(n)):
                actual,_ = decoder(scores[:,perm])
                torch.testing.assert_close(actual,reference[:,perm],atol=1e-12,rtol=1e-12)

    def test_ir_and_zero_values(self):
        model = MLPMechanism(2,3)
        x = generate(32,2,3,.5,20)["x"]
        out = model(x)
        self.assertTrue((utility(x,out) >= -1e-6).all())
        self.assertTrue((out["p"] >= 0).all())
        torch.testing.assert_close(model(torch.zeros_like(x))["p"],torch.zeros(32,2))

    def test_unilateral_utilities_independent_loop(self):
        model = MLPMechanism(3,3)
        truth = generate(4,3,3,.5,22)["x"]
        deviations = torch.rand(2,4,3,7)*6
        actual = deviation_utilities(model,truth,deviations)
        for r in range(2):
            for i in range(3):
                report = truth.clone()
                report[:,i] = deviations[r,:,i]
                out = model(report)
                expected = (truth[:,i]*out["g"][:,i]).sum(-1)-out["p"][:,i]
                torch.testing.assert_close(actual[r,:,i],expected)

    def test_analytic_manipulable_and_truthful_mechanisms(self):
        class PayOwnBid(torch.nn.Module):
            report_upper = 1.
            def forward(self,x):
                return {"g": torch.ones_like(x), "p": x.sum(-1)*.5}
        x = torch.tensor([[[.8]],[[.3]]])
        model = PayOwnBid()
        response = best_responses(model,x,steps=5,random_starts=1,lr=.05,seed=1)
        torch.testing.assert_close(response,torch.zeros_like(x))
        torch.testing.assert_close(regrets(model,x,response),x.squeeze(-1)*.5)
        class FreeAllocation(PayOwnBid):
            def forward(self,x):
                return {"g":torch.ones_like(x),"p":torch.zeros_like(x.sum(-1))}
        free = FreeAllocation()
        response = best_responses(free,x,steps=5,random_starts=1,lr=.05,seed=1)
        torch.testing.assert_close(regrets(free,x,response),torch.zeros(2,1))

    def test_attack_nested_budgets_and_outer_gradients(self):
        torch.manual_seed(1)
        model = MLPMechanism(2,3)
        truth = generate(6,2,3,.5,3)["x"]
        a = best_responses(model,truth,steps=2,random_starts=1,lr=.05,seed=2)
        b = best_responses(model,truth,steps=8,random_starts=2,lr=.05,seed=2,initial=a)
        ra, rb = regrets(model,truth,a),regrets(model,truth,b)
        self.assertTrue((rb >= ra-1e-6).all())
        self.assertTrue((b >= 0).all() and (b <= 6).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        rb.mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0.)

    def test_interior_best_response_against_analytic_and_grid(self):
        class QuadraticPayment(torch.nn.Module):
            report_upper = 1.
            def forward(self,x):
                return {"g": x, "p": x.square().sum(-1)}
        model = QuadraticPayment()
        truth = torch.tensor([[[.8]],[[.3]]],dtype=torch.float64)
        response = best_responses(model,truth,steps=150,random_starts=2,lr=.03,seed=3)
        # Utility v*b-b^2: interior maximizer b=v/2, regret v^2/4.
        torch.testing.assert_close(response,truth/2,atol=2e-4,rtol=0)
        measured = regrets(model,truth,response)
        torch.testing.assert_close(measured,truth.squeeze(-1).square()/4,atol=1e-7,rtol=0)
        grid = torch.linspace(0,1,10001,dtype=torch.float64)
        exhaustive = (truth.flatten()[:,None]*grid-grid.square()).max(-1).values
        torch.testing.assert_close(measured.flatten(),exhaustive,atol=1e-7,rtol=0)

    def test_regret_gradient_finite_difference_fixed_response(self):
        torch.manual_seed(4)
        model = MLPMechanism(2,3).double()
        truth = generate(4,2,3,.5,9)["x"].double()
        response = torch.zeros_like(truth)
        parameter = model.payment[-1].bias
        value = regrets(model,truth,response).mean()
        gradient, = torch.autograd.grad(value,parameter)
        eps = 1e-5
        with torch.no_grad():
            parameter[0] += eps
            plus = float(regrets(model,truth,response).mean())
            parameter[0] -= 2*eps
            minus = float(regrets(model,truth,response).mean())
            parameter[0] += eps
        self.assertAlmostEqual(float(gradient[0]),(plus-minus)/(2*eps),places=7)

    def test_permutation_metric_detects_mlp_error(self):
        torch.manual_seed(1)
        model = MLPMechanism(2,3)
        errors = permutation_errors(model,generate(10,2,3,.5,1)["x"])
        self.assertGreater(errors["permutation_allocation"],1e-4)
        self.assertGreater(errors["permutation_payment"],1e-4)

    def test_welfare_oracle_additive_manual(self):
        data=generate(12,3,4,0.,71)
        model=MLPMechanism(3,4,report_upper=18)
        enumerated=(data["x"].flatten(1)@model.decoder.incidence.flatten(1).T).max(-1).values
        # With additive values, each item can be assigned to its highest valuer.
        independent=data["a"].max(1).values.sum(-1)
        torch.testing.assert_close(enumerated,independent,atol=1e-6,rtol=1e-6)

    def test_normalized_permutation_metric_manual(self):
        class FixedBidder(torch.nn.Module):
            n=2
            def forward(self,x):
                return {"g":torch.tensor([[[1.],[0.]]]).expand(len(x),-1,-1),"p":torch.tensor([[.5,0.]]).expand(len(x),-1)}
        actual=permutation_errors(FixedBidder(),torch.tensor([[[1.],[2.]]]))
        for value in actual.values(): self.assertAlmostEqual(value,2**.5,places=6)


if __name__ == "__main__":
    unittest.main()
