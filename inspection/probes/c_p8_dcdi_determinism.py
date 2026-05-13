"""C-P8: Deterministic repeatability on CPU for a tiny controlled run."""
import sys

import numpy as np
import torch

sys.path.insert(0, "external/source_inspection/dcdi")
from dcdi.dag_optim import compute_dag_constraint
from dcdi.models.learnables import LearnableModel_NonLinGaussANM
from dcdi.utils.penalty import compute_penalty


def tiny_train() -> torch.Tensor:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_default_tensor_type(torch.FloatTensor)

    m = LearnableModel_NonLinGaussANM(
        num_vars=3,
        num_layers=2,
        hid_dim=8,
        nonlin="leaky-relu",
        intervention=False,
        intervention_type="perfect",
        intervention_knowledge="known",
        num_regimes=1,
    )

    x = torch.randn(32, 3)
    opt = torch.optim.RMSprop(m.parameters(), lr=1e-3)

    full_adj = torch.ones(3, 3) - torch.eye(3)
    constraint_norm = float(compute_dag_constraint(full_adj).item())

    for _ in range(5):
        weights, biases, extra_params = m.get_parameters(mode="wbx")
        log_lik = m.compute_log_likelihood(x, weights, biases, extra_params)
        nll = -log_lik.mean()
        w_adj = m.get_w_adj()
        h = compute_dag_constraint(w_adj) / constraint_norm
        reg = 0.1 * compute_penalty([w_adj], p=1) / (3 ** 2)
        aug = nll + reg + h
        opt.zero_grad()
        aug.backward()
        opt.step()

    return m.gumbel_adjacency.log_alpha.detach().clone()


result1 = tiny_train()
result2 = tiny_train()

identical = bool(torch.equal(result1, result2))
print("identical?", identical)
if not identical:
    diff = (result1 - result2).abs()
    print("max abs diff:", float(diff.max().item()))
    print("close within 1e-6?", bool(torch.allclose(result1, result2, atol=1e-6)))
    print("close within 1e-4?", bool(torch.allclose(result1, result2, atol=1e-4)))
print("result1[0, 1]:", float(result1[0, 1].item()))
print("result2[0, 1]:", float(result2[0, 1].item()))
