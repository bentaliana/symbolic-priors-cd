"""C-P7: Minimal ancestral sampling sketch with one variable clamped."""
import sys

import torch

sys.path.insert(0, "external/source_inspection/dcdi")
from dcdi.models.learnables import LearnableModel_NonLinGaussANM

torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)

model = LearnableModel_NonLinGaussANM(
    num_vars=3,
    num_layers=2,
    hid_dim=8,
    nonlin="leaky-relu",
    intervention=False,
    intervention_type="perfect",
    intervention_knowledge="known",
    num_regimes=1,
)
model.eval()

# Force a known DAG: 0 -> 1 -> 2 in row-source/column-destination convention.
adj = torch.zeros(3, 3)
adj[0, 1] = 1.0
adj[1, 2] = 1.0
with torch.no_grad():
    model.adjacency.copy_(adj)
    # Set log_alpha so sigmoid(log_alpha) is ~1 on edges and ~0 elsewhere.
    saturated = adj * 100.0 + (1.0 - adj) * -100.0
    model.gumbel_adjacency.log_alpha.copy_(saturated)

print("model.adjacency:")
print(model.adjacency.tolist())
print("model.get_w_adj():")
print(model.get_w_adj().tolist())

bs = 5
weights, biases, extra_params = model.get_parameters(mode="wbx")
ext_t = model.transform_extra_params(model.extra_params)


def cond_sample(x_in: torch.Tensor, var_index: int) -> torch.Tensor:
    """Sample variable var_index from its conditional given parent values in x_in."""
    with torch.no_grad():
        density_params = model.forward_given_params(x_in, weights, biases)
        dp = list(torch.unbind(density_params[var_index], 1))
        dp.extend(list(torch.unbind(ext_t[var_index], 0)))
        return model.get_distribution(dp).sample()


x = torch.zeros(bs, 3)
target = 1
value = 0.5
print(f"applying do(X_{target}={value})")

# Topological order is [0, 1, 2] under our forced DAG.
# Sample X_0 (root) given zeros in other columns.
x[:, 0] = cond_sample(x, 0)
# Clamp X_1 to value.
x[:, 1] = value
# Sample X_2 given X_0 and X_1 (clamped).
x[:, 2] = cond_sample(x, 2)

print("samples under do(X_1 = 0.5):")
print(x.tolist())

invariant = torch.allclose(x[:, 1], torch.full((bs,), value))
print("clamping invariant holds?", invariant)
print("X_2 mean:", float(x[:, 2].mean().item()))
print("X_2 std:", float(x[:, 2].std().item()))

if invariant:
    print("ancestral-sampling sketch SUCCEEDED")
else:
    print("ancestral-sampling sketch FAILED: clamping invariant broken")
