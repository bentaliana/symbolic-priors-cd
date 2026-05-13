"""C-P6: Conditional Normal construction and sampling."""
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

bs = 4
x = torch.randn(bs, 3)
weights, biases, extra_params = model.get_parameters(mode="wbx")

with torch.no_grad():
    density_params = model.forward_given_params(x, weights, biases)
    extra_params_t = model.transform_extra_params(model.extra_params)

print("density_params type:", type(density_params).__name__)
print("density_params length:", len(density_params))
for i, dp in enumerate(density_params):
    print(f"  density_params[{i}] shape:", tuple(dp.shape))

print("extra_params_t type:", type(extra_params_t).__name__)
print("extra_params_t length:", len(extra_params_t))
for i, ep in enumerate(extra_params_t):
    print(f"  extra_params_t[{i}] shape:", tuple(ep.shape),
          "value(s):", ep.tolist())

i = 0
dp_i = list(torch.unbind(density_params[i], 1))
print(f"after unbind dim 1: {len(dp_i)} tensors, "
      f"shapes={[tuple(t.shape) for t in dp_i]}")
dp_i.extend(list(torch.unbind(extra_params_t[i], 0)))
print(f"after extend with extra_params: {len(dp_i)} tensors, "
      f"shapes={[tuple(t.shape) for t in dp_i]}")

dist = model.get_distribution(dp_i)
print("dist type:", type(dist).__name__)
print("dist.loc shape:", tuple(dist.loc.shape))
print("dist.scale shape:", tuple(dist.scale.shape))

with torch.no_grad():
    sample = dist.sample()
print("sample shape:", tuple(sample.shape))
print("first few values:", sample[:4].tolist())
