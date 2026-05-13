"""C-P5: forward_given_params on a tiny batch in eval mode.

Inspect the signature first, then try call patterns from simple to explicit.
"""
import inspect
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

print("forward_given_params signature:",
      inspect.signature(model.forward_given_params))

bs = 4
x = torch.randn(bs, 3)
weights, biases, extra_params = model.get_parameters(mode="wbx")

# Attempt 1: minimal call (no mask, no regime).
try:
    with torch.no_grad():
        density_params = model.forward_given_params(x, weights, biases)
    call_pattern = "x, weights, biases (mask=None, regime=None)"
    print("Attempt 1 succeeded:", call_pattern)
except TypeError as e:
    print("Attempt 1 (no mask/regime) failed with TypeError:", e)
    # Attempt 2: with explicit mask and regime.
    try:
        mask = torch.ones(bs, model.num_vars)
        regime = torch.zeros(bs, dtype=torch.long).numpy()
        with torch.no_grad():
            density_params = model.forward_given_params(
                x, weights, biases, mask=mask, regime=regime
            )
        call_pattern = "x, weights, biases, mask=ones(bs,d), regime=zeros(bs)"
        print("Attempt 2 succeeded:", call_pattern)
    except Exception as e2:
        print("Attempt 2 also failed:", type(e2).__name__, str(e2))
        raise

print("call pattern that worked:", call_pattern)
print("density_params type:", type(density_params).__name__)
print("density_params length:", len(density_params))
for i, dp in enumerate(density_params):
    print(f"  density_params[{i}] shape:", tuple(dp.shape))
    print(f"  density_params[{i}] dtype:", dp.dtype)
