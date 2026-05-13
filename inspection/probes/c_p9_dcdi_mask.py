"""C-P9: DCDI structural mask handling for parent exclusion.

Goal: determine whether wrapper-side sampling can force DCDI's
forward_given_params to ignore non-surviving parents of a target node,
so the DCDI MMD sampler can honour the thresholded valid DAG and not
soft edges in P.

Strategy:
1. Inspect the forward_given_params signature.
2. Test whether the `mask=` argument is the structural mask or an
   intervention mask. The source comment in base_model.py describes
   `mask` as `tensor, batch_size x num_vars`; this is the intervention
   mask, not a parent mask. Test both shapes to confirm.
3. Force structural masking via `model.adjacency` plus saturated
   `log_alpha`. Vary an excluded parent and confirm the target's
   density parameters are invariant. Vary an included parent and
   confirm sensitivity.

If excluded-parent invariance holds under step 3, Option A is feasible:
the wrapper sets model.adjacency to the thresholded valid DAG before
sampling.
"""
import inspect
import sys

import torch

sys.path.insert(0, "external/source_inspection/dcdi")
from dcdi.models.learnables import LearnableModel_NonLinGaussANM


torch.set_default_tensor_type(torch.FloatTensor)


def make_model():
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
    return model


model = make_model()
print("forward_given_params signature:",
      inspect.signature(model.forward_given_params))

bs = 4
x = torch.randn(bs, 3)
weights, biases, extra_params = model.get_parameters(mode="wbx")

# Convention: model.adjacency[parent, child] = 1 if edge parent->child exists.
# Build a structural mask: parent 0 -> child 2 (included),
# parent 1 -> child 2 (EXCLUDED), parent 0 -> child 1 (any, not key here).
struct_mask = torch.zeros(3, 3)
struct_mask[0, 2] = 1.0  # parent 0 included for target 2
struct_mask[0, 1] = 1.0  # parent 0 included for target 1 (peripheral)
# struct_mask[1, 2] left at 0 -> parent 1 EXCLUDED for target 2


# ---------------------------------------------------------------------------
# Step 1: try the `mask=` argument with the structural-mask shape
# ---------------------------------------------------------------------------
print("\n=== Step 1: mask= argument with structural-mask shape (3, 3) ===")
try:
    with torch.no_grad():
        out = model.forward_given_params(
            x, weights, biases, mask=struct_mask
        )
    print("  call accepted with structural-shape mask")
    print("  density_params shapes:", [tuple(dp.shape) for dp in out])
except Exception as e:
    print("  failed:", type(e).__name__, str(e)[:300])

# ---------------------------------------------------------------------------
# Step 2: try the `mask=` argument with the documented intervention shape
# ---------------------------------------------------------------------------
print("\n=== Step 2: mask= argument with intervention-mask shape (bs, num_vars) ===")
import numpy as np
try:
    interv_mask = torch.ones(bs, 3)  # all variables observed (no intervention)
    regime = np.zeros(bs, dtype=np.int64)
    with torch.no_grad():
        out = model.forward_given_params(
            x, weights, biases, mask=interv_mask, regime=regime
        )
    print("  call accepted with intervention-shape mask")
    print("  density_params shapes:", [tuple(dp.shape) for dp in out])
    print("  note: this is the INTERVENTION mask path, not structural masking")
except Exception as e:
    print("  failed:", type(e).__name__, str(e)[:300])

# ---------------------------------------------------------------------------
# Step 3: enforce structural mask via model.adjacency + saturated log_alpha
# ---------------------------------------------------------------------------
print("\n=== Step 3: enforce structural mask via model.adjacency + log_alpha ===")
model = make_model()
with torch.no_grad():
    model.adjacency.copy_(struct_mask)
    saturated = struct_mask * 100.0 + (1.0 - struct_mask) * -100.0
    model.gumbel_adjacency.log_alpha.copy_(saturated)

print("model.adjacency:")
print(model.adjacency.tolist())
print("model.get_w_adj():")
print(model.get_w_adj().tolist())

weights, biases, extra_params = model.get_parameters(mode="wbx")

# Two batches differing only in the EXCLUDED parent's column (column 1).
torch.manual_seed(123)
x_a = torch.randn(bs, 3)
x_excluded = x_a.clone()
x_excluded[:, 1] = x_a[:, 1] + 5.0  # large change in excluded parent

torch.manual_seed(42)
with torch.no_grad():
    dp_a = model.forward_given_params(x_a, weights, biases)
torch.manual_seed(42)
with torch.no_grad():
    dp_excl = model.forward_given_params(x_excluded, weights, biases)

print(f"\ndensity_params[2] for x_a       : "
      f"{[round(v, 6) for v in dp_a[2].squeeze().tolist()]}")
print(f"density_params[2] for x_excluded: "
      f"{[round(v, 6) for v in dp_excl[2].squeeze().tolist()]}")

excluded_diff = float((dp_a[2] - dp_excl[2]).abs().max().item())
print(f"\nmax |delta| target 2 when EXCLUDED parent varied: {excluded_diff:.6e}")
excluded_invariant = excluded_diff < 1e-6
print(f"excluded-parent invariance holds? {excluded_invariant}")

# Two batches differing only in the INCLUDED parent's column (column 0).
x_included = x_a.clone()
x_included[:, 0] = x_a[:, 0] + 5.0  # large change in included parent

torch.manual_seed(42)
with torch.no_grad():
    dp_inc = model.forward_given_params(x_included, weights, biases)

print(f"density_params[2] for x_included: "
      f"{[round(v, 6) for v in dp_inc[2].squeeze().tolist()]}")

included_diff = float((dp_a[2] - dp_inc[2]).abs().max().item())
print(f"\nmax |delta| target 2 when INCLUDED parent varied: {included_diff:.6e}")
included_sensitive = included_diff > 1e-3
print(f"included-parent sensitivity observed? {included_sensitive}")

if not included_sensitive:
    print("note: included-parent sensitivity may be weak because of random "
          "MLP weights and a small input perturbation; the load-bearing "
          "result is the excluded-parent invariance check above.")
