"""C-P4: Access log_alpha and get_w_adj on a tiny DCDI-G model."""
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

log_alpha = model.gumbel_adjacency.log_alpha
print("log_alpha shape:", tuple(log_alpha.shape))
print("log_alpha dtype:", log_alpha.dtype)
print("log_alpha requires_grad:", log_alpha.requires_grad)
print("log_alpha[0, 1] init value:", float(log_alpha[0, 1].item()))
print("log_alpha unique values:", torch.unique(log_alpha).tolist())

w_adj = model.get_w_adj()
print("w_adj shape:", tuple(w_adj.shape))
print("w_adj dtype:", w_adj.dtype)
print("w_adj diagonal:", w_adj.diag().tolist())
print("w_adj[0, 1] off-diag value:", float(w_adj[0, 1].item()))
print("w_adj unique off-diag values:",
      torch.unique(w_adj[~torch.eye(3, dtype=torch.bool)]).tolist())
print("expected sigmoid(5) approx:", float(torch.sigmoid(torch.tensor(5.0))))
