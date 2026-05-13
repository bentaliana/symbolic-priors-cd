"""C-P3: Instantiate LearnableModel_NonLinGaussANM in observational mode on CPU."""
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
print("instantiated:", type(model).__name__)
print("num_vars:", model.num_vars)
print("num_layers:", model.num_layers)
print("hid_dim:", model.hid_dim)
print("intervention:", model.intervention)
print("intervention_type:", model.intervention_type)
print("intervention_knowledge:", model.intervention_knowledge)
