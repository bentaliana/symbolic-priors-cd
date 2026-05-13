"""C-P2: sys.modules diff to confirm cdt is not imported by targeted imports."""
import sys

sys.path.insert(0, "external/source_inspection/dcdi")

before = set(sys.modules)
from dcdi.models.learnables import LearnableModel_NonLinGaussANM  # noqa: F401
from dcdi.dag_optim import GumbelAdjacency, compute_dag_constraint  # noqa: F401
from dcdi.utils.penalty import compute_penalty  # noqa: F401
after = set(sys.modules)

newly_imported = after - before
cdt_modules = sorted(m for m in newly_imported if "cdt" in m.lower())
r_modules = sorted(m for m in newly_imported if m.startswith("cdt.utils.R"))

print("newly imported count:", len(newly_imported))
print("cdt-related newly imported:", cdt_modules)
print("cdt.utils.R newly imported:", r_modules)
print("newly imported dcdi modules:",
      sorted(m for m in newly_imported if m.startswith("dcdi")))
