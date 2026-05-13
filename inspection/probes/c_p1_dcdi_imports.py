"""C-P1: Targeted DCDI imports without dcdi.train."""
import sys

sys.path.insert(0, "external/source_inspection/dcdi")

try:
    from dcdi.models.learnables import LearnableModel_NonLinGaussANM
    from dcdi.dag_optim import GumbelAdjacency, compute_dag_constraint
    from dcdi.utils.penalty import compute_penalty
    print("DCDI low-level imports OK")
    print("LearnableModel_NonLinGaussANM:", LearnableModel_NonLinGaussANM)
    print("GumbelAdjacency:", GumbelAdjacency)
    print("compute_dag_constraint:", compute_dag_constraint)
    print("compute_penalty:", compute_penalty)
except Exception as e:
    print("import failed:", type(e).__name__, str(e))
    raise
