"""D-P1: Import feasibility from local source clone."""
import sys

sys.path.insert(0, "external/source_inspection/dagma/src")
from dagma.linear import DagmaLinear
from dagma import utils

print("DAGMA import OK", DagmaLinear)
print("utils.set_random_seed exists:", hasattr(utils, "set_random_seed"))
