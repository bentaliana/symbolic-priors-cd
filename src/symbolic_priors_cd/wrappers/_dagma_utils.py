"""Low-level DAGMA helpers for the wrapper layer.

DAGMA is loaded from the project's pinned source clone at
``external/source_inspection/dagma/src/dagma`` and not from any
installed DAGMA package. ``dagma.utils`` is intentionally not
imported; the wrapper does not call ``dagma.utils.set_random_seed``
because DAGMA's fit is deterministic for fixed input and
hyperparameters, and that helper mutates global NumPy state.

The module-level guard verifies the resolved import path so that an
installed DAGMA cannot silently shadow the pinned source.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve the DAGMA source path relative to this file.
# This file lives at src/symbolic_priors_cd/wrappers/_dagma_utils.py,
# so four parent steps reach the project root.
_DAGMA_SRC = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "external"
    / "source_inspection"
    / "dagma"
    / "src"
)

_linear_path = _DAGMA_SRC / "dagma" / "linear.py"
if not _DAGMA_SRC.exists() or not _linear_path.exists():
    raise ImportError(
        f"Pinned DAGMA source not found at '{_DAGMA_SRC}'. "
        "The inspected DAGMA source must be present at "
        "external/source_inspection/dagma/src relative to the project root."
    )
del _linear_path

# Drop any cached dagma.* entries that resolve outside the pinned
# source so the next import binds to the pinned location via the
# prepended sys.path entry below.
for _name in list(sys.modules):
    if _name != "dagma" and not _name.startswith("dagma."):
        continue
    _module = sys.modules[_name]
    _module_file = getattr(_module, "__file__", None)
    if _module_file is None:
        del sys.modules[_name]
        continue
    if not Path(_module_file).resolve().is_relative_to(_DAGMA_SRC):
        del sys.modules[_name]

if str(_DAGMA_SRC) not in sys.path:
    sys.path.insert(0, str(_DAGMA_SRC))

from dagma.linear import DagmaLinear  # noqa: E402

_actual = Path(sys.modules["dagma.linear"].__file__).resolve()
if not _actual.is_relative_to(_DAGMA_SRC):
    raise ImportError(
        f"dagma.linear was imported from '{_actual}', "
        f"not from the expected source at '{_DAGMA_SRC}'. "
        "A different DAGMA installation may be shadowing the pinned source. "
        "Check sys.path ordering."
    )

#: Absolute path of the resolved ``dagma.linear`` module. Used by the
#: wrapper diagnostics so audits can confirm the pinned source was used.
DAGMA_SOURCE_PATH: Path = _actual
del _actual


__all__ = ["DagmaLinear", "DAGMA_SOURCE_PATH"]
