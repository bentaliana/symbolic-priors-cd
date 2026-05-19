"""Wrappers for external causal discovery models.

Exports the status taxonomy, WrapperDiagnostics TypedDict, preprocessing
transforms, and the public wrapper classes and configurations.
Importing this package does not trigger the DAGMA source-import shim
nor the DCDI source-import shim; both shims run only when the
corresponding low-level helper modules are imported. ``DCDIConfig`` is
re-exported lazily via module ``__getattr__`` so importing this package
does not load DCDI source either.
"""

from symbolic_priors_cd.wrappers.dagma import DAGMAConfig, DAGMAWrapper
from symbolic_priors_cd.wrappers.dcdi import DCDIWrapper
from symbolic_priors_cd.wrappers.preprocessing import (
    CentredOnlyTransform,
    StandardisedTransform,
)
from symbolic_priors_cd.wrappers.status import (
    GraphStatus,
    SamplerStatus,
    TrainingStatus,
    WrapperDiagnostics,
)


def __getattr__(name: str):
    """Lazy access to ``DCDIConfig`` to keep the package import light.

    ``DCDIConfig`` is defined in ``_dcdi_training``; importing that
    module eagerly triggers the pinned DCDI source-import chain.
    Deferring the import means ``import symbolic_priors_cd.wrappers``
    does not load DCDI source.
    """
    if name == "DCDIConfig":
        from symbolic_priors_cd.wrappers.dcdi import DCDIConfig as _DCDIConfig
        return _DCDIConfig
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "CentredOnlyTransform",
    "DAGMAConfig",
    "DAGMAWrapper",
    "DCDIConfig",
    "DCDIWrapper",
    "GraphStatus",
    "SamplerStatus",
    "StandardisedTransform",
    "TrainingStatus",
    "WrapperDiagnostics",
]
