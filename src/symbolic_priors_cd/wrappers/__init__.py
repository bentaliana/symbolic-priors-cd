"""Wrappers for external causal discovery models.

Exports the status taxonomy, WrapperDiagnostics TypedDict, preprocessing
transforms, and the public wrapper classes and configurations.
Importing this package does not trigger the DAGMA source-import shim;
that shim runs only when ``_dagma_utils`` is imported.
"""

from symbolic_priors_cd.wrappers.dagma import DAGMAConfig, DAGMAWrapper
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

__all__ = [
    "CentredOnlyTransform",
    "DAGMAConfig",
    "DAGMAWrapper",
    "GraphStatus",
    "SamplerStatus",
    "StandardisedTransform",
    "TrainingStatus",
    "WrapperDiagnostics",
]
