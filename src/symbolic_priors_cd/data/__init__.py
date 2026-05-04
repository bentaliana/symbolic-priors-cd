"""Public data layer for the symbolic-priors-cd evaluator.

Exports the SCM data contract, generation functions, sampling functions,
and the intervention API. The private helper ``_ancestral_sample`` is
intentionally not re-exported.
"""

from symbolic_priors_cd.data.interventions import (
    Intervention,
    InterventionalSampler,
    intervene,
)
from symbolic_priors_cd.data.scm_generator import (
    GenerationSpec,
    LinearGaussianSCM,
    generate_er_dag,
    generate_linear_gaussian_scm,
    sample_edge_weights,
    sample_observational,
)

__all__ = [
    "GenerationSpec",
    "Intervention",
    "InterventionalSampler",
    "LinearGaussianSCM",
    "generate_er_dag",
    "generate_linear_gaussian_scm",
    "intervene",
    "sample_edge_weights",
    "sample_observational",
]
