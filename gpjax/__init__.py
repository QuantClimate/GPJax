# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from warnings import filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

from gpjax import (
    gps,
    integrators,
    kernels,
    likelihoods,
    mean_functions,
    models,
    natural_gradients,
    objectives,
    parameters,
    plotting,
    state_space,
    typing as typing,
    variational_families,
)
from gpjax.citation import cite
from gpjax.conditioning import Posterior
from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.fit import (
    fit,
    fit_lbfgs,
    fit_natgrads,
    fit_scipy,
)
from gpjax.gps import (
    ConjugateModel,
    HeteroscedasticModel,
    JointModel,
    NonConjugateModel,
    Prior,
    construct_model,
)
from gpjax.summary import summarise

__license__ = "MIT"
__description__ = "Gaussian processes in JAX"
__url__ = "https://github.com/QuantClimate/GPJax"
__contributors__ = "https://github.com/QuantClimate/GPJax/graphs/contributors"
__version__ = "1.0.0"

__all__ = [
    "ConjugateModel",
    "Dataset",
    "GaussianDistribution",
    "HeteroscedasticModel",
    "JointModel",
    "NonConjugateModel",
    "Posterior",
    "Prior",
    "cite",
    "construct_model",
    "fit",
    "fit_lbfgs",
    "fit_natgrads",
    "fit_scipy",
    "gps",
    "integrators",
    "kernels",
    "likelihoods",
    "mean_functions",
    "models",
    "natural_gradients",
    "objectives",
    "parameters",
    "plotting",
    "state_space",
    "summarise",
    "typing",
    "variational_families",
]
