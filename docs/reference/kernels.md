# Kernels

```{eval-rst}
.. currentmodule:: gpjax.kernels

.. autosummary::
   :toctree: generated/
   :nosignatures:

   RBF
   RFF
   AbstractKernel
   ArcCosine
   BasisFunctionComputation
   Constant
   ConstantDiagonalKernelComputation
   DenseKernelComputation
   DiagonalKernelComputation
   EigenKernelComputation
   Gibbs
   Gneiting
   GraphKernel
   ICMKernel
   LCMKernel
   Linear
   Matern12
   Matern32
   Matern52
   MultiOutputKernel
   MultiOutputKernelComputation
   OrthogonalAdditiveKernel
   Periodic
   Polynomial
   PoweredExponential
   ProductKernel
   RationalQuadratic
   SumKernel
   VaryingAmplitude
   White
```

## Location functions

A location function gives a kernel parameter that changes with input location,
such as the standard deviation of {class}`~gpjax.kernels.VaryingAmplitude` or
the lengthscale of {class}`~gpjax.kernels.Gibbs`. It is not a mean function: a
mean function describes the Gaussian process, but a location function describes
its covariance. A location function evaluates one input point, selects its own
columns with `active_dims`, and returns a value on the log scale.

```{eval-rst}
.. currentmodule:: gpjax.kernels.location_functions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AbstractLocationFunction
   Constant
   Linear
```
