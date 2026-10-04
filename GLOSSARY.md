# GPJax

GPJax is a Gaussian process library built on JAX. This glossary gives the
canonical terms for concepts that are specific to GPJax.

## Nonstationary kernels

**Location function**:
A kernel parameter that has a value at each input location, for example a
standard deviation or a lengthscale that changes over space. It is not a mean
function: a mean function describes the process, but a location function
describes the covariance.
_Avoid_: mean function, warping function, parameter field

**Base kernel**:
The stationary kernel that a nonstationary kernel modifies with a location
function.
_Avoid_: inner kernel, wrapped kernel

**Varying amplitude kernel**:
A base kernel whose standard deviation changes with location, so that
variability is larger in some regions than in others.
_Avoid_: heteroscedastic kernel (GPJax uses "heteroscedastic" for noise),
scaled kernel

**Gibbs kernel**:
A base kernel whose lengthscale changes with location, so that correlation
decays faster in some regions than in others. The literature also calls it the
Paciorek–Schervish kernel.
_Avoid_: nonstationary Matérn, Paciorek–Schervish kernel (as a name)

## Space–time kernels

**Gneiting kernel**:
A nonseparable space–time kernel. The spatial correlation changes with the
time lag, so that it decays more slowly at longer lags.
_Avoid_: space–time Matérn, separable space–time kernel

**Space columns**:
The input columns over which a space–time kernel measures spatial distance.

**Time columns**:
The input columns over which a space–time kernel measures the time lag.
