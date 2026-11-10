# xarray

Convert labelled xarray data to a {class}`~gpjax.dataset.Dataset` and map
predictions back onto the grid. Requires the optional extra:
`pip install "gpjax[xarray]"`.

```{eval-rst}
.. currentmodule:: gpjax.xarray

.. autosummary::
   :toctree: generated/
   :nosignatures:

   from_xarray
   GridSpec
```

## Input transforms

Transforms turn the named inputs into the columns of `X`. Pass them to
{func}`~gpjax.xarray.from_xarray`; the {class}`~gpjax.xarray.GridSpec` applies the
same fitted transforms to every new grid.

```{eval-rst}
.. currentmodule:: gpjax.xarray

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Standardise
   UnitSphere
   Cyclic
   InputTransform
```
