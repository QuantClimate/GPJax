from hypothesis import settings
from jax import config
from jaxtyping import install_import_hook

config.update("jax_enable_x64", True)

# import gpjax within import hook to apply beartype everywhere, before running tests
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax  # noqa: F401

settings.register_profile(
    "gpjax-default",
    settings(deadline=None, max_examples=20),
)
settings.load_profile("gpjax-default")
