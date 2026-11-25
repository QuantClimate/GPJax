from hypothesis import settings
from jax import config

config.update("jax_enable_x64", True)

settings.register_profile(
    "gpjax-default",
    settings(deadline=None, max_examples=20),
)
settings.load_profile("gpjax-default")
