from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from skgp._types import WholeNumber


class Config(BaseModel):
    """skgp's global settings. There is one object, `skgp.config`; assign to its
    attributes to change a setting for the rest of the session. A value that is not valid
    for its setting is refused when you assign it, and so is a name that is not a setting."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    size_warning_rows: Annotated[WholeNumber, Field(ge=0)] = 10_000
    """How many training rows `GPRegressor.fit` and `GPClassifier.fit` accept without a
    warning. It controls only whether the warning is issued, never which model is fitted or
    what it predicts."""


config = Config()
