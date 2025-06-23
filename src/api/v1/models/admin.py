from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, List, Literal


class DeleteDatasetParams(BaseModel):
    dataset_name: str
