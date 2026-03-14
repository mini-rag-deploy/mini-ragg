from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional
from bson.objectid import ObjectId

class project(BaseModel):
    id: Optional[ObjectId] = Field(default=None, alias="_id")
    project_id: str = Field(..., min_length=1)

    @field_validator('project_id')
    @classmethod
    def validate_project_id(cls, v):
        if not v.isalnum():
            raise ValueError('project_id must be alphanumeric')
        
        return v

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True
    )