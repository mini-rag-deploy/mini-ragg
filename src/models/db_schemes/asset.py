from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime

class Asset(BaseModel):
    id: Optional[ObjectId] = Field(default=None, alias="_id")
    asset_project_id: ObjectId
    asset_type: str = Field(..., min_length=1)
    asset_name: str = Field(..., min_length=1)
    asset_size: int = Field(ge=0,default=None)
    asset_config: dict = Field(default=None)
    asset_pushed_at: datetime = Field(default=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
        # populate_by_name = True

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [("asset_project_id", 1)], # 1 for ascending order and -1 for descending order
                "name": "asset_project_id_index_1",
                "unique": False
            },
            {
                "key": [("asset_project_id", 1), ("asset_name", 1)], # 1 for ascending order and -1 for descending order
                "name": "asset_project_id_name_index_1",
                "unique": True
            }
        ]