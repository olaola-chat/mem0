from enum import Enum
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field, field_validator


class MetricType(str, Enum):
    euclidean = "euclidean"
    dotproduct = "dotproduct"
    COSINE = "cosine"


class DashVectorConfig(BaseModel):
    """Configuration for DashVector vector store."""

    url: str = Field(..., description="DashVector endpoint URL")
    api_key: str = Field(..., description="DashVector API key")
    collection_name: str = Field("mem0", description="DashVector collection name")
    embedding_model_dims: int = Field(1536, description="Dimensions of embedding model")
    metric_type: MetricType = Field(
        MetricType.COSINE, description="Metric type for similarity search"
    )

    @field_validator("url")
    def url_must_be_valid(cls, v: str) -> str:
        """Validate URL format."""
        # 基本验证，确保URL不为空
        if not v.strip():
            raise ValueError("URL cannot be empty")
        return v

    @field_validator("api_key")
    def api_key_must_be_valid(cls, v: str) -> str:
        """Validate API key format."""
        # 基本验证，确保API key不为空
        if not v.strip():
            raise ValueError("API key cannot be empty")
        return v

    @field_validator("embedding_model_dims")
    def dims_must_be_positive(cls, v: int) -> int:
        """Validate embedding dimensions."""
        if v <= 0:
            raise ValueError("Embedding dimensions must be positive")
        return v 