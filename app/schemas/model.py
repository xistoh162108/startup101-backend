"""
Pydantic schemas for Model-related requests and responses.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ModelInfoResponse(BaseModel):
    """
    Response schema for model information.
    GET /api/model/info
    """

    request_id: str = Field(..., alias="requestId", description="Request ID for tracing")
    model_version: str = Field(..., alias="modelVersion", description="Active model version")
    model_type: str = Field(..., alias="modelType", description="Model type (e.g., 'tfidf_sgd')")
    threshold: float = Field(..., description="Classification threshold")
    last_trained_at: Optional[datetime] = Field(
        None, alias="lastTrainedAt", description="When the model was last trained"
    )
    features_used: List[str] = Field(
        ..., alias="featuresUsed", description="Features used by the model"
    )
    status: str = Field(..., description="Model status")

    model_config = ConfigDict(populate_by_name=True)


class TrainModelRequest(BaseModel):
    """
    Request schema for triggering model training.
    POST /api/admin/model/train
    """

    admin_id: str = Field(..., alias="adminId", description="Admin user ID requesting training")
    mode: str = Field("full", description="Training mode: 'full' or 'incremental'")
    min_new_labels: int = Field(
        20,
        alias="minNewLabels",
        ge=1,
        description="Minimum number of new labels required to proceed",
    )
    activate_if_better: bool = Field(
        True, alias="activateIfBetter", description="Whether to activate model if it performs better"
    )

    model_config = ConfigDict(populate_by_name=True)


class TrainModelResponse(BaseModel):
    """
    Response schema for training trigger.
    """

    request_id: str = Field(..., alias="requestId", description="Request ID for tracing")
    accepted: bool = Field(..., description="Whether training request was accepted")
    job_id: Optional[str] = Field(None, alias="jobId", description="Training job ID (if async)")
    message: Optional[str] = Field(None, description="Additional message or error details")

    model_config = ConfigDict(populate_by_name=True)
