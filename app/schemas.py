from pydantic import BaseModel, Field

class DiseasePredictionResponse(BaseModel):
    disease: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0) 
    message: str
    recommendation: str | None = None
    image_url: str | None = None
    crop_type: str | None = None  

    class Config:
        from_attributes = True 


class PestPredictionResponse(BaseModel):
    pest: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)  
    message: str
    recommendation: str | None = None
    image_url: str | None = None

    class Config:
        from_attributes = True 

class ErrorResponse(BaseModel):
    error: str
    status_code: int | None = None
    detail: str | None = None