from pydantic import BaseModel

class DiseasePredictionResponse(BaseModel):
    disease: str | None = None
    confidence: str
    message: str
    recommendation: str
    image_url:str


class PestPredictionResponse(BaseModel):
    pest: str | None = None
    confidence: str
    message: str
    recommendation: str
    image_url:str

class ErrorResponse(BaseModel):
    error: str