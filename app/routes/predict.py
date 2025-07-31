from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from ..schemas import PestPredictionResponse, ErrorResponse, DiseasePredictionResponse
from ..repository.pest_prediction import PestPredictor
from ..repository.disease_prediction import DiseasePredictor
from ..models import PestDetectionResult, DiseaseDetectionResult
from ..utils.logger import setup_logger
from ..database import get_db
from sqlalchemy.orm import Session
from pathlib import Path
from datetime import datetime, timezone
import uuid

router = APIRouter(prefix="/predict", tags=["prediction"])
logger = setup_logger()

_pest_predictor = PestPredictor()
_disease_predictor = DiseasePredictor()

def get_pest_predictor():
    return _pest_predictor

def get_disease_predictor():
    return _disease_predictor

@router.post("/pest", response_model=PestPredictionResponse)
async def predict_pest(
    file: UploadFile = File(...),
    predictor: PestPredictor = Depends(get_pest_predictor),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Processing pest prediction for file: {file.filename}")
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=ErrorResponse(error="File must be an image").dict())
        allowed_extensions = {".jpg", ".jpeg", ".png"}
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail=ErrorResponse(error="Invalid file extension").dict())

        upload_dir = Path("data/uploads")
        upload_dir.mkdir(exist_ok=True)
        file_id = str(uuid.uuid4())
        file_ext = file.filename.split(".")[-1]
        file_path = upload_dir / f"{file_id}.{file_ext}"
        image_data = await file.read()
        with open(file_path, "wb") as f:
            f.write(image_data)

        result = predictor.predict(image_data)
        required_keys = {"pest", "confidence", "recommendation"}
        if "error" in result:
            file_path.unlink() 
            raise HTTPException(status_code=400, detail=ErrorResponse(error=result["error"]).dict())
        if not all(key in result for key in required_keys):
            file_path.unlink()
            raise HTTPException(status_code=500, detail=ErrorResponse(error="Invalid predictor output").dict())

        prediction = PestDetectionResult(
            predicted_class=result["pest"],
            date=datetime.now(timezone.utc),
            confidence=float(result["confidence"].rstrip("%")) / 100, 
            image_url=str(file_path),
            recommendation=result["recommendation"]
        )
        db.add(prediction)
        db.commit()
        logger.info(f"Pest prediction successful: {result['pest']}, confidence: {result['confidence']}")

        return PestPredictionResponse(
            pest=result["pest"],
            confidence=str(float(result["confidence"].rstrip("%")) / 100),
            message="Pest prediction successful",
            recommendation=result["recommendation"],
            image_url=str(file_path)
        )
    except Exception as e:
        if file_path.exists():
            file_path.unlink() 
        logger.error(f"Pest prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=ErrorResponse(error=f"Prediction failed: {str(e)}").dict())

@router.post("/disease", response_model=DiseasePredictionResponse)
async def predict_disease(
    file: UploadFile = File(...),
    predictor: DiseasePredictor = Depends(get_disease_predictor),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Processing disease prediction for file: {file.filename}")
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=ErrorResponse(error="File must be an image").dict())
        allowed_extensions = {".jpg", ".jpeg", ".png"}
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail=ErrorResponse(error="Invalid file extension").dict())

        upload_dir = Path("data/uploads")
        upload_dir.mkdir(exist_ok=True)
        file_id = str(uuid.uuid4())
        file_ext = file.filename.split(".")[-1]
        file_path = upload_dir / f"{file_id}.{file_ext}"
        image_data = await file.read()
        with open(file_path, "wb") as f:
            f.write(image_data)

        result = predictor.predict(image_data)
        required_keys = {"disease", "confidence", "recommendation"}
        if "error" in result:
            file_path.unlink()
            raise HTTPException(status_code=400, detail=ErrorResponse(error=result["error"]).dict())
        if not all(key in result for key in required_keys):
            file_path.unlink()
            raise HTTPException(status_code=500, detail=ErrorResponse(error="Invalid predictor output").dict())

        prediction = DiseaseDetectionResult(
            predicted_class=result["disease"],
            crop_type=None,  
            date=datetime.now(timezone.utc),
            confidence=str(float(result["confidence"].rstrip("%")) / 100),  
            image_url=str(file_path),
            recommendation=result["recommendation"]
        )
        db.add(prediction)
        db.commit()
        logger.info(f"Disease prediction successful: {result['disease']}, confidence: {result['confidence']}")

        return DiseasePredictionResponse(
            disease=result["disease"],
            confidence=float(result["confidence"].rstrip("%")) / 100, 
            message="Disease prediction successful",
            recommendation=result["recommendation"],
            image_url=str(file_path)
        )
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        logger.error(f"Disease prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=ErrorResponse(error=f"Prediction failed: {str(e)}").dict())