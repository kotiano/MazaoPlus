from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from ..schemas import PestPredictionResponse, ErrorResponse, DiseasePredictionResponse
from ..repository.pest_prediction import PestPredictor
from ..repository.disease_prediction import DiseasePredictor
from ..models import PestDetectionResult, DiseaseDetectionResult
from .utils.logging import setup_logger
from ..database import get_db
from sqlalchemy.orm import Session
import uuid
from pathlib import Path
from datetime import datetime

router = APIRouter(prefix="/predict", tags=["prediction"])
logger = setup_logger()

def get_pest_predictor():
    return PestPredictor()

def get_disease_predictor():
    return DiseasePredictor()

@router.post("/pest", response_model=PestPredictionResponse)
async def predict_pest(
    file: UploadFile = File(...),
    predictor: PestPredictor = Depends(get_pest_predictor),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Processing pest prediction for file: {file.filename}")
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(exist_ok=True)
        file_id = str(uuid.uuid4())
        file_ext = file.filename.split(".")[-1]
        file_path = upload_dir / f"{file_id}.{file_ext}"
        image_data = await file.read()
        with open(file_path, "wb") as f:
            f.write(image_data)
        

        result = predictor.predict(image_data)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        prediction = PestDetectionResult(

            predicted_class=result["pest"],
            date=datetime.utcnow(),
            confidence=float(result["confidence"].rstrip("%")),
            image_url=str(file_path),
            reccomendation=result['reccomendation']

        )
        db.add(prediction)
        db.commit()
        
        return PestPredictionResponse(**result)
    except Exception as e:
        logger.error(f"Pest prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/disease", response_model=DiseasePredictionResponse)
async def predict_disease(
    file: UploadFile = File(...),
    predictor: DiseasePredictor = Depends(get_disease_predictor),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Processing disease prediction for file: {file.filename}")
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(exist_ok=True)
        file_id = str(uuid.uuid4())
        file_ext = file.filename.split(".")[-1]
        file_path = upload_dir / f"{file_id}.{file_ext}"
        image_data = await file.read()
        with open(file_path, "wb") as f:
            f.write(image_data)
        
        result = predictor.predict(image_data)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        prediction = DiseaseDetectionResult(
            predicted_class=result["disease"],
            date=datetime.utcnow(),
            confidence=float(result["confidence"].rstrip("%")),
            image_url=str(file_path),
            reccomendation=result['reccomendation']
        )
        db.add(prediction)
        db.commit()
        
        return DiseasePredictionResponse(**result)
    except Exception as e:
        logger.error(f"Disease prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")