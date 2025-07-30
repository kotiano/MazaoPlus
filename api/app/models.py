from sqlalchemy import Column, Date, Float, Integer, String
from database import Base

class PestDetectionResult(Base):
    __tablename__ = "pest_detection_results"

    id = Column(Integer, primary_key=True, index=True)
    predicted_class  = Column(String, index=True)
    date = Column(Date)
    confidence = Column(Float)
    image_url = Column(String, nullable=True)
    reccomendation = Column(String, nullable=True)

    __tablename__ = "diease_results"

class DiseaseDetectionResult(Base):

    __tablename__= 'disease_detection_results'
    
    id = Column(Integer, primary_key=True, index=True)
    predicted_class  = Column(String, index=True)
    crop_type = Column(String, nullable=True)
    date = Column(Date)
    confidence = Column(Float)
    image_url = Column(String, nullable=True)
    reccomendation = Column(String, nullable=True)