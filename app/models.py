from sqlalchemy import Column, Date, Float, Integer, String
from app.database import Base

class PestDetectionResult(Base):
    __tablename__ = "pest_detection_results"

    id = Column(Integer, primary_key=True, index=True)
    predicted_class = Column(String, index=True)
    date = Column(Date)
    confidence = Column(Float)
    image_url = Column(String, nullable=True)
    recommendation = Column(String, nullable=True)

    __table_args__ = {'extend_existing': True} 
    
    def __repr__(self):
        return f"<PestDetectionResult(id={self.id}, predicted_class='{self.predicted_class}', date={self.date})>"

class DiseaseDetectionResult(Base):
    __tablename__ = "disease_detection_results"

    id = Column(Integer, primary_key=True, index=True)
    predicted_class = Column(String, index=True)
    crop_type = Column(String, nullable=True)
    date = Column(Date)
    confidence = Column(Float)
    image_url = Column(String, nullable=True)
    recommendation = Column(String, nullable=True)

    __table_args__ = {'extend_existing': True}  

    def __repr__(self):
        return f"<DiseaseDetectionResult(id={self.id}, predicted_class='{self.predicted_class}', crop_type='{self.crop_type}', date={self.date})>"