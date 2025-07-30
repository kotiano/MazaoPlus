from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .utils.logging import setup_logger

logger = setup_logger()

SQLALCHEMY_DATABASE_URL = "sqlite:///data/results.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


Base.metadata.create_all(bind=engine)
logger.info("Database initialized")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()