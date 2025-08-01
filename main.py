from fastapi import FastAPI
import uvicorn
from app.utils.config import load_config
from app.database import Base, engine
from app.routes import predict
from app.utils.logger import setup_logger

app = FastAPI(
    title="Mazao-Plus API",
    description="API for pest and disease prediction, optimal planting, and yield forecasting for agricultural support",
    version="1.0.0"
)


app.include_router(predict.router, prefix="/v1")

logger = setup_logger()


try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Mazao-Plus API")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Mazao-Plus API")

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy", "Model Loaded":True}

if __name__ == "__main__":

    config = load_config()
    port = config.get("api_port", 8000)
    uvicorn.run(app, host="0.0.0.0", port=port)

