from pathlib import Path
import yaml
from dotenv import load_dotenv
import os
from .logger import setup_logger

logger = setup_logger()

def load_config() -> dict:
    """
    Load configuration from params.yaml and .env files.
    Returns a dictionary with merged settings.
    """
    config = {}
    
    params_path = Path("params.yaml")
    try:
        if params_path.exists():
            with open(params_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
                config.update(yaml_config)
            logger.info(f"Loaded configuration from {params_path}")
        else:
            logger.warning(f"{params_path} not found, skipping YAML config")
    except Exception as e:
        logger.error(f"Error loading {params_path}: {str(e)}")
        raise
    
    try:
        load_dotenv()
        config["api_port"] = int(os.getenv("API_PORT", 8000))
        config["pest_model_path"] = os.getenv("PEST_MODEL_PATH", "data/models/pest_classifier.h5")
        config["disease_model_path"] = os.getenv("DISEASE_MODEL_PATH", "data/models/disease_model.keras")
        config["database_url"] = os.getenv("DATABASE_URL", "sqlite:///data/results.db")
        logger.info("Loaded environment variables from .env")
    except Exception as e:
        logger.error(f"Error loading .env variables: {str(e)}")
        raise
    
    return config


