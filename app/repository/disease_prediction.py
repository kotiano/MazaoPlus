import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict
from PIL import Image
import io
from app.utils.logger import setup_logger
from .reccomendations import disease_reccomendations

logger = setup_logger()

CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
DISEASE_RECOMMENDATIONS=disease_reccomendations()


class DiseasePredictor:
    def __init__(self, model_path: str = "/home/jude/Projects/MazaoPlus/src/models/disease_model.keras"):
        self.model_path = Path(model_path)
        self.model = None

    def load_model(self) -> tf.keras.Model:
        logger.info(f"Loading disease model from {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        return self.model

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        try:
            image = tf.image.resize(image, [224, 224])
            image = image.numpy().astype("float32") / 255.0  
            image = np.expand_dims(image, axis=0)
            return image
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError("Failed to process the image")

    def predict(self, image_file: bytes) -> Dict:
        try:
            if self.model is None:
                self.load_model()
            
            logger.info("Processing disease prediction")
            image = Image.open(io.BytesIO(image_file)).convert("RGB")
            image_array = np.array(image)
            processed_image = self.preprocess_image(image_array)
            
            predictions = self.model.predict(processed_image, verbose=0)[0]
            top_index = np.argmax(predictions)
            top_class = CLASS_NAMES[top_index]
            confidence = round(float(predictions[top_index]) * 100, 2)
            
            result = {
                "disease": top_class,
                "confidence": f"{confidence}%",
                "message": f"I’m {confidence}% sure it’s {top_class}.",
                "recommendation": DISEASE_RECOMMENDATIONS[top_class]
            }
            logger.info(f"Prediction: {top_class} with {confidence}% confidence")
            return result
        except Exception as e:
            logger.error(f"Disease prediction error: {str(e)}")
            return {"error": f"Something went wrong: {str(e)}"}