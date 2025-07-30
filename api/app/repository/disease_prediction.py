import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict
from PIL import Image
import io
from api.app.utils.logging import setup_logger

logger = setup_logger()

CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

CORN_DISEASE_RECOMMENDATIONS = {
    "Blight": (
        "Northern Corn Leaf Blight (NCLB), caused by Exserohilum turcicum, forms 1-6 inch cigar-shaped, gray-green to tan lesions on leaves, starting lower.\n"
        "Control methods:\n"
        "- Choose hybrids with partial or race-specific resistance (Ht1, Ht2, or HtN genes).\n"
        "- Scout fields weekly before silking; look for lesions on lower leaves.\n"
        "- Rotate with non-host crops (e.g., wheat) for 1-2 years.\n"
        "- Bury crop debris by plowing.\n"
        "- Plant early to avoid peak humidity periods.\n"
        "- Use fungicides (e.g., Delaro® Complete) if lesions reach the third leaf below the ear on 50% of plants at tasseling."
    ),
    "Common_Rust": (
        "Common Rust, caused by Puccinia sorghi, appears as small, oval, dark-reddish-brown pustules on both leaf surfaces.\n"
        "Control methods:\n"
        "- Plant resistant corn hybrids.\n"
        "- Scout fields weekly from V10-V14; remove affected leaves if limited.\n"
        "- Rotate crops yearly with non-hosts (e.g., soybeans).\n"
        "- Plow crop residues.\n"
        "- Avoid humid areas; ensure good air circulation.\n"
        "- Apply foliar fungicides (e.g., azoxystrobin) if pustules cover 50% of leaves before tasseling."
    ),
    "Gray_Leaf_Spot": (
        "Gray Leaf Spot, caused by Cercospora zeae-maydis, appears as rectangular, grayish-tan lesions with a gray-white center, often along veins.\n"
        "Control methods:\n"
        "- Plant resistant hybrids.\n"
        "- Scout fields weekly during warm, humid conditions (75-85°F).\n"
        "- Rotate with non-host crops for 1-2 years.\n"
        "- Bury crop debris to reduce fungal spores.\n"
        "- Apply fungicides (e.g., strobilurins) at early disease onset, typically at tasseling."
    ),
    "Healthy": (
        "No disease detected. Continue regular monitoring.\n"
        "Recommendations:\n"
        "- Scout fields weekly for early signs of disease.\n"
        "- Maintain crop rotation and soil health.\n"
        "- Ensure proper irrigation."
    )
}

class DiseasePredictor:
    def __init__(self, model_path: str = "data/models/disease_model.keras"):
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
                "recommendation": CORN_DISEASE_RECOMMENDATIONS[top_class]
            }
            logger.info(f"Prediction: {top_class} with {confidence}% confidence")
            return result
        except Exception as e:
            logger.error(f"Disease prediction error: {str(e)}")
            return {"error": f"Something went wrong: {str(e)}"}