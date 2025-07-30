import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict
from PIL import Image
import io
from api.app.utils.logging import setup_logger

logger = setup_logger()

CLASS_NAMES = ["aphids", "bollworm", "fall armyworm", "stem borer", "weevil"]

PEST_RECOMMENDATIONS = {
    "aphids": (
        "Aphids are tiny sap-sucking bugs that weaken plants and spread viruses.\n"
        "Control methods:\n"
        "- Release ladybugs or lacewings (available from garden stores) to eat aphids naturally.\n"
        "- Spray leaves with soapy water (1 tsp dish soap per liter water) or neem oil weekly.\n"
        "- Blast aphids off plants with a strong hose spray; repeat every 2–3 days.\n"
        "- Rotate crops yearly to prevent aphid buildup.\n"
        "- Plant garlic or marigolds nearby to repel aphids.\n"
        "- Inspect undersides of leaves weekly; squash any aphids found."
    ),
    "bollworm": (
        "Bollworms are worms that chew into buds and fruits, especially cotton and maize.\n"
        "Control methods:\n"
        "- Set up pheromone traps (from farm suppliers) to catch moths before egg-laying.\n"
        "- Use Bt crops (e.g., Bt maize) to kill worms naturally when they feed.\n"
        "- Check plants weekly; remove white egg clusters or worms by hand (wear gloves).\n"
        "- Spray spinosad (natural pesticide) if damage is heavy; follow label instructions.\n"
        "- Plant crops early to avoid peak bollworm season (warm months).\n"
        "- Burn or bury crop residues after harvest to kill hiding worms."
    ),
    "fall armyworm": (
        "Fall armyworms strip leaves and devastate maize, rice, or sorghum.\n"
        "Control methods:\n"
        "- Inspect fields weekly in warm, wet weather; look for young worms or leaf damage.\n"
        "- Release Trichogramma wasps (from suppliers) to destroy armyworm eggs.\n"
        "- Plant Napier grass around fields as a trap crop to lure worms away.\n"
        "- Grow desmodium nearby to repel armyworms with its scent.\n"
        "- Apply Bt spray or Beauveria bassiana (fungal biopesticide) on young worms.\n"
        "- Mow and burn crop debris after harvest to eliminate hiding worms."
    ),
    "stem borer": (
        "Stem borers are larvae that tunnel into stems, weakening or killing plants.\n"
        "Control methods:\n"
        "- Burn or bury crop residues (e.g., maize stalks) after harvest to kill larvae.\n"
        "- Release Cotesia wasps (from suppliers) to attack stem borer larvae.\n"
        "- Plant sorghum as a trap crop near main crops to divert borers.\n"
        "- Spray spinosad when plants are young to protect stems.\n"
        "- Choose borer-resistant crop varieties (check with seed suppliers).\n"
        "- Check stems for holes or frass (sawdust-like waste) weekly; remove larvae."
    ),
    "weevil": (
        "Weevils are beetles that damage grains, fruits, or roots, especially in storage.\n"
        "Control methods:\n"
        "- Remove and burn infested plants or grains to stop weevil spread.\n"
        "- Use sticky traps (from farm stores) to catch adult weevils.\n"
        "- Heat grains to 50°C for 2–3 hours (use solar dryer or oven) to kill eggs.\n"
        "- Mix neem leaves or diatomaceous earth with stored grains to deter weevils.\n"
        "- Store grains in airtight containers to block new weevils.\n"
        "- Check stored grains monthly; discard any with small holes or weevil signs."
    )
}

class PestPredictor:
    def __init__(self, model_path: str = "data/models/pest_classifier.h5"):
        self.model_path = Path(model_path)
        self.model = None
        self.preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

    def load_model(self) -> tf.keras.Model:
        logger.info(f"Loading pest model from {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        return self.model

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        try:
            image = tf.image.resize(image, [224, 224])
            image = image.numpy().astype("float32")
            image = np.expand_dims(image, axis=0)
            image = self.preprocess_input(image)
            return image
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError("Failed to process the image")

    def predict(self, image_file: bytes) -> Dict:
        try:
            if self.model is None:
                self.load_model()
            
            logger.info("Processing pest prediction")
            image = Image.open(io.BytesIO(image_file)).convert("RGB")
            image_array = np.array(image)
            processed_image = self.preprocess_image(image_array)
            
            predictions = self.model.predict(processed_image, verbose=0)[0]
            top_index = np.argmax(predictions)
            top_class = CLASS_NAMES[top_index]
            confidence = round(float(predictions[top_index]) * 100, 2)
            
            result = {
                "pest": top_class,
                "confidence": f"{confidence}%",
                "message": f"I’m {confidence}% sure it’s {top_class}.",
                "recommendation": PEST_RECOMMENDATIONS[top_class]
            }
            logger.info(f"Prediction: {top_class} with {confidence}% confidence")
            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": f"Something went wrong: {str(e)}"}