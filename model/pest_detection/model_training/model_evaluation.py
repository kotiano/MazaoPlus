import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import logging
from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelEvaluationStrategy(ABC):
    
    @abstractmethod
    def evaluate_model(self, model: tf.keras.Model, test_images: List[np.ndarray], 
                      test_labels: List[np.ndarray], class_names: List[str]) -> Dict:

        pass

class ModelEvaluation(ModelEvaluationStrategy):
    
    def __init__(self, output_dir: str = "data/evaluation"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(self, model: tf.keras.Model, test_images: List[np.ndarray], 
                      test_labels: List[np.ndarray], class_names: List[str]) -> Dict:
        

        logging.info("Evaluating model on test set")
        
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
        metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
        logging.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
        
        pred_probs = model.predict(test_images, verbose=0)
        pred_labels = np.argmax(pred_probs, axis=1)
        true_labels = np.argmax(test_labels, axis=1)
        
        report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
        with open(self.output_dir / "classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        logging.info(f"Saved classification report to {self.output_dir / 'classification_report.json'}")
        
        self.make_confusion_matrix(
            y_true=true_labels,
            y_pred=pred_labels,
            classes=class_names,
            savefig=True,
            filepath=self.output_dir / "confusion_matrix.png"
        )
        
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Saved metrics to {self.output_dir / 'metrics.json'}")
        
        return metrics

    def make_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, classes: List[str] = None, 
                             figsize: tuple = (15, 7), text_size: int = 10, norm: bool = False, 
                             savefig: bool = False, filepath: Path = None) -> None:

        cm = confusion_matrix(y_true, y_pred)
        if norm:
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        else:
            cm_norm = cm
        
        n_classes = cm.shape[0]
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        labels = classes if classes else np.arange(n_classes)
        ax.set(
            title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=labels,
            yticklabels=labels
        )

        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        plt.xticks(rotation=90, fontsize=text_size)
        plt.yticks(fontsize=text_size)

        threshold = (cm.max() + cm.min()) / 2.
        for i, j in itertools.product(range(n_classes), range(n_classes)):
            text = f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)" if norm else f"{cm[i, j]}"
            plt.text(
                j, i, text,
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size
            )

        if savefig and filepath:
            fig.savefig(filepath, bbox_inches='tight')
            logging.info(f"Saved confusion matrix to {filepath}")
        plt.close(fig)

def main() -> None:
    try:
        logging.info("Starting model evaluation")
        
       
        data_dir = Path("data/processed")
        test_images = np.load(data_dir / "test_images.npy")
        test_labels = np.load(data_dir / "test_labels.npy")
        test_images = test_images.tolist()  
        test_labels = test_labels.tolist()
        
 
        class_names = [f"class_{i}" for i in range(8)] 
        
        model_path = Path("data/models/pest_classifier.h5")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = tf.keras.models.load_model(model_path)
        logging.info(f"Loaded model from {model_path}")
        
        evaluator = ModelEvaluation(output_dir="data/evaluation")
        metrics = evaluator.evaluate_model(
            model=model,
            test_images=test_images,
            test_labels=test_labels,
            class_names=class_names
        )
        
        logging.info("Model evaluation completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()