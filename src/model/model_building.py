import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BaseModel(ABC):
    
    @abstractmethod
    def __init__(self, model_path: str, target_size: tuple):
        pass
    
    @abstractmethod
    def train_model(self, cleaned_images: List[np.ndarray]) -> tf.keras.Model:
        pass

class Model(BaseModel):
    
    def __init__(self, model_path: str, target_size: tuple, num_classes: int = 8):

        self.model_path = Path(model_path)
        self.input_shape = target_size
        self.num_classes = num_classes
        self.inputs = tf.keras.layers.Input(shape=self.input_shape)
        self.model = None
        
    def train_model(self, cleaned_images: List[np.ndarray], train_labels: List[np.ndarray], 
                   val_images: List[np.ndarray], val_labels: List[np.ndarray]) -> tf.keras.Model:
 
        logging.info("Building and training the model")
        
        checkpoint_callback = ModelCheckpoint(
            str(self.model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        pretrained_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=self.inputs,
            pooling='max'
        )

        for layer in pretrained_model.layers[-20:]:
            layer.trainable = True

        x = tf.keras.layers.Dense(256)(pretrained_model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=self.inputs, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        self.model.fit(
            x=np.array(cleaned_images),
            y=np.array(train_labels),
            validation_data=(np.array(val_images), np.array(val_labels)),
            epochs=20,
            callbacks=[early_stopping, checkpoint_callback]
        )
        mlflow.log_param('Epochs', 20)
        mlflow.log_param('Learning Rate', 1e-4)
        mlflow.log_param('No. of classes', self.num_classes)
        mlflow.log_param('target size', self.input_shape)
        mlflow.log_metric('Training Loss', None)
        mlflow.log_metric('Validation Loss', None)
        mlflow.log_metric('Training Accuracy', None)
        mlflow.log_metric('Validation Accuracy', None)
        return self.model

def main() -> tf.keras.Model:
    try:
        model_path = "data/models/pest_classifier.h5"
        target_size = (224, 224, 3)
        num_classes = 8

        from src.data.data_ingestion import load_data
        from src.data.data_preprocessing import preprocess_data

        logging.info("Loading and preprocessing data")
        train_images, train_labels, val_images, val_labels = load_data()  
        train_images, val_images = preprocess_data(train_images, val_images)  
        model_instance = Model(model_path=model_path, target_size=target_size, num_classes=num_classes)
        model = model_instance.train_model(
            cleaned_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels
        )

        return model

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()