import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Tuple, List
from pathlib import Path
import mlflow 

from data_ingestion import load_data


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Strategy(ABC):
    
    @abstractmethod
    def handle_data(self, data: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
       
        pass

class DataPreprocessing(Strategy):
    
    def __init__(self, target_size: tuple = (224, 224), batch_size: int = 32):

        self.target_size = target_size
        self.batch_size = batch_size
    
    def handle_data(self, data: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
     
        logging.info("Preprocessing data with ImageDataGenerator")
        
        train_df, test_df = data
        
        for df, name in [(train_df, 'train_df'), (test_df, 'test_df')]:
            if not {'Filepath', 'Label'}.issubset(df.columns):
                raise ValueError(f"{name} must contain 'Filepath' and 'Label' columns")

        train_generator = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            horizontal_flip=True,
            rotation_range=10,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1],
            validation_split=0.2
        )
        
        test_generator = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
        )

        train_flow = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col='Filepath',
            y_col='Label',
            target_size=self.target_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            subset='training'
        )

        val_flow = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col='Filepath',
            y_col='Label',
            target_size=self.target_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            subset='validation'
        )

        test_flow = test_generator.flow_from_dataframe(
            dataframe=test_df,
            x_col='Filepath',
            y_col='Label',
            target_size=self.target_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False
        )

        logging.info("Extracting images and labels from generators")
        train_images, train_labels = self._extract_data(train_flow)
        val_images, val_labels = self._extract_data(val_flow)
        test_images, test_labels = self._extract_data(test_flow)

        self._save_preprocessed_data(train_images, train_labels, val_images, val_labels, test_images, test_labels)

        logging.info(f"Preprocessed {len(train_images)} training, {len(val_images)} validation, and {len(test_images)} test samples")
        return train_images, train_labels, val_images, val_labels

    def _extract_data(self, generator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
        images, labels = [], []
        for batch_images, batch_labels in generator:
            images.extend(batch_images)
            labels.extend(batch_labels)
            if len(images) >= generator.n:  
                break
        return images, labels

    def _save_preprocessed_data(self, train_images: List[np.ndarray], train_labels: List[np.ndarray], 
                              val_images: List[np.ndarray], val_labels: List[np.ndarray], 
                              test_images: List[np.ndarray], test_labels: List[np.ndarray]) -> None:
        
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / "train_images.npy", np.array(train_images))
        np.save(output_dir / "train_labels.npy", np.array(train_labels))
        np.save(output_dir / "val_images.npy", np.array(val_images))
        np.save(output_dir / "val_labels.npy", np.array(val_labels))
        np.save(output_dir / "test_images.npy", np.array(test_images))
        np.save(output_dir / "test_labels.npy", np.array(test_labels))
        logging.info(f"Saved preprocessed data to {output_dir}")

        mlflow.log_param('Target Size',self.target_size )
        mlflow.log_param('Batch Size', self.batch_size)


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
   
    try:
        logging.info("Starting data preprocessing")
        processor = DataPreprocessing(target_size=(224, 224), batch_size=32)
        preprocessed_data = processor.handle_data(data=(train_df, test_df))
        return preprocessed_data
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

def main() -> None:
    try:
        logging.info("Starting preprocessing pipeline")
        
        logging.info("Loading data")
        train_df, test_df = load_data(data_dir="data/raw")

        logging.info("Preprocessing data")
        train_images, train_labels, val_images, val_labels = preprocess(train_df, test_df)

        logging.info("Preprocessing completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()