import pandas as pd
import logging
from pathlib import Path
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:

    logging.info("Loading data from %s", data_dir)
    data_dir = Path(data_dir)
    
    train_files, train_labels = [], []
    test_files, test_labels = [], []
    
    for split in ['train', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                label = class_dir.name
                for img_path in class_dir.glob("*.jpg"):
                    if split == 'train':
                        train_files.append(str(img_path))
                        train_labels.append(label)
                    else:
                        test_files.append(str(img_path))
                        test_labels.append(label)
    
    train_df = pd.DataFrame({'Filepath': train_files, 'Label': train_labels})
    test_df = pd.DataFrame({'Filepath': test_files, 'Label': test_labels})
    
    if train_df.empty:
        raise ValueError("No training data found")
    if test_df.empty:
        raise ValueError("No test data found")
    
    logging.info("Loaded %d training samples and %d test samples", len(train_df), len(test_df))
    return train_df, test_df

def main() -> None:
    """Main function to load data for DVC pipeline."""
    try:
        logging.info("Starting data ingestion")
        train_df, test_df = load_data(data_dir="data/raw")
        
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_dir / "train_data.csv", index=False)
        test_df.to_csv(output_dir / "test_data.csv", index=False)
        logging.info(f"Saved DataFrames to {output_dir}")
        
        logging.info("Data ingestion completed successfully")
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()