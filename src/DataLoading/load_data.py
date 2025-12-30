import logging

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pathlib import Path 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("src.DataLoading.load_data")

def fetch_data() -> pd.DataFrame:
    """"
    Fetch the breast cancer dataset and convert to DataFrame.

    returns: 
    pd.DataFrame: DataFrame containing the breast cncer data with features and target
    """
    logger.info("fetching data...")
    dataset = load_breast_cancer()

    #features columns
    data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

    #Introduce random NaN values
    np.random.seed(42)
    for col in data.columns:
        mask = np.random.random(len(data)) < 0.05 # 5% chance of NaN
        data.loc[mask, col] = np.nan 
    
    # Target column
    data['target'] = dataset.target

    return data

def save_data(data: pd.DataFrame, output_dir: str = "data/raw", filename: str = "raw.csv") -> None:
    """
    save the raw data to disk

    Args:
        data (pd.DataFrame): Raw breast cancer dataset to save
    """

    #create path object
    path = Path(output_dir)

    # Create directories if they dont't exist
    path.mkdir(parents=True, exist_ok= True)

    full_path = path/filename

    logger.info(f"Saving  raw data to {full_path}")
    data.to_csv(full_path,index=False)


def main() -> None:
    """Main function to orchastrate the data loading process."""
    try: 
        raw_data = fetch_data()
        save_data(raw_data)
        logger.info("Data loading  completed sucessfully.")
    except Exception as e:
        logger.error(f"An error ocurred during data loading: {e}")
        raise e 

#
if __name__=="__main__":
    main()