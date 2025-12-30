import logging
import yaml
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pathlib import Path

# Configuração básica do log para aparecer no terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("src.DataPreprocessing.preprocess_data")

def load_data() -> pd.DataFrame:
    """Load the raw data from disk."""
    # Uso do Path para garantir compatibilidade
    input_path = Path("data/raw/raw.csv")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado em: {input_path}. Rode o script de load_data primeiro.")

    logger.info(f"Loading raw data from {input_path}")
    data = pd.read_csv(input_path)
    return data

def load_params() -> dict:
    """Load preprocessing parameters from params.yaml."""
    params_path = Path("params.yaml")
    
    if not params_path.exists():
        raise FileNotFoundError("O arquivo params.yaml não foi encontrado.")

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    
    return params["preprocess_data"]

def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets using parameters from params.yaml."""
    params = load_params()
    logger.info("Splitting data into train and test sets...")
    
    train_data, test_data = train_test_split(
        data, 
        test_size=params["test_size"], 
        random_state=params["random_seed"]
    )
    return train_data, test_data

def preprocess_data(
        train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]: 
    """Perform preprocessing steps on train and test sets."""

    logger.info("Preprocessing data...")

    # Separate target column
    train_target = train_data['target']
    test_target = test_data['target']
    train_features = train_data.drop('target', axis=1)
    test_features = test_data.drop('target', axis=1)

    # Apply imputation 
    imputer = SimpleImputer(strategy="mean")
    
    # O fit é feito APENAS no treino para evitar Data Leakage
    train_features_processed = pd.DataFrame(
        imputer.fit_transform(train_features), columns=train_features.columns
    )
    # O teste é apenas transformado
    test_features_processed = pd.DataFrame(
        imputer.transform(test_features), columns=test_features.columns
    )

    # Merge target back. Usamos tolist() para ignorar desalinhamento de índices causado pelo split
    train_processed = train_features_processed.assign(target=train_target.tolist())
    test_processed = test_features_processed.assign(target=test_target.tolist())

    return train_processed, test_processed, imputer

def save_artifacts(
        train_data: pd.DataFrame, test_data: pd.DataFrame, imputer: SimpleImputer
) -> None:
    
    """Save processed data and preprocessing artifacts."""
    
    # Definindo caminhos com Pathlib
    data_dir = Path("data/preprocessed")
    artifacts_dir = Path("artifacts")
    
    # Criar pastas se não existirem (essencial!)
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed data to {data_dir}")
    train_data.to_csv(data_dir / "train_preprocessed.csv", index=False)
    test_data.to_csv(data_dir / "test_preprocessed.csv", index=False) # Corrigido typo no nome

    # Save imputer
    imputer_path = artifacts_dir / "features_mean_imputer.joblib"
    logger.info(f"Saving imputer to {imputer_path}")
    joblib.dump(imputer, imputer_path)

def main() -> None:
    """Main function to orchestrate the preprocessing pipeline."""
    try:
        raw_data = load_data()
        train_data, test_data = split_data(raw_data)
        train_processed, test_processed, imputer = preprocess_data(train_data, test_data)
        save_artifacts(train_processed, test_processed, imputer)
        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Failed to run preprocessing: {e}")
        raise e

if __name__ == "__main__":
    main()