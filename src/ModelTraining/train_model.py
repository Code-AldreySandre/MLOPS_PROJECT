import json
import logging
import os
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from dotenv import load_dotenv
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# 1. Configuração Inicial e Variáveis de Ambiente
load_dotenv()

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("src.ModelTraining.train_model")


def load_data() -> pd.DataFrame:
    """Load the feature-engineered training data."""
    path = "data/processed/train_processed.csv"
    try:
        logger.info(f"Carregando dados de treino de: {path}")
        train_data = pd.read_csv(path)
        return train_data
    except FileNotFoundError:
        logger.error(f"Arquivo {path} não encontrado. Verifique se o pipeline anterior rodou.")
        sys.exit(1)


def load_params() -> dict[str, float | int]:
    """Load model hyperparameters for the train stage from params.yaml."""
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        return params["train"]
    except Exception as e:
        logger.error(f"Erro ao ler params.yaml: {e}")
        sys.exit(1)


def prepare_data(train_data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, OneHotEncoder]:
    """Prepare data for neural network training."""
    # Separate features and target
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    # One-hot encode the target variable
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))

    return X_train, y_train_encoded, encoder


def create_model(
    input_shape: int, num_classes: int, params: dict[str, int | float]
) -> tf.keras.Model:
    """Create a Keras Dense Neural Network model."""
    model = Sequential(
        [
            Dense(
                params["hidden_layer_1_neurons"],
                activation="relu",
                input_shape=(input_shape,)
            ),
            Dropout(params["dropout_rate"]),
            Dense(
                params["hidden_layer_2_neurons"],
                activation="relu",
            ),
            Dropout(params["dropout_rate"]),
            Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = Adam(learning_rate=params["learning_rate"])

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def save_training_artifacts(model: tf.keras.Model, encoder: OneHotEncoder) -> None:
    """Save model artifacts to disk."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    model_path = os.path.join("models", "model.keras")
    # Mantendo o padrão [target] solicitado
    encoder_path = os.path.join("artifacts", "[target]_one_hot_encoder.joblib")

    logger.info(f"Salvando modelo em {model_path}")
    model.save(model_path)

    logger.info(f"Salvando encoder em {encoder_path}")
    joblib.dump(encoder, encoder_path)


def setup_mlflow_run(experiment_name: str):
    """Configura o experimento e define a hierarquia de Runs (DVC vs Standalone)."""
    
    # Define URI (garante http se não estiver no env)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    
    # Configura Experimento e obtém ID
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
    
    # Ativa Autolog do Keras
    mlflow.keras.autolog()

    # Lógica para DVC (Detecta se estamos rodando via 'dvc exp run')
    dvc_exp_name = os.getenv("DVC_EXP_NAME")
    extra_args = {}
    
    if dvc_exp_name:
        logger.info(f"Execução DVC detectada: {dvc_exp_name}")
        # Busca se já existe um run "pai" para este experimento DVC
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.dvc_exp = 'True'",
            order_by=["start_time DESC"],
        )
        
        if runs.empty:
            # Se não existe pai, cria um run pai e marca com a tag
            logger.info("Criando Run Pai para o experimento DVC...")
            parent_run = mlflow.start_run(experiment_id=experiment_id, run_name="DVC_Parent_Run")
            mlflow.set_tag("dvc_exp", True)
            parent_run_id = parent_run.info.run_id
            mlflow.end_run() # Fecha o pai, vamos usar só o ID
        else:
            parent_run_id = runs.iloc[0].run_id
            
        # Define que o run atual será filho (nested) desse pai
        extra_args = {
            "parent_run_id": parent_run_id,
            "run_name": dvc_exp_name,
            "nested": True,
            "experiment_id": experiment_id
        }
    else:
        # Execução normal (python train_model.py manual)
        extra_args = {"experiment_id": experiment_id}

    return mlflow.start_run(**extra_args)


def train_model(train_data: pd.DataFrame, params: dict[str, int | float]) -> None:
    """Train a Keras model, logging metrics and artifacts with MLflow."""
    
    # Inicia o contexto do MLflow com a lógica correta
    with setup_mlflow_run("ml_classification"):
        
        # Log de Hiperparâmetros explícito (embora o autolog pegue alguns, é bom garantir)
        mlflow.log_params(params)
        
        # Configura semente aleatória
        seed = params.pop("random_seed", 42)
        tf.keras.utils.set_random_seed(seed)
        
        # Prepara dados
        X_train, y_train, encoder = prepare_data(train_data)
        
        # Cria modelo
        model = create_model(
            input_shape=X_train.shape[1], 
            num_classes=y_train.shape[1], 
            params=params
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        logger.info("Iniciando treinamento...")
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[early_stopping],
            verbose=1 # Necessário para ver o progresso no log do DVC
        )

        # Salva artefatos locais
        save_training_artifacts(model, encoder)
        
        # Loga artefatos adicionais no MLflow (input artifacts)
        # Verificamos se existem antes de tentar logar
        if os.path.exists("artifacts/features_scaler.joblib"):
            mlflow.log_artifact("artifacts/features_scaler.joblib")
        if os.path.exists("artifacts/[target]_one_hot_encoder.joblib"):
            mlflow.log_artifact("artifacts/[target]_one_hot_encoder.joblib")

        # Salva métricas para o DVC (json local)
        metrics = {
            metric: float(history.history[metric][-1]) 
            for metric in history.history
        }
        
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/training.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info("Treinamento finalizado com sucesso.")


def main() -> None:
    """Main function to orchestrate the model training process."""
    train_data = load_data()
    params = load_params()
    train_model(train_data, params)


if __name__ == "__main__":
    main()