import json
import logging
import os
import sys

import joblib
import mlflow
import dagshub
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Configuração Inicial e Variáveis de Ambiente
load_dotenv()

dagshub.init(
    repo_owner="Code-AldreySandre",
    repo_name="MLOPS_PROJECT",
    mlflow=True
)

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("src.model_evaluation.evaluate_model")

def load_artifacts() -> tuple[tf.keras.Model, LabelEncoder]:
    """Carrega o modelo treinado e o encoder."""
    try:
        logger.info("Carregando artefatos do modelo...")
        model = tf.keras.models.load_model("models/model.keras")
        encoder = joblib.load("artifacts/[target]_one_hot_encoder.joblib")
        return model, encoder
    except Exception as e:
        logger.error(f"Erro ao carregar artefatos: {e}")
        sys.exit(1)

def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carrega os dados de teste processados."""
    try:
        path = "data/processed/test_processed.csv"
        logger.info(f"Carregando dados de teste de: {path}")
        df = pd.read_csv(path)
        
        X = df.drop("target", axis=1)
        y = df["target"]
        return X, y
    except FileNotFoundError:
        logger.error(f"Arquivo de teste não encontrado em {path}. Verifique se o dvc repro rodou corretamente.")
        sys.exit(1)

def log_to_mlflow(y_true, y_pred, metrics_dict: dict, report_dict: dict):
    """Loga as métricas no MLflow anexando ao último run de treino."""
    
    # Nome do experimento deve ser O MESMO usado no train_model.py
    exp_name = "ml_classification"
    #mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.warning("URI do MLflow não encontrada no .env. Verifique suas configurações!")

    mlflow.set_tracking_uri(tracking_uri)
    
    # 1. Pegar o ID do experimento corretamente
    experiment = mlflow.set_experiment(exp_name)
    experiment_id = experiment.experiment_id

    # 2. Buscar o último run deste experimento para anexar as métricas
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"]
    )

    if runs.empty:
        logger.warning("Nenhum run anterior encontrado. Criando um novo run isolado para avaliação.")
        run_id = None # Vai criar um novo
    else:
        run_id = runs.iloc[0].run_id
        logger.info(f"Anexando métricas ao Run ID: {run_id}")

    # 3. Logar
    with mlflow.start_run(run_id=run_id):
        # Log das métricas resumidas
        mlflow.log_metrics(metrics_dict)
        
        # Opcional: Logar o report completo como um artefato (arquivo texto ou json)
        mlflow.log_dict(report_dict, "evaluation_report.json")
        
        logger.info("Métricas enviadas ao MLflow com sucesso.")

def evaluate():
    """Função principal de avaliação."""
    
    # 1. Carregar recursos
    model, encoder = load_artifacts()
    X_test, y_test = load_test_data()
    
    # 2. Previsão
    logger.info("Realizando inferência no conjunto de teste...")
    y_pred_probs = model.predict(X_test)
    
    # Converter probabilidades para índices (0, 1, 2...)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    

    y_pred_labels = encoder.categories_[0][y_pred_indices]

    # 3. Cálculos de Métricas
    logger.info("Calculando métricas...")
    
    # Classification Report (gera dicionário para JSON e string para Log)
    report_dict = classification_report(y_test, y_pred_labels, output_dict=True)
    report_str = classification_report(y_test, y_pred_labels)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_labels).tolist()
    
    # Métricas pontuais para o MLflow
    metrics_for_mlflow = {
        "test_accuracy": report_dict["accuracy"],
        "test_f1_weighted": report_dict["weighted avg"]["f1-score"],
        "test_precision_weighted": report_dict["weighted avg"]["precision"],
        "test_recall_weighted": report_dict["weighted avg"]["recall"]
    }

    # 4. Salvar outputs Locais (DVC)
    metrics_output = {
        "metrics": metrics_for_mlflow,
        "confusion_matrix": cm,
        "full_report": report_dict
    }
    
    # Salvando apenas o que o DVC espera (metrics/evaluation.json)
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/evaluation.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
        
    logger.info(f"\nRelatório de Classificação:\n{report_str}")
    
    # 5. Enviar para MLflow
    log_to_mlflow(y_test, y_pred_labels, metrics_for_mlflow, report_dict)

if __name__ == "__main__":
    evaluate()