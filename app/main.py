import io
import logging
import os
import sys

import joblib
import mlflow
import dagshub
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, current_app
from sklearn.datasets import load_breast_cancer # <--- Voltamos com ele!
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# 1. Carregar variáveis de ambiente (Crucial para o MLflow funcionar)
load_dotenv()
#mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app.main")


class ModelService:
    def __init__(self) -> None:
        self.features_imputer = None
        self.features_scaler = None
        self.target_encoder = None
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load the registered model and artifacts from MLflow."""
        
        # Nome do modelo (deve ser o mesmo do register_artifacts.py)
        model_name = "model" 
        stage = "Staging"  # Tenta pegar o que está em Staging

        logger.info(f"Carregando modelo '{model_name}' do MLflow...")
        
        try:
            # 1. Carrega o modelo Keras
            self.model = mlflow.keras.load_model(f"models:/{model_name}/{stage}")
            logger.info("Modelo Keras carregado.")

            # 2. Descobre o Run ID original
            client = MlflowClient()
            latest_versions = client.get_latest_versions(model_name, stages=[stage])
            
            if not latest_versions:
                 # Se não tiver Staging, tenta None (última versão)
                 latest_versions = client.get_latest_versions(model_name, stages=["None"])
            
            if not latest_versions:
                raise Exception("Modelo não encontrado no Registry.")

            run_id = latest_versions[0].run_id
            logger.info(f"Baixando artefatos do Run ID: {run_id}")

            # 3. Baixa os artefatos auxiliares
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="artifacts")
            
            # Ajuste para garantir a pasta correta
            if os.path.basename(local_path) != "artifacts":
                 base_path = os.path.join(local_path, "artifacts")
            else:
                 base_path = local_path

            # --- AQUI ESTÁ A CORREÇÃO DOS NOMES PARA O SEU PROJETO ---
            
            
            self.features_imputer = joblib.load(os.path.join(base_path, "features_mean_imputer.joblib"))
            
            
            self.features_scaler = joblib.load(os.path.join(base_path, "features_scaler.joblib"))
            
            encoder_file = [f for f in os.listdir(base_path) if "[target]_one_hot_encoder.joblib" in f][0]
            self.target_encoder = joblib.load(os.path.join(base_path, encoder_file))
            
            logger.info("Artefatos carregados com sucesso!")

        except Exception as e:
            logger.critical(f"Erro ao carregar artefatos: {e}")
            raise e

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Pipeline de predição."""
        features = features.astype(float) # Garante numérico
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)

        y_pred_probs = self.model.predict(X_scaled)
        y_pred_indices = np.argmax(y_pred_probs, axis=1)

        # Decodifica
        if hasattr(self.target_encoder, 'categories_'):
             y_decoded = self.target_encoder.categories_[0][y_pred_indices]
        else:
             y_decoded = self.target_encoder.inverse_transform(y_pred_indices)

        return pd.DataFrame({"Prediction": y_decoded}, index=features.index)


def create_routes(app: Flask) -> None:
    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        file = request.files.get("file")
        if not file or not file.filename.endswith(".csv"):
            return render_template("index.html", error="Envie um CSV válido.")

        try:
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            
            expected_features = list(load_breast_cancer().feature_names)
            
            missing_cols = [col for col in expected_features if col not in features.columns]
            if missing_cols:
                return render_template("index.html", error=f"Colunas faltando: {', '.join(missing_cols)}")
            
            features = features[expected_features]
            # ------------------------------------------------------

            predictions = current_app.model_service.predict(features)
            
            return render_template(
                "index.html", 
                predictions=predictions.to_html(classes='table table-striped', border=0, justify='center')
            )

        except Exception as e:
            logger.error(f"Erro: {e}", exc_info=True)
            return render_template("index.html", error=f"Erro: {str(e)}")


app = Flask(__name__)

# Inicializa o serviço
try:
    with app.app_context():
        app.model_service = ModelService()
except Exception as e:
    logger.error(f"Erro na inicialização do serviço: {e}")

create_routes(app)

def main() -> None:
    app.run(host="0.0.0.0", port=5001, debug=True)

if __name__ == "__main__":
    main()