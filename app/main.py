import io
import logging
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, current_app # <--- Importar current_app
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("app.main")


class ModelService:
    def __init__(self) -> None:
        self.features_imputer = None
        self.features_scaler = None
        self.target_encoder = None
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        logger.info("Loading artifacts from local project folder")

        # Verifica se estamos na pasta raiz ou dentro de app (ajuste de caminho relativo)
        if os.path.exists("artifacts"):
            base_dir = ""
        elif os.path.exists("../artifacts"):
            base_dir = "../"
        else:
            raise FileNotFoundError("Could not find 'artifacts' folder. Are you running from project root?")

        artifacts_dir = os.path.join(base_dir, "artifacts")
        models_dir = os.path.join(base_dir, "models")

        # Caminhos com os nomes EXATOS que você confirmou
        features_imputer_path = os.path.join(artifacts_dir, "features_mean_imputer.joblib")
        features_scaler_path = os.path.join(artifacts_dir, "features_scaler.joblib")
        target_encoder_path = os.path.join(artifacts_dir, "[target]_one_hot_encoder.joblib") # Com colchetes
        model_path = os.path.join(models_dir, "model.keras")

        # Carrega tudo (se falhar aqui, o app nem inicia)
        self.features_imputer = joblib.load(features_imputer_path)
        self.features_scaler = joblib.load(features_scaler_path)
        self.target_encoder = joblib.load(target_encoder_path)
        self.model = load_model(model_path)
        logger.info("Successfully loaded all artifacts")

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)
        y_pred_probs = self.model.predict(X_scaled)
        y_pred_indices = np.argmax(y_pred_probs, axis=1)
        y_decoded = self.target_encoder.categories_[0][y_pred_indices]
        return pd.DataFrame({"Prediction": y_decoded}, index=features.index)


def create_routes(app: Flask) -> None:
    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        file = request.files["file"]
        if not file or not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a valid CSV file")

        try:
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            # Usa current_app para acessar o serviço de forma segura
            model_service = current_app.model_service

            if hasattr(model_service.features_scaler, "feature_names_in_"):
                expected_features = list(model_service.features_scaler.feature_names_in_)
            else:
                expected_features = load_breast_cancer().feature_names

            missing_cols = [col for col in expected_features if col not in features.columns]
            
            if missing_cols:
                return render_template(
                    "index.html",
                    error=f"Missing required columns: {', '.join(missing_cols)}",
                )
            
            features = features[expected_features]
            predictions_df = model_service.predict(features)
            
            result_html = predictions_df.to_html(classes='table table-striped table-hover', border=0)
            return render_template("index.html", predictions=result_html)

        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return render_template("index.html", error=f"Error: {str(e)}")


app = Flask(__name__)

# --- MUDANÇA CRÍTICA AQUI ---
# Removemos o try/except. Se der erro, queremos ver no terminal agora!
logger.info("Initializing ModelService...")
app.model_service = ModelService()
logger.info("ModelService initialized successfully!")

create_routes(app)

def main() -> None:
    app.run(host="0.0.0.0", port=5001, debug=True)

if __name__ == "__main__":
    main()