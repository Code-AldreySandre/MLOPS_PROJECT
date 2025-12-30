import io
import logging
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import load_model

logger = logging.getLogger("app.main")


class ModelService:
    def __init__(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all artifacts from the local project folder."""
        logger.info("Loading artifacts from local project folder")

        artifacts_dir = "artifacts"
        models_dir = "models"

        
        features_imputer_path = os.path.join(artifacts_dir, "features_mean_imputer.joblib")
        features_scaler_path = os.path.join(artifacts_dir, "features_scaler.joblib")
        target_encoder_path = os.path.join(artifacts_dir, "[target]_one_hot_encoder.joblib")
        model_path = os.path.join(models_dir, "model.keras")

        self.features_imputer = joblib.load(features_imputer_path)
        self.features_scaler = joblib.load(features_scaler_path)
        self.target_encoder = joblib.load(target_encoder_path)
        self.model = load_model(model_path)

        logger.info("Successfully loaded all artifacts")

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the full pipeline.

        Args:
            features: DataFrame containing the input features

        Returns:
            DataFrame containing the predictions
        """
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)

        y_pred_probs = self.model.predict(X_scaled)
        
        # Converte probabilidades em índices e depois em labels
        y_pred_indices = np.argmax(y_pred_probs, axis=1)
        y_decoded = self.target_encoder.categories_[0][y_pred_indices]

        return pd.DataFrame({"Prediction": y_decoded}, index=features.index)


def create_routes(app: Flask) -> None:
    """Create all routes for the application."""

    @app.route("/")
    def index() -> str:
        """Serve the HTML upload interface."""
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        """Handle CSV file upload, validate features, and return predictions."""
        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a CSV file")

        try:
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            # Validação: Usa as features que o scaler espera (mais seguro)
            # Se preferir usar o load_breast_cancer, pode manter, mas feature_names_in_ é dinâmico
            if hasattr(app.model_service.features_scaler, "feature_names_in_"):
                expected_features = app.model_service.features_scaler.feature_names_in_
            else:
                expected_features = load_breast_cancer().feature_names

            missing_cols = [col for col in expected_features if col not in features.columns]
            
            if missing_cols:
                return render_template(
                    "index.html",
                    error=f"Missing required columns: {', '.join(missing_cols)}",
                )
            
            features = features[expected_features]
            predictions = app.model_service.predict(features)
            
            return render_template("index.html", predictions=predictions.to_html(classes='table table-striped'))

        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return render_template("index.html", error=f"Error processing file: {str(e)}")


app = Flask(__name__)

try:
    app.model_service = ModelService()
    logger.info("Application initialized with model service")
except Exception as e:
    logger.error(f"Failed to initialize ModelService: {e}")

create_routes(app)


def main() -> None:
    """Run the flask development server."""
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()