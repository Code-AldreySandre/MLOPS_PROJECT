import json
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score

def evaluate():
    # 1. Carregar dados de TESTE (você precisa ter salvo isso em algum lugar ou dividir agora)
    # Assumindo que você tem um test.csv ou vai dividir o dataset original
    df = pd.read_csv("data/processed/train_processed.csv") # Idealmente seria test.csv
    
    # 2. Carregar artefatos
    model = load_model("models/model.keras")
    encoder = joblib.load("artifacts/[target]_one_hot_encoder.joblib")
    
    # 3. Preparar dados (X e y)
    X = df.drop("target", axis=1)
    y_true = df["target"]
    
    # 4. Previsão
    y_pred_probs = model.predict(X)
    y_pred_indices = y_pred_probs.argmax(axis=1)
    y_pred_labels = encoder.categories_[0][y_pred_indices]
    
    # 5. Métricas
    acc = accuracy_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels, average="weighted")
    
    # 6. Salvar evaluation.json
    metrics = {"accuracy": acc, "f1_score": f1}
    
    with open("metrics/evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    evaluate()