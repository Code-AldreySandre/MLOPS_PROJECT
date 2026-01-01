import logging
import os
import sys

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# 1. Carregar variáveis de ambiente (Crucial para achar o servidor correto!)
load_dotenv()

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("src.register_artifacts")

# Inicializa o cliente usando a URI do .env
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
client = MlflowClient()

def get_best_run(experiment_id: str, parent_run_id: str) -> pd.Series:
    """Retorna o melhor run filho baseado na acurácia de teste."""
    try:
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            order_by=["metrics.test_accuracy DESC"], # Garanta que essa métrica existe no evaluate_model
            max_results=1
        )
        
        if not child_runs:
            logger.warning("Nenhum run filho encontrado para este pai.")
            return None
            
        return child_runs[0]
    except Exception as e:
        logger.error(f"Erro ao buscar melhor run: {e}")
        return None

def register_model() -> None:
    """Registra o melhor modelo da última execução do DVC."""
    
    experiment_name = "ml_classification"
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        logger.error(f"Experimento '{experiment_name}' não encontrado.")
        return

    experiment_id = experiment.experiment_id

    logger.info("Buscando a última execução pai do DVC...")
    
    # ESTRATÉGIA MAIS SEGURA: 
    # Em vez de pegar qualquer 'latest run', pegamos o último run marcado como PAI pelo DVC
    dvc_parents = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.dvc_exp = 'True'", # Lembra que colocamos essa tag no train_model?
        order_by=["start_time DESC"],
        max_results=1
    )

    if not dvc_parents:
        logger.error("Nenhum experimento DVC encontrado.")
        return

    parent_run_id = dvc_parents[0].info.run_id
    logger.info(f"Processando Experimento Pai ID: {parent_run_id}")

    # Usa a função auxiliar para achar o melhor filho
    best_run = get_best_run(experiment_id, parent_run_id)
    
    if best_run is None:
        logger.error("Não foi possível determinar o melhor modelo.")
        return

    run_id = best_run.info.run_id
    best_acc = best_run.data.metrics.get('test_accuracy', 'N/A')
    
    logger.info(f"Melhor Run encontrado: {run_id} (Acurácia: {best_acc})")
    
    # Nome do modelo no Registry
    model_name = "model" 
    
    # Criar o Modelo Registrado (se não existir)
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        logger.debug(f"Modelo '{model_name}' já existe no registro.")

    # Criar a Versão do Modelo
    # O caminho "model" aqui refere-se à pasta DENTRO do artefato do run. 
    # O autolog do Keras salva na pasta "model" ou "model/data"? 
    # Geralmente 'runs:/{id}/model' funciona se usou autolog padrão.
    model_uri = f"runs:/{run_id}/model"
    
    try:
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        logger.info(f"Modelo registrado com sucesso! Versão: {model_version.version}")
        
        # (Opcional) Adicionar transição de estágio para 'Staging'
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info("Modelo movido para o estágio 'Staging'.")
        
    except Exception as e:
        logger.error(f"Falha ao registrar versão do modelo: {e}")

def main() -> None:
    register_model()
    logger.info("Processo de registro finalizado.")

if __name__ == "__main__":
    main()