import yaml
import sys
import logging
from pathlib import Path
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# --- AJUSTE DE CAMINHO ---
# Se o arquivo está em: .../MLOPS_Project/dags/ml_pipeline_dag.py
# parents[0] = dags
# parents[1] = MLOPS_Project (Raiz do Projeto)
project_root = Path(__file__).resolve().parents[1]

# Adiciona a raiz ao Python Path para conseguir importar 'src'
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Função para ler os estágios do dvc.yaml
def get_dvc_stages():
    dvc_yaml_path = project_root / "dvc.yaml"
    if not dvc_yaml_path.exists():
        logging.error(f"dvc.yaml não encontrado em {dvc_yaml_path}")
        return []
        
    with open(dvc_yaml_path) as f:
        dvc_config = yaml.safe_load(f)
    return list(dvc_config.get("stages", {}).keys())

# Função para registrar artefatos (já que você confirmou que tem o arquivo)
def register_artifacts_callable():
    try:
        from src.register_artifacts import main
        main()
        logging.info("Artefatos registrados com sucesso!")
    except ImportError as e:
        logging.error(f"Erro ao importar src.register_artifacts: {e}")
        raise e

default_args = {
    "owner": "aldrey",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "ml_pipeline_project",
    default_args=default_args,
    description='Pipeline MLOps: DVC + MLflow (Sem Docker)',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=['mlops', 'dvc', 'portfolio']
) as dag:
    
    # 1. Lê os estágios do DVC dinamicamente
    dvc_stages = get_dvc_stages()

    # 2. Cria tasks do DVC (BashOperator)
    dvc_tasks = []
    for stage in dvc_stages:
        task = BashOperator(
            task_id=f"dvc_{stage}",
            cwd=str(project_root), # Executa na raiz do projeto
            bash_command=f"dvc repro {stage}",
            append_env=True # Mantém credenciais do ambiente
        )
        dvc_tasks.append(task)

    # 3. Task de Registro no MLflow (PythonOperator)
    register_artifacts = PythonOperator(
        task_id="register_artifacts",
        python_callable=register_artifacts_callable
    )

    # --- DEFININDO A ORDEM ---
    
    # Encadeia os estágios do DVC: dvc_load >> dvc_train >> ...
    if dvc_tasks:
        for i in range(len(dvc_tasks) - 1):
            dvc_tasks[i] >> dvc_tasks[i + 1]
        
        # O último estágio do DVC chama o registro de artefatos
        dvc_tasks[-1] >> register_artifacts
    else:
        # Fallback caso não ache estágios (para não quebrar a DAG)
        register_artifacts