# MLOps Pipeline: Breast Cancer Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-3.0%2B-red?style=for-the-badge&logo=apache-airflow)
![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-9cf?style=for-the-badge&logo=dvc)
![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-blueviolet?style=for-the-badge)

Este projeto implementa um pipeline completo de Machine Learning (End-to-End) do dataset nativo Breast Cancer. O objetivo é demonstrar práticas modernas de **MLOps**, garantindo reprodutibilidade, versionamento de dados e orquestração automatizada.

---

## Arquitetura do Pipeline

O pipeline é orquestrado pelo **Apache Airflow** e gerencia o ciclo de vida dos dados usando **DVC**. Abaixo, a visualização do fluxo de execução:

![Airflow Graph View](images/airflow_graph_view.png)

### Etapas do Pipeline:
1.  **Ingestion (`dvc_load_data`)**: Carregamento dos dados brutos versionados.
2.  **Preprocessing (`dvc_preprocess_data`)**: Limpeza e normalização de texto (NLP).
3.  **Feature Engineering (`dvc_engineer_features`)**: Transformação de dados (TF-IDF/Embeddings).
4.  **Training (`dvc_train_model`)**: Treinamento do modelo (Random Forest/XGBoost).
5.  **Evaluation (`dvc_evaluate_model`)**: Geração de métricas e validação.
6.  **Registry (`register_artifacts`)**: Log de modelos e métricas no MLflow.

---

## Tecnologias Utilizadas

* **Linguagem:** Python 3.12
* **Gerenciamento de Dependências:** Poetry
* **Versionamento de Dados:** DVC (Data Version Control), DAGSHUB (repositório para versionamento dos dados)
* **Orquestração:** Apache Airflow (Standalone Mode)
* **Rastreamento de Experimentos:** MLflow (Integrado via DAG)
* **Bibliotecas de ML:** Scikit-learn, Pandas, Tensorflow, NumPY.

---

## Como Rodar o Projeto

### 1. Pré-requisitos
Certifique-se de ter instalado:
* Git
* Python 3.12+
* Poetry (`pip install poetry`)

### 2. Instalação
Clone o repositório e instale as dependências:

```bash
git clone [https://github.com/Code-AldreySandre/MLOPS_Project.git](https://github.com/Code-AldreySandre/MLOPS_Project.git)
cd MLOPS_Project

# Instala todas as libs do pyproject.toml
poetry install
```
### 3. Configuração do Airflow (Standalone)
Este projeto utiliza uma estrutura customizada onde o Airflow reside dentro da pasta raiz.
```bash
# 1. Defina a variável de ambiente (Crucial!)
export AIRFLOW_HOME=$(pwd)/airflow_home

# 2. Inicialize o Airflow no modo Standalone (DB + Scheduler + Webserver)
poetry run airflow standalone
```
-**obs:** Ao rodar pela primeira vez, o terminal exibirá a senha do usuário admin. Essa senha também fica salva em airflow_home/standalone_admin_password.txt.

### 4. Acessando a Interface
- Abra o navegador em: **http://localhost:8080**

- Faça login com admin e a senha gerada.

- Busque pela DAG: **ml_pipeline_project**.

- Ative o toggle (Unpause) e clique em ▶️ Trigger DAG.

---
## Estrutura do Projeto
```bash
MLOPS_Project/
├── airflow_home/       # Configurações e Logs do Airflow (Local)
├── dags/               # Definição dos Pipelines (DAGs)
│   └── ml_pipeline_dag.py
├── data/               # Dados versionados pelo DVC (ignorados no git)
├── src/                # Código fonte do projeto
│   ├── data_load.py
│   ├── train_model.py
│   └── ...
├── dvc.yaml            # Definição dos estágios do DVC
├── dvc.lock            # Hash exato dos dados e modelos (Reprodutibilidade)
├── pyproject.toml      # Dependências do Poetry
└── README.md           # Este arquivo
```
---
## Segurança e Boas Práticas
- **Credenciais:** Senhas e chaves de API não são versionadas (via .gitignore).

- **Dados:** Apenas os arquivos .dvc sobem para o GitHub; os dados reais ficam no armazenamento remoto (S3/DagsHub).
