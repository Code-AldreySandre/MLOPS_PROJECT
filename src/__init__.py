import logging
from dotenv import load_dotenv
import dagshub

load_dotenv()

#Initialize Dagshub with credentials
dagshub.init(
    repo_owner="Code-AldreySandre",
    repo_name="MLOPS_PROJECT"
)

# LogGin strategy
logging.basicConfig(
    level=logging.INFO, # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)