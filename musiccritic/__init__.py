import logging
from pathlib import Path

from dotenv import load_dotenv

from musiccritic.config import Configs

# Load environment variables from .env file
environment_file_path = Path(__file__).parents[1] / ".env"
load_dotenv(environment_file_path)

# Initiate logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configurations
configs = Configs()
