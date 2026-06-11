import os
from pathlib import Path
from dotenv import load_dotenv

# Load the .env that sits next to this file (src/utrace/.env), independent of the current working directory.
load_dotenv(Path(__file__).resolve().parent / ".env")

USE_JAX = os.getenv("USE_JAX", "false").lower() == "true"