import os
from dotenv import load_dotenv

load_dotenv()  # root .env

USE_JAX = os.getenv("USE_JAX", "false").lower() == "true"