import os
from dotenv import load_dotenv
env = os.getenv("ENV", "dev")
load_dotenv(f".env.{env}")

DATABASE_URL = os.getenv("DATABASE_URL")