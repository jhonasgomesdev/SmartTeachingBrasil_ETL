import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_postgres_engine():
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")
    db = os.getenv("PG_DATABASE")

    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(conn_str)
    return engine