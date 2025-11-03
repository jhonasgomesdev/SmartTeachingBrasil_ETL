import requests
import os
import warnings

API_URL = os.getenv("API_URL", "https://localhost:7033/api/export/alunos-detalhados")

def extract_data():
    # DEV: se estiver com self-signed cert, usamos verify=False (inseguro, só dev)
    verify = os.getenv("API_VERIFY_SSL", "false").lower() == "true"
    if not verify:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    resp = requests.get(API_URL, verify=verify, timeout=30)
    resp.raise_for_status()
    return resp.json()