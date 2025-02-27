from fastapi import FastAPI
from app.routers import api
import ee
import os

service_account = 'gee-service-account@wise-scene-427306-q3.iam.gserviceaccount.com'

# Get the absolute path of the key file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of main.py
KEY_PATH = os.path.join(BASE_DIR, "config", "gee-key.json")

credentials = ee.ServiceAccountCredentials(service_account, KEY_PATH)
ee.Initialize(credentials)


app = FastAPI(title="GEE Parameter Extraction Microservice")
app.include_router(api.router)

# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload