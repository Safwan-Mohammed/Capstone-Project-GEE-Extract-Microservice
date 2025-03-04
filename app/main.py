from fastapi import FastAPI
from app.routers import api
import ee
import os
import uvicorn

service_account = 'gee-service-account@wise-scene-427306-q3.iam.gserviceaccount.com'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
KEY_PATH = os.path.join(BASE_DIR, "config", "gee-key.json")

credentials = ee.ServiceAccountCredentials(service_account, KEY_PATH)
ee.Initialize(credentials)

app = FastAPI(title="GEE Parameter Extraction Microservice")
app.include_router(api.router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=4000, reload=True)

""" 
RUN THIS FILE USING THE COMMAND 
    python -m app.main 
"""