from fastapi import FastAPI
from app.routers import api
import ee

service_account = 'gee-service-account@wise-scene-427306-q3.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, '../gee-key.json')
ee.Initialize(credentials)

app = FastAPI(title="GEE Parameter Extraction Microservice")
app.include_router(api.router)