from fastapi import APIRouter, UploadFile, File, Form
from app.services.geojson_service import process_geojson
from app.services.gee_service import extract_parameters

router = APIRouter()

@router.post("/extract-parameters")
async def extract_parameters_endpoint(
    geojson: UploadFile = File(..., description="GeoJSON file defining the region of interest"),
    start_date: str = Form(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Form(..., description="End date in YYYY-MM-DD format")
):
    geometry = process_geojson(geojson)
    results = extract_parameters(geometry, start_date, end_date)
    return results