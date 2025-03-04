from fastapi import APIRouter, UploadFile, File, Form, Body
from app.services.geojson_service import process_geojson
from app.services.s1_service import extract_s1_parameters
from app.services.s2_service import extract_s2_parameters

router = APIRouter()

@router.post("/extract-s1-parameters")
async def extract_s1_parameters_endpoint(
    geojson: dict = Body(..., description="GeoJSON file defining the region of interest"),
    start_date: str = Body(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Body(..., description="End date in YYYY-MM-DD format")
):
    # geometry = process_geojson(geojson)
    geometry = geojson.get('geometry')
    results = extract_s1_parameters(geometry, start_date, end_date)
    return results

@router.post("/extract-s2-parameters")
async def extract_s2_parameters_endpoint(
    geojson: dict = Body(..., description="GeoJSON file defining the region of interest"),
    start_date: str = Body(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Body(..., description="End date in YYYY-MM-DD format")
):
    # geometry = process_geojson(geojson)
    geometry = geojson.get('geometry')
    results = extract_s2_parameters(geometry, start_date, end_date)
    return results
