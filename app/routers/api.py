from fastapi import APIRouter, Body
from app.services.s1_service import extract_s1_parameters
from app.services.s2_service import extract_s2_parameters
import asyncio
import ee, random, json

router = APIRouter()

@router.post("/extract-s1-parameters")
async def extract_s1_parameters_endpoint(
    geojson: dict = Body(..., description="GeoJSON file defining the region of interest"),
    start_date: str = Body(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Body(..., description="End date in YYYY-MM-DD format")
):
    geometry = geojson.get('geometry')
    results = await asyncio.to_thread(extract_s1_parameters, geometry, start_date, end_date)
    return results

@router.post("/extract-s2-parameters")
async def extract_s2_parameters_endpoint(
    geojson: dict = Body(..., description="GeoJSON file defining the region of interest"),
    start_date: str = Body(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Body(..., description="End date in YYYY-MM-DD format")
):
    geometry = geojson.get('geometry')
    results = await asyncio.to_thread(extract_s2_parameters, geometry, start_date, end_date)
    return results

@router.post("/mock-results")
def generate_mock_crop_predictions(geojson_polygon: dict, point_spacing: int = 10) -> dict:
    try:
        # Convert input GeoJSON geometry to ee.Geometry
        polygon = ee.Geometry(geojson_polygon['geometry'])

        # Create pixel coordinate image and sample points within polygon
        image = ee.Image.pixelCoordinates(ee.Projection('EPSG:4326')).clip(polygon)
        sampled_points = image.sample(region=polygon, scale=point_spacing, geometries=True)

        # Assign random labels to each point
        def assign_label(feature):
            label = random.choice(['crop', 'not_crop'])
            return feature.set({'prediction': label})
        
        def assign_label_2(feature):
            rand = ee.Number.random()
            label = ee.Algorithms.If(rand.gt(0.5), 'crop', 'not_crop')
            return feature.set({'prediction': label})
        
        labeled_points = sampled_points.map(assign_label)

        # Convert to client-side GeoJSON format
        features = labeled_points.getInfo()['features']
        geojson_output = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": feature['geometry'],
                    "properties": {
                        "prediction": feature['properties']['prediction']
                    }
                }
                for feature in features
            ]
        }

        return geojson_output
    
    except Exception as e:
        raise RuntimeError(f"Error generating mock predictions: {e}")

