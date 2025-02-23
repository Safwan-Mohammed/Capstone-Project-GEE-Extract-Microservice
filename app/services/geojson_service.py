import json
import ee
from io import BytesIO
from fastapi import UploadFile

def process_geojson(geojson: UploadFile) -> ee.Geometry:
    geojson_content = geojson.file.read().decode('utf-8')
    geojson_dict = json.loads(geojson_content)
    
    if geojson_dict['type'] == 'FeatureCollection':
        features = geojson_dict['features']
        geometry = ee.Geometry(features[0]['geometry'])
    elif geojson_dict['type'] == 'Feature':
        geometry = ee.Geometry(geojson_dict['geometry'])
    else:
        raise ValueError("Invalid GeoJSON: Must be a Feature or FeatureCollection")
    
    return geometry