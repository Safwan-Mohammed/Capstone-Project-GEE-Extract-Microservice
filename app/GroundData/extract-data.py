import ee
import os
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Initialize Earth Engine with Service Account Credentials
service_account = 'gee-service-account@wise-scene-427306-q3.iam.gserviceaccount.com'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
KEY_PATH = os.path.join(BASE_DIR, "..", "config", "gee-key.json")

credentials = ee.ServiceAccountCredentials(service_account, KEY_PATH)
try:
    ee.Initialize(credentials)
    print("GEE successfully initialized")
except Exception as e:
    print("Error in initializing GEE:", e)

# Load GeoJSON region
with open(os.path.join(BASE_DIR, "..", "config", "Tumkur.geojson")) as f:
    geojson_data = json.load(f)
    feature = ee.Feature(geojson_data)
    geojson = ee.FeatureCollection([feature])
    region = geojson.geometry().bounds()

def extract_s1_s2_data(points_list):
    """
    Extracts Sentinel-1 and Sentinel-2 data, processes them, and saves the output.
    """
    s1_results = []
    s2_results = []

    for point in points_list:
        coords, date = point["coords"], point["date"]
        date_ee = ee.Date(date)

        # Sentinel-1 Processing
        s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                  .filterDate(date_ee.advance(-6, "day"), date_ee.advance(6, "day"))
                  .filterBounds(region))
        s1_img = s1_col.first()
        if s1_img:
            vh_vv_ratio = s1_img.expression("VH / VV", {"VH": s1_img.select("VH"), "VV": s1_img.select("VV")}).rename("VH_VV_ratio")
            s1_img = s1_img.addBands(vh_vv_ratio)
            vh_db = s1_img.select("VH").log10().multiply(10).rename("VH_dB")
            vv_db = s1_img.select("VV").log10().multiply(10).rename("VV_dB")
            vh_vv_db = s1_img.select("VH_VV_ratio").log10().multiply(10).rename("VH_VV_dB")
            s1_img = s1_img.addBands([vh_db, vv_db, vh_vv_db])

            point_feature = ee.Feature(ee.Geometry.Point(coords), {"date": date})
            s1_values = s1_img.sampleRegions(collection=ee.FeatureCollection([point_feature]), scale=10).getInfo()
            for feature in s1_values["features"]:
                props = feature["properties"]
                s1_results.append({
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "date": date,
                    "VH_dB": props.get("VH_dB"),
                    "VV_dB": props.get("VV_dB"),
                    "VH_VV_dB": props.get("VH_VV_dB")
                })

        # Sentinel-2 Processing
        s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterDate(date_ee.advance(-2, "day"), date_ee.advance(3, "day"))
                  .filterBounds(region)
                  .sort("CLOUDY_PIXEL_PERCENTAGE"))
        s2_img = s2_col.first()
        if s2_img:
            ndvi = s2_img.expression("(B8 - B4) / (B8 + B4)", {"B8": s2_img.select("B8"), "B4": s2_img.select("B4")}).rename("NDVI")
            evi = s2_img.expression("2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1)", {"B8": s2_img.select("B8"), "B4": s2_img.select("B4"), "B2": s2_img.select("B2")}).rename("EVI")
            s2_img = s2_img.addBands([ndvi, evi])
            s2_values = s2_img.sampleRegions(collection=ee.FeatureCollection([point_feature]), scale=10).getInfo()
            for feature in s2_values["features"]:
                props = feature["properties"]
                s2_results.append({
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "date": date,
                    "NDVI": props.get("NDVI"),
                    "EVI": props.get("EVI")
                })
    return s1_results, s2_results

def save_results(s1_results, s2_results):
    """Combine and save the results."""
    combined_results = []
    s1_dict = {(r["longitude"], r["latitude"], r["date"]): r for r in s1_results}
    s2_dict = {(r["longitude"], r["latitude"], r["date"]): r for r in s2_results}
    all_keys = set(s1_dict.keys()).union(set(s2_dict.keys()))

    for key in all_keys:
        longitude, latitude, date = key
        s1_data = s1_dict.get(key, {})
        s2_data = s2_dict.get(key, {})
        combined_results.append({
            "Longitude": longitude,
            "Latitude": latitude,
            "Date": date,
            "VH_dB": s1_data.get("VH_dB"),
            "VV_dB": s1_data.get("VV_dB"),
            "VH_VV_dB": s1_data.get("VH_VV_dB"),
            "NDVI": s2_data.get("NDVI"),
            "EVI": s2_data.get("EVI")
        })

    df = pd.DataFrame(combined_results)
    csv_filename = os.path.join(BASE_DIR, "sentinel_data.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Results exported to {csv_filename}")

points_list = [
    {"coords": [-122.4194, 37.7749], "date": "2025-01-05"},
    {"coords": [-87.6298, 41.8781], "date": "2025-01-10"},
    {"coords": [2.3522, 48.8566], "date": "2025-01-15"},
    {"coords": [13.4050, 52.5200], "date": "2025-01-20"}
]

s1_results, s2_results = extract_s1_s2_data(points_list)
save_results(s1_results, s2_results)
