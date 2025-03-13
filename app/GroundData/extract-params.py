import ee
import os
import csv
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd

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

with open(os.path.join(BASE_DIR, "..", "config", "Tumkur.geojson")) as f:
    geojson_data = json.load(f)
    feature = ee.Feature(geojson_data)
    geojson = ee.FeatureCollection([feature])

def preprocess_s2_collection_for_date(date, region):

    def get_cloud_parameters(date):
        try:
            month = datetime.strptime(date, "%Y-%m-%d").month
            if 7 <= month <= 9:
                return 80, 50, 0.15, 1, 50
            else:
                return 40, 40, 0.15, 1, 100
        except Exception as e:
            raise Exception(f"Failed to parse date in get_cloud_parameters: {e}")

    try:
        CLOUD_FILTER, CLD_PRB_THRESH, NIR_DRK_THRESH, CLD_PRJ_DIST, BUFFER = get_cloud_parameters(date)
    except Exception as e:
        raise Exception(f"Error setting cloud parameters for {date}: {e}")
    
    def add_cloud_bands(img):
        try:
            cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
            is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
            return img.addBands(ee.Image([cld_prb, is_cloud]))
        except Exception as e:
            raise Exception(f"Error in add_cloud_bands for {date}: {e}")

    def add_shadow_bands(img):
        try:
            not_water = img.select('SCL').neq(6)
            SR_BAND_SCALE = 1e4
            dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
            shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
            cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                .select('distance')
                .mask()
                .rename('cloud_transform'))
            shadows = cld_proj.multiply(dark_pixels).rename('shadows')
            return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))
        except Exception as e:
            raise Exception(f"Error in add_shadow_bands for {date}: {e}")

    def add_cld_shdw_mask(img):
        try:
            img_cloud = add_cloud_bands(img)
            img_cloud_shadow = add_shadow_bands(img_cloud)
            is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
            is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER * 2 / 20)
                .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                .rename('cloudmask'))
            return img_cloud_shadow.addBands(is_cld_shdw)
        except Exception as e:
            raise Exception(f"Error in add_cld_shdw_mask for {date}: {e}")

    def apply_cld_shdw_mask(img):
        try:
            not_cld_shdw = img.select('cloudmask').Not()
            return img.select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']).updateMask(not_cld_shdw)
        except Exception as e:
            raise Exception(f"Error in apply_cld_shdw_mask for {date}: {e}")

    date_ee = ee.Date(date)
    start_date = date_ee.advance(-2, "day")
    end_date = date_ee.advance(3, "day")

    s2_sr_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                 .filterDate(start_date, end_date)
                 .select(["B2", "B3", "B4", "B5", "B8", "B11", "B12", "SCL"])
                 .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER))
                 .filterBounds(region)
                 )

    s2_cloudless_col = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
                        .filterDate(start_date, end_date)
                        .filterBounds(region)
                        )

    s2_sr_cld_col = ee.ImageCollection(ee.Join.saveFirst("s2cloudless").apply(**{
        "primary": s2_sr_col,
        "secondary": s2_cloudless_col,
        "condition": ee.Filter.equals(leftField="system:index", rightField="system:index")
    }))

    return s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)

def compute_indices(image):
    try:
        nir = image.select("B8")
        red = image.select("B4")
        blue = image.select("B2")
        green = image.select("B3")
        rededge = image.select("B5")
        swir1 = image.select("B11")

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        evi = nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).rename("EVI")
        gndvi = nir.subtract(green).divide(nir.add(green)).rename("GNDVI")
        savi = nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename("SAVI")
        ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
        ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")
        rendvi = rededge.subtract(red).divide(rededge.add(red)).rename("RENDVI")

        return image.addBands([ndvi, evi, gndvi, savi, ndwi, ndmi, rendvi])
    except Exception as e:
        print(f"str{e}")

def extract_s1_parameters(points_list):

    points_by_date = defaultdict(list)
    for p in points_list:
        points_by_date[p["date"]].append(p["coords"])

    s1_results = []
    for date, coords_list in points_by_date.items():
        date_ee = ee.Date(date)
        region = ee.Geometry.Point(coords_list[0])
        s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.eq("resolution_meters", 10))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))  # Ensure VV exists
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))  # Ensure VH exists
                  .filterDate(date_ee.advance(-6, "day"), date_ee.advance(6, "day"))
                  .filterBounds(region)
                  )
        s1_img = s1_col.median()
        if s1_img:
            try:
                vh_vv_ratio = s1_img.expression("VH / VV", {"VH": s1_img.select("VH"), "VV": s1_img.select("VV")}).rename("VH_VV_ratio")
                s1_img = s1_img.addBands(vh_vv_ratio)
                vh_db = s1_img.select("VH").log10().multiply(10).rename("VH_dB")
                vv_db = s1_img.select("VV").log10().multiply(10).rename("VV_dB")
                vh_vv_db = s1_img.select("VH_VV_ratio").log10().multiply(10).rename("VH_VV_dB")
                s1_img = s1_img.addBands([vh_db, vv_db, vh_vv_db])

                features = [ee.Feature(ee.Geometry.Point(coords), {"date": date}) for coords in coords_list]
                feature_collection = ee.FeatureCollection(features)
                s1_values = s1_img.sampleRegions(collection=feature_collection, scale=10, projection="EPSG:4326", geometries = True).getInfo()

                # Store results
                for feature in s1_values["features"]:
                    props = feature["properties"]
                    s1_results.append({
                        "longitude": feature["geometry"]["coordinates"][0],
                        "latitude": feature["geometry"]["coordinates"][1],
                        "date": props["date"],
                        "s1_values":{
                            "VH_dB": props.get("VH_dB"),
                            "VV_dB": props.get("VV_dB"),
                            "VH_VV_dB": props.get("VH_VV_dB"),
                        }
                    })
            except ee.EEException as e:
                print(f"Error processing Sentinel-1 for {date}: {str(e)}")
                continue
    return s1_results

def extract_s2_parameters(points_list):

    points_by_date = defaultdict(list)
    for p in points_list:
        points_by_date[p["date"]].append(p["coords"])

    s2_results = []
    for date, coords_list in points_by_date.items():
        region = ee.Geometry.Point(coords_list[0])
        s2_col = preprocess_s2_collection_for_date(date, region)
        s2_img = s2_col.median()
        if s2_img:
            try:
                s2_img = compute_indices(s2_img)
                features = [ee.Feature(ee.Geometry.Point(coords), {"date": date}) for coords in coords_list]
                feature_collection = ee.FeatureCollection(features)
                s2_values = s2_img.sampleRegions(collection=feature_collection, scale=10, projection="EPSG:4326", geometries = True).getInfo()
                for feature in s2_values["features"]:
                    props = feature["properties"]
                    s2_results.append({
                        "longitude": feature["geometry"]["coordinates"][0],
                        "latitude": feature["geometry"]["coordinates"][1],
                        "date": props["date"],
                        "s2_values": {
                            "NDVI": props.get("NDVI"),
                            "EVI": props.get("EVI"),
                            "GNDVI": props.get("GNDVI"),
                            "SAVI": props.get("SAVI"),
                            "NDWI": props.get("NDWI"),
                            "NDMI": props.get("NDMI"),
                            "RENDVI": props.get("RENDVI"),
                        }
                    })
            except Exception as e:
                print(f"Error processing Sentinel-2 for {date}: {str(e)}")
                continue
    return s2_results

def combine_s1_s2_parameters(s1_results, s2_results):

    combined_results = []

    s1_dict = {(r["longitude"], r["latitude"], r["date"]): r for r in s1_results}

    s2_dict = {(r["longitude"], r["latitude"], r["date"]): r for r in s2_results}

    all_keys = set(s1_dict.keys()).union(set(s2_dict.keys()))

    for key in all_keys:
        longitude, latitude, date = key
        s1_data = s1_dict.get(key, {"longitude": longitude, "latitude": latitude, "date": date, "s1_values": None})
        s2_data = s2_dict.get(key, {"longitude": longitude, "latitude": latitude, "date": date, "s2_values": None})

        combined_results.append({
            "longitude": longitude,
            "latitude": latitude,
            "date": date,
            "s1_values": s1_data["s1_values"],
            "s2_values": s2_data["s2_values"]
        })

    return combined_results

points_list = [
    {"coords": [-87.6298, 41.8781], "date": "2025-01-10"},   # Chicago, IL, USA
    {"coords": [2.3522, 48.8566], "date": "2025-01-15"},    # Paris, France
    {"coords": [13.4050, 52.5200], "date": "2025-01-20"},   # Berlin, Germany
    {"coords": [139.6917, 35.6895], "date": "2025-02-01"},  # Tokyo, Japan
    {"coords": [77.2090, 28.6139], "date": "2025-02-05"},   # New Delhi, India
    {"coords": [-46.6333, -23.5505], "date": "2025-02-10"},  # São Paulo, Brazil
    {"coords": [31.2357, 30.0444], "date": "2025-02-15"},    # Cairo, Egypt
]

s1_results = extract_s1_parameters(points_list)
s2_results = extract_s2_parameters(points_list)

combined_results = combine_s1_s2_parameters(s1_results, s2_results)

df = pd.DataFrame([
    {
        "Longitude": item["longitude"],
        "Latitude": item["latitude"],
        "Date": item["date"],
        "VV_dB": item["s1_values"]["VV_dB"] if item["s1_values"] else None,
        "VH_dB": item["s1_values"]["VH_dB"] if item["s1_values"] else None,
        "VH_VV_dB": item["s1_values"]["VH_VV_dB"] if item["s1_values"] else None,
        "NDVI": item["s2_values"]["NDVI"] if item["s2_values"] else None,
        "EVI": item["s2_values"]["EVI"] if item["s2_values"] else None,
        "GNDVI": item["s2_values"]["GNDVI"] if item["s2_values"] else None,
        "SAVI": item["s2_values"]["SAVI"] if item["s2_values"] else None,
        "NDWI": item["s2_values"]["NDWI"] if item["s2_values"] else None,
        "NDMI": item["s2_values"]["NDMI"] if item["s2_values"] else None,
        "RENDVI": item["s2_values"]["RENDVI"] if item["s2_values"] else None,
    }
    for item in combined_results
])

csv_filename = os.path.join(BASE_DIR ,"sentinel_data.csv")
df.to_csv(csv_filename, index=False)
print(f"Results exported to {csv_filename}")