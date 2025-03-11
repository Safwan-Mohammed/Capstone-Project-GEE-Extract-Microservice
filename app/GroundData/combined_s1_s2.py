import ee, os
from datetime import datetime
from s1_pointwise import get_sentinel1_data
from s2_pointwise import get_sentinel2_data 

# Initialize GEE
service_account = 'gee-service-account@wise-scene-427306-q3.iam.gserviceaccount.com'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
KEY_PATH = os.path.join(BASE_DIR, "..", "config", "gee-key.json")

credentials = ee.ServiceAccountCredentials(service_account, KEY_PATH)

try:
    ee.Initialize(credentials)
    print("GEE successfully initialized")
except Exception as e:
    print("Error in initializing GEE:", e)

def process_coordinates(coordinate_date_list):
    results = []
    for coord, date in coordinate_date_list:
        print(f"Processing coordinate {coord} for date {date}...")
        try:
            s2_data = get_sentinel2_data(ee, coord, date)
        except Exception as e:
            print(f"Sentinel-2 processing failed for {coord} on {date}: {e}")
            s2_data = {}
        try:
            s1_data = get_sentinel1_data(ee, coord, date)
        except Exception as e:
            print(f"Sentinel-1 processing failed for {coord} on {date}: {e}")
            s1_data = {}

        combined_data = {
            "coordinate": coord,
            "date": date,
            "sentinel_2": s2_data,
            "sentinel_1": s1_data
        }
        results.append(combined_data)

    return results

coordinate_date_list = [
    ((77.2090, 28.6139), "2024-06-01"),  # Delhi
    ((-122.4194, 37.7749), "2025-01-05"),  # San Francisco
    ((78.9629, 20.5937), "2024-07-10")  # India (general location)
]

# Run processing
final_results = process_coordinates(coordinate_date_list)

# Print or save results
for res in final_results:
    print(res)
