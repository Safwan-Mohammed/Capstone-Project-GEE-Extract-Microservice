import ee, os

def get_sentinel1_data(ee, coordinate: tuple, date: str):
    # Initialize GEE
    # service_account = 'gee-service-account@wise-scene-427306-q3.iam.gserviceaccount.com'
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    # KEY_PATH = os.path.join(BASE_DIR, "..", "config", "gee-key.json")

    # credentials = ee.ServiceAccountCredentials(service_account, KEY_PATH)

    # try:
    #     ee.Initialize(credentials)
    #     print("GEE successfully initialized")
    # except Exception as e:
    #     print("Error in initializing GEE:", e)

    # Create a Point geometry
    point = ee.Geometry.Point(coordinate)

    # Load Sentinel-1 Image Collection
    s1_collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(point) \
        .filterDate(ee.Date(date).advance(-3, "day"), ee.Date(date).advance(3, "day")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")) \
        .select(["VV", "VH"])  # Select radar bands

    # Get the first image (or you can use median)
    s1_img = s1_collection.median()

    # Extract band values at the point
    s1_values = s1_img.sampleRegions(
        collection=ee.FeatureCollection([ee.Feature(point)]),
        scale=10,  # Adjust scale based on resolution
        projection=s1_img.projection()
    ).first().toDictionary().getInfo()

    return s1_values
