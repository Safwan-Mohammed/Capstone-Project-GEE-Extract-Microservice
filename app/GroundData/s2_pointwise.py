import ee, os
from datetime import datetime

# Initialize GEE

def get_sentinel2_data(ee, coordinate: tuple, date: str) -> dict:

    # service_account = 'gee-service-account@wise-scene-427306-q3.iam.gserviceaccount.com'
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    # KEY_PATH = os.path.join(BASE_DIR, "..", "config", "gee-key.json")

    # credentials = ee.ServiceAccountCredentials(service_account, KEY_PATH)

    # try:
    #     ee.Initialize(credentials)
    #     print("GEE successfully initialized")
    # except Exception as e:
    #     print("Error in initializing GEE:", e)

    AOI = ee.Geometry.Point(coordinate)
    START_DATE = date
    END_DATE = date  # Single date

    def get_cloud_parameters(date):
        month = datetime.strptime(date, "%Y-%m-%d").month
        return (80, 50, 0.15, 1, 50) if 7 <= month <= 9 else (40, 40, 0.15, 1, 100)

    CLOUD_FILTER, CLD_PRB_THRESH, NIR_DRK_THRESH, CLD_PRJ_DIST, BUFFER = get_cloud_parameters(START_DATE)

    def add_cloud_bands(img):
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
        is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def add_shadow_bands(img):
        not_water = img.select('SCL').neq(6)
        dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * 1e4).multiply(not_water).rename('dark_pixels')
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        cld_proj = (img.select('clouds')
            .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
            .reproject(crs=img.select(0).projection(), scale= 100)
            .select('distance')
            .mask()
            .rename('cloud_transform'))
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def add_cld_shdw_mask(img):
        img_cloud = add_cloud_bands(img)
        img_cloud_shadow = add_shadow_bands(img_cloud)
        is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
        is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER * 2 / 20)
            .reproject(crs=img.select(0).projection(), scale= 20)
            .rename('cloudmask'))
        return img_cloud_shadow.addBands(is_cld_shdw)

    def apply_cld_shdw_mask(img):
        return img.select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']).updateMask(img.select('cloudmask').Not())

    def get_s2_sr_cld_col(aoi, date):
        s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(date, ee.Date(date).advance(5, 'day'))  # Ensure we get an image for the given date
            .select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12', 'SCL'])
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(aoi)
            .filterDate(date, ee.Date(date).advance(5, 'day')))

        return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_sr_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
        }))
    def compute_indices(image):
        try:
            nir = image.select('B8')
            red = image.select('B4')
            blue = image.select('B2')
            green = image.select('B3')
            rededge = image.select('B5')
            swir1 = image.select('B11')
            
            ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
            evi = nir.subtract(red).multiply(2.5).divide(
                nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).rename('EVI')
            gndvi = nir.subtract(green).divide(nir.add(green)).rename('GNDVI')
            L = 0.5
            savi = nir.subtract(red).divide(nir.add(red).add(L)).multiply(1 + L).rename('SAVI')
            ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
            ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename('NDMI')
            rendvi = rededge.subtract(red).divide(rededge.add(red)).rename('RENDVI')
            
            return image.addBands([ndvi, evi, gndvi, savi, ndwi, ndmi, rendvi])
        except Exception as e:
            raise Exception(f"Error in compute_indices: {e}")

    # Process Sentinel-2 imagery
    s2_sr_cld_col = get_s2_sr_cld_col(AOI, START_DATE)
    s2_sr_clean = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)
    s2_sr_median = s2_sr_clean.median().clip(AOI)

    s2_median_with_indices = compute_indices(s2_sr_median)

    # Extract band values for the point
    band_values = s2_median_with_indices.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=AOI,
        scale=10,  # Sentinel-2 resolution
        bestEffort=True
    ).getInfo()

    return band_values