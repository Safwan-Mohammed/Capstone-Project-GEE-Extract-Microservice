import ee
import math
from .wrapper import s1_preproc
from .helper import lin_to_db2

def preprocess_s1(geometry: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    parameters = {
        'START_DATE': start_date,
        'STOP_DATE': end_date,
        'POLARIZATION': 'VVVH',  
        'GEOMETRY': geometry,    
        'ORBIT': 'BOTH',         
        'APPLY_ADDITIONAL_BORDER_NOISE_CORRECTION': True,
        'APPLY_SPECKLE_FILTERING': True,
        'SPECKLE_FILTER_FRAMEWORK': 'MULTI',
        'SPECKLE_FILTER': 'REFINED LEE',
        'SPECKLE_FILTER_KERNEL_SIZE': 15,
        'SPECKLE_FILTER_NR_OF_IMAGES': 10,
        'APPLY_TERRAIN_FLATTENING': True,
        'DEM': ee.Image('USGS/SRTMGL1_003'),
        'TERRAIN_FLATTENING_MODEL': 'VOLUME',
        'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER': 0,
        'FORMAT': 'DB',
        'CLIP_TO_ROI': True,
        'SAVE_ASSETS': False     
    }

    parameters['ROI'] = parameters.pop('GEOMETRY')
    s1_processed = s1_preproc(parameters)

    s1_median = s1_processed.median()
    s1_with_ratio = s1_median.addBands(
        s1_median.expression('VH / VV', {'VH': s1_median.select('VH'), 'VV': s1_median.select('VV')}).rename('VH_VV_ratio')
    )
    s1_final = lin_to_db2(s1_with_ratio)
    return s1_final.clip(geometry)

def calculate_vwc(image: ee.Image) -> ee.Image:
    A = 0.12
    B = 0.04
    Omega = 0.3
    VV = image.select('VV')
    VWC = VV.subtract(A).divide(B).multiply(Omega).rename('VWC')
    return image.addBands(VWC)

def preprocess_s2(geometry: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    CLOUD_FILTER = 80
    CLD_PRB_THRESH = 50
    NIR_DRK_THRESH = 0.15
    CLD_PRJ_DIST = 1
    BUFFER = 50
    SR_BAND_SCALE = 1e4

    s2_sr_col = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))

    s2_cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date)

    joined = ee.Join.saveFirst('s2cloudless').apply(
        primary=s2_sr_col,
        secondary=s2_cloudless_col,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    )

    def add_cloud_bands(img):
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
        is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
        return img.addBands([cld_prb, is_cloud])

    s2_sr_cld = joined.map(add_cloud_bands)

    def add_shadow_bands(img):
        not_water = img.select('SCL').neq(6)
        dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        cld_proj = img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10).reproject({'crs': img.select(0).projection(), 'scale': 100}).select('distance').mask().rename('cloud_transform')
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')
        return img.addBands([dark_pixels, cld_proj, shadows])

    s2_sr_cld_shdw = s2_sr_cld.map(add_shadow_bands)

    def mask_clouds_shadows(img):
        is_cld_shdw = img.select('clouds').add(img.select('shadows')).gt(0)
        is_cld_shdw = is_cld_shdw.focalMin(2).focalMax(BUFFER * 2 / 20).reproject({'crs': img.select(0).projection(), 'scale': 20}).rename('cloudmask')
        not_cld_shdw = is_cld_shdw.not()
        return img.select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11']).updateMask(not_cld_shdw)

    s2_sr_clean = s2_sr_cld_shdw.map(mask_clouds_shadows)
    s2_median = s2_sr_clean.median().clip(geometry)
    return s2_median

def calculate_indices(image: ee.Image) -> ee.Image:
    B2 = image.select('B2')
    B3 = image.select('B3')
    B4 = image.select('B4')
    B5 = image.select('B5')
    B8 = image.select('B8')
    B11 = image.select('B11')

    NDVI = B8.subtract(B4).divide(B8.add(B4)).rename('NDVI')
    EVI = image.expression(
        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
        {'NIR': B8, 'RED': B4, 'BLUE': B2}
    ).rename('EVI')
    GNDVI = B8.subtract(B3).divide(B8.add(B3)).rename('GNDVI')
    SAVI = B8.subtract(B4).divide(B8.add(B4).add(0.5)).multiply(1.5).rename('SAVI')
    NDWI = B3.subtract(B8).divide(B3.add(B8)).rename('NDWI')
    NDMI = B8.subtract(B11).divide(B8.add(B11)).rename('NDMI')
    RENDVI = B5.subtract(B4).divide(B5.add(B4)).rename('RENDVI')

    return image.addBands([NDVI, EVI, GNDVI, SAVI, NDWI, NDMI, RENDVI])

def extract_parameters(geometry: ee.Geometry, start_date: str, end_date: str) -> dict:
    s1_median = preprocess_s1(geometry, start_date, end_date)
    s1_with_vwc = calculate_vwc(s1_median)
    vwc_median = s1_with_vwc.select('VWC').reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=geometry,
        scale=10,
        maxPixels=1e13
    ).get('VWC').getInfo()

    s2_median = preprocess_s2(geometry, start_date, end_date)
    s2_with_indices = calculate_indices(s2_median)
    indices_median = s2_with_indices.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=geometry,
        scale=10,
        maxPixels=1e13
    )

    results = {
        'VWC': vwc_median,
        'NDVI': indices_median.get('NDVI').getInfo(),
        'EVI': indices_median.get('EVI').getInfo(),
        'GNDVI': indices_median.get('GNDVI').getInfo(),
        'SAVI': indices_median.get('SAVI').getInfo(),
        'NDWI': indices_median.get('NDWI').getInfo(),
        'NDMI': indices_median.get('NDMI').getInfo(),
        'RENDVI': indices_median.get('RENDVI').getInfo()
    }
    return results