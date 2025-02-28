import ee
from app.helpers.wrapper import s1_preproc
from app.helpers.helper import lin_to_db2

def extract_s1_parameters(geometry: ee.Geometry, start_date: str, end_date: str) -> dict:
    s1_median = preprocess_s1(geometry, start_date, end_date)
    s1_with_vwc = calculate_vwc(s1_median)
    
    vwc_median = s1_with_vwc.select('VWC').reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=geometry,
        scale=10,
        maxPixels=1e13
    ).get('VWC').getInfo()

    return {'VWC': vwc_median}

def preprocess_s1(geometry: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    parameters = {
        'START_DATE': start_date,
        'STOP_DATE': end_date,
        'POLARIZATION': 'VVVH',  
        'GEOMETRY': geometry,    
        'ORBIT': 'BOTH',         
        'APPLY_BORDER_NOISE_CORRECTION': True,
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
    print(s1_processed.size())
    s1_median = s1_processed.median()
    s1_with_ratio = s1_median.addBands(
        s1_median.expression('VH / VV', {'VH': s1_median.select('VH'), 'VV': s1_median.select('VV')}).rename('VH_VV_ratio')
    )
    s1_final = lin_to_db2(s1_with_ratio)
    print("Finished processing S1")
    return s1_final.clip(geometry)

def calculate_vwc(image: ee.Image) -> ee.Image:
    A = 0.12
    B = 0.04
    Omega = 0.3
    VV = image.select('VV')
    VWC = VV.subtract(A).divide(B).multiply(Omega).rename('VWC')
    return image.addBands(VWC)