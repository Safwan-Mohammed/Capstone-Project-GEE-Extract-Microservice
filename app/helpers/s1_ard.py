import wrapper as wp
import ee

parameter = {  'START_DATE': '2018-01-01',
            'STOP_DATE': '2018-02-01',        
            'POLARIZATION': 'VVVH',
            'ORBIT' : 'DESCENDING',
            'ROI': ee.Geometry.Rectangle([-47.1634, -3.00071, -45.92746, -5.43836]),
            'APPLY_BORDER_NOISE_CORRECTION': False,
            'APPLY_SPECKLE_FILTERING': True,
            'SPECKLE_FILTER_FRAMEWORK':'MULTI',
            'SPECKLE_FILTER': 'GAMMA MAP',
            'SPECKLE_FILTER_KERNEL_SIZE': 9,
            'SPECKLE_FILTER_NR_OF_IMAGES':10,
            'APPLY_TERRAIN_FLATTENING': True,
            'DEM': ee.Image('USGS/SRTMGL1_003'),
            'TERRAIN_FLATTENING_MODEL': 'VOLUME',
            'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER':0,
            'FORMAT': 'DB',
            'CLIP_TO_ROI': False,
            'SAVE_ASSET': True,
            'ASSET_ID': "users/amullissa"
            }

s1_processed = wp.s1_preproc(parameter)