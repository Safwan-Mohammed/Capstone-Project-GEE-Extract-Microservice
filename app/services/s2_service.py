import ee
from datetime import datetime

def preprocess_s2(geometry: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    AOI = geometry
    START_DATE = start_date  
    END_DATE = end_date    

    def get_cloud_parameters(start_date):
        month = datetime.strptime(start_date, "%Y-%m-%d").month
        
        if 7 <= month <= 9:
            return 80, 50, 0.15, 1, 50
        else:
            return 40, 40, 0.15, 1 , 100

    CLOUD_FILTER, CLD_PRB_THRESH, NIR_DRK_THRESH, CLD_PRJ_DIST, BUFFER = get_cloud_parameters(START_DATE)

    def add_cloud_bands(img):
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
        is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def add_shadow_bands(img):
        not_water = img.select('SCL').neq(6)
        SR_BAND_SCALE = 1e4
        dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
            .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
            .select('distance')
            .mask()
            .rename('cloud_transform'))
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def add_cld_shdw_mask(img):
        img_cloud = add_cloud_bands(img)
        img_cloud_shadow = add_shadow_bands(img_cloud)
        is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
        is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
            .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
            .rename('cloudmask'))
        return img_cloud_shadow.addBands(is_cld_shdw)

    def apply_cld_shdw_mask(img):
        not_cld_shdw = img.select('cloudmask').Not()
        return img.select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']).updateMask(not_cld_shdw)

    def get_s2_sr_cld_col(aoi, start_date, end_date):
        s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12', 'SCL'])
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
        
        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(aoi)
            .filterDate(start_date, end_date))
        
        return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_sr_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
        }))
    s2_sr_cld_col = get_s2_sr_cld_col(AOI, START_DATE, END_DATE)
    s2_sr_clean = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)
    s2_sr_median = s2_sr_clean.median().clip(AOI)
    return s2_sr_median

def generate_tile_grid(image, aoi):
    PIXEL_SIZE = 2560
    region = image.geometry()
    bounds = region.bounds().getInfo()
    bounds = bounds["coordinates"]
    
    coords = ee.List(bounds).get(0)
    minLon = ee.Number(ee.List(ee.List(coords).get(0)).get(0))
    minLat = ee.Number(ee.List(ee.List(coords).get(0)).get(1))
    maxLon = ee.Number(ee.List(ee.List(coords).get(2)).get(0))
    maxLat = ee.Number(ee.List(ee.List(coords).get(2)).get(1))

    centroid = region.centroid()
    latitude = ee.Number(centroid.coordinates().get(1))

    tile_size_meters = ee.Number(PIXEL_SIZE)
    meters_per_degree = ee.Number(111320).multiply(latitude.cos())
    tile_size_degrees = tile_size_meters.divide(meters_per_degree).multiply(10000).round().divide(10000).getInfo()

    lon_seq = ee.List.sequence(minLon, maxLon, tile_size_degrees)
    
    def make_feature(lon):
        lon = ee.Number(lon)
        lat_seq = ee.List.sequence(minLat, maxLat, tile_size_degrees)
        
        def make_tile(lat):
            lat = ee.Number(lat)
            return ee.Feature(ee.Geometry.Rectangle([lon, lat, lon.add(tile_size_degrees), lat.add(tile_size_degrees)]))
        
        return lat_seq.map(make_tile)
    
    grid = ee.FeatureCollection(ee.List(lon_seq.map(make_feature)).flatten())
    tilesInAOI = grid.filterBounds(aoi)
    clipped_tiles = tilesInAOI.map(lambda tile: tile.setGeometry(tile.geometry().intersection(aoi)))
    return clipped_tiles

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

def extract_s2_parameters(geometry: ee.Geometry, start_date: str, end_date: str) -> dict:
    s2_median = preprocess_s2(geometry, start_date, end_date)
    grid = generate_tile_grid(s2_median, geometry)
    print("Success")
    return grid

    # BELOW PART HAS TO BE TESTED

    # s2_with_indices = calculate_indices(s2_median)
    
    # indices_median = s2_with_indices.reduceRegion(
    #     reducer=ee.Reducer.median(),
    #     geometry=geometry,
    #     scale=10,
    #     maxPixels=1e13
    # )

    # results = {
    #     'NDVI': indices_median.get('NDVI').getInfo(),
    #     'EVI': indices_median.get('EVI').getInfo(),
    #     'GNDVI': indices_median.get('GNDVI').getInfo(),
    #     'SAVI': indices_median.get('SAVI').getInfo(),
    #     'NDWI': indices_median.get('NDWI').getInfo(),
    #     'NDMI': indices_median.get('NDMI').getInfo(),
    #     'RENDVI': indices_median.get('RENDVI').getInfo()
    # }
    # return results