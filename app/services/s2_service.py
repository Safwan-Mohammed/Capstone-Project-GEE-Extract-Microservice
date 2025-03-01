import ee
from datetime import datetime
import json
from app.helpers.generate_tile_grid import generate_tile_grid

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

def compute_indices(image):
    nir = image.select('B8')
    red = image.select('B4')
    blue = image.select('B2')
    green = image.select('B3')
    rededge = image.select('B5')
    swir1 = image.select('B11')
    
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    evi = nir.subtract(red).multiply(2.5).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    ).rename('EVI')
    
    gndvi = nir.subtract(green).divide(nir.add(green)).rename('GNDVI')
    
    L = 0.5
    savi = nir.subtract(red).divide(nir.add(red).add(L)).multiply(1 + L).rename('SAVI')
    
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename('NDMI')
    
    rendvi = rededge.subtract(red).divide(rededge.add(red)).rename('RENDVI')
    
    return image.addBands([ndvi, evi, gndvi, savi, ndwi, ndmi, rendvi])

def extract_s2_parameters(geometry: ee.Geometry, start_date: str, end_date: str) -> str:
    
    s2_median = preprocess_s2(geometry, start_date, end_date)
    grid = generate_tile_grid(s2_median, geometry)
    tiles_list = grid.toList(grid.size())
    total_tiles = grid.size().getInfo()
    print(f"Total tiles to process: {total_tiles}")

    indices = ['NDVI', 'EVI', 'GNDVI', 'SAVI', 'NDWI', 'NDMI', 'RENDVI']
    
    all_tiles_data = []

    for i in range(total_tiles):
        tile = ee.Feature(tiles_list.get(i))
        tile_geometry = tile.geometry()
        
        clipped_image = s2_median.clip(tile_geometry)
        indexed_image = compute_indices(clipped_image)
        image = indexed_image.reproject(crs='EPSG:4326', scale=10)
        
        pixels = image.sampleRectangle(region=tile_geometry, properties=indices, defaultValue=0)
        pixel_data = pixels.getInfo()
        
        ref_index = 'NDVI'
        grid = pixel_data['properties'][ref_index]
        grid_height = len(grid)
        grid_width = len(grid[0])
        total_pixels = grid_height * grid_width
        
        tile_data = {
            "tile_index": i,
            "geometry": tile_geometry.getInfo(),
            "grid_size": {"height": grid_height, "width": grid_width},
            "indices": {}
        }
        
        for index in indices:
            print(f"Processing Index: {index}")
            try:
                index_values = pixels.get(index).getInfo()
                if index_values:
                    flat_values = [val for row in index_values for val in row]
                    tile_data["indices"][index] = flat_values
                else:
                    raise Exception("No Data")
            except Exception as e:
                print(f"Error in index {index}: {e}")
                tile_data["indices"][index] = [None] * total_pixels
        
        all_tiles_data.append(tile_data)
    
    return json.dumps({"tiles": all_tiles_data})