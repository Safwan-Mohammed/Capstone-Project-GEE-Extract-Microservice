import ee
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.helpers.generate_tile_grid import generate_tile_grid

def preprocess_s2(geometry: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    AOI = geometry
    START_DATE = start_date  
    END_DATE = end_date    

    CLOUD_FILTER, CLD_PRB_THRESH, NIR_DRK_THRESH, CLD_PRJ_DIST, BUFFER = 70, 70, 0.15, 1, 40

    def add_cloud_bands(img):
        try:
            cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
            is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
            return img.addBands(ee.Image([cld_prb, is_cloud]))
        except Exception as e:
            raise Exception(f"Error in add_cloud_bands: {e}")

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
            raise Exception(f"Error in add_shadow_bands: {e}")

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
            raise Exception(f"Error in add_cld_shdw_mask: {e}")

    def apply_cld_shdw_mask(img):
        try:
            not_cld_shdw = img.select('cloudmask').Not()
            return img.select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']).updateMask(not_cld_shdw)
        except Exception as e:
            raise Exception(f"Error in apply_cld_shdw_mask: {e}")
        
    def get_s2_sr_cld_col(aoi, start_date, end_date):
        try:
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
        except Exception as e:
            raise Exception(f"Error in get_s2_sr_cld_col: {e}")

    try:
        s2_sr_cld_col = get_s2_sr_cld_col(AOI, START_DATE, END_DATE)
        if s2_sr_cld_col is None:
            raise Exception("Failed to retrieve Sentinel-2 collection")
        s2_sr_clean = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)
        s2_sr_median = s2_sr_clean.median().clip(AOI)
        return s2_sr_median
    except Exception as e:
        raise Exception(f"Error in preprocess_s2: {e}")

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

def process_tile(tile_idx, tile, s2_median_with_indices, indices, month_name):
    try:
        tile_geometry = tile.geometry()
        clipped_image = s2_median_with_indices.clip(tile_geometry)
        samples = clipped_image.sample(
            region=tile_geometry,
            scale=10,
            geometries=True,
            numPixels=3000,
        ).getInfo()

        pixel_dict = {}
        for feat in samples['features']:
            props = feat['properties']
            coords = feat['geometry']['coordinates']
            coord_key = f'{coords[0]},{coords[1]}'
            if coord_key not in pixel_dict:
                pixel_dict[coord_key] = {}
            pixel_dict[coord_key][month_name] = {index: props.get(index, 0) for index in indices}
        return pixel_dict
    except Exception as e:
        raise Exception(f"Error processing tile {tile_idx}: {e}")

def get_monthly_ranges(start: str, end: str):
    start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
    months = []
    while start_dt <= end_dt:
        month_start = start_dt.replace(day=1)
        if start_dt.month == 12:
            month_end = start_dt.replace(year=start_dt.year + 1, month=1, day=1)
        else:
            month_end = start_dt.replace(month=start_dt.month + 1, day=1)
        months.append((month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"), month_start.strftime("%B")))
        start_dt = month_end
    return months

def process_month(geometry, month_start, month_end, month_name):
    indices = ['NDVI', 'EVI', 'GNDVI', 'SAVI', 'NDWI', 'NDMI', 'RENDVI']
    all_tiles_data = {}
    
    try:
        s2_median = preprocess_s2(geometry, month_start, month_end)
        if s2_median is None:
            raise Exception("Failed to preprocess Sentinel-2 data")

        s2_median_with_indices = compute_indices(s2_median)
        if s2_median_with_indices is None:
            raise Exception("Failed to compute indices")

        grid = generate_tile_grid(s2_median_with_indices, geometry)
        tiles_list = grid.toList(grid.size())
        total_tiles = grid.size().getInfo()
        tiles_to_process = total_tiles 

        max_workers = 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {
                executor.submit(process_tile, i, ee.Feature(tiles_list.get(i)), s2_median_with_indices, indices, month_name): i
                for i in range(tiles_to_process)
            }
            
            for future in as_completed(future_to_tile):
                tile_idx = future_to_tile[future]
                try:
                    tile_data = future.result()
                    all_tiles_data.update(tile_data)
                except Exception as e:
                    print(f"Tile {tile_idx} processing failed: {e}")
                    # If tile processing fails, populate with zeros for all coordinates in the tile
                    tile = ee.Feature(tiles_list.get(tile_idx))
                    tile_geometry = tile.geometry()
                    samples = s2_median_with_indices.sample(
                        region=tile_geometry,
                        scale=10,
                        geometries=True,
                        numPixels=3000,
                    ).getInfo()
                    for feat in samples['features']:
                        coords = feat['geometry']['coordinates']
                        coord_key = f'{coords[0]},{coords[1]}'
                        if coord_key not in all_tiles_data:
                            all_tiles_data[coord_key] = {}
                        all_tiles_data[coord_key][month_name] = {index: 0 for index in indices}

        # Check if any coordinates are missing data for this month
        if not all_tiles_data:
            # If no data is available for the entire month, populate all coordinates with zeros
            samples = s2_median_with_indices.sample(
                region=geometry,
                scale=10,
                geometries=True,
                numPixels=3000 * tiles_to_process,
            ).getInfo()
            for feat in samples['features']:
                coords = feat['geometry']['coordinates']
                coord_key = f'{coords[0]},{coords[1]}'
                if coord_key not in all_tiles_data:
                    all_tiles_data[coord_key] = {}
                all_tiles_data[coord_key][month_name] = {index: 0 for index in indices}

    except Exception as e:
        # If the entire month fails, populate all coordinates with zeros
        samples = ee.Image().sample(
            region=geometry,
            scale=10,
            geometries=True,
            numPixels=3000 * 10,  # Arbitrary large number to cover the area
        ).getInfo()
        for feat in samples['features']:
            coords = feat['geometry']['coordinates']
            coord_key = f'{coords[0]},{coords[1]}'
            if coord_key not in all_tiles_data:
                all_tiles_data[coord_key] = {}
            all_tiles_data[coord_key][month_name] = {index: 0 for index in indices}

    return all_tiles_data

def extract_s2_parameters(geometry: ee.Geometry, start_date: str, end_date: str) -> dict:
    """Endpoint to extract Sentinel-2 indices for all tiles using 4 processes and return as JSON."""
    try:
        all_data = {}
        for month_start, month_end, month_name in get_monthly_ranges(start_date, end_date):
            month_data = process_month(geometry, month_start, month_end, month_name)
            for coord, data in month_data.items():
                if coord not in all_data:
                    all_data[coord] = {}
                all_data[coord].update(data)
        
        return all_data
    
    except Exception as e:
        print(f"Error in extract_s2_parameters: {e}")
        raise Exception(f"Internal Server Error: {e}")