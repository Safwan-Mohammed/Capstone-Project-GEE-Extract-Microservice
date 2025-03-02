import ee
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.helpers.generate_tile_grid import generate_tile_grid

def extract_s1_parameters(geometry: ee.Geometry, start_date: str, end_date: str) -> dict:

    s1 = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT').filter(ee.Filter.eq('instrumentMode', 'IW')).filter(ee.Filter.eq('resolution_meters', 10)).filterDate(start_date, end_date).filterBounds(geometry)

    def add_vh_vv_ratio(image):
        vh_vv_ratio = image.expression('VH / VV', {'VH': image.select('VH'), 'VV': image.select('VV')}).rename('VH_VV_ratio')
        return image.addBands(vh_vv_ratio)

    def convert_to_db(image):
        vh_db = image.select('VH').log10().multiply(10).rename('VH_dB')
        vv_db = image.select('VV').log10().multiply(10).rename('VV_dB')
        vh_vv_db = image.select('VH_VV_ratio').log10().multiply(10).rename('VH_VV_dB')
        
        return image.addBands([vh_db, vv_db, vh_vv_db])

    s1_with_ratio = s1.map(add_vh_vv_ratio)
    s1_median = s1_with_ratio.median().clip(geometry)
    s1_median_db = convert_to_db(s1_median)

    grid = generate_tile_grid(s1_median_db, geometry)
    tiles_list = grid.toList(grid.size())
    indices = ['VH_dB', 'VV_dB', 'VH_VV_dB']

    def process_tile(tile_idx, tile, indices):
        tile_geometry = tile.geometry()
        clipped_image = s1_median_db.clip(tile_geometry)
        image = clipped_image.reproject(crs='EPSG:4326', scale=10)
        
        pixels = image.sampleRectangle(region=tile_geometry, properties=indices, defaultValue=0)
        pixel_data = pixels.getInfo()

        ref_index = 'VV_dB'
        grid = pixel_data['properties'][ref_index]
        grid_height = len(grid)
        grid_width = len(grid[0])
        total_pixels = grid_height * grid_width
        
        tile_data = {
            "tile_index": tile_idx,
            "geometry": tile_geometry.getInfo(),
            "grid_size": {"height": grid_height, "width": grid_width},
            "indices": {}
        }
        
        for index in indices:
            index_values = pixel_data['properties'][index]
            if index_values:
                flat_values = [val for row in index_values for val in row]
                tile_data["indices"][index] = flat_values
            else:
                tile_data["indices"][index] = [None] * total_pixels
        
        return tile_data

    total_tiles = tiles_list.size().getInfo()
    all_tiles_data = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_tile = {
            executor.submit(process_tile, i, ee.Feature(tiles_list.get(i)), indices): i
            for i in range(total_tiles)
        }
        
        tile_results = [None] * total_tiles
        for future in as_completed(future_to_tile):
            tile_idx = future_to_tile[future]
            tile_results[tile_idx] = future.result()
        all_tiles_data.extend(tile_results)

    return {"tiles": all_tiles_data}