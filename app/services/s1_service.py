import ee
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.helpers.generate_tile_grid import generate_tile_grid

def extract_s1_parameters(geometry: ee.Geometry, start_date: str, end_date: str) -> dict:
    s1 = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
    )

    def add_vh_vv_ratio(image):
        ratio = image.expression('VH / VV', {
            'VH': image.select('VH'),
            'VV': image.select('VV')
        }).rename('VH_VV')
        return image.addBands(ratio)


    s1_with_ratio = s1.map(add_vh_vv_ratio)
    s1_median = s1_with_ratio.median().clip(geometry)
    # s1_median = convert_to_db(s1_median)

    grid = generate_tile_grid(s1_median, geometry)
    tiles_list = grid.toList(grid.size())
    indices = ['VV', 'VH', 'VH_VV']

    def process_tile(tile_idx, tile, indices):
        tile_geometry = tile.geometry()
        clipped_image = s1_median.clip(tile_geometry)

        sampled = clipped_image.sample(
            region=tile_geometry,
            scale=10,
            geometries=True,  # Include coordinates
            numPixels=3000,
            seed=tile_idx     # For reproducibility
        )

        features = sampled.getInfo()["features"]

        pixel_dict = {}
        for feat in features:
            props = feat["properties"]
            coords = feat["geometry"]["coordinates"]
            coord_key = f'{coords[0]},{coords[1]}'
            pixel_dict[coord_key] = {index:props.get(index) for index in indices}
        
        return pixel_dict

    total_tiles = tiles_list.size().getInfo()
    all_pixels = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_tile = {
            executor.submit(process_tile, i, ee.Feature(tiles_list.get(i)), indices): i
            for i in range(total_tiles)
        }

        for future in as_completed(future_to_tile):
            tile_data = future.result()
            all_pixels.update(tile_data)

    return all_pixels
