import ee
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.helpers.generate_tile_grid import generate_tile_grid

def extract_s1_parameters(geometry: ee.Geometry, start_date: str, end_date: str) -> dict:
    def add_vh_vv_ratio(image):
        ratio = image.expression('VH / VV', {
            'VH': image.select('VH'),
            'VV': image.select('VV')
        }).rename('VH_VV')
        return image.addBands(ratio)

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

    def process_month(month_start, month_end, month_name):
        s1_month = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.eq('resolution_meters', 10))
            .filterDate(month_start, month_end)
            .filterBounds(geometry)
            .map(add_vh_vv_ratio)
        )

        if s1_month.size().getInfo() == 0:
            return {}

        s1_median = s1_month.median().clip(geometry)

        # Generate tile grid from current month's image
        grid = generate_tile_grid(s1_median, geometry)
        tiles_list = grid.toList(grid.size())
        total_tiles = tiles_list.size().getInfo()

        def process_tile(tile_idx, tile):
            tile_geometry = tile.geometry()
            clipped = s1_median.clip(tile_geometry)

            sampled = clipped.sample(
                region=tile_geometry,
                scale=10,
                geometries=True,
                numPixels=3000, 
                seed=tile_idx
            )

            features = sampled.getInfo().get("features", [])

            month_pixels = {}
            for feat in features:
                props = feat["properties"]
                coords = feat["geometry"]["coordinates"]
                coord_key = f"{coords[0]},{coords[1]}"
                if coord_key not in month_pixels:
                    month_pixels[coord_key] = {}
                month_pixels[coord_key][month_name] = {
                    "VV": props.get("VV"),
                    "VH": props.get("VH"),
                    "VH_VV": props.get("VH_VV")
                }
            return month_pixels

        month_results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_tile, i, ee.Feature(tiles_list.get(i)))
                for i in range(total_tiles)
            ]

            for future in as_completed(futures):
                tile_data = future.result()
                for coord, data in tile_data.items():
                    if coord not in month_results:
                        month_results[coord] = {}
                    month_results[coord].update(data)

        return month_results

    # Combine all months' results
    all_data = {}
    for month_start, month_end, month_name in get_monthly_ranges(start_date, end_date):
        month_data = process_month(month_start, month_end, month_name)
        for coord, value in month_data.items():
            if coord not in all_data:
                all_data[coord] = {}
            all_data[coord].update(value)

    return all_data
