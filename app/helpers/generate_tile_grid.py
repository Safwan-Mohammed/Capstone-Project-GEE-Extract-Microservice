import ee

def generate_tile_grid(image, aoi):
    PIXEL_SIZE = 1280
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