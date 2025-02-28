import ee
import math

def slope_correction(collection, TERRAIN_FLATTENING_MODEL, DEM, TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER):
    ninetyRad = ee.Image.constant(90).multiply(math.pi/180)

    def _volumetric_model_SCF(theta_iRad, alpha_rRad):
        nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
        denominator = (ninetyRad.subtract(theta_iRad)).tan()
        return nominator.divide(denominator)

    def _direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad):
        nominator = (ninetyRad.subtract(theta_iRad)).cos()
        denominator = alpha_azRad.cos().multiply((ninetyRad.subtract(theta_iRad).add(alpha_rRad)).cos())
        return nominator.divide(denominator)

    def _erode(image, distance):
        d = (image.Not().unmask(1).fastDistanceTransform(30).sqrt()
             .multiply(ee.Image.pixelArea().sqrt()))

        return image.updateMask(d.gt(distance))

    def _masking(alpha_rRad, theta_iRad, buffer):
        layover = alpha_rRad.lt(theta_iRad).rename('layover')
        shadow = alpha_rRad.gt(ee.Image.constant(-1)
                        .multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')
        mask = layover.And(shadow)
        if (buffer > 0):
            mask = _erode(mask, buffer)
        return mask.rename('no_data_mask')

    def _correct(image):
        bandNames = image.bandNames()
        geom = image.geometry()
        proj = image.select(1).projection()
        elevation = DEM.resample('bilinear').reproject(proj,None, 10).clip(geom)
        heading = ee.Terrain.aspect(image.select('angle')).reduceRegion(ee.Reducer.mean(), image.geometry(), 1000)
        heading = ee.Dictionary(heading).combine({'aspect': 0}, False).get('aspect')
        heading = ee.Algorithms.If(
            ee.Number(heading).gt(180),
            ee.Number(heading).subtract(360),
            ee.Number(heading)
        )
        
        theta_iRad = image.select('angle').multiply(math.pi/180)
        phi_iRad = ee.Image.constant(heading).multiply(math.pi/180)
        
        alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(math.pi / 180)

        aspect = ee.Terrain.aspect(elevation).select('aspect').clip(geom)
        
        aspect_minus = aspect.updateMask(aspect.gt(180)).subtract(360)
        
        phi_sRad = aspect.updateMask(aspect.lte(180)).unmask().add(aspect_minus.unmask()).multiply(-1).multiply(math.pi / 180)
          
        phi_rRad = phi_iRad.subtract(phi_sRad)

        alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

        alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

        gamma0 = image.divide(theta_iRad.cos())

        if (TERRAIN_FLATTENING_MODEL == 'VOLUME'):
            scf = _volumetric_model_SCF(theta_iRad, alpha_rRad)

        if (TERRAIN_FLATTENING_MODEL == 'DIRECT'):
            scf = _direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)

        gamma0_flat = gamma0.multiply(scf)

        mask = _masking(alpha_rRad, theta_iRad, TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER)
        output = gamma0_flat.mask(mask).rename(bandNames).copyProperties(image)
        output = ee.Image(output).addBands(image.select('angle'), None, True)

        return output.set('system:time_start', image.get('system:time_start'))
    return collection.map(_correct)