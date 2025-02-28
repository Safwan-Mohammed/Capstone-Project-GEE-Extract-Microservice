import ee
import math

def boxcar(image, KERNEL_SIZE):
    bandNames = image.bandNames().remove('angle')
    kernel = ee.Kernel.square((KERNEL_SIZE/2), units='pixels', normalize=True)
    output = image.select(bandNames).convolve(kernel).rename(bandNames)
    return image.addBands(output, None, True)

def leefilter(image, KERNEL_SIZE):
    bandNames = image.bandNames().remove('angle')
    enl = 5
    eta = 1.0/math.sqrt(enl)
    eta = ee.Image.constant(eta)
    oneImg = ee.Image.constant(1)
    reducers = ee.Reducer.mean().combine(
                      reducer2= ee.Reducer.variance()
                      ,sharedInputs= True
                      )
    stats = (image.select(bandNames).reduceNeighborhood(
                      reducer= reducers
                          ,kernel= ee.Kernel.square(KERNEL_SIZE/2, 'pixels')
                              ,optimization= 'window'))
    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
    varBand = bandNames.map(lambda bandName:  ee.String(bandName).cat('_variance'))

    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)
    varx = (varz.subtract(z_bar.pow(2).multiply(eta.pow(2)))).divide(oneImg.add(eta.pow(2)))
    b = varx.divide(varz)
    new_b = b.where(b.lt(0), 0)
    output = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(image.select(bandNames)))
    output = output.rename(bandNames)
    return image.addBands(output, None, True)


def gammamap(image,KERNEL_SIZE): 
    enl = 5
    bandNames = image.bandNames().remove('angle')
    reducers = ee.Reducer.mean().combine( \
                      reducer2= ee.Reducer.stdDev(), \
                      sharedInputs= True
                      )
    stats = (image.select(bandNames).reduceNeighborhood( \
                      reducer= reducers, \
                          kernel= ee.Kernel.square(KERNEL_SIZE/2,'pixels'), \
                              optimization= 'window'))
    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
    stdDevBand = bandNames.map(lambda bandName:  ee.String(bandName).cat('_stdDev'))
        
    z = stats.select(meanBand)
    sigz = stats.select(stdDevBand)
    ci = sigz.divide(z)
    cu = 1.0/math.sqrt(enl)
    cmax = math.sqrt(2.0) * cu
    cu = ee.Image.constant(cu)
    cmax = ee.Image.constant(cmax)
    enlImg = ee.Image.constant(enl)
    oneImg = ee.Image.constant(1)
    twoImg = ee.Image.constant(2)

    alpha = oneImg.add(cu.pow(2)).divide(ci.pow(2).subtract(cu.pow(2)))
    q = image.select(bandNames).expression('z**2 * (z * alpha - enl - 1)**2 + 4 * alpha * enl * b() * z', { 'z': z,  'alpha':alpha,'enl': enl})
    rHat = z.multiply(alpha.subtract(enlImg).subtract(oneImg)).add(q.sqrt()).divide(twoImg.multiply(alpha))
    zHat = (z.updateMask(ci.lte(cu))).rename(bandNames)
    rHat = (rHat.updateMask(ci.gt(cu)).updateMask(ci.lt(cmax))).rename(bandNames)
    x = image.select(bandNames).updateMask(ci.gte(cmax)).rename(bandNames)  
    output = ee.ImageCollection([zHat,rHat,x]).sum()
    return image.addBands(output, None, True)

def RefinedLee(image):
    bandNames = image.bandNames().remove('angle')

    def inner(b):

        img = image.select([b]);
        weights3 = ee.List.repeat(ee.List.repeat(1,3),3);
        kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False);
  
        mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3);
        variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3);  
        sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]]);
  
        sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False);
  
        sample_mean = mean3.neighborhoodToBands(sample_kernel); 
        sample_var = variance3.neighborhoodToBands(sample_kernel);
  
        gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs();
        gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
        gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
        gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());
  
        max_gradient = gradients.reduce(ee.Reducer.max());
  
        gradmask = gradients.eq(max_gradient);
  
        gradmask = gradmask.addBands(gradmask);
  
        directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1);
        directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
        directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
        directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
        directions = directions.addBands(directions.select(0).Not().multiply(5));
        directions = directions.addBands(directions.select(1).Not().multiply(6));
        directions = directions.addBands(directions.select(2).Not().multiply(7));
        directions = directions.addBands(directions.select(3).Not().multiply(8));
  
        directions = directions.updateMask(gradmask);
  
        directions = directions.reduce(ee.Reducer.sum());  
  
        sample_stats = sample_var.divide(sample_mean.multiply(sample_mean));
  
        sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0]);
  
        rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4));
  
        diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0], [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]]);
  
        rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False);
        diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False);
  
        dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1));
        dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1));
  
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)));
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)));
  
        for i in range(1, 4):
            dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
            dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))

  
        dir_mean = dir_mean.reduce(ee.Reducer.sum());
        dir_var = dir_var.reduce(ee.Reducer.sum());
  
        varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))
  
        b = varX.divide(dir_var)
        result = dir_mean.add(b.multiply(img.subtract(dir_mean)))
  
        return result.arrayProject([0]).arrayFlatten([['sum']]).float()
    
    result = ee.ImageCollection(bandNames.map(inner)).toBands().rename(bandNames).copyProperties(image)
    
    return image.addBands(result, None, True) 



def leesigma(image,KERNEL_SIZE):
    Tk = ee.Image.constant(7) 
    sigma = 0.9
    enl = 4
    target_kernel = 3
    bandNames = image.bandNames().remove('angle')
   
    z98 = ee.Dictionary(image.select(bandNames).reduceRegion(
                reducer= ee.Reducer.percentile([98]),
                geometry= image.geometry(),
                scale=10,
                maxPixels=1e13
            )).toImage()
  

    brightPixel = image.select(bandNames).gte(z98)
    K = brightPixel.reduceNeighborhood(ee.Reducer.countDistinctNonNull()
            ,ee.Kernel.square(target_kernel/2)) 
    retainPixel = K.gte(Tk)
  
    eta = 1.0/math.sqrt(enl) 
    eta = ee.Image.constant(eta)
    reducers = ee.Reducer.mean().combine(reducer2= ee.Reducer.variance(),sharedInputs= True)
    stats = image.select(bandNames).reduceNeighborhood(reducer= reducers, kernel= ee.Kernel.square(target_kernel/2,'pixels'), optimization= 'window')
    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
    varBand = bandNames.map(lambda bandName:  ee.String(bandName).cat('_variance'))
        
    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)
    
    oneImg = ee.Image.constant(1)
    varx = (varz.subtract(z_bar.abs().pow(2).multiply(eta.pow(2)))).divide(oneImg.add(eta.pow(2)))
    b = varx.divide(varz)
    xTilde = oneImg.subtract(b).multiply(z_bar.abs()).add(b.multiply(image.select(bandNames)))
  
    LUT = ee.Dictionary({0.5: ee.Dictionary({'I1': 0.694,'I2': 1.385,'eta': 0.1921}),
                                 0.6: ee.Dictionary({'I1': 0.630,'I2': 1.495,'eta': 0.2348}),
                                 0.7: ee.Dictionary({'I1': 0.560,'I2': 1.627,'eta': 0.2825}),
                                 0.8: ee.Dictionary({'I1': 0.480,'I2': 1.804,'eta': 0.3354}),
                                 0.9: ee.Dictionary({'I1': 0.378,'I2': 2.094,'eta': 0.3991}),
                                 0.95: ee.Dictionary({'I1': 0.302,'I2': 2.360,'eta': 0.4391})});
  
    sigmaImage = ee.Dictionary(LUT.get(str(sigma))).toImage()
    I1 = sigmaImage.select('I1')
    I2 = sigmaImage.select('I2')
    nEta = sigmaImage.select('eta')
    I1 = I1.multiply(xTilde)
    I2 = I2.multiply(xTilde)
  
    mask = image.select(bandNames).gte(I1).Or(image.select(bandNames).lte(I2))
    z = image.select(bandNames).updateMask(mask)
  
    stats = z.reduceNeighborhood(reducer= reducers, kernel= ee.Kernel.square(KERNEL_SIZE/2,'pixels'), optimization= 'window')
        
    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)
    
    
    varx = (varz.subtract(z_bar.abs().pow(2).multiply(nEta.pow(2)))).divide(oneImg.add(nEta.pow(2)))
    b = varx.divide(varz)

    new_b = b.where(b.lt(0), 0)
    xHat = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(z))
  
    xHat = image.select(bandNames).updateMask(retainPixel).unmask(xHat)
    output = ee.Image(xHat).rename(bandNames)
    return image.addBands(output, None, True)

def MonoTemporal_Filter(coll,KERNEL_SIZE, SPECKLE_FILTER) :
    def _filter(image):    
       if (SPECKLE_FILTER=='BOXCAR'):
          _filtered = boxcar(image, KERNEL_SIZE)
       elif (SPECKLE_FILTER=='LEE'):
          _filtered = leefilter(image, KERNEL_SIZE)
       elif (SPECKLE_FILTER=='GAMMA MAP'):
          _filtered = gammamap(image, KERNEL_SIZE)
       elif (SPECKLE_FILTER=='REFINED LEE'):
          _filtered = RefinedLee(image)
       elif (SPECKLE_FILTER=='LEE SIGMA'):
          _filtered = leesigma(image, KERNEL_SIZE)
       return _filtered
    return coll.map(_filter)

def MultiTemporal_Filter(coll,KERNEL_SIZE, SPECKLE_FILTER,NR_OF_IMAGES):
    def Quegan(image) :
        def setresample(image):
                return image.resample()
            
        def get_filtered_collection(image):
            s1_coll = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT').filterBounds(image.geometry()).filter(ee.Filter.eq('instrumentMode', 'IW')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', ee.List(image.get('transmitterReceiverPolarisation')).get(-1))).filter(ee.Filter.Or(ee.Filter.eq('relativeOrbitNumber_stop', image.get('relativeOrbitNumber_stop')), ee.Filter.eq('relativeOrbitNumber_stop', image.get('relativeOrbitNumber_start')))).map(setresample)
      
            def check_overlap(_image):
                s1 = s1_coll.filterDate(_image.date(), _image.date().advance(1, 'day'))
                intersect = image.geometry().intersection(s1.geometry().dissolve(), 10)
                valid_date = ee.Algorithms.If(intersect.area(10).divide(image.geometry().area(10)).gt(0.95), _image.date().format('YYYY-MM-dd'))
                return ee.Feature(None, {'date': valid_date})
            
            dates_before = s1_coll.filterDate('2014-01-01', image.date().advance(1, 'day')).sort('system:time_start', False).limit(5*NR_OF_IMAGES).map(check_overlap).distinct('date').aggregate_array('date')
    
            dates = ee.List(ee.Algorithms.If(dates_before.size().gte(NR_OF_IMAGES), dates_before.slice(0, NR_OF_IMAGES), s1_coll.filterDate(image.date(), '2100-01-01').sort('system:time_start', True).limit(5*NR_OF_IMAGES).map(check_overlap).distinct('date').aggregate_array('date').cat(dates_before).distinct().sort().slice(0, NR_OF_IMAGES)))
    
            return ee.ImageCollection(dates.map(lambda date: s1_coll.filterDate(date, ee.Date(date).advance(1,'day')).toList(s1_coll.size())).flatten())
      
        s1 = get_filtered_collection(image)
  
        bands = image.bandNames().remove('angle')
        s1 = s1.select(bands)
        meanBands = bands.map(lambda bandName: ee.String(bandName).cat('_mean'))
        ratioBands = bands.map(lambda bandName: ee.String(bandName).cat('_ratio'))
        count_img = s1.reduce(ee.Reducer.count())

        def inner(image):
            if (SPECKLE_FILTER=='BOXCAR'):
                _filtered = boxcar(image, KERNEL_SIZE).select(bands).rename(meanBands) 
            elif (SPECKLE_FILTER=='LEE'):
                _filtered = leefilter(image, KERNEL_SIZE).select(bands).rename(meanBands)
            elif (SPECKLE_FILTER=='GAMMA MAP'):
                _filtered = gammamap(image, KERNEL_SIZE).select(bands).rename(meanBands)
            elif (SPECKLE_FILTER=='REFINED LEE'):
                _filtered = RefinedLee(image).select(bands).rename(meanBands)
            elif (SPECKLE_FILTER=='LEE SIGMA'):
                _filtered = leesigma(image, KERNEL_SIZE).select(bands).rename(meanBands)
    
            _ratio = image.select(bands).divide(_filtered).rename(ratioBands) 
            return _filtered.addBands(_ratio)

        isum = s1.map(inner).select(ratioBands).reduce(ee.Reducer.sum())
        filtered = inner(image).select(meanBands)
        divide = filtered.divide(count_img)
        output = divide.multiply(isum).rename(bands)

        return image.addBands(output, None, True)
    return coll.map(Quegan)