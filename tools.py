#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version 0.1, Apr 23 2018

@author: fabio (dot) oriani (at) protonmail (dot) com
"""
#%%
#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import rankdata
#from scipy.stats import norm
from scipy.ndimage.measurements import label
#from scipy.interpolate import interp1d
from skimage.measure import regionprops
from itertools import product, permutations
#from PIL import Image
from scipy.ndimage import rotate
from shapefile import Reader
from datetime import datetime, timedelta
from scipy.signal import fftconvolve
from scipy.ndimage import distance_transform_edt
from scipy import interpolate
import matplotlib.path as mplp
#from pickle import dump,load
from osgeo import gdal
from osgeo import osr
import struct
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import shapefile
import cmasher as cmr

try:
	import mkl_fft as fft
except ImportError:
	try:
		import pyfftw.interfaces.numpy_fft  as fft
	except ImportError:
		import numpy.fft as fft

def plot_shp(fname,col="default",enc='utf-8'): # import and plot a shapefile (.shp) given the local/global file path. col = color (optional)
    ch=Reader(fname,encoding=enc)
    n=0
    for shape in list(ch.iterShapes()): # iterate oavell all shapes
        npoints=len(shape.points) # total points
        nparts = len(shape.parts) # total parts
        if nparts == 1:
           x_lon = np.zeros((len(shape.points),1))
           y_lat = np.zeros((len(shape.points),1))
           for ip in range(len(shape.points)):
               x_lon[ip] = shape.points[ip][0]
               y_lat[ip] = shape.points[ip][1]
           n=n+1
           if n==1 and col=="default":  # fetch color from first part
               p=plt.plot(x_lon,y_lat)
               col=p[0].get_color()
           elif n==1:
               p=plt.plot(x_lon,y_lat,color=col)
           else:
               plt.plot(x_lon,y_lat,color=col)
    
        else: # loop over parts of each shape, plot separately
           for ip in range(nparts): # loop over parts, plot separately
               i0=shape.parts[ip]
               if ip < nparts-1:
                  i1 = shape.parts[ip+1]-1
               else:
                  i1 = npoints
    
               seg=shape.points[i0:i1+1]
               x_lon = np.zeros((len(seg),1))
               y_lat = np.zeros((len(seg),1))
               for ip in range(len(seg)):
                   x_lon[ip] = seg[ip][0]
                   y_lat[ip] = seg[ip][1]
               n=n+1
               if n==1 and col=="default":
                   p=plt.plot(x_lon,y_lat)
                   col=p[0].get_color()
               elif n==1:
                   p=plt.plot(x_lon,y_lat,color=col)
               else:
                   plt.plot(x_lon,y_lat,color=col)

def shp_to_line(fname,enc='utf-8'): # return a (list of) line array from a (multiple-shape) shapefile 
    ch=Reader(fname,encoding=enc)
    line=[] # output list of lines arrays [N,(x,y)]
    for shape in list(ch.iterShapes()): # iterate over all shapes
        npoints=len(shape.points) # total points
        nparts = len(shape.parts) # total parts
        # loop over parts of each shape, plot separately
        for ip in range(nparts): # loop over parts, plot separately
           i0=shape.parts[ip]
           if ip < nparts-1:
              i1 = shape.parts[ip+1]-1
           else:
              i1 = npoints
           seg=shape.points[i0:i1+1]
           x_lon = np.zeros((len(seg),1))
           y_lat = np.zeros((len(seg),1))
           for ip in range(len(seg)):
               x_lon[ip] = seg[ip][0]
               y_lat[ip] = seg[ip][1]
           line.append(np.hstack((x_lon,y_lat)))
    return line           

def shp_to_mask(fname,xv,yv): # return a mask from a shapefile over a given grid xv,xy (from meshgrid)
    line=shp_to_line(fname)
    mask=np.empty_like(xv)*0
    for n in range(len(line)):
        mpath = mplp.Path(line[n])
        points = np.array((xv.flatten(), yv.flatten())).T
        mask = mask+mpath.contains_points(points).reshape(xv.shape)
    mask=mask>0
    return mask
    ## to test the result
    # plt.scatter(xv,yv,c=mask)
    # ft.plot_shp('data/CNTR_RG_60M_2020_4326.shp',col="brown") # world boundaries

def write_point_shp(base_name,x,y,field_name,field_type,records,esri_code=None):
    """
    WRITES A POINT SHAPEFILE (.shp .shx .dbf .prj)
    base_name [str] the base name for the files (no extension)
    x [num vector] vector of x / lon coordinates
    y [num vector] vector of y / lat coordinates
    field_name [str list] field names
    field_type [str list] field types: 'L' boolean, 'N' numerical , 'C' custom
    records [iterables list] list of iterables (vectors or lists) containing the records. len(list)=N for N fields, and len(list[i])=M for M records (points)
    esri_code [int] ESRI code of the used crs, if not given the .prj file is not created (unreferenced shape file)
    
    """
        
    # create shapefile object
    w = shapefile.Writer(base_name + '.shp')
    # write fields
    for i in range(len(field_name)):
        w.field(field_name[i],field_type[i]) # field
    # record points
    r=np.vstack(records) # array from records list
    for i in range(len(records[0])):
        w.point(x[i],y[i])
        w.record(*r[:,i])
    w.close()
    
    # georeferenziation file
    if esri_code!=None:
        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromEPSG(esri_code)
        spatialRef.MorphToESRI()
        file = open(base_name + '.prj', 'w')
        file.write(spatialRef.ExportToWkt())
        file.close()

def read_geotiff(fname):
    # import geotiff image with georef class R, usage: im,R=read_geotiff(filename)
    dataset = gdal.Open(fname, gdal.GA_ReadOnly)
    nb=dataset.RasterCount # number of bands
    xs=dataset.RasterXSize # raster x size
    ys=dataset.RasterYSize # raster y size
    im=np.empty([ys,xs,nb])*np.nan
    for i in range(nb):
        band = dataset.GetRasterBand(i+1)
        raster = band.ReadRaster(xoff=0, yoff=0, # origin 
                               xsize=xs, ysize=ys, # extension
                               buf_xsize=xs, buf_ysize=ys, # buffer = extension for full resolution
                               buf_type=gdal.GDT_Float32)
        raster_float = struct.unpack('f'*xs*ys, raster)
        im[:,:,i]=np.reshape(raster_float,[ys,xs])
    im=np.squeeze(im)
    geotr = dataset.GetGeoTransform()
    dataset=None
    band=None
    class make_R:
        def __init__(self,im,geotr):
            self.ncols=im.shape[1]
            self.nrows=im.shape[0]
            self.cellsize=geotr[1]
            self.xllc=geotr[0]
            self.yllc=geotr[3]-self.cellsize*self.nrows
    R=make_R(im,geotr)
    return im,R

def write_tiff(filename,im,dtype="Float32"):
    driver = gdal.GetDriverByName("GTiff")   
    if np.ndim(im)==3:
        outdata = driver.Create(filename, np.shape(im)[1], np.shape(im)[0], np.shape(im)[2], gdal.GDT_Float32)
        for i in range(np.shape(im)[2]):
            outdata.GetRasterBand(i+1).WriteArray(im[:,:,i].astype("Float32"))
        #    outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    else:
        outdata = driver.Create(filename, np.shape(im)[1], np.shape(im)[0], 1, gdal.GDT_Float32)
        outdata.GetRasterBand(1).WriteArray(im.astype(dtype))
    outdata.FlushCache() ##saves to disk!!
    outdata=None

def write_geotiff(filename,im,xmin,ymax,xres,yres,epsg=None,dtype=None,nodata_val=None,band_name=None):
    # nodata_val, band_name should be lists/array even if containing one value
    if dtype==None: # set data type from given format
        dtype=im.dtype.name
    if np.ndim(im)<3:
        nb=1
        im=im[:,:,None]
    else:
        nb=np.shape(im)[2]
    
    driver =  gdal.GetDriverByName('GTiff') # generate driver    
    outdata = driver.Create(filename, np.shape(im)[1], np.shape(im)[0], nb, gdal.GetDataTypeByName(dtype))
    for i in range(nb): # write data
        outdata.GetRasterBand(i+1).WriteArray(im[:,:,i].astype(dtype))
        outdata.GetRasterBand(i+1).SetNoDataValue(nodata_val[i])
        outdata.GetRasterBand(i+1).SetDescription(band_name[i])
        
    # GEOREFERENTIATION
     # coords
    geotransform = (xmin, # x-coordinate of the upper-left corner of the upper-left pixel.
                    xres, # W-E pixel resolution / pixel width.
                    0, #row rotation (typically zero).
                    ymax, # y-coordinate of the upper-left corner of the upper-left pixel.
                    0, # column rotation (typically zero).
                    -yres) # N-S pixel resolution / pixel height (negative value for a north-up image).
    outdata.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(epsg)                # set CRS from EPSG code 
    outdata.SetProjection(srs.ExportToWkt())
    # save data
    outdata.FlushCache()
    outdata=None

def extract_grid_points(V, # grid values
                        xll, # lower left grid CENTER x coord
                        yll, # lower left grid CENTER y coord (increasing upward)
                        resx, # x resolution
                        resy, # y resolution
                        px, # query point x coordinate [numpy vector or list]
                        py, # query point y coordinate [numpy vector or list]
                        plot_xy = False, # if true plot all points and grids
                        ):
    # extract pixels values from grid given some of query points (e.g. dataset coordinates).
    # it returns an nan if a point is out of the grid
    nx = V.shape[1]
    ny = V.shape[0]
    vx = np.arange(xll, xll + nx*resx, resx)
    vy = np.arange(yll, yll + ny*resy, resy)
    pv = np.zeros_like(px).astype('float') # grid to point data
    pvx = np.copy(pv) # grid to point data x coord
    pvy = np.copy(pv) # grid to point data y coord
    for i in range(len(pv)):
        if px[i]<vx[0] or px[i]>vx[-1] or py[i]<vy[0] or py[i]>vy[-1]:
            pv[i] = np.nan
            pvx[i] = np.nan
            pvy[i] = np.nan
        else:        
            indx = np.argmin(np.abs(px[i]-vx))
            indy = np.argmin(np.abs(py[i]-vy))
            pv[i] = np.copy(V[ny-indy-1,indx])
            pvx[i] = np.copy(vx[indx])
            pvy[i] = np.copy(vy[indy])
    
    # plot
    if plot_xy ==True:
      Gx, Gy = np.meshgrid(vx, vy, sparse=False, indexing='xy')
      plt.figure()
      plt.imshow(V,extent=[xll-resx/2,xll-resx/2+resx*nx,yll-resy/2,yll-resy/2+resy*ny],label='grid')
      plt.scatter(px,py,facecolors='white',edgecolors='green',s=40,label='query points')
      plt.scatter(pvx,pvy,marker='s',edgecolors='red',c=pv,s=40,label='extracted grids')
      plt.colorbar()
      plt.legend()
    
    return pv

def point_vario(x,y,z,nlag,lagseq='lin',q=0.99,maxlag=None):
    # x = x coordinate
    # y = y coordinate
    # z = z coordinate
    # nlag = number of lags (scalar)
    # lagseq = type of lag sequence linear or log
    # q = quantile value of the distance among points to put as max lag limit
    # maxlag = given max lag limit (if given, q is not used)
    
    X=np.repeat(x[:,None],len(x),axis=1)
    dX=X-np.transpose(X)
    del X
    Y=np.repeat(y[:,None],len(y),axis=1)
    dY=Y-np.transpose(Y)
    del Y
    D=np.sqrt(dX*dX+dY*dY)
    del dX,dY
    Z=np.repeat(z[:,None],len(z),axis=1)
    dZ=Z-np.transpose(Z)
    del Z
    if maxlag==None:
        maxlag=np.quantile(D,q)
    if lagseq=="lin":
        lags=np.arange(0,maxlag,maxlag/nlag)
    else:
        minlag=np.ceil(np.log10(np.min(D[D>0])))
        lags=np.logspace(minlag,np.log10(maxlag),num=nlag)
        lags=np.append(lags,0)
    lags=np.unique(lags)
    
    v=np.empty(len(lags)-1)*np.nan
    s=np.empty(len(lags)-1)*np.nan
    for i in range(len(lags)-1):
        dZtmp=dZ[np.logical_and(D>lags[i],D<=lags[i+1])] # select couples for i-th lag
        v[i]=np.nanmean(np.power(dZtmp,2))/2
        s[i]=np.nanstd(np.power(dZtmp,2))/2
    lags=(lags[:-1]+lags[1:])/2
    return lags, maxlag, v, s # lags, maxlag, variogram,  vario standard deviation


# variogram model functions: d: x data, return f(x) 
def linVario(d, slope, nugget):
    return slope * d + nugget

def powerVario(d,scale,exponent,nugget):
    return scale * d ** exponent + nugget

def gaussVario(d, psill, range_, nugget):
    return psill * (1 - np.exp(-(d ** 2.0) / (range_ * 4.0 / 7.0) ** 2.0)) + nugget

def spheriVario(d, psill, range_, nugget):
    return np.piecewise(
        d,
        [d <= range_, d > range_],
        [
            lambda x: psill
            * ((3.0 * x) / (2.0 * range_) - (x ** 3.0) / (2.0 * range_ ** 3.0))
            + nugget,
            psill + nugget,
        ],
    )

def expVario(d, psill, range_, nugget): 
    return psill * (1.0 - np.exp(-d / (range_ / 3.0))) + nugget


def vario_optim(obs_x,obs_y,obs_data,nlags=20,q=1,lagseq='lin',plot=False): 
    # automatic choice of variogram model and optimize params
    
    # SAMPLE VARIOGRAM
    lags, maxlag, v, s=point_vario(obs_x,obs_y,obs_data,nlags,q=q,lagseq=lagseq)
    ind = np.logical_not(np.isnan(v))
    lags=lags[ind]
    v=v[ind]
    
    # inizialize output params
    fp=[] # fitted params
    mv=[] # variogram values
    err=[] # error
    mn=[] # model name
    
    
    # LINEAR MODEL
    try:
        # optimize params: slope,nugget
        initialParameters = np.array([0, v[0]])
        lowerBounds = (0,0)#(-np.Inf,-np.Inf)
        upperBounds = (np.Inf,np.Inf)
        parameterBounds = [lowerBounds, upperBounds]
        fittedParams, pcov = curve_fit(linVario, lags, v, initialParameters, bounds = parameterBounds) # fitting
        slope, nugget = fittedParams # fitted params
        fp.append(np.copy(fittedParams))
        vv=linVario(lags,slope, nugget) # # fitted function
        mv.append(vv) # fitted function
        err.append(np.sqrt(np.mean((vv-v)*(vv-v)))) # RMSE
        mn.append('linear')
    except: 
        pass
    
    # POWER MODEL
    try:
        # optimize params: scale, exponent, nugget
        initialParameters = np.array([0, 1,v[0]])
        lowerBounds = (0,0,0) #(-np.Inf,-np.Inf,-np.Inf)
        upperBounds = (np.Inf,2,np.Inf)
        parameterBounds = [lowerBounds, upperBounds]
        fittedParams, pcov = curve_fit(powerVario, lags, v, initialParameters, bounds = parameterBounds) # fitting
        scale, exponent, nugget = fittedParams # fitted params
        fp.append(np.copy(fittedParams))
        vv=powerVario(lags,scale,exponent,nugget) # fitted function
        mv.append(vv) # fitted function
        err.append(np.sqrt(np.mean((vv-v)*(vv-v)))) # RMSE
        mn.append('power')
    except: 
        pass
    
    # GAUSSIAN MODEL
    try:
        # optimize params: psill, range_, nugget
        initialParameters = np.array([np.var(obs_data)/2,max(lags)/2,v[0]])
        lowerBounds = (0,0,0)
        upperBounds = (np.var(obs_data), max(lags), np.var(obs_data))
        parameterBounds = [lowerBounds, upperBounds]
        fittedParams, pcov = curve_fit(gaussVario, lags, v, initialParameters, bounds = parameterBounds) # fitting
        psill, range_, nugget = fittedParams # fitted params
        fp.append(np.copy(fittedParams))
        vv=gaussVario(lags,psill, range_, nugget) # fitted function
        mv.append(vv) # fitted function
        err.append(np.sqrt(np.mean((vv-v)*(vv-v)))) # RMSE
        mn.append('gaussian')
    except:
        pass
    
    # SPHERICAL MODEL
    try:       
        # optimize params: psill, range_, nugget
        initialParameters = np.array([np.var(obs_data)/2,max(lags)/2,v[0]])
        lowerBounds = (0,0,0)
        upperBounds = (np.var(obs_data), max(lags), np.var(obs_data))
        parameterBounds = [lowerBounds, upperBounds]
        fittedParams, pcov = curve_fit(spheriVario, lags, v, initialParameters, bounds = parameterBounds) # fitting
        psill, range_, nugget = fittedParams # fitted params
        fp.append(np.copy(fittedParams))
        vv=gaussVario(lags,psill, range_, nugget) # fitted function
        mv.append(vv) # fitted function
        err.append(np.sqrt(np.mean((vv-v)*(vv-v)))) # RMSE
        mn.append('spherical')
    except:
        pass
    
    # EXPONENTIAL MODEL
    try:
        # optimize params: psill, range_, nugget
        initialParameters = np.array([np.var(obs_data)/2,max(lags)/2,v[0]])
        lowerBounds = (0,0,0)
        upperBounds = (np.var(obs_data), max(lags), np.var(obs_data))
        parameterBounds = [lowerBounds, upperBounds]
        fittedParams, pcov = curve_fit(expVario, lags, v, initialParameters, bounds = parameterBounds) # fitting
        psill, range_, nugget = fittedParams # fitted params
        fp.append(np.copy(fittedParams))
        vv=expVario(lags,psill, range_, nugget)
        mv.append(vv) # fitted function
        err.append(np.sqrt(np.mean((vv-v)*(vv-v)))) # RMSE
        mn.append('exponential')
    except:
        pass
    
    if len(mn)==0:
        raise Exception('No model could fit the sample variogram!')
    
    # PLOT
    if plot==True:
        plt.figure()
        plt.scatter(lags,v,label='sample variogram')
        for i in range(len(mn)):
            plt.plot(lags,mv[i],label='%s , RMSE=%.3f' % (mn[i],err[i]))
        plt.xlabel('lag [cells]')
        plt.ylabel('v(lag)')
        plt.title('Sample variogram and model')
        plt.legend()
    
    ind=np.argmin(err)
    mtype=mn[ind]
    mparams=fp[ind]
    if mtype=='gaussian' or mtype=='exponential' or mtype=='spherical':
        mparams = {'psill':mparams[0],'range':mparams[1],'nugget':mparams[2]}
    elif mtype=='linear':
        mparams = {'slope':mparams[0],'nugget':mparams[1]}
    elif mtype=='power':
        mparams = {'scale': mparams[0], 'exponent':mparams[1], 'nugget': mparams[2]}
    return mtype, mparams 

# reliability plot
def reliability(p,o,q,plot=False):
    """
    RELIABILITY
    Given a vector of prediction probabibilites p [0-1] and a respective occurrences o [boolean], compute
    the frequency of occurrence for q equally spaced classes of probability. A prediction
    is considered reliable if for a probability class q_i (e.g. 50%), the occurrence frequency o_i is
    ~= q_i.
    
    USAGE:
    f,qc=reliability(p,o,q) gives the (conditional) occurrence frequency vector for q probabilty classes and 
      prediction vector p.
    
    f,qc=reliability(p,o,q,plot=True) also output the realiability plot of f vs q
    
    INPUT:
        p [float in [0-1]] = Nx1 vector of predicted probability
        o [boolean] = Nx1 correspondent vector of occurrences for each prediction
        q [int] = number of probability classes used to group the predictions
    
    OUTPUT:
        f [float in [0-1]] = qx1 occurrence freqeuency for each probability class, derived from o
        qc = qx1 vector of probability class centers
    
    """
   
    qb=np.linspace(0,1,q+1) # class bounds
    qc=(qb[1:]+qb[:-1])/2 # class centers
    f=np.empty_like(qc)*np.nan
    for i in range(len(qc)): # computin occurrence frequency
        o_tmp=o[np.logical_and(p>=qb[i],p<=qb[i+1])]
        f[i]=np.sum(o_tmp)/len(o_tmp)
    
    if plot==True:
        plt.plot(qc,f,'-o')
        plt.xlabel('prediction probability')
        plt.ylabel('occurrence probability')
        plt.axis('square')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot([0,1],[0,1],'--')
        plt.show()
    return qc,f