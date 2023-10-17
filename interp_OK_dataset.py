#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
font = {'family' : 'DejaVu Sans',
        'size'   : 10}
rc('font', **font)
import fabio_tools as ft
from pykrige import OrdinaryKriging
from osgeo import osr
#import zipfile
import os
from joblib import Parallel,delayed

#%% IMPORT DATA

# database
pcount=np.genfromtxt("data/counts.csv",dtype="float",delimiter=";",usecols=3,skip_header=1) # pollen count
psample=np.genfromtxt("data/counts.csv",dtype="str",delimiter=";",usecols=0,skip_header=1) # sample name (linked to sname)
pname=np.genfromtxt("data/counts.csv",dtype="str",delimiter=";",usecols=2,skip_header=1) # pollen name (linked to poll)
lon=np.genfromtxt("data/samples_coords.csv",delimiter=",",usecols=1,skip_header=1) # coords
lat=np.genfromtxt("data/samples_coords.csv",delimiter=",",usecols=2,skip_header=1)

# possible values lists
sname=np.genfromtxt("data/samples_coords.csv",dtype="str",delimiter=",",usecols=0,skip_header=1) # sample name list
#poll=np.genfromtxt("data/poll_groupid.csv",delimiter=";",dtype="str",usecols=1,skip_header=1) # pollen name list
poll=np.genfromtxt("data/EMPD2_count_v1.csv",delimiter=",",dtype="str",usecols=1,skip_header=4) # pollen name list
#poll_gr_id=np.genfromtxt("data/poll_groupid.csv",delimiter=";",dtype="str",usecols=2,skip_header=1) # pollen name list
poll_gr_id=np.genfromtxt("data/EMPD2_count_v1.csv",delimiter=",",dtype="str",usecols=2,skip_header=4) # pollen name list
groupidok=np.array(['DWAR','HERB','LIAN','PALM','SUCC','TRSH','UPHE']) # selected group_ids

# transform data from EPSG:4326  WGS84 to EPSG:3034 ETRS89-extended / LCC Europe
crs_in = osr.SpatialReference()
crs_out = osr.SpatialReference()
crs_in.ImportFromEPSG(4326)
crs_out.ImportFromEPSG(3034)
transform = osr.CoordinateTransformation(crs_in, crs_out)
for i in range(len(lon)):
    lat[i],lon[i] = transform.TransformPoint(lat[i],lon[i])[0:2]

# remove samples with nan coordinates
coords=np.hstack([lat[:,None],lon[:,None]]).sum(axis=1)
ind=np.isfinite(coords)
sname=sname[ind]
lat=lat[ind]
lon=lon[ind]

# consider samples with the same coordinates as a unique sample
coords=np.hstack([lat[:,None],lon[:,None]])
coords1, ind1,ind_inv=np.unique(coords,axis=0,return_index=True,return_inverse=True)
sname1=sname[ind1].copy()
lat1=lat[ind1].copy()
lon1=lon[ind1].copy()
psample2=np.copy(psample)
for i in range(len(psample)):
    ind2=np.nonzero(psample[i]==sname)[0]
    if ind2.size!=0:
        psample2[i]=sname1[ind_inv[ind2]][0]
psample=psample2.copy()
sname=sname1.copy()
lat=lat1.copy()
lon=lon1.copy()


# select pollen types belonging to selected groups
p_list=[]
p_list_gr_id=[]
for i in range(len(poll_gr_id)):
    if np.in1d(poll_gr_id[i],groupidok):
        p_list.append(poll[i])
        p_list_gr_id.append(poll_gr_id[i])
        
p_list,uind=np.unique(p_list,return_index=True)
p_list_gr_id=np.array(p_list_gr_id)
p_list_gr_id=p_list_gr_id[uind]
p_list=np.delete(p_list,[0])
p_list_gr_id=np.delete(p_list_gr_id,[0])

# group to consolidated names in both sample dataset and list
#lut=np.genfromtxt("data/accvar-to-consolidated_name.txt",dtype='str',delimiter=',',skip_footer=455)
lut=np.genfromtxt("data/EMPD2_LegacyName.csv",dtype='str',delimiter=',',skip_header=1,usecols=[0,2])
isin_pname=np.in1d(pname,lut[:,0])
for i in range(len(pname)):
    if isin_pname[i]:
        pname[i]=lut[pname[i]==lut[:,0],1][0]

isin_plist=np.in1d(p_list,lut[:,0])
for i in range(len(p_list)):
    if isin_plist[i]:
        p_list[i]=lut[p_list[i]==lut[:,0],1][0]
        
isin_lut2=np.in1d(p_list,lut[:,1])
p_list=np.delete(p_list,np.logical_not(isin_lut2))
p_list=np.unique(p_list)
p_list=np.delete(p_list,np.in1d(p_list,['Delete','Diospyros','Libanotis']))

#%% MESHGRID FOR INTERPOLATION

# in ETRS89-extended
xst=2006492#-10.5#5#-15
xen=6845416#20#+56
yst=462152  #31
yen=5218004 #55#73

res=25000 # metres
x=np.arange(xst,xen,res).astype("float")
y=np.arange(yst,yen,res).astype("float")
xv, yv = np.meshgrid(x, y)

#%% plot
f=plt.figure()
ft.plot_shp('data/CNTR_RG_60M_2020_3034.shp',col="brown") # world boundaries
plt.scatter(xv,yv,marker='.',label='interpolation grid',s=1) # grid
plt.scatter(lon,lat,label='pollen') # pollen samples
deltax=(xen-xst)*0.05
deltay=deltax
plt.axis('equal')
plt.xlim([xst-deltax,xen+deltax])
plt.ylim([yen+deltay,yst-deltay])
plt.legend()
plt.gca().invert_yaxis()

# MASK for GRID
mask=ft.shp_to_mask('data/CNTR_RG_60M_2020_3034.shp',xv,yv)
plt.scatter(xv,yv,c=mask,s=1)
ft.plot_shp('data/CNTR_RG_60M_2020_3034.shp',col="brown") # world boundaries
plt.xlim([xst-deltax,xen+deltax])
plt.ylim([yen+deltay,yst-deltay])
plt.legend()
plt.gca().invert_yaxis()
plt.axis('equal')


# add a slight delta [+-10 m] to data coordinates too close to grid points
for i in range(len(lon)):
    if np.any(abs(lon[i]-x)<1):
        lon[i]=lon[i]+np.random.rand()*10-5
    if np.any(abs(lat[i]-x)<1):
        lat[i]=lat[i]+np.random.rand()*10-5
    


#%% DEFINE SAVE-FIGURE FUNCTIONS
# parallerl/meridians lines
par1lon=np.arange(-15,50)
par1lat=np.zeros_like(par1lon)+40
par2lon=np.arange(-25,78)
par2lat=np.zeros_like(par2lon)+60
mer1lat=np.arange(28.3,75)
mer1lon=np.zeros_like(mer1lat)+0
mer2lat=np.arange(26.6,74)
mer2lon=np.zeros_like(mer2lat)+30
for i in range(len(par1lon)):
    par1lat[i],par1lon[i] = transform.TransformPoint(par1lat[i].astype('float'),par1lon[i].astype('float'))[0:2]
for i in range(len(par2lon)):
    par2lat[i],par2lon[i] = transform.TransformPoint(par2lat[i].astype('float'),par2lon[i].astype('float'))[0:2]
for i in range(len(mer1lon)):
    mer1lat[i],mer1lon[i] = transform.TransformPoint(mer1lat[i].astype('float'),mer1lon[i].astype('float'))[0:2]
for i in range(len(mer2lon)):
    mer2lat[i],mer2lon[i] = transform.TransformPoint(mer2lat[i].astype('float'),mer2lon[i].astype('float'))[0:2]

def plot_save_fig(pred,ss,obs_x,obs_y,obs_data,OK,pname_tmp):
    
    try:
        os.mkdir('export/dataset/' + pname_tmp)
    except:
        pass
    
    # indicator kriging mean
    fig, ax = plt.subplots(2, 2, figsize=([12.55,  9.5 ]))
    plt.axes(ax[0,0])
    plt.imshow(pred,extent=(x[0],x[-1],y[-1],y[0]),vmin=0,vmax=1) # interpolation
    plt.scatter(obs_x,obs_y,c=obs_data,edgecolors="gray",linewidths=0.5,vmin=0,vmax=1,s=10,label="pollen samples") # samples
    plt.xlim([xst-deltax,xen+deltax*2.5]) # figure limits
    plt.ylim([yst-deltay*1.2,yen+deltay])
    plt.colorbar() # colorbar
    plt.title('a) ' + pname_tmp + ' - Occurrence probability (Kriging mean)')
    plt.legend(loc='upper left')
    plt.xlabel('easting [m]')
    plt.ylabel('northing [m]')
    ax[0,0].ticklabel_format(style='scientific',scilimits=[0,3],useMathText=True)
    ax[0,0].text(1.1e6,-50300,'$\\times\\mathdefault{10^{6}}\\mathdefault{}$')
    ax[0,0].yaxis.get_offset_text().set_visible(False)
    ax[0,0].xaxis.get_offset_text().set_visible(False)
    
    # parallels and meridians
    plt.plot(par1lon,par1lat,'--',linewidth=0.5,color="gray") 
    plt.text(par1lon[0],par1lat[0],'40°N',rotation=-15.0,fontsize=5.5,color='grey')
    plt.plot(par2lon,par2lat,'--',linewidth=0.5,color="gray") 
    plt.text(par2lon[0],par2lat[0],'60°N',rotation=-15.0,fontsize=5.5,color='grey')
    plt.plot(mer1lon,mer1lat,'--',linewidth=0.5,color="gray") 
    plt.text(mer1lon[0]+25e3,mer1lat[0],'0°E',rotation=0.0,fontsize=5.5,color='grey')
    plt.plot(mer2lon,mer2lat,'--',linewidth=0.5,color="gray") 
    plt.text(mer2lon[0]+10e3,mer2lat[0],'30°E',rotation=0.0,fontsize=5.5,color='grey')
        
    ## variogram
    plt.axes(ax[0,1])
    plt.plot(OK.lags,OK.semivariance, 'r*',label='experimental')
    plt.plot(OK.lags,OK.variogram_function(OK.variogram_model_parameters,OK.lags), 'k-',label= OK.variogram_model + ' model')
    maxlag = 3121950.7115684478
    plt.plot([maxlag,maxlag],[np.min(OK.semivariance), np.max(OK.semivariance)],'--',color='gray')
    plt.text(maxlag-190000,(np.min(OK.semivariance)+np.max(OK.semivariance))/3,"model fitting limit",rotation=90,color="gray")
    plt.xlabel('lag [m]')
    plt.ylabel('semivariance')
    plt.title('b) ' + pname_tmp + ' - Semivariogram')
    ax[0,1].ticklabel_format(style='scientific',scilimits=[0,3],useMathText=True)
    plt.legend()
    
    
    #% threshold interpolation using mean of the indicator
    plt.axes(ax[1,0])
    p=[0,0.2,0.40,0.50,0.60,0.8,1]#np.mean(obs_data)
    pred_th=np.ma.copy(pred)
    pred_th.mask=pred.mask
    cmap = mpl.cm.viridis
    norm = mpl.colors.BoundaryNorm(p, cmap.N)
    plt.imshow(pred_th,extent=(x[0],x[-1],y[-1],y[0]),cmap=cmap,norm=norm) #
    
    # parallels and meridians
    plt.plot(par1lon,par1lat,'--',linewidth=0.5,color="gray") 
    plt.text(par1lon[0],par1lat[0],'40°N',rotation=-15.0,fontsize=5.5,color='grey')
    plt.plot(par2lon,par2lat,'--',linewidth=0.5,color="gray") 
    plt.text(par2lon[0],par2lat[0],'60°N',rotation=-15.0,fontsize=5.5,color='grey')
    plt.plot(mer1lon,mer1lat,'--',linewidth=0.5,color="gray") 
    plt.text(mer1lon[0]+25e3,mer1lat[0],'0°E',rotation=0.0,fontsize=5.5,color='grey')
    plt.plot(mer2lon,mer2lat,'--',linewidth=0.5,color="gray") 
    plt.text(mer2lon[0]+10e3,mer2lat[0],'30°E',rotation=0.0,fontsize=5.5,color='grey')
    
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),)
    plt.xlim([xst-deltax,xen+deltax*2.5]) # figure limits
    plt.ylim([yst-deltay*1.2,yen+deltay])
    plt.title('c) ' + pname_tmp + ' - Occurrence map (probability thresholds)')
    plt.xlabel('easting [m]')
    plt.ylabel('northing [m]')
    ax[1,0].ticklabel_format(style='scientific',scilimits=[0,3],useMathText=True)
    ax[1,0].text(1.1e6,-50300,'$\\times\\mathdefault{10^{6}}\\mathdefault{}$')
    ax[1,0].yaxis.get_offset_text().set_visible(False)
    ax[1,0].xaxis.get_offset_text().set_visible(False)

    #% kriging variance
    plt.axes(ax[1,1])
    plt.imshow(ss,extent=(x[0],x[-1],y[-1],y[0]),vmin=np.quantile(ss[np.logical_not(ss.mask)],0.01),vmax=np.quantile(ss[np.logical_not(ss.mask)],0.99)) # interpolation
    plt.colorbar() # colorbar
    plt.scatter(obs_x,obs_y,c=obs_data,edgecolors="gray",linewidths=0.5,vmin=0,vmax=1,s=10,label="pollen samples") # samples

    # parallels and meridians
    plt.plot(par1lon,par1lat,'--',linewidth=0.5,color="gray") 
    plt.text(par1lon[0],par1lat[0],'40°N',rotation=-15.0,fontsize=5.5,color='grey')
    plt.plot(par2lon,par2lat,'--',linewidth=0.5,color="gray") 
    plt.text(par2lon[0],par2lat[0],'60°N',rotation=-15.0,fontsize=5.5,color='grey')
    plt.plot(mer1lon,mer1lat,'--',linewidth=0.5,color="gray") 
    plt.text(mer1lon[0]+25e3,mer1lat[0],'0°E',rotation=0.0,fontsize=5.5,color='grey')
    plt.plot(mer2lon,mer2lat,'--',linewidth=0.5,color="gray") 
    plt.text(mer2lon[0]+10e3,mer2lat[0],'30°E',rotation=0.0,fontsize=5.5,color='grey')
    
    plt.xlim([xst-deltax,xen+deltax*2.5]) # figure limits
    plt.ylim([yst-deltay*1.2,yen+deltay])
    plt.title('d) ' + pname_tmp + ' - Occurrence uncertainty (Kriging variance)')
    plt.xlabel('easting [m]')
    plt.ylabel('northing [m]')
    plt.legend()
    ax[1,1].ticklabel_format(style='scientific',scilimits=[0,3],useMathText=True)
    ax[1,1].text(1.1e6,-50300,'$\\times\\mathdefault{10^{6}}\\mathdefault{}$')
    ax[1,1].yaxis.get_offset_text().set_visible(False)
    ax[1,1].xaxis.get_offset_text().set_visible(False)
    plt.tight_layout()
    
    # save fig
    plt.savefig('export/dataset/' + pname_tmp + '/' + pname_tmp + '.pdf')
    plt.close('all')
    
    # WRITE GEOTIFF
    im1=np.flipud(pred.data).copy() # occurrence probability (Kriging mean)
    im1[im1<0]=0 # set probability limits 
    im1[im1>1]=1
    im1[np.flipud(pred.mask)]=-999 # put back no data values
    
    im2=np.copy(im1)
    p=[0,0.2,0.40,0.50,0.60,0.8,1]#np.mean(obs_data)
    im2[im1>=p[0]]=p[1]
    for i in range(1,len(p)-1):
        im2[im1>p[i]]=p[i+1]
                
    im2[im1==-999]=-999 # put back no data values
    
    im3=np.flipud(ss.data).copy() # occurrence uncertainty (Kriging variance)
    im3[np.flipud(pred.mask)]=-999 # put no data values
    
    im=np.stack((im1,im2,im3),axis=2)
    ft.write_geotiff(filename='export/dataset/' + pname_tmp + '/' + pname_tmp + '.tiff',
                 im=im,
                 xmin=x[0]-res/2,
                 ymax=y[-1]+res/2,
                 xres=res,
                 yres=res,
                 epsg=3034,
                 dtype='float32',
                 nodata_val=-999,
                 band_name=['Occurrence probability (Kriging mean)','Occurrence map (<= probability thresholds)','Occurrence uncertainty (Kriging variance)']
                 )
    # WRITE POINT DATA SHAPEFILE
    ft.write_point_shp('export/dataset/' + pname_tmp +'/' + pname_tmp ,obs_x,obs_y,['POLLEN_PRESENCE'],['L'],[obs_data],esri_code=3034)

def plot_save_cost(c,obs_x,obs_y,obs_data,pname_tmp): # save image in case od costant field (no interpolation)
    pred=np.ma.empty_like(mask)*c
    pred.mask=np.logical_not(mask).copy()
    ss=np.ma.copy(pred)*0
    
    try:
        os.mkdir('export/dataset/' + pname_tmp)
    except:
        pass
    # indicator kriging mean
    fig, ax = plt.subplots(2, 2, figsize=([12.55,  9.5 ]))
    plt.axes(ax[0,0])
    plt.imshow(pred,extent=(x[0],x[-1],y[-1],y[0]),vmin=0,vmax=1) # interpolation
    plt.scatter(obs_x,obs_y,c=obs_data,edgecolors="gray",linewidths=0.5,vmin=0,vmax=1,s=10,label="pollen samples") # samples
    plt.xlim([xst-deltax,xen+deltax*2.5]) # figure limits
    plt.ylim([yst-deltay*1.2,yen+deltay])
    plt.colorbar() # colorbar
    plt.title('a) ' + pname_tmp + ' - Occurrence probability (Kriging mean)')
    plt.legend(loc='upper left')
    plt.xlabel('easting [m]')
    plt.ylabel('northing [m]')
    ax[0,0].ticklabel_format(style='scientific',scilimits=[0,3],useMathText=True)
    ax[0,0].text(1.1e6,-50300,'$\\times\\mathdefault{10^{6}}\\mathdefault{}$')
    ax[0,0].yaxis.get_offset_text().set_visible(False)
    ax[0,0].xaxis.get_offset_text().set_visible(False)
    
    ## variogram
    plt.axes(ax[0,1])
    plt.plot(np.linspace(2,48,20),np.linspace(0,48,20)*0, 'r*',label='experimental')
    plt.plot(np.linspace(2,48,20),np.linspace(0,48,20)*0, 'k',label='constant model')
    plt.xlabel('lag [m]')
    plt.ylabel('semivariance')
    plt.xlim([0,50])
    plt.title('b) ' + pname_tmp + ' - Semivariogram')
    plt.legend()
    
    
    #% threshold interpolation using mean of the indicator
    plt.axes(ax[1,0])
    p=[0,0.2,0.40,0.50,0.60,0.8,1];#np.mean(obs_data)
    pred_th=np.ma.copy(pred)
    cmap = mpl.cm.viridis
    norm = mpl.colors.BoundaryNorm(p, cmap.N)
    plt.imshow(pred_th,extent=(x[0],x[-1],y[-1],y[0]),cmap=cmap,norm=norm) #
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),)
    plt.xlim([xst-deltax,xen+deltax*2.5]) # figure limits
    plt.ylim([yst-deltay*1.2,yen+deltay])
    plt.title('d) ' + pname_tmp + ' - Occurrence map (probability thresholds)')
    plt.xlabel('easting [m]')
    plt.ylabel('northing [m]')
    ax[1,0].ticklabel_format(style='scientific',scilimits=[0,3],useMathText=True)
    ax[1,0].text(1.1e6,-50300,'$\\times\\mathdefault{10^{6}}\\mathdefault{}$')
    ax[1,0].yaxis.get_offset_text().set_visible(False)
    ax[1,0].xaxis.get_offset_text().set_visible(False)
    
    #% kriging variance
    plt.axes(ax[1,1])
    plt.imshow(ss,extent=(x[0],x[-1],y[-1],y[0]),vmin=np.quantile(ss[np.logical_not(ss.mask)],0.01),vmax=np.quantile(ss[np.logical_not(ss.mask)],0.99)) # interpolation
    plt.colorbar() # colorbar
    plt.scatter(obs_x,obs_y,c=obs_data,edgecolors="gray",linewidths=0.5,vmin=0,vmax=1,s=10,label="pollen samples") # samples
    plt.xlim([xst-deltax,xen+deltax*2.5]) # figure limits
    plt.ylim([yst-deltay*1.2,yen+deltay])
    plt.title('d) ' + pname_tmp + ' - Occurrence uncertainty (Kriging variance)')
    plt.xlabel('easting [m]')
    plt.ylabel('northing [m]')
    plt.legend()
    ax[1,1].ticklabel_format(style='scientific',scilimits=[0,3],useMathText=True)
    ax[1,1].text(1.1e6,-50300,'$\\times\\mathdefault{10^{6}}\\mathdefault{}$')
    ax[1,1].yaxis.get_offset_text().set_visible(False)
    ax[1,1].xaxis.get_offset_text().set_visible(False)
    plt.tight_layout()
    
    # save fig
    plt.savefig('export/dataset/' + pname_tmp + '/' + pname_tmp + '.pdf')
    plt.close('all')
    
    # WRITE GEOTIFF
    im1=np.flipud(pred.data).copy() # occurrence probability (Kriging mean)
    im1[im1<0]=0 # set probability limits 
    im1[im1>1]=1
    im1[np.flipud(pred.mask)]=-999 # put back no data values
    
    im2=np.copy(im1)
    p=[0,0.2,0.40,0.50,0.60,0.8,1];#np.mean(obs_data)
    im2[im1>=p[0]]=p[1]
    for i in range(1,len(p)-1):
        im2[im1>p[i]]=p[i+1]
                
    im2[im1==-999]=-999 # put back no data values
    
    im3=np.flipud(ss.data).copy() # occurrence uncertainty (Kriging variance)
    im3[np.flipud(pred.mask)]=-999 # put no data values
    
    im=np.stack((im1,im2,im3),axis=2)
    ft.write_geotiff(filename='export/dataset/' + pname_tmp + '/' + pname_tmp + '.tiff',
                 im=im,
                 xmin=x[0]-res/2,
                 ymax=y[-1]+res/2,
                 xres=res,
                 yres=res,
                 epsg=3034,
                 dtype='float32',
                 nodata_val=-999,
                 band_name=['Occurrence probability (Kriging mean)','Occurrence map (<= probability thresholds)','Occurrence uncertainty (Kriging variance)']
                 )
    # WRITE POINT DATA SHAPEFILE
    ft.write_point_shp('export/dataset/' + pname_tmp +'/' + pname_tmp ,obs_x,obs_y,['POLLEN_PRESENCE'],['L'],[obs_data],esri_code=3034)


#%% DEFINE INTERPOLATION FUNCTION
ss_tot=np.empty((np.shape(mask)[0],np.shape(mask)[1],np.shape(p_list)[0]))*0
f_tot=np.empty([np.shape(p_list)[0],10])*np.nan

m_frac=0.50
def interp(p_list,j):
    pname_tmp=p_list[j]
    print('interpolation ' + str(j) + '/' + str(len(p_list)-1) + ': ' + pname_tmp + '\n')
    if np.all(pname!=pname_tmp):
        print('no pollen data present for ' + pname_tmp)
        return
    else:
        pcount_tmp=np.empty(np.size(sname))*np.nan
        for i in range(np.size(sname)):
            ind=np.logical_and(psample==sname[i],pname==pname_tmp)
            if len(pcount[ind])==0:
                pcount_tmp[i]=0    
            else:
                pcount_tmp[i]=pcount[ind][0] # if more counts are present, take the 1st
        
        if np.all(pcount_tmp==0):
            print('All pollen data in the interpolation area are 0')
            return
        #% KRIGING INTERPOLATION
        # generate indicator data
        data=pcount_tmp.copy()
        data[pcount_tmp>0]=1 # 1= pollen present, 0=absent
        data[pcount_tmp<=0]=0
        
        # exclude missing data/coordinates
        ind=np.logical_and(np.logical_and(np.isfinite(data),np.isfinite(lon)),np.isfinite(lat))
        obs_x=lon[ind].copy()
        obs_y=lat[ind].copy()
        obs_data=data[ind].copy()
        
        #select data within and near the interpolation grid
        indx=np.logical_and(obs_x>x[0]-deltax,obs_x<x[-1]+deltax)
        indy=np.logical_and(obs_y>y[0]-deltay,obs_y<y[-1]+deltay)
        ind=np.logical_and(indx,indy)
        obs_x=obs_x[ind].copy()
        obs_y=obs_y[ind].copy()
        obs_data=obs_data[ind].copy()
        
        # check if all data 0 or 1
        if np.all(obs_data==0):
            print('All pollen data in the interpolation area are 0')
            plot_save_cost(0,obs_x,obs_y,obs_data,pname_tmp)
            f=np.empty([1,10])*np.nan
            f[0]=1
        
        elif np.all(obs_data==1):
            print('All pollen data in the interpolation area are 1')
            plot_save_cost(1,obs_x,obs_y,obs_data,pname_tmp)
            f=np.empty([1,10])*np.nan
            f[-1]=1
            
        else:            

            #% OPTIMIZE ISO KRIGING VARIOGRAM TYPE
            mtype,mparams=ft.vario_optim(obs_x,obs_y,obs_data,nlags=20,q=0.80,lagseq='lin',plot=False)
            
            # EXECUTE KRIGING
            OK = OrdinaryKriging(obs_x, 
                     obs_y, 
                     obs_data, #x,y,value
                     variogram_model=mtype, #estimator.best_estimator_.variogram_model,#'gaussian',#'spherical',#'exponential',
                     variogram_parameters=mparams,
                     nlags=20,
                     verbose=False,
                     enable_plotting=False,
                     enable_statistics=False
                     )
            # apply interpolator
            pred, ss = OK.execute('masked',x,y,mask=np.logical_not(mask))
            
            # save ss for total variance map
            ss_tot[:,:,j]=ss.copy()
            
            # SAVE OUTPUT
            plot_save_fig(pred,ss,obs_x,obs_y,obs_data,OK,pname_tmp)
            
            
            #% KRIGING CROSS VALIDATION
            # generate reference randon missing data
            ref_ind=np.random.rand(len(obs_data))<m_frac
            ref_x=obs_x[ref_ind].copy()
            ref_y=obs_y[ref_ind].copy()
            ref_data=obs_data[ref_ind].copy()
            
            #remaining cdata
            c_ind=np.logical_not(ref_ind)
            c_x=obs_x[c_ind].copy()
            c_y=obs_y[c_ind].copy()
            c_data=obs_data[c_ind].copy()           
    
            # check if all data 0 or 1
            if np.all(c_data==0):
                f=np.empty([1,10])*np.nan
                f[0,0]=1
            
            elif np.all(c_data==1):
                f=np.empty([1,10])*np.nan
                f[0,-1]=1
                
            else:            
                #% OPTIMIZE ISO KRIGING VARIOGRAM TYPE
                mtype,mparams=ft.vario_optim(c_x,c_y,c_data,nlags=20,q=0.9,lagseq='lin',plot=False)
                
                # EXECUTE KRIGING
                OK = OrdinaryKriging(c_x, 
                         c_y, 
                         c_data, #x,y,value
                         variogram_model=mtype, #estimator.best_estimator_.variogram_model,#'gaussian',#'spherical',#'exponential',
                         variogram_parameters=mparams,
                         nlags=20,
                         verbose=False,
                         enable_plotting=False,
                         enable_statistics=False
                         )
                
                # apply interpolator
                pred, ss = OK.execute('masked',x,y,mask=np.logical_not(mask))
                
                # reliability plot
                pred_p = ft.extract_grid_points(np.flipud(pred),x[0]-res/2,y[0]-res/2,res,res,ref_x,ref_y,plot_xy=False) # co-located dem cells
                qc,f = ft.reliability(pred_p,ref_data,10,plot=False)
        
        f_tot[j,:]=f.copy()
        print('done!')
    
#%% FOR LOOP  

# UNCOMMENT to initialize saved matricies and counter
n = 0
np.save('f_tot.npy',f_tot)
np.save('ss_tot.npy',ss_tot)
np.save('counter.npy',n)

# load latest saved matricies and counter to resume script (initialization must be commented)
n = np.load('counter.npy')
f_tot = np.load('f_tot.npy')
ss_tot = np.load('ss_tot.npy')

for j in range(n,len(p_list)):
    
    # intepolate pollen for j-th taxum
    interp(p_list,j)
    
    # save updated matricies and counter
    np.save('f_tot.npy',f_tot)
    np.save('ss_tot.npy',ss_tot)
    n= n+1
    np.save('counter.npy',n)
               
#% save final matricies and counter
f_tot2=np.array(f_tot).T
np.save('ss_tot.npy',ss_tot)
np.save('f_tot.npy',f_tot2)