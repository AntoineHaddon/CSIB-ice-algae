import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import cartopy.crs as ccrs
from datetime import datetime, timedelta

from glob import glob 

import warnings

###################### Dask
# from dask.distributed import Client, LocalCluster

# def startDask(port=':8181',**kw):
#     cluster = LocalCluster(dashboard_address=port,**kw)
#     return cluster,Client(cluster)

# def stopDask(cluster,client):
#     cluster.close()
#     client.close()







days=[datetime(2001, 1, 1) + timedelta(d) for d in range(365)]
middledayofmonth = [datetime(2001, 1, 15) + timedelta(hours=730*m) for m in range(12)]
fstdayofmonth = [d.replace(day=1) for d in middledayofmonth]

    
    
    
#########################################################################
#
#  File IO
#
#############################################################################



def openNCfile(dadic,scenario,dataFile):
    """
        Open NAA output files, stored in dictonary dadic with key scenario+dataFile
        dataFile : diad, ptrc, grid, icemod, biolog
    """
    # datadir = '/net/venus/kenes/...'
    datadir = '/tsanta/ahaddon/data/'
    
    # daily output needs smaller chunks to avoid memory problems with dask
    if dataFile == 'biolog':
        dadic[scenario+dataFile] = xr.open_mfdataset(datadir+scenario+'/NAA_1d_*_biolog.nc',
                                                     combine='by_coords', parallel=True,
                                                     chunks={'time_counter':120})
    #monthly output
    else:
        if dataFile == 'diad':
            fn = '/NAA_730h_*_diad_T.nc' 
        elif dataFile == 'ptrc':
            fn =+ '/NAA_730h_*_ptrc_T.nc'
        elif dataFile == 'grid':
            fn='/NAA_730h_*_grid_T.nc'
        elif dataFile == 'icemod':
            fn='/NAA_730h_*icemod.nc'
        
        dadic[scenario+dataFile] = xr.open_mfdataset(datadir+scenario+'/'+fn, combine='by_coords', parallel=True)


        
    
def loadNAAmesh(dadic):
    meshdir='/tsanta/ahaddon/NAA/mesh/'
    # meshdir = '/home/ahaddon/Work/NAA/mesh/'    
    dadic['mesh'] = xr.open_mfdataset(meshdir+'mesh_mask_naa1_rn_hmin7.nc')
    dadic['landMask'] = dadic['mesh'].tmask.isel(t=0, z= 0)
    # for area integrals, over ocean surface 
    dadic['didj'] = (dadic['mesh'].e1t * dadic['mesh'].e2t).isel(t=0).where((dadic['landMask']>0)) 
    
        
      
        
###############################################################################
#
#     Climatologies
#
#############################################################################



def climatology(dadic, scn, dataFile, varName,
                timeperiod=None, months=None, depthLevel=None,
                areaMask=None, convFac=1., unitName=None,
                group=None, fillna=None,
                gca=False, 
                dask=False, verb=True,
               ):
    """ 
        Compute climatology of `varName` as mean over years `timeperiod` for each grid cell
        Options : 
            timeperiod : slice('y0','yf') default 1980-2009 for historical and 2056-2085 for future runs
            months : restrict to months, array (Jan=1, .. Dec=12)
            depthLevel : int, default none for 2d variables
            areaMask : default above 60 North
            group : to compute mean for each day of year or month
            gca : for variables that have already been multiplied by sea ice concentration, gca=True will divide by sea ice concentration to get in situ value, assuming daily output, false does nothing
    """
    if dask: cluster, client = startDask()
    
    if verb: print(varName, end=' ')
    with xr.set_options(keep_attrs=True):

        if timeperiod is None:
            if scn == 'historical-DFS-G510.00':
                timeperiod=slice('1981','2000')
            else:
                timeperiod=slice('2066','2085')
        var=dadic[scn+dataFile][varName].sel(time_counter=timeperiod)

        if months is not None:
            var =var.sel(time_counter=var.time_counter.dt.month.isin(months) )

        if depthLevel is not None: # 3d variables
            var=var.isel(deptht=depthLevel)

        if areaMask is None:
            areaMask = (dadic['mesh'].nav_lat>60) & (dadic['landMask']) 
        var=var.where(areaMask)

        if gca:
            if timeperiod is None:
                if scn == 'historical-DFS-G510.00':
                    timeperiod=slice('1981','2000')
                else:
                    timeperiod=slice('2066','2085')
            ileadfrac=dadic[scn+'biolog']['ileadfra2'].sel(time_counter=timeperiod).where(areaMask)
            if months is not None:
                ileadfrac =ileadfrac.sel(time_counter=ileadfrac.time_counter.dt.month.isin(months) )
            vname=var.name
            var = var / ileadfrac.where(ileadfrac>0)
            var.name=vname # need to have a name for xhisto


        if unitName is not None:
            var.attrs['units']=unitName

        if fillna is not None: var = var.fillna(fillna).where(areaMask)
        
        if group is None:
            var= var.mean(dim='time_counter')
        elif group=='day':
            var= var.groupby('time_counter.dayofyear').mean()
        else:
            var= var.groupby(group).mean(dim='time_counter')
            
        var=(var*convFac).compute()
        
    if dask: stopDask(cluster,client)
    return var


def resampleClimDaily(clim,timedim='month'):
    clim = clim.rename({timedim:'dayofyear'})
    clim['dayofyear']=middledayofmonth
    clim=clim.resample(dayofyear="1D").interpolate("linear") 
    clim=clim.reindex(dayofyear=pd.date_range(start=days[0],end=days[-1],freq="1D"), method='nearest')
    clim['dayofyear']=clim.dayofyear.dt.dayofyear
    return clim

