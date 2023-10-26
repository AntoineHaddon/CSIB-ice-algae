import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import datetime


from glob import glob 


def shiftCMAP(vmin, vmax, midpoint,cmap,nticks=4):
    nl=256
    levels = np.linspace(vmin, vmax, nl)
    if isinstance(cmap , str): clmap = plt.colormaps[cmap]
    else: clmap=cmap
    cmap_shifted, norm = colors.from_levels_and_colors(np.linspace(vmin, vmax, nl-1), 
                                         clmap(np.interp(levels, [vmin, midpoint, vmax], [0, 0.5, 1])), extend='both')
    return cmap_shifted, norm, [t for t in np.linspace(vmin,vmax,nticks)] 






import matplotlib.path as mpath

def roundBoundary(ax):
    # Compute a circle in axes coordinates, which we can use as a boundary for the map. 
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)



def initmaps(r=1,c=1,extent=[-180,180, 60,90], rnd=False, central_longitude=0,
             landcolor=cfeature.COLORS['land_alt1'],
             # landcolor='tab:gray',
             **kw): 
    fig, ax = plt.subplots(r,c,**kw, 
                       subplot_kw={'projection':ccrs.NorthPolarStereo(central_longitude=central_longitude)} )    

    for a in fig.axes: 
        a.coastlines(linewidth=0.5,resolution='50m')
        a.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor=None,# don't use this for coast line problem with 180 degree longitude line drawn on north polar stereo
                                        facecolor=landcolor,alpha=0.7) )
        if extent is not None: a.set_extent( extent, ccrs.PlateCarree())
        if rnd: roundBoundary(a)
    
    return fig,ax





def singleMap(var, dadic, 
              fig=None, ax=None,
              cbar=True, shrinkcbar=0.4, cbarorient='horizontal', extend='both',
              unitName=None, title=None, 
              mask=xr.DataArray(True), landMask=1, 
              hatchMsk=None, noHatchNan=True, hatchlw=0.1, hlbl=None, hatchColor='k',
              **pltkw
             ):

    if fig is None: fig,ax=initmaps(figsize=(7,7))

    try: ax.set_title(var.attrs['standard_name'] if title is None else title)
    except: pass

    pl = ax.pcolormesh(dadic['mesh'].nav_lon, dadic['mesh'].nav_lat,
                       var.where((dadic['landMask']>=landMask)).where(mask),
                       transform=ccrs.PlateCarree(), 
                       **pltkw
                        )
    
    if hatchMsk is not None:
        plt.rcParams['hatch.linewidth'] = hatchlw
        # add hatching everywhere withn a circle up to ~60N (done in axis coordinates)
        theta = np.linspace(-np.pi/2 *0.68, np.pi/2 *0.45 , 100)
        center, radius = [0.5, 0.5], 0.54
        cx,cyp = np.sin(theta)* radius + center[0], np.cos(theta)* radius + center[1]
        ax.fill_between(cx,cyp,1-cyp, transform=ax.transAxes,
                         hatch='xxx',color="none",edgecolor=hatchColor,lw=0,)#label=hlbl)
        # draw on land (+NE pacific, baltic, etc) to mask off hatching
        ax.pcolormesh(dadic['mesh'].nav_lon, dadic['mesh'].nav_lat, dadic['landMask'].where(dadic['landMask']==0), 
                      cmap='binary', transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, zorder=1, facecolor=[1,1,1])#, edgecolor='black')
        # redraw variable over hatching in areas without hatching
        ax.pcolormesh(dadic['mesh'].nav_lon, dadic['mesh'].nav_lat, var.where(np.logical_not(hatchMsk)).where((dadic['landMask']>=landMask)).where(mask),
                      transform=ccrs.PlateCarree(), **pltkw)
        # redraw white over hatching in areas where variable is nan
        if noHatchNan: 
            ax.pcolormesh(dadic['mesh'].nav_lon, dadic['mesh'].nav_lat, 
                      var.isnull().where(var.isnull()), cmap='binary', transform=ccrs.PlateCarree())

    
    if cbar:
        cl = fig.colorbar(pl, ax=ax, pad=0.05, extend=extend, orientation=cbarorient, shrink=shrinkcbar)
        try: 
            cl.set_label(var.attrs['units'] if unitName is None else unitName) 
        except: pass
        return fig,ax,pl,cl
    else:
        return fig,ax,pl






def pltRgnBdnry(fig,ax,rgnmsk, dmaskNAA, landMask,
               **kwargs):

    drdxy =(np.abs(dmaskNAA.where((landMask>0)).region.differentiate("x")) +  np.abs(dmaskNAA.where((landMask>0)).region.differentiate("y")))
    
    # with pcolormesh
    # border = drdxy.where(drdxy>0).where(rgnmsk)
    # ax.pcolormesh(dmaskNAA.nav_lon, dmaskNAA.nav_lat, border,
    #               cmap=colors.ListedColormap(['k']),
    #               transform=ccrs.PlateCarree(), 
    #               # shading='nearest'#, 'gouraud', 'flat'
    #              )
    
    # with scatter
    lons = dmaskNAA.nav_lon.where(drdxy>0).where(rgnmsk).values.ravel()
    lats = dmaskNAA.nav_lat.where(drdxy>0).where(rgnmsk).values.ravel()
    ax.scatter(lons[~np.isnan(lons)], lats[~np.isnan(lats)],
                **kwargs,
              transform=ccrs.PlateCarree(),
              )

















def OpenPlotSingleClim(dataFileName,varName,timeperiod,months,
               depthLevel=None,cmap=plt.colormaps['coolwarm'],
               unitName=None, convFac=1.0,
               logScale=None):

 
    # Mesh
    dmesh = xr.open_dataset('/tsanta/ahaddon/NAA/mesh/mesh_mask_naa1_rn_hmin7.nc')
    landMask = dmesh.tmask.isel(t=0, z= 0)
    didj = (dmesh.e1t * dmesh.e2t).isel(t=0).where((landMask>0)) # for area integrals, over ocean surface 

    # datadir = '/net/venus/kenes/user/jlanger/'
    datadir = '/tsanta/ahaddon/data/'
    ds = xr.open_mfdataset(datadir+dataFileName, combine='by_coords', parallel=True)

    if depthLevel is None:
        var=ds[varName].sel(time_counter=timeperiod).where(dmesh.nav_lat>60)
    else:
        var=ds[varName].isel(deptht=depthLevel).sel(time_counter=timeperiod).where(dmesh.nav_lat>60)
    
    with xr.set_options(keep_attrs=True):
        var=var.sel(time_counter=var.time_counter.dt.month.isin(months) ).mean(dim='time_counter').compute() *convFac


    fig, ax = plt.subplots(figsize=(8,8),
                       subplot_kw={'projection':ccrs.NorthPolarStereo()} )    
    ax.coastlines(); ax.set_extent( [-180,180, 60,90], ccrs.PlateCarree())
    pl = ax.pcolormesh(dmesh.nav_lon, dmesh.nav_lat, var.where((landMask>0)),
                          transform=ccrs.PlateCarree(), cmap=cmap,
                           norm=logScale
                        )
    
    cbar = fig.colorbar(pl, ax=ax, extend='both',orientation='horizontal',shrink=0.5)
    if unitName is None:
        ax.set_title(var.attrs['standard_name'])
        cbar.set_label(var.attrs['units'])
    else:
        ax.set_title(var.attrs['standard_name'])
        cbar.set_label(unitName)
    
    
   
    
    
    
    
def pltwithhatch(ax,lon,lat,var,
                 hatchMsk, hatchColor='k', hatchlw=0.1, noHatchNan=True, hatchStyle='xxx',
                 **pltkw):
    # plot variable 
    pl=ax.pcolormesh(lon, lat, var, transform=ccrs.PlateCarree(), **pltkw)

    plt.rcParams['hatch.linewidth'] = hatchlw
    # add hatching everywhere withn a circle up to ~60N (done in axis coordinates)
    theta = np.linspace(-np.pi/2 *0.68, np.pi/2 *0.45 , 100)
    center, radius = [0.5, 0.5], 0.54
    cx,cyp = np.sin(theta)* radius + center[0], np.cos(theta)* radius + center[1]
    ax.fill_between(cx,cyp,1-cyp, transform=ax.transAxes,
                        hatch=hatchStyle,color="none",edgecolor=hatchColor,lw=0,)#label=hlbl)
    # draw on land (+NE pacific, baltic, etc) to mask off hatching
    ax.add_feature(cfeature.LAND, zorder=1, facecolor=[1,1,1])#, edgecolor='black')
    # redraw variable over hatching in areas without hatching
    ax.pcolormesh(lon, lat, var.where(np.logical_not(hatchMsk)),#.where((dadic['landMask']>=landMask)).where(mask),
                    transform=ccrs.PlateCarree(), **pltkw)
    # redraw white over hatching in areas where variable is nan
    if noHatchNan: 
        ax.pcolormesh(lon, lat, 
                    var.isnull().where(var.isnull()), cmap='binary', transform=ccrs.PlateCarree())
    return pl