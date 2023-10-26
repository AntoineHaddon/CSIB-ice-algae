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

import statsmodels.api as sm
lowess = sm.nonparametric.lowess
lreg= lambda y,x : sm.OLS(y, sm.add_constant(x)).fit()



regions = ['Arctic Basin',
 'Greenland Shelf',
 'Baffin Bay',
 'Canadian Arctic Archipelago',
 'S. Beaufort Sea',
 'N. Beaufort Sea',
 'Bering Sea',
 'S. Chuckchi Sea',
 'N. Chuckchi Sea',
 'S. East Siberian Sea',
 'N. East Siberian Sea',
 'Kara Sea',
 'Barents Sea',
 'Nordic Sea']

cList = plt.colormaps['tab20b'](np.arange(14)/13)
clrs={r:c for r,c in zip(regions,cList)}



days=[datetime(2001, 1, 1) + timedelta(d) for d in range(365)]
middledayofmonth = [datetime(2001, 1, 15) + timedelta(hours=730*m) for m in range(12)]
fstdayofmonth = [d.replace(day=1) for d in middledayofmonth]
times = {'1d': days, '730h' : middledayofmonth}


def loadRegions():
    meshdir='/tsanta/ahaddon/NAA/mesh/'
    # meshdir = '/home/ahaddon/Work/NAA/mesh/'    

    dmaskNAA = xr.open_mfdataset(meshdir+'RegionalMasksNAA.nc')
    regions = list(dmaskNAA.attrs.values())

    dmesh = xr.open_mfdataset(meshdir+'mesh_mask_naa1_rn_hmin7.nc')
    landMask = dmesh.tmask.isel(t=0, z= 0)
    didj = (dmesh.e1t * dmesh.e2t).isel(t=0).where((landMask>0)) 
    rgnMsk = { rn: dmaskNAA.region == int(irn) for irn,rn in dmaskNAA.attrs.items()}
    rgnArea = {}
    for rgn in regions:
        rgnArea[rgn] = didj.where(rgnMsk[rgn]).sum(dim=['y','x']).compute() 

    return regions, rgnMsk, rgnArea






################################################################################################################################
#
#   Selection
#
###################################################################################################################################

def selMonth(m,dts,rgn,scns,varName):
    return xr.concat([dts[scn+rgn+varName] for scn in scns],dim='time_counter').groupby('time_counter.month')[m] 

def selPeak(dts,rgn,scns,varName):
    return xr.concat([dts[scn+rgn+varName] for scn in scns],dim='time_counter').groupby('time_counter.year').max() 

def selMin(dts,rgn,scns,varName):
    return xr.concat([dts[scn+rgn+varName] for scn in scns],dim='time_counter').groupby('time_counter.year').min() 

def selargPeak(mths,dts,rgn,scns,varName):
    var= xr.concat([dts[scn+rgn+varName] for scn in scns],dim='time_counter')
    var = var.sel(time_counter=var.time_counter.dt.month.isin(mths))
    dateargmax = var.groupby('time_counter.year').apply(lambda c: c.time_counter.isel(time_counter=c.argmax(dim="time_counter")) )
    return dateargmax.dt.dayofyear


def selargMin(mths,dts,rgn,scns,varName):
    var= xr.concat([dts[scn+rgn+varName] for scn in scns],dim='time_counter')
    var = var.sel(time_counter=var.time_counter.dt.month.isin(mths))
    dateargmax = var.groupby('time_counter.year').apply(lambda c: c.time_counter.isel(time_counter=c.argmin(dim="time_counter")) )
    return dateargmax.dt.dayofyear


def selSIbreak(dts,rgn,scns,rgnArea):
    # sea ice break day : when sea ice concentration is <0.15
    # argmin returns first occurence so can compute break up date as (first) minimun of sea ice concentration floored at 0.15 
    # take after day of max sea ice in case freeze up occurs at begening of year
    # check if max is > 0.15 for ice free years
    # check if min < 0.15 for year round ice 
    iceConc= xr.concat([dts[scn+rgn+'iceSurf1d'] for scn in scns],dim='time_counter')/rgnArea 
    # index of maximum ice from jan to aug
    idatemaxIce= iceConc.sel(time_counter=iceConc.time_counter.dt.month.isin(range(9))).groupby('time_counter.year').apply(lambda c: c.argmax(dim="time_counter") )
    def icebreakdate(si):
        y= si.time_counter.dt.year.values[0]
        # ice after the maximum
        si = si.isel(time_counter=slice(idatemaxIce.sel(year=y).values,-1) )
        if (si[0]>0.15) and (si.min()<0.15): #if (sea ice max > 0.15) and (sea ice min < 0.15)
            return si.time_counter.isel(time_counter=np.maximum(0.15, si).argmin() ).dt.dayofyear 
        else: #return nan
            return xr.DataArray(np.nan,coords={'time_counter':np.nan},name='dayofyear')
    return iceConc.groupby('time_counter.year').apply(icebreakdate)



def selRangeSum(mths,dts,rgn,scns,varName):
    var= xr.concat([dts[scn+rgn+varName] for scn in scns],dim='time_counter')
    return var.sel(time_counter=var.time_counter.dt.month.isin(mths)).groupby('time_counter.year').sum()

def selRangexISsum(mths,dts,rgn,scns,varName):
    var= xr.concat([dts[scn+rgn+varName] for scn in scns],dim='time_counter')
    isurf = xr.concat([dts[scn+rgn+'iceSurf730h'] for scn in scns],dim='time_counter')
    return (var*isurf).sel(time_counter=var.time_counter.dt.month.isin(mths)).groupby('time_counter.year').sum()





################################################################################################################################
#
#   Seasonal cycle
#
###################################################################################################################################



def pltmeanqtl(axis,dseacyl,scn,rgn,varName,freq,color,ls='-',SICmin=0):
    if SICmin>0: SImask= dseacyl[scn+rgn+'ileadfra2_OSav'+'1d'+'mean'] >SICmin
    else: SImask=True
    m = dseacyl[scn+rgn+varName+freq+'mean'].where(SImask)
    q25 = dseacyl[scn+rgn+varName+freq+'q25'].where(SImask)
    q75 = dseacyl[scn+rgn+varName+freq+'q75'].where(SImask)
    axis.plot(times[freq], m, color=color,ls=ls)
    axis.fill_between(times[freq], q25, q75, facecolor=color, alpha=0.2)

    
def spines_ticks_Options(ax, pos, c):       
    ax.spines[["right",'top','left','bottom']].set_visible(False)
    if pos is None:
        ax.set_yticks([]), ax.set_yticks([],minor=True)
    else:
        ax.spines[pos].set_visible(True)
        ax.spines[pos].set_color(c)
        ax.yaxis.set_label_position(pos)
        ax.yaxis.label.set_color(c)
        ax.yaxis.set_ticks_position(pos)
        ax.tick_params(colors=c)
    ax.set_xticks([], [] )
    ax.set_xlim(days[0],days[-1])

    
def removeFirstYtick(ax):
    y_ticks = ax.yaxis.get_major_ticks()
    y_ticks[0].set_visible(False)


def pltSeasonCycl(rgn,dseacyl,scenarios, SICmin=0):

    gs = gridspec.GridSpec(ncols=2, nrows=2, hspace = 0,wspace = 0.1)
    fig=plt.figure(figsize=(10,5))
    fig.suptitle(rgn)

    axsH = plt.subplot(gs[0,0])
    axiH = axsH.twinx()
    axicH = axsH.twinx()

    axs85 = plt.subplot(gs[0,1])
    axi85 = axs85.twinx()
    axic85 = axs85.twinx()


    c='tab:blue'
    axsH.set_ylabel('Snow thickness [m]')
    spines_ticks_Options(axsH, 'left', c)
    axsH.set_ylim(0,0.4)
    pltmeanqtl(axsH,dseacyl,scenarios[0],rgn,'isnowthigca_SIav','1d',c,SICmin=SICmin)

    spines_ticks_Options(axs85, None, c)
    axs85.set_ylim(0,0.4)
    pltmeanqtl(axs85,dseacyl,scenarios[1],rgn,'isnowthigca_SIav','1d',c,SICmin=SICmin)


    c='k'
    axiH.set_ylabel('Ice thickness [m]')
    spines_ticks_Options(axiH, 'left', c)
    axiH.spines["left"].set_position(("axes", -.2))
    axiH.set_ylim(0,4)
    pltmeanqtl(axiH,dseacyl,scenarios[0],rgn,'iicethicgca_SIav','1d',c,SICmin=SICmin)

    spines_ticks_Options(axi85, None, c)
    axi85.set_ylim(0,4)
    pltmeanqtl(axi85,dseacyl,scenarios[1],rgn,'iicethicgca_SIav','1d',c,SICmin=SICmin)
    
#     axiH.set_ylabel('Ice Volume [10$^3$ km$^3$]')
#     spines_ticks_Options(axiH, 'left', c)
#     axiH.spines["left"].set_position(("axes", -.2))
#     axiH.set_ylim(1e-3,10)
#     axiH.set_yscale('log')
#     axiH.set_yticks([1,10],[1,10])
#     pltmeanqtl(axiH,dseacyl,scenarios[0],rgn,'iceVol','1d',c)

#     axi85.set_yscale('log')
#     spines_ticks_Options(axi85, None, c)
#     axi85.set_ylim(1e-3,10)
#     pltmeanqtl(axi85,dseacyl,scenarios[1],rgn,'iceVol','1d',c)


    c='tab:orange'
    axicH.set_ylabel('Ice concentration [-]')
    spines_ticks_Options(axicH, 'left', c)
    axicH.spines["left"].set_position(("axes", -.4))
    axicH.set_ylim(0,1)
    pltmeanqtl(axicH,dseacyl,scenarios[0],rgn,'ileadfra2_OSav','1d',c)

    spines_ticks_Options(axic85, None, c)
    axic85.set_ylim(0,1)
    pltmeanqtl(axic85,dseacyl,scenarios[1],rgn,'ileadfra2_OSav','1d',c)
   


    axiaH = plt.subplot(gs[1,0])
    axinH = axiaH.twinx()
    axpH = axiaH.twinx()

    axia85 = plt.subplot(gs[1,1])
    axin85 = axia85.twinx()
    axp85 = axia85.twinx()
    
    
    
    c='tab:green'
    axiaH.set_ylabel('Ice Algae [mmol C m$^{-2}$]')
    spines_ticks_Options(axiaH, 'left', c)
    removeFirstYtick(axiaH)
    axiaH.set_ylim(0,20)
    # axiaH.set_ylim(1e-2,15)
    # axiaH.set_yscale('log')
    # axiaH.set_yticks([1e-1,1,10],[1e-1,1,10])
    axiaH.invert_yaxis()
    pltmeanqtl(axiaH,dseacyl,scenarios[0],rgn,'icediagca_SIav','1d',c,SICmin=SICmin)

    axia85.set_ylim(0,20)
    # axia85.set_yscale('log')
    # axia85.set_ylim(1e-2,15)
    spines_ticks_Options(axia85, None, c)
    axia85.invert_yaxis()
    pltmeanqtl(axia85,dseacyl,scenarios[1],rgn,'icediagca_SIav','1d',c,SICmin=SICmin)




    c='tab:purple'
    axinH.set_ylabel('Bottom ice NO$_3$ [mmol m$^{-2}$]')
    spines_ticks_Options(axinH, 'left', c)
    axinH.spines["left"].set_position(("axes", -.2))
    removeFirstYtick(axinH)
    axinH.set_ylim(0,0.5)
    axinH.invert_yaxis()
    pltmeanqtl(axinH,dseacyl,scenarios[0],rgn,'iceno3gca_SIav','1d',c,SICmin=SICmin)

    spines_ticks_Options(axin85, None, c)
    axin85.set_ylim(0,0.5)
    axin85.invert_yaxis()
    pltmeanqtl(axin85,dseacyl,scenarios[1],rgn,'iceno3gca_SIav','1d',c,SICmin=SICmin)


    c='tab:red'
    axpH.set_ylabel('Bottom ice PAR [W m$^{-2}$]')
    spines_ticks_Options(axpH, 'left', c)
    axpH.spines["left"].set_position(("axes", -.4))
    removeFirstYtick(axpH)
    axpH.set_ylim(0,13)
    axpH.invert_yaxis()
    pltmeanqtl(axpH,dseacyl,scenarios[0],rgn,'fstricgca_SIav','1d',c,SICmin=SICmin)

    spines_ticks_Options(axp85, None, c)
    axp85.set_ylim(0,13)
    axp85.invert_yaxis()
    pltmeanqtl(axp85,dseacyl,scenarios[1],rgn,'fstricgca_SIav','1d',c,SICmin=SICmin)


#     axdmsH= axiaH.twinx()
#     axdfH= axiaH.twinx()
#     axdms85= axia85.twinx()
#     axdf85= axia85.twinx()

#     c='y'
#     axdmsH.set_ylabel('Bottom ice DMS [$\mu$mol S m$^{-2}$]')
#     spines_ticks_Options(axdmsH, 'left', c)
#     axdmsH.spines["left"].set_position(("axes", -.4))
#     removeFirstYtick(axdmsH)
#     axdmsH.set_ylim(0,5)
#     axdmsH.invert_yaxis()
#     pltmeanqtl(axdmsH,dseacyl,scenarios[0],rgn,'icedms_SIav','730h',c)

#     spines_ticks_Options(axdms85, None, c)
#     axdms85.set_ylim(0,5)
#     axdms85.invert_yaxis()
#     pltmeanqtl(axdms85,dseacyl,scenarios[1],rgn,'icedms_SIav','730h',c)


#     c='tab:brown'
#     axdfH.set_ylabel('Ice to sea DMS flux [Gg S month$^{-1}$]')
#     spines_ticks_Options(axdfH, 'left', c)
#     axdfH.spines["left"].set_position(("axes", -.6))
#     # removeFirstYtick(axdfH)
#     axdfH.set_ylim(0,0.4)
#     # axdmsH.invert_yaxis()
#     pltmeanqtl(axdfH,dseacyl,scenarios[0],rgn,'icedmsrls_SItot','730h',c)

#     spines_ticks_Options(axdf85, None, c)
#     axdf85.set_ylim(0,0.4)
#     # axdms85.invert_yaxis()
#     pltmeanqtl(axdf85,dseacyl,scenarios[1],rgn,'icedmsrls_SItot','730h',c)




    axsH.set_title('Historical')
    axsH.set_xticks(fstdayofmonth, [] )
    axsH.grid(axis='x',ls=':')

    axs85.set_title('RCP 8.5')
    axs85.set_xticks(fstdayofmonth, [] )
    axs85.grid(axis='x',ls=':')

    axiaH.grid(axis='x',ls=':')
    axiaH.spines[['bottom','top']].set_visible(True)
    axiaH.set_xticks(fstdayofmonth, [] )
    axiaH.set_xticks(middledayofmonth, [d.strftime(format='%b')[0] for d in middledayofmonth ],minor=True )
    axiaH.tick_params(axis='x', which='minor', length=0)
    axiaH.set_xlim(days[0],days[-1])

    axia85.grid(axis='x',ls=':')
    axia85.spines[['bottom','top']].set_visible(True)
    axia85.set_xticks(fstdayofmonth, [] )
    axia85.set_xticks(middledayofmonth, [d.strftime(format='%b')[0] for d in middledayofmonth ],minor=True )
    axia85.tick_params(axis='x', which='minor', length=0)
    axia85.set_xlim(days[0],days[-1])

    
    
    
    
    

################################################################################################################################
#
#   Trends
#
###################################################################################################################################


trend = lambda var : sm.OLS(var.dropna(dim='year').values,sm.add_constant(var.dropna(dim='year').year.values)).fit().params[1]


def autolabel(ax,bars,label):
    # attach some text labels
    xl=ax.get_xlim() ; axwidth=xl[1]-xl[0]
    for ib, bs in enumerate(bars):
        for bar in bs:
            width, height = bar.get_width(), bar.get_height()
            sign=width/np.abs(width)
            if (np.abs(width)/axwidth <= 0.3):
                # Shift the text outside 
                xloc1 = width + sign*axwidth*0.02
                clr = 'black'
                if sign>0 : align = 'left'
                else: align = 'right'
            else:
                # Shift the text to the edge
                xloc1 = width - sign*axwidth*0.02
                clr = 'white'
                if sign>0 : align = 'right'
                else: align = 'left'
            # print(xloc1)
            ax.text(xloc1,bar.get_y() + height /2.0, label[ib] ,horizontalalignment=align, verticalalignment='center', fontsize=9,color=clr)



def pltTrends(dvars,regions):
    nvars = len(dvars)
    nrows = int((nvars-1)/4)+1
    nregions=len(regions)
    
    # ncols=4
    # fig,ax=plt.subplots( nrows, 4, figsize=(25,nregions*nrows*0.4+1+1.2*nrows) )
    ncols=3
    fig,ax=plt.subplots( nrows, ncols, figsize=(12,nregions*nrows*0.2+1+1*nrows) )
    
    ax=ax.ravel()
    
    for iv, vname in enumerate(dvars):
        labels=['']*nregions
        bars=[]
        for ir,rgn in enumerate(regions):
            var = dvars[vname](rgn)
            ddec = trend(var)*10 # convert to change per decade
            bars.append( ax[iv].barh(ir, ddec, color=clrs[rgn] ) )
            relchange = (ddec/var.sel(year=slice(1980,2009)).mean() *100).values # relative change % per decade
            # if np.abs(relchange)>0.01:
            #     if relchange>0: ax[iv].bar_label(bar, labels=[f'+{relchange:.2f} %'] ,label_type='center' )
            #     else: ax[iv].bar_label(bar, labels=[f'{relchange:.2f} %'] ,label_type='center' )
            if vname.split('[')[-1][:-1] == 'days': # for days relative change is a bit weird 
                if ddec>0: labels[ir]=f'+{ddec:.2f} '
                else: labels[ir]=f'{ddec:.2f} '
            else:
                if np.abs(relchange)>0.01:
                    if relchange>0: labels[ir]=f'+{relchange:.2f} %'
                    else: labels[ir]=f'{relchange:.2f} %'
        
        # makfe plots nearly symetric
        xl=ax[iv].get_xlim(); ax[iv].set_xlim(min(xl[0],-xl[1]*0.5),max(xl[1],-xl[0]*0.5))
        autolabel(ax[iv],bars,labels )

        
    for iv, vn in enumerate(dvars):
        if iv%ncols ==0:
            ax[iv].set_yticks(range(len(regions)), [r+' '*3 for r in regions])
        else:
            ax[iv].set_yticks([])
        ax[iv].set_ylim(-1,len(regions) )
        ax[iv].invert_yaxis()
        ax[iv].tick_params(axis='y', length=0) 
        ax[iv].set_title(vn.split('[')[0])
        ax[iv].set_xlabel(vn.split('[')[-1][:-1] + ' decade$^{-1}$')
        ax[iv].spines[["left", "top", "right"]].set_visible(False)
        ax[iv].plot([0,0],[-1,13.5],'k:',lw=0.5)
    
    # plt.suptitle('Regional trends and % change per decade relative to 1980-2009 mean',fontsize=16)
    # plt.subplots_adjust(wspace=0.25,hspace=.3,top=0.93)
    plt.tight_layout()
    return fig,ax
    
    

    
    
    
    
    
    

################################################################################################################################
#
#   Timeseries
#
###################################################################################################################################



def pltwtrend(axis,var,cl='k',ls='-',lbl=''):
    if 'time_counter' in var.dims:
        t = np.arange(var.time_counter.dt.year.values[0], var.time_counter.dt.year.values[-1]+1)
    else:
        # t = np.arange(var.year.values[0], var.year.values[-1]+1)
        var=var.dropna(dim='year')
        t= var.year.values
        
    axis.scatter(t, var, marker='.', color=cl, alpha=0.5)
    axis.plot(t, lowess(var, t)[:,1], c=cl, lw=2, label=lbl)
    
    
    
    
def pltTimeseries(dvars,regions):
    nvars = len(dvars)
    nrows = int((nvars-1)/3)+1
    nregions=len(regions)

    fig,ax=plt.subplots( nrows, 3, figsize=(18,nrows*4+1) )
    ax=ax.ravel()
    

    
    for iv, vname in enumerate(dvars):
        for ir,rgn in enumerate(regions):
            var = dvars[vname](rgn)
            pltwtrend(ax[iv],var,cl=clrs[rgn],lbl=rgn)
            

    for iv, vn in enumerate(dvars):
        ax[iv].set_title(vn.split('[')[0])
        ax[iv].set_ylabel(vn.split('[')[-1][:-1])
    
    for a in ax:
        ylim = a.get_ylim()
        a.plot((2015.5,2015.5), ylim, 'k',lw=0.5)
        a.set_ylim(ylim)
        a.set_xlim((1979,2085))

    ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)   

    # plt.subplots_adjust(wspace=0.25,hspace=.3,top=0.93)
    plt.tight_layout()
    # return ax
    
    
    
    
    
    
###########################################################################################################################
#
#      Regions map
#
########################################################################################################################
import matplotlib.path as mpath

def regionsMap(regions=regions,sidegraphs=True):
    nregions=len(regions)
    
    dmesh = xr.open_dataset('/tsanta/ahaddon/NAA/mesh/mesh_mask_naa1_rn_hmin7.nc')
    landMask = dmesh.tmask.isel(t=0, z= 0)
    didj = (dmesh.e1t * dmesh.e2t).isel(t=0).where((landMask>0)) 

    dmaskNAA = xr.open_dataset('/tsanta/ahaddon/NAA/mesh/RegionalMasksNAA.nc')
    irgn = [int(irn) for irn,rn in dmaskNAA.attrs.items() if rn in regions ]
    regions = [rn for irn,rn in dmaskNAA.attrs.items() if rn in regions ] # make sure that ordering of regions is always the same
    rgnMsk = { rn: dmaskNAA.region == int(irn) for irn,rn in dmaskNAA.attrs.items() if rn in regions  }
    rgnArea = {}
    for rgn in regions:
        rgnArea[rgn] = didj.where(rgnMsk[rgn]).sum(dim=['y','x']).compute() 

    if sidegraphs:
        gs = gridspec.GridSpec(3, 3, width_ratios=[3, 1, 1], height_ratios=[0.5-(nregions/14)/2,(nregions/14),0.6-(nregions/14)/2 ],hspace=.3) 
        fig=plt.figure(figsize=(15,7))
    else:
        gs = gridspec.GridSpec(3, 2, width_ratios=[3, 0], hspace=.3) 
        fig=plt.figure(figsize=(9,7))

    ax0 = plt.subplot(gs[0:3,0],projection=ccrs.NorthPolarStereo(),frameon=False)
    for i,rgn in enumerate(regions):
        ax0.pcolormesh( dmaskNAA.nav_lon, dmaskNAA.nav_lat, dmaskNAA.region.where(rgnMsk[rgn]), 
                     transform=ccrs.PlateCarree(), cmap =colors.ListedColormap(plt.colormaps['tab20b']( (irgn[i]-1)/13) ) )
        if irgn[i]==1:
            lon,lat = 0, 89
        else:
            lon,lat = dmaskNAA.nav_lon.where(rgnMsk[rgn]).median(),dmaskNAA.nav_lat.where(rgnMsk[rgn]).median()
        ax0.text(lon,lat,str(irgn[i]), transform=ccrs.PlateCarree(),fontsize=12,
                  bbox={'fc':'w','lw':0,'alpha':0.8, 'pad': 0.2,'boxstyle':'circle'})
    ax0.set_extent( [-180,180, 58,90], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary for the map. 
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax0.set_boundary(circle, transform=ax0.transAxes)
    ax0.coastlines()

    ax1 = plt.subplot(gs[1,1])
    if sidegraphs:
        ax1.barh( range(nregions), [ra*1e-6*1e-6 for ra in rgnArea.values()] , color= plt.colormaps['tab20b'](np.arange(14)/13))
        ax1.set_xlabel('Surface area [10$^{6}$ km$^2$]')
        ax1.spines[["left", "top", "right"]].set_visible(False)
    else:
        ax1.spines[["left", "top", "right",'bottom']].set_visible(False)
        ax1.set_xticks([])

    ax1.set_yticks(range(nregions), [str(irgn[i])+' : '+regions[i] for i in range(nregions)], wrap=True)
    ax1.invert_yaxis(), ax1.tick_params(axis='y', length=0) 

    if sidegraphs:
        ax2 = plt.subplot(gs[1,2])
        ax2.barh(range(nregions), [dmesh.hdept.isel(t=0).where(rgnMsk[rgn]).median() for rgn in regions], 
                 color= plt.colormaps['tab20b'](np.arange(14)/13))
        ax2.set_yticks(range(nregions), [])
        ax2.invert_yaxis()
        ax2.tick_params(axis='y', length=0) 
        ax2.set_xlabel('Median depth [m]')
        ax2.spines[["left", "top", "right"]].set_visible(False)
    