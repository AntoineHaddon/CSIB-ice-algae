import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

import cartopy.crs as ccrs
import palettable.cmocean.sequential as cmOcSeq

from datetime import datetime, timedelta

from xhistogram.xarray import histogram as xhist


scenarios = ['historical-DFS-G510.00','RCP85-G510.14-G515.00-01']
hist,rcp85 = scenarios
periods = ['1981-2000','2023-2042','2066-2085']
past,pres,futur=periods
runs={past:hist, pres:rcp85, futur:rcp85}
timeperiod = {past: slice('1981','2000'), pres:slice('2023','2042'), futur: slice('2066','2085') }


days=[datetime(2001, 1, 1) + timedelta(d) for d in range(365)]
middledayofmonth = [datetime(2001, 1, 15) + timedelta(hours=730*m) for m in range(12)]
fstdayofmonth = [d.replace(day=1) for d in middledayofmonth]

# for reindex from sept to aug
doys_sept = np.concatenate((np.arange(244,366),np.arange(1,244)) )
days_sept = [datetime(2001, 9, 1) + timedelta(d) for d in range(365)]
middledayofmonth_sept = [datetime(2001, 9, 15) + timedelta(hours=730*m) for m in range(12)]
fstdayofmonth_sept = [d.replace(day=1) for d in middledayofmonth_sept]



def monthTicks(ax,grid=False,start_sept=False):
    if start_sept:
        ax.set_xticks(fstdayofmonth_sept, [] )
        ax.set_xticks(middledayofmonth_sept, [d.strftime(format='%b')[0] for d in middledayofmonth_sept ],minor=True )
    else:
        ax.set_xticks(fstdayofmonth, [] )
        ax.set_xticks(middledayofmonth, [d.strftime(format='%b')[0] for d in middledayofmonth ],minor=True )
    ax.tick_params(axis='x', which='minor', length=0)
    if grid: ax.grid(axis='x',ls=':')

def rescale( y, ymin=None,ymax=None):
    if ymin is None: ymin = np.min(y)
    if ymax is None: ymax = np.max(y)
    return (y - ymin) / (ymax - ymin) 








def pltdayHist(dayhist, varDay, color, varsDayName,
                        suptitle=None, freqMax=0.1, spring=False,
                        fig=None,ax=None,
                       ):

    if fig is None: fig,ax=plt.subplots(figsize=(10,3.5))
    if suptitle is not None: fig.suptitle(suptitle)

    monthTicks(ax,grid=True)

    ax.plot(days, dayhist, c=color)
    meanday=days[int(varDay.load().median()-1)]
    # ax.plot([meanday,meanday],[0,freqMax],'--',color=color, label = f'Mean day {varsDayName}: ' + meanday.strftime(format='%-d %b') )
    ax.plot([meanday,meanday],[0,freqMax],'--',color=color, label = meanday.strftime(format='%-d %b') )

    if spring: ax.set_xlim(days[0],days[244])
    else: ax.set_xlim(days[0],days[-1])
    ax.set_ylim(0,freqMax)
    # ax.set_yticks([0,0.05,0.1])
    # ax.set_ylabel('Frequency')
    ax.legend(fontsize=10,loc='upper right',title="Median dates")



def pltperiodHist(periodhist, varPeriod, color, varsPerName='', xlim=(-75,210),
                        suptitle=None, freqMax=0.1, 
                        fig=None,ax=None,
                       ):

    if fig is None: fig,ax=plt.subplots(figsize=(10,3.5))
    if suptitle is not None: fig.suptitle(suptitle)


    bins=[v for k,v in periodhist.coords.items()][0]
    ax.plot(bins, periodhist, c=color)
    meanday=varPeriod.load().median()
    # ax.plot([meanday,meanday],[0,freqMax],'--',color=color, label = f'Mean day {varsDayName}: ' + meanday.strftime(format='%-d %b') )
    ax.plot([meanday,meanday],[0,freqMax],'--',color=color, label = f'{meanday.values:.1f}' )

    ax.grid(axis='x',ls=':')
    ax.set_xlim(xlim)
    ax.set_ylim(0,freqMax)
    # ax.set_yticks([0,0.05,0.1])
    # ax.set_ylabel('Frequency')
    ax.legend(fontsize=10,loc='upper right',title="Median")




















def pltHistDly(histo,cmap='Reds', 
                        suptitle=None, unit='', 
                        vmin=0,vmax=100, alpha=1,
                       threshold=None, thresholdName=None,
                        fig=None,ax=None, cbar=True, leftSpines=True,
                        start_sept=False, logscale=False
                       ):
    """
    fig,ax = pltHistDly(hist[scn+'icedia'], 
                    cmap='Greens', clFreq='tab:brown', clProp='tab:gray',
                    suptitle=f'Bottom ice algae distribution - {rgn}',
                    unit='mmol C m$^{-2}$', varDayhistName='PAR lim = N lim',
                        vmin=0,vmax=100, 
                       )
    """
    binNames = [k for k in histo.coords if k != 'dayofyear'][0]
    bins=histo[binNames] 
    binsize = bins[1]-bins[0]

    if fig is None: fig,ax=plt.subplots(figsize=(10,3.5))
    if suptitle is not None: fig.suptitle(suptitle)

    if start_sept:
        xday=days_sept
        hplt=histo.reindex(dayofyear=doys_sept)
    else:
        xday = days
        hplt=histo
    
    if logscale:
        pl=ax.pcolormesh(xday, bins, hplt.T, cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=alpha)
    else:
        pl=ax.pcolormesh(xday, bins, hplt.T, vmin=vmin,vmax=vmax, cmap=cmap , alpha=alpha)
        
    ax.set_xlim(xday[0],xday[-1])
    monthTicks(ax,grid=True,start_sept=start_sept)
    if leftSpines: ax.set_ylabel(unit)
    else: ax.set_yticks([])

    if threshold is not None :ax.plot([xday[0],xday[-1]],[threshold]*2,'-.',label= thresholdName )
    # ax[i].legend()
            
    if cbar:
        cbar_ax = fig.add_axes([0., 0.25, 0.02, 0.5])  
        cl=fig.colorbar(pl,cax=cbar_ax,extend='max')
        cl.set_label('10$^3$ km$^{2}$')
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')

    return fig,ax,pl








def pltHistDly_single(histo,cmap='Reds', 
                        suptitle=None, unit='', 
                        vmin=0,vmax=100, alpha=1,
                       threshold=None, thresholdName=None,
                        fig=None,ax=None, cbar=True, leftSpines=True,
                        start_sept=False,
                       ):
    """
    fig,ax = pltHistDly(hist[scn+'icedia'], 
                    cmap='Greens', clFreq='tab:brown', clProp='tab:gray',
                    suptitle=f'Bottom ice algae distribution - {rgn}',
                    unit='mmol C m$^{-2}$', varDayhistName='PAR lim = N lim',
                        vmin=0,vmax=100, 
                       )
    """
    binNames = [k for k in histo.coords if k != 'dayofyear'][0] 
    bins=histo[binNames]
    binsize = bins[1]-bins[0]

    if fig is None: fig,ax=plt.subplots(1,1,figsize=(10,3.5))
    if suptitle is not None: fig.suptitle(suptitle)
    
    if start_sept: xday=days_sept
    else: xday = days

    if start_sept:
        pl=ax.pcolormesh(xday, bins, histo.reindex(dayofyear=doys_sept).T, vmin=vmin,vmax=vmax, cmap=cmap , alpha=alpha)
    else:
        pl=ax.pcolormesh(xday, bins, histo.T, vmin=vmin,vmax=vmax, cmap=cmap , alpha=alpha)
    ax.set_xlim(xday[0],xday[-1])
    # ax.set_title(scn.split('-')[0])
    monthTicks(ax,grid=True,start_sept=start_sept)
    if leftSpines: ax.set_ylabel(unit)
    else: ax.set_yticks([])

    if threshold is not None :ax.plot([xday[0],xday[-1]],[threshold]*2,'k-.',label= thresholdName )
    # ax.legend()
            
    if cbar:
        cbar_ax = fig.add_axes([0., 0.25, 0.02, 0.5])  
        cl=fig.colorbar(pl,cax=cbar_ax,extend='max')
        cl.set_label('10$^3$ km$^{2}$')
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')

    return fig,ax,pl


















def pltHistDly_wyearly_wdays(histo, cmaps=['Blues','Reds'], daysplot =[115,190,298],
                             suptitle='', unit='', vmin=0,vmax=100, yearVmax=False, dayVmax=False,cbarlabel='10$^3$ km$^{2}$'):
    """
    fig,ax=pltHistDly_wyearly_wdays({scn:hist[scn+'icedia'] for scn in scenarios}, 
                                cmaps=['Greens','Purples'], daysplot =[85,125,298],
                                unit='mmol C m$^{-2}$', vmin=0,vmax=100,yearVmax=True,
                                   suptitle=f'Sea ice algae distribution - {rgn} (total area {rgnArea[rgn].values *1e-6*1e-6:.2f}$\cdot$10$^6$ km$^2$)',
                               )
    """
    binNames = {scn : [k for k in histo[scn].coords if k != 'dayofyear'][0] for scn in scenarios}
    bins={scn:histo[scn][binNames[scn]] for scn in scenarios}
    binsize = bins[scenarios[0]][1]-bins[scenarios[0]][0]

    cl=[plt.colormaps[cmap](256) for cmap in cmaps]

    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1],hspace=.35, top=0.92) 
    fig=plt.figure(figsize=(15,11))
    fig.suptitle(suptitle)
    
    ax= np.array([[plt.subplot(gs[i,j]) for j in [0,1,2]] for i in [0,1,2]])
    ax[0,0]= plt.subplot(gs[0,:2])
    ax[1,0]= plt.subplot(gs[1,:2])

    for i,scn in enumerate(scenarios):
        pl=ax[i,0].pcolormesh(days, bins[scn], histo[scn].T, vmin=vmin,vmax=vmax, cmap=cmaps[i] )
        ax[i,0].set_ylabel(unit)
        ax[i,0].set_title(scn.split('-')[0])
        monthTicks(ax[i,0],grid=True)
        ax[i,0].set_xlim(days[0],days[-1])
    
        mean_per_y=histo[scn].T.mean(dim='dayofyear')#/rgnArea[rgn]
        ax[i,2].barh(bins[scn], mean_per_y , height=binsize,
                    color=plt.colormaps[cmaps[i]](rescale( mean_per_y,vmin,vmax))
                    )
        ax[i,2].plot(mean_per_y, bins[scn], color=cl[i], alpha=0.8 )
        ax[i,2].set_ylim(ax[i,0].get_ylim())
        if yearVmax: ax[i,2].set_xlim(vmin,vmax)
        ax[i,2].set_ylabel(unit)

        cbar_ax = fig.add_axes([0.05, 0.71-0.3*i, 0.01, 0.2])  
        cbar=fig.colorbar(pl,cax=cbar_ax,extend='max')
        cbar.set_label(cbarlabel)
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')

    ax[0,2].set_title('Yearly distribution')

    

    for j in [0,1,2]:
        ax[2,j].set_title(days[daysplot[j]-1].strftime(format='%-d %B'))
        for i,scn in enumerate(scenarios):
            ax[2,j].barh(bins[scn], histo[scn].sel(dayofyear=daysplot[j]) , height=binsize,
                        alpha=0.8, #label=scn.split('-')[0],
                        # color=cl[i], 
                         color=plt.colormaps[cmaps[i]](rescale(histo[scn].sel(dayofyear=daysplot[j]), vmin,vmax) )
                        )
            ax[2,j].plot(histo[scn].sel(dayofyear=daysplot[j]), bins[scn],
                        alpha=0.8, label=scn.split('-')[0], c=cl[i]
                        )
            ax[2,j].set_ylim(ax[1,0].get_ylim())
            if dayVmax: ax[2,j].set_xlim(vmin,vmax)
            ax[2,j].set_ylabel(unit)
            ax[2,j].set_xlabel(cbarlabel)

            ax[i,0].plot( [days[daysplot[j]-1], days[daysplot[j]-1]], ax[1,0].get_ylim(), 'k:',lw=0.8)

        # line to day histograms
        con = ConnectionPatch(xyA=((daysplot[j]-1)/364,-0.1), coordsA=ax[1,0].transAxes, xyB=(0.5,1.1), coordsB=ax[2,j].transAxes, lw=0.8,ls=':') 
        fig.add_artist(con)
        con = ConnectionPatch(xyA=((daysplot[j]-1)/364,-0.1), coordsA=ax[1,0].transAxes, xyB=((daysplot[j]-1)/364,0), coordsB=ax[1,0].transAxes, lw=0.8,ls=':')
        fig.add_artist(con)

    # ax[2,0].legend(loc='center right', bbox_to_anchor=(-0.1, .5), ncol=1)   
    return fig,ax



def pltHistDly_wdays(histo, cmaps=['Blues','Reds'], daysplot =[115,190,298],
                     suptitle=None, unit='', vmin=0,vmax=100, dayVmax=False,
                     cbar=True, cbarlabel='10$^3$ km$^{2}$',
                     fig=None,ax=None):
    """
    fig,ax=pltHistDly_wdays({scn:hist[scn+'icedia'] for scn in scenarios}, 
                                cmaps=['Greens','Purples'], daysplot =[85,125,298],
                                unit='mmol C m$^{-2}$', vmin=0,vmax=100,
                                   suptitle=f'Sea ice algae distribution - {rgn} (total area {rgnArea[rgn].values *1e-6*1e-6:.2f}$\cdot$10$^6$ km$^2$)',
                               )
    """
    binNames = {scn : [k for k in histo[scn].coords if k != 'dayofyear'][0] for scn in scenarios}
    bins={scn:histo[scn][binNames[scn]] for scn in scenarios}
    binsize = bins[scenarios[0]][1]-bins[scenarios[0]][0]

    cl=[plt.colormaps[cmap](256) for cmap in cmaps]
    
    if fig is None:
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1],hspace=.35, top=0.92) 
        fig=plt.figure(figsize=(10,11))
        ax= np.array([[plt.subplot(gs[i,j]) for j in [0,1]] for i in [0,1,2]])
        ax[0,0]= plt.subplot(gs[0,:2])
        ax[1,0]= plt.subplot(gs[1,:2])

    if suptitle is not None: fig.suptitle(suptitle)

    for i,scn in enumerate(scenarios):
        pl=ax[i,0].pcolormesh(days, bins[scn], histo[scn].T, vmin=vmin,vmax=vmax, cmap=cmaps[i] )
        ax[i,0].set_ylabel(unit)
        # ax[i,0].set_title(scn.split('-')[0])
        monthTicks(ax[i,0],grid=True)
        ax[i,0].set_xlim(days[0],days[-1])

        if cbar:
            cbar_ax = fig.add_axes([0.0, 0.71-0.3*i, 0.01, 0.2])  
            cbar=fig.colorbar(pl,cax=cbar_ax,extend='max')
            cbar.set_label(cbarlabel)
            cbar_ax.yaxis.set_ticks_position('left')
            cbar_ax.yaxis.set_label_position('left')

    

    for j in [0,1]:
        ax[2,j].set_title(days[daysplot[j]-1].strftime(format='%-d %B'))
        for i,scn in enumerate(scenarios):
            ax[2,j].barh(bins[scn], histo[scn].sel(dayofyear=daysplot[j]) , height=binsize,
                        alpha=0.8, #label=scn.split('-')[0],
                        # color=cl[i], 
                         color=plt.colormaps[cmaps[i]](rescale(histo[scn].sel(dayofyear=daysplot[j]), vmin,vmax) )
                        )
            ax[2,j].plot(histo[scn].sel(dayofyear=daysplot[j]), bins[scn],
                        alpha=0.8, label=scn.split('-')[0], c=cl[i]
                        )
            ax[2,j].set_ylim(ax[1,0].get_ylim())
            if dayVmax: ax[2,j].set_xlim(vmin,vmax)
            if j==0: ax[2,j].set_ylabel(unit)
            ax[2,j].set_xlabel(cbarlabel)

            ax[i,0].plot( [days[daysplot[j]-1], days[daysplot[j]-1]], ax[1,0].get_ylim(), 'k:',lw=1)

        # line to day histograms
        # con = ConnectionPatch(xyA=((daysplot[j]-1)/364,0), coordsA=ax[0,0].transAxes, xyB=((daysplot[j]-1)/364,0), coordsB=ax[1,0].transAxes, lw=1,ls=':') 
        # fig.add_artist(con)
        con = ConnectionPatch(xyA=((daysplot[j]-1)/364,-0.1), coordsA=ax[1,0].transAxes, xyB=(0.5,1.1), coordsB=ax[2,j].transAxes, lw=1,ls=':') 
        fig.add_artist(con)
        con = ConnectionPatch(xyA=((daysplot[j]-1)/364,-0.1), coordsA=ax[1,0].transAxes, xyB=((daysplot[j]-1)/364,0), coordsB=ax[1,0].transAxes, lw=1,ls=':')
        fig.add_artist(con)

    # ax[2,0].legend(loc='center right', bbox_to_anchor=(-0.1, .5), ncol=1)   
    return fig,ax







def pltHistDly_wdayHist(histo, dayHist, dayClim, threshold=np.nan,
                        cmap='Reds', clFreq='tab:orange', clProp='tab:gray', lw=1,
                        suptitle='', unit='', varDayhistName='',
                        vmin=0,vmax=100, freqMax=0.1,
                        fig=None,ax=None, cbar=True, rightSpines=True, leftSpines=True,
                        start_sept=False,
                       ):
    """
    fig,ax = pltHistDly_wdayHist({scn:hist[scn+'icedia'] for scn in scenarios}, 
    {scn:hist[scn+'doyPARlimGTnlim'] for scn in scenarios}, 
                    {scn:clim[scn+'doyPARlimGTnlim'] for scn in scenarios}, threshold=None,
                    cmap='Greens', clFreq='tab:brown', clProp='tab:gray',
                    suptitle=f'Bottom ice algae distribution - {rgn}',
                    unit='mmol C m$^{-2}$', varDayhistName='PAR lim = N lim',
                        vmin=0,vmax=100, freqMax=0.1,
                       )
    """
    binNames = {scn : [k for k in histo[scn].coords if k != 'dayofyear'][0] for scn in scenarios}
    bins={scn:histo[scn][binNames[scn]] for scn in scenarios}
    binsize = bins[scenarios[0]][1]-bins[scenarios[0]][0]

    # cl=[plt.colormaps[cmap](256) for cmap in cmaps]

    if fig is None: fig,ax=plt.subplots(2,1,figsize=(10,7))
    fig.suptitle(suptitle)
    ax2 = [a.twinx() for a in ax]
    ax3 = [a.twinx() for a in ax]
    
    if start_sept: xday=days_sept
    else: xday = days

    for i,scn in enumerate(scenarios):
        if start_sept:
            pl=ax[i].pcolormesh(xday, bins[scn], histo[scn].reindex(dayofyear=doys_sept).T, vmin=vmin,vmax=vmax, cmap=cmap )
        else:
            pl=ax[i].pcolormesh(xday, bins[scn], histo[scn].T, vmin=vmin,vmax=vmax, cmap=cmap )
        ax[i].set_xlim(xday[0],xday[-1])
        ax[i].set_title(scn.split('-')[0])
        monthTicks(ax[i],grid=True,start_sept=start_sept)
        if leftSpines: ax[i].set_ylabel(unit)
        else: ax[i].set_yticks([])
        
        if start_sept: 
            meandoy=dayClim[scn].mean()
            if meandoy<244: meanday=xday[int(meandoy+(365-244))]
            else: meanday=xday[int(meandoy-244)]
        else: meanday=xday[int(dayClim[scn].mean()-1)]
        ax[i].plot([meanday,meanday],[bins[scn][0],bins[scn][-1]],c=clFreq, ls='--',
                   label=f'Mean day of \n{varDayhistName} \n' + meanday.strftime(format='%-d %B') )
        # medianday=days[int(dayClim[scn].median()-1)]
        # ax[i].plot([medianday,medianday],[0,vmax],'k--', label=f'Median day of {varDayhistName}')
    
        if threshold is not None :ax[i].plot([xday[0],xday[-1]],[threshold]*2,'k--',label= varDayhistName )
        
        ax[i].legend()
        
        if start_sept:
            indexname = [k for k in dayHist[scn].indexes][0]
            reindexHist = dayHist[scn].sel({indexname:doys_sept-0.5})
            ax2[i].plot(xday, reindexHist, clFreq, lw=lw, label = f'Frequency of day {varDayhistName}') 
            ax3[i].plot(xday, [reindexHist[:d].sum().values for d in range(365)], clProp)  
        else:
            ax2[i].plot(xday, dayHist[scn],clFreq, lw=lw, label = f'Frequency of day {varDayhistName}') 
            ax3[i].plot(xday, [dayHist[scn][:d].sum().values for d in range(365)], clProp)
            
        ax2[i].set_ylim(0,freqMax)
        if rightSpines:
            ax2[i].set_ylabel(f'Day of {varDayhistName} frequency')
            ax2[i].yaxis.label.set_color(clFreq)
        else: ax2[i].set_yticks([])

        ax3[i].set_ylim(0,1)
        if rightSpines:
            ax3[i].set_ylabel(f'Proportion of {varDayhistName}')
            ax3[i].spines["right"].set_position(("axes", 1.1))
            ax3[i].yaxis.label.set_color(clProp)
        else: ax3[i].set_yticks([])
    
    if cbar:
        cbar_ax = fig.add_axes([0., 0.25, 0.02, 0.5])  
        cl=fig.colorbar(pl,cax=cbar_ax,extend='max')
        cl.set_label('10$^3$ km$^{2}$')
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')

    return fig,ax


def addMeanDay(fig,ax, dayClim, dayName, col, start_sept=False, legend=True):

    yl=ax.get_ylim()

    if start_sept: 
        meandoy=dayClim.mean()
        if meandoy<244: meanday=days_sept[int(meandoy+(365-244))]
        else: meanday=days_sept[int(meandoy-244)]
    else: meanday=days[int(dayClim.mean()-1)]
    # ax.plot([meanday,meanday],[bins[0],bins[-1]],c=clFreq, ls='--',
    #            label=f'Mean day of \n{varDayhistName} \n' + meanday.strftime(format='%-d %B') )
    ax.plot([meanday,meanday],yl,'--', color=col, 
               label=f'Mean day of \n{dayName} \n' + meanday.strftime(format='%-d %B') )

    if legend: ax.legend()
    ax.set_ylim(yl)
    
    return fig,ax



def pltdayHists(hist, climdiag, varsDay, varsColor, varsDayName,
                suptitle=None, freqMax=0.1, rgn='',
                fig=None,ax=None,
               ):

    if fig is None: fig,ax=plt.subplots(2,1,figsize=(10,7))
    if suptitle is not None: fig.suptitle(suptitle)

    for i,scn in enumerate(scenarios):
        monthTicks(ax[i],grid=True)
        
        for vari in varsDay:
            ax[i].plot(days, hist[rgn+scn+vari], varsColor[vari])#, label = f'Frequency of {varsDayName[vars]}')
            meanday=days[int(climdiag[rgn+scn][vari].mean()-1)]
            ax[i].plot([meanday,meanday],[0,freqMax],'--',color=varsColor[vari], label = f'Mean day of {varsDayName[vari]}\n' + meanday.strftime(format='%-d %B') )
    
        ax[i].set_xlim(days[0],days[-1])
        ax[i].set_ylim(0,freqMax)
        ax[i].set_ylabel('Frequency')
        ax[i].legend()
        
        
def pltdayHists_singlescn(scn, hist, climdiag, varsDay, varsColor, varsDayName,
                        suptitle=None, freqMax=0.1, rgn='',
                        fig=None,ax=None,
                       ):

    if fig is None: fig,ax=plt.subplots(figsize=(10,3.5))
    if suptitle is not None: fig.suptitle(suptitle)

    monthTicks(ax,grid=True)

    for vari in varsDay:
        ax.plot(days, hist[rgn+scn+vari], varsColor[vari])#, label = f'Frequency of {varsDayName[vars]}')
        meanday=days[int(climdiag[rgn+scn][vari].mean()-1)]
        ax.plot([meanday,meanday],[0,freqMax],'--',color=varsColor[vari], label = f'Mean day of {varsDayName[vari]}\n' + meanday.strftime(format='%-d %B') )

    ax.set_xlim(days[0],days[-1])
    ax.set_ylim(0,freqMax)
    ax.set_ylabel('Frequency')
    ax.legend()