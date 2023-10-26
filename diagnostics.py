import xarray as xr
import numpy as np
import pandas as pd

from datetime import datetime, timedelta




######  IA growth limitation

def limPAR(par): return np.tanh( 2.0 * par)
def limN(n): return n/0.03 / ( n/0.03 + 1.0) # for values in mmol m2 multply by z bottom ice = 0.03m



def doyIApeak(ia,timedim='dayofyear'):
    """ first day max IA is reached
        computed on 14 day rolling mean, before sept 1"""
    
    if timedim=='dayofyear': doys=ia[timedim]
    else: doys=ia[timedim].dt.dayofyear
   
    # for finding day of max, smooth daily signal with 14 day rolling mean
    smth_ia = ia.rolling(dim={timedim:14},center=True,min_periods=1).mean()
    
    # max ; before day 244 ; need to fill nan because argmax is stupid
    iday = smth_ia.where(doys<244).fillna(0).argmax(dim=timedim)+1    
   
    # iday > 1 to filter NaNs
    res= iday.where(iday>1)
    
    del smth_ia, iday
    return res


def doyIAstd(ia,timedim='dayofyear'):
    """ first day IA reached 1 standard deviation
        computed on 14 day rolling mean, before sept 1"""
    
    if timedim=='dayofyear': doys=ia[timedim]
    else: doys=ia[timedim].dt.dayofyear
   
    # for finding day, smooth daily signal with 14 day rolling mean
    smth_ia = ia.rolling(dim={timedim:14},center=True,min_periods=1).mean()
    
    # std of IA before sept 1
    std = smth_ia.where(doys<244).fillna(0).std(dim=timedim,keep_attrs=True)
    
    # IA capped at 1 std
    smth_ia_capped = xr.where(smth_ia>std, std,smth_ia)
    # first day of reaching 1 std is first max of IA capped at std
    iday = smth_ia_capped.fillna(0).argmax(dim=timedim,keep_attrs=True)+1 
   
    # iday > 1 to filter NaNs
    res= iday.where(iday>1)
    
    del smth_ia, smth_ia_capped, iday
    return res



#### Day of reaching limitation threshold

def doyPARlim(lim, threshold=0.5,timedim='dayofyear'):
    """ first day PAR limitation reaches threshold : first max of lim capped at threshold """
    
    # smoth daily signal with 14 day rolling mean
    smth_lim = lim.rolling(dim={timedim:14},center=True,min_periods=1).mean()
    # lim capped at threshold
    min_lim_thrshld = xr.where(smth_lim<threshold, smth_lim, threshold)
    # for grid cell always lower than threshold : set =0, argmax will be first day and can be filtered out 
    min_lim_thrshld = xr.where(min_lim_thrshld.max(dim=[timedim]) <threshold, 0, min_lim_thrshld)
    # argmax returns index : +1 to get day of year
    iday = min_lim_thrshld.argmax(dim=timedim,keep_attrs=True)+1 
    # for grid cell with only nan : argmin is 1
    res=iday.where(iday>1)
    
    del smth_lim, min_lim_thrshld
    return res


def doySIbreak(sic,threshold=0.5,timedim='dayofyear'):
    """ first day after winter max SI concetration is below threshold
        return Nan for cells with max above or min below for 30 day rolling mean
    """
    if timedim=='dayofyear': doys=sic[timedim]
    else: doys=sic[timedim].dt.dayofyear
   
    # set =0 if SI always below or  above threshold- will return day 0 which is then changed to nan
    # use 30 day rolling mean for this
    smth30_sic = sic.rolling(dim={timedim:30},center=True,min_periods=1).mean()
    sic_dropbelow = xr.where((smth30_sic.max(dim=[timedim]) <threshold) | (smth30_sic.min(dim=[timedim]) >threshold), 0, sic) 
    # for finding day of threshold, smooth daily signal with 14 day rolling mean
    smth_sic = sic_dropbelow.rolling(dim={timedim:14},center=True,min_periods=1).mean()
    
    # winter max ; before day 200 ; need to fill nan because argmax is stupid
    wmax = smth_sic.where(doys<200).fillna(0).argmax(dim=timedim)+1    
    # SIc afetr winter max
    sic_afterwintermax = smth_sic.where(doys>=wmax)
    
    # floor at threshold
    sic_floored = xr.where( sic_afterwintermax<threshold, threshold, sic_afterwintermax)
    # # first min is day threshood is reached
    iday= sic_floored.fillna(1).argmin(dim=timedim)+1
    # iday > 1 to filter NaNs
    res= iday.where(iday>1)
    
    del smth30_sic, sic_dropbelow, smth_sic, wmax, sic_afterwintermax, sic_floored, iday
    return res



def doyNlim(lim,threshold=0.5,timedim='dayofyear'):
    """ first day after winter max Nlim is below threshold 
        possibility for a recharge with 2nd max above threshold for more than 5 days
    """
    if timedim=='dayofyear': doys=lim[timedim]
    else: doys=lim[timedim].dt.dayofyear

    # smoth daily signal with 14 day rolling mean
    smth_lim = lim.rolling(dim={timedim:14},center=True,min_periods=1).mean()
    
    # set =0 if lim always below threshold - will return day 0 which is then changed to nan
    lim_dropbelow = xr.where(smth_lim.max(dim=timedim) <threshold, 0, smth_lim)
    
    # winter max ; before day 200 ; need to fill nan because argmax is stupid
    wmax = lim_dropbelow.where(doys<200).fillna(0).argmax(dim=timedim)+1
    lim_afterwintermax = lim_dropbelow.where(doys>=wmax)
    
    # floor at threshold
    lim_floored = xr.where( lim_afterwintermax<threshold, threshold, lim_afterwintermax)
    # # first min is day threshood is reached
    doyNfrst = lim_floored.fillna(1).argmin(dim=timedim)+1
    
    # check if there is a another max after that is greater that threshold
    lim_after = lim_floored.where((doys>doyNfrst) & (doys<200))
    max_after = lim_after.max(dim=timedim)
    dmax_after = lim_after.fillna(0).argmax(dim=timedim)+1
    # mask cells that have a 2nd max lower than threshold and then select days after 2nd max and before 200 and 
    lim_after_2ndmax = lim_after.where(max_after>threshold).where( (doys>dmax_after) & (doys<250))
    # take min to get day threshold is reached
    doyNscnd = lim_after_2ndmax.fillna(1).argmin(dim=timedim)+1
    
    # drop spikes : when 2nd time the threshold is reached is less than 5 days after 2nd max
    doyNscnd = doyNscnd.where(doyNscnd>dmax_after+5)
    
    # doyNfrst == 1 to filter NaNs and drop after day 250
    doyNfrst = doyNfrst.where((doyNfrst>1)&(doyNfrst<250))
    doyNscnd = doyNscnd.where(doyNscnd<250)
    res= xr.where(doyNscnd>1, doyNscnd, doyNfrst )
    
    del smth_lim,lim_dropbelow, wmax, lim_afterwintermax, lim_floored, doyNfrst, lim_after, max_after, dmax_after, lim_after_2ndmax, doyNscnd
    return res


def doyPARlimGTnlim(limPAR,limN,timedim='dayofyear'):
    """ first day PAR lim is greater that N lim : first day PARlim - Nlim >0 """
    
    # smoth daily signal with 14 day rolling mean
    smth_limPAR = limPAR.rolling(dim={timedim:14},center=True,min_periods=1).mean()
    smth_limN = limN.rolling(dim={timedim:14},center=True,min_periods=1).mean()

    # first min of max(0, PARlim - Nlim )
    diffLim_floored = xr.where(smth_limPAR-smth_limN<0, smth_limPAR-smth_limN, 0 )
    iday= diffLim_floored.argmax(dim=timedim)+1
    iday.name='dayoflim'
    res = iday.where(iday>1)
    
    del smth_limPAR, smth_limN, diffLim_floored, iday
    return res


def lim_daylimequal(limpar, doyPARlimGTnlim,timedim='dayofyear'):
    """ limitation on day of swith from PAR to N limitation"""
    
    # can't select with Nan so add value of PAR limitation on day 1 (ie jan 1) 
    lim_daylimequal = limpar.isel({timedim:doyPARlimGTnlim.fillna(1).astype(int)-1})
    # filter out Nans : areas with lim_daylimequal = lim PAR jan 1 
    res= lim_daylimequal.where(lim_daylimequal>limpar.isel({timedim:0}))
    
    del lim_daylimequal
    return res


def least_lim(limpar, limN,timedim='dayofyear'):
    """ least limitation : max of min(lim par, lim N) before day 250 """

    # smoth daily signal with 14 day rolling mean
    smth_limpar = limpar.rolling(dim={timedim:14},center=True,min_periods=1).mean()
    smth_limN = limN.rolling(dim={timedim:14},center=True,min_periods=1).mean()

    lim = xr.where(smth_limpar<smth_limN, smth_limpar, smth_limN)
    res= lim.isel({timedim:slice(0,250)}).max(dim=timedim)
    
    del lim, smth_limpar, smth_limN
    return res



def lenghtNonLimGrwth(doyLimPAR, doylimN, doySIbreak):
    """ days between PAR half limitation and earliest of N half limitation or sea ice break up, up to sept"""
    
    # earliest of N half limitation or sea ice break up
    # filter out where doySIbreak is nan by filling nan with last day we want
    end = xr.where(doylimN<doySIbreak.fillna(250), doylimN, doySIbreak)
    return end - doyLimPAR



def lengthPAR(sic, doySIbreak, doyPARlim15):
    """ days from PAR lim =.15 to SI break up, up to sept """
    
    # for areas where SI break not defined, fill nan with 250, but only if ice is always there
    # areas with ice defined as areas where max of 30 day rolling mean is greater than 50% """
    smth_sic = sic.rolling(dim={'time_counter':30},center=True,min_periods=1).mean() 
    max_smth_sic = smth_sic.groupby('time_counter.year').max('time_counter')
    dsib = xr.where( max_smth_sic>0.5, doySIbreak.fillna(244), np.nan)
    
    return dsib - doyPARlim15



def growthDays(limpar, limN, doySIbreak):
    """ sum of limitation up to sea ice break or sept """

    # smoth daily signal with 5 day rolling mean
    smth_limpar = limpar.rolling(dayofyear=5,center=True,min_periods=1).mean()
    smth_limN = limN.rolling(dayofyear=5,center=True,min_periods=1).mean()

    lim = xr.where(smth_limpar<limN, smth_limpar, smth_limN)
    # filter out where doySIbreak is nan by filling nan with last day we want
    lim_before_SIbreak = lim.where((lim.dayofyear<doySIbreak.fillna(250)) & (lim.dayofyear<250))
    gd= lim_before_SIbreak.sum('dayofyear')
    return gd.where(gd>0)





def doySIbreak_fromend(sic,threshold=0.5,timedim='dayofyear'):
    """ last day before sept SI concetration is above threshold   """

    # smoth daily signal with 14 day rolling mean
    smth_sic = sic.rolling(dim={timedim:14},center=True,min_periods=1).mean()
    # set =0 if SI always below or  above threshold- will return day 0 which is then changed to nan
    sic_dropbelow = xr.where((smth_sic.max(dim=[timedim]) <threshold) | (smth_sic.min(dim=[timedim]) >threshold), 
                             0, smth_sic)
    # reverse time indexing starting at sept 1 and ending jan 1
    sic_reversed =sic_dropbelow.reindex({timedim:sic_dropbelow[timedim][243::-1]})
    # cap at threshold
    sic_rev_cap = xr.where(sic_reversed<=threshold,sic_reversed,threshold)
    # argmax is first day sic is above threshold
    ndays_beforsept1 = sic_rev_cap.fillna(0).argmax(dim=timedim) 
    # day of year : sept 1 - ndays_beforsept1
    doysib = 244-ndays_beforsept1
    # filter nans with sept 1 
    return doysib.where(doysib<244)



def doySIfreezeUp(sic,threshold=0.5,timedim='dayofyear'):
    """ first day after sept 1 SI concetration is above threshold   """
    
    if timedim=='dayofyear': doys=sic[timedim]
    else: doys=sic[timedim].dt.dayofyear
   
    # set =0 if SI always below or  above threshold- will return day 0 which is then changed to nan
    # use 30 day rolling mean for this
    smth30_sic = sic.rolling(dim={timedim:30},center=True,min_periods=1).mean()
    sic_dropbelow = xr.where((smth30_sic.max(dim=[timedim]) <threshold) | (smth30_sic.min(dim=[timedim]) >threshold), 0, sic) 
    # for finding day of threshold, smooth daily signal with 14 day rolling mean
    smth_sic = sic_dropbelow.rolling(dim={timedim:14},center=True,min_periods=1).mean()
   
    # after septembre 1 : day of year 244
    sic_aftersept1 = smth_sic.where(doys>244)
    # cap at threshold
    sic_floored = xr.where( sic_aftersept1>threshold, threshold, sic_aftersept1)
    # # first max is day threshold is reached
    iday= sic_floored.fillna(0).argmax(dim=timedim)+1
    # iday == 1 to filter NaNs
    return iday.where(iday>1)


