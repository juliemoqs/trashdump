import sys
from os import path, getcwd

import numpy as np
from scipy.signal import fftconvolve, resample, medfilt, find_peaks, resample_poly, convolve
from scipy.stats import trim_mean, sigmaclip
import pandas as pd
from astropy.io import fits
import glob

from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm

import batman
from wotan import flatten

import warnings

from collections import deque
from bisect import insort, bisect_left
from itertools import islice




def get_lc_files(kic, directory='data/lightcurves'):

    lc_files = '{1}/{0}/kplr*_llc.fits'.format(kic, directory)

    return sorted(glob.glob( lc_files ) )

  
#############################################################################
#
## SAVING/LOADING h5 data with Pandas following guide here:
## https://stackoverflow.com/questions/29129095/save-additional-attributes-in-pandas-dataframe/29130146#29130146
#
############################################################################
def h5store(filename, df, **kwargs):
    
    store = pd.HDFStore(filename, )
    store.put('data', df, format='f')
    store.get_storer('data').attrs.metadata = kwargs
    store.close()

#def h5load(store):
#    data = store['data']
#    metadata = store.get_storer('data').attrs.metadata
#    return data, metadata

def loadh5file(filename):

    if path.exists(filename):
    
        with pd.HDFStore(filename) as store:
            data = store['data']
            metadata = store.get_storer('data').attrs.metadata
            store.close()

        data.astype(np.float64)
        detrend_keys = []
        for col in data.columns:
            if 'detrend_mask' in col:
                detrend_keys.append(col)
        data[detrend_keys].astype(np.bool)
            
        return data, metadata
    else:
        print('NO FILE {} FOUND!!'.format(filename) )
        return None

def pandas_read_hdf(fname):

   return  pd.read_hdf(fname)


def create_df_from_lc(lc):

    df_dic = {}

    all_keys = np.concatenate( [lc.detrend_mask_keys, lc.detrend_flux_keys, lc.ses_keys, lc.num_keys, lc.den_keys])
    all_detrend_masks = np.concatenate(lc.detrend_masks)
    all_detrend_flux = np.concatenate(lc.detrend_flux)
    
    all_ses =  np.concatenate(lc.ses)
    all_num =  np.concatenate(lc.num)
    all_den =  np.concatenate(lc.den)

    all_arr = np.concatenate([all_detrend_masks, all_detrend_flux, all_ses, all_num, all_den])

    reshaped_arr = all_arr.reshape( (-1, len(all_keys)), order='F'  )

    print(np.shape(lc.detrend_masks ) )
    
    ses_df = pd.DataFrame( reshaped_arr, columns = all_keys,  )
        

    lc_dic = { 'time':lc.time,'pdcsap':lc.pdcsap, 'mom_centr1':lc.mom_centr1, 'mom_centr1_err':lc.mom_centr1_err, 'mom_centr2':lc.mom_centr2, 'mom_centr2_err':lc.mom_centr2_err}

    lc_df = pd.DataFrame( data=lc_dic )
    df = pd.concat([lc_df, ses_df], axis=1)    

    return df


def sigma_clip(x, y, upper_sig=4., lower_sig=np.inf, use_mad=False):

    with warnings.catch_warnings():
        # Ignore RuntimeWarning: invalid value encountered in less_equal."
        warnings.simplefilter("ignore", RuntimeWarning)
        
        if use_mad:
            cut_upper = y - np.nanmedian(y) < upper_sig*mad(y)
            cut_lower = np.nanmedian(y) - y < lower_sig*mad(y)
        else:
            y_nonans = y[~np.isnan(y)]
            clipped, _, _ = sigmaclip(y_nonans, low=lower_sig, high=upper_sig)
            cut_upper = y <= np.max(clipped)
            cut_lower = y >= np.min(clipped)

    cut = np.logical_and(cut_upper, cut_lower)
    
    return x[cut], y[cut], cut



def make_binned_flux(t, f, texp, P=None):

    if P is None:
        P = max(t)
        fold_t = t
    else:
        fold_t = np.mod(t,P)

    t_bins = np.arange(0, P+texp, texp)
    f_bin, _ = np.histogram(fold_t, bins=t_bins, weights=f)
    tr_num, t_binedge = np.histogram(fold_t,bins=t_bins, )
    
    f_bin_norm = f_bin/tr_num
    t_bin_mid = 0.5*(t_binedge[1:] + t_binedge[:-1] )
    
    return t_bin_mid, f_bin_norm




def moving_average(a, n) :
    
    ret = np.cumsum(a, dtype=float, )
    ret[n:] = ret[n:] - ret[:-n]
    #print(len(a), n )
    return np.pad(ret[n - 1:] / n, int(n/2), mode='reflect')




def rolling_trim_mean(x, n, percentile=0.1):
    
    lo, hi = int(percentile*n),int( percentile*n)
    
    x_sorted = np.sort(x[:n])
    res = np.repeat(np.nan, len(x))
    
    for i in range(n, len(x)):
        res[i-1] = x_sorted[lo:hi].mean()
        if i != len(x):
            idx_old = np.searchsorted(x_sorted, x[i-50])
            x_sorted[idx_old] = x[i]
            x_sorted.sort()
    return res




def running_median_insort(seq, window_size):
    """Contributed by Peter Otten"""
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return np.array(result)



def mad(arr, scale=1.4826):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    med = np.nanmedian(arr)
    return scale*np.nanmedian(np.abs(arr - med))



def median_detrend(flux, t_dur, exp_time = 0.020417):
    
    kernel = 2*int(10.*t_dur/exp_time)+1
    smoothed_flux = medfilt(flux, kernel)
    
    return flux/smoothed_flux



def flag_gap_edges(x, y, min_dif=0.3, sig=2.5, gap_size=1., npoints=15):

    dx = x[1:] - x[:-1]
    gap_idx = np.array(range(len(dx)))[dx>min_dif]

    dy_std = mad(y)
    cut = x==x

    y_median = np.median(y)
    
    for j in gap_idx:

        # cut left edge
        if np.nanmean((y[j-npoints:j]-y_median)**2./dy_std**2. ) > sig:
             cut &=  np.abs(x-x[j])>gap_size
         
        # cut right edge
        if np.nanmean((y[j+1:j+npoints+1]-y_median)**2. )  > sig:
             cut &= np.abs(x-x[j+1])>gap_size


    # check edge of data:
    if np.nanmean((y[-npoints:]-y_median)**2./dy_std**2. ) > sig:
        cut &= x[-1]-x > gap_size
    
        
    return x[cut], y[cut], cut





def fill_gap_AR(y, n_add, gap):
    
    nlag2 = min([5*n_add, len(y)-gap])
    nlag1 = min([5*n_add, gap])

    yreg = y[gap - nlag1:gap]
    yreg2 = y[gap+1:gap+nlag2][::-1]    
    
    #if np.sum((yreg-np.mean(yreg))**2.)>np.sum((yreg2-np.mean(yreg2))**2.):
    fill2 = AutoReg(yreg2, lags=int(0.37*nlag2) ).fit().predict( start=len(yreg2), end=len(yreg2)+n_add, )
    #else:
    fill = AutoReg(yreg, lags=int(0.37*nlag1) ).fit().predict( start=len(yreg), end=len(yreg)+n_add, )

    
    weights = np.linspace(0,1,len(fill))
    fillgap = fill * weights + fill2 * weights[::-1]

    return fillgap
    


def fill_all_gaps_AR(x, y, gap_size=3, cadence=1):
    
    dx = x[1:]-x[:-1]
        
    gap_inds = np.arange(len(dx), dtype=int)[dx>1.001*cadence]    
    
    y_tofill = []
    x_tofill = []
    fill_ind = []

    print('Filling Gaps ... ')
    
    for gap in tqdm(gap_inds):
        
        n_add = int(dx[gap]//cadence) 
        fill_ind.extend(np.zeros(n_add)+gap)
        x_tofill.extend(np.linspace(x[gap]+cadence, x[gap+1], n_add) )

        if n_add>gap_size:
            y_tofill.extend( fill_gap_AR(y, n_add-1, gap ) ) 
        else:
            y_tofill.extend( y[gap-n_add:gap][::-1])  
    
    y_filled = np.insert(y, obj=fill_ind, values=y_tofill, axis=0)
    x_filled = np.insert(x, obj=fill_ind, values=x_tofill, axis=0)
            
    return x_filled, y_filled





def pad_time_series(x, y, min_dif=0.021,cadence=0.02043361, in_mode='reflect', pad_end=False, fill_gaps=True, constant_val=0., len_pad=None):

    padded_y = y
    padded_x = x

    truth_array = np.ones_like(x, dtype=bool)


    if fill_gaps:
        
        
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        bad_idx = np.arange(len(dx))[dx>min_dif]
        x_add = []
        y_add = []

        idx2add = []
        


        for j in bad_idx:

            x_add.extend( list(np.arange(x[j]+cadence, x[j+1], cadence) ) )
            n_add = len(np.arange(x[j]+cadence, x[j+1], cadence))

            idx2add.extend( [j+1]*n_add )

            if in_mode=='line':
                y1 = np.median(y[j-10:j])
                y2 = np.median(y[j:j+10])
                y_add.extend( list(y1 + ((y2-y1)/dx[j]) * (np.arange(x[j]+cadence, x[j+1], cadence) - x[j] ) ) )

            elif in_mode=='mean':
                y_add.extend( [np.mean(y)]*n_add )

            elif in_mode=='constant':
                y_add.extend( [constant_val]*n_add )

                
            elif in_mode=='reflect':
                n_left = int(np.ceil(n_add/2.))
                n_right = int(np.floor(n_add/2.))

                y_negative = -1*y + 2.*np.median(y)

                if j-n_left<0:
                    y_repeat_begin = np.concatenate( [y_negative[::-1],y]  )
                    y_add.extend(  y_repeat_begin[range(j+len(y)-n_left,j+len(y))][::-1] )
                else:
                    y_add.extend( y_negative[range(j-n_left,j)][::-1]  )

                if j+n_right > len(y):
                    y_repeat_end = np.concatenate( [ y,y_negative[::-1] ]  )
                    y_add.extend(  y_repeat_end[range(j,j+n_right)][::-1] )
                else:
                    y_add.extend(  y_negative[range(j,j+n_right)][::-1] )

            elif in_mode=='autoreg':

                y_add.extend( fill_gap_AR(y, n_add, j) )

                
                if n_add>3:  
                    y_add.extend( fill_gap_AR(y, n_add, j ) ) 
                else:
                    y_add.extend( y[int(j-n_add):int(j)][::-1])


        if in_mode=='AR':
            padded_x, padded_y  = fill_all_gaps_AR(x, y, gap_size=3, cadence=cadence)
            
        else:
            padded_x = np.insert(x, idx2add, x_add    )
            padded_y = np.insert(y, idx2add, y_add  )

        truth_array = np.insert(truth_array, idx2add, np.zeros_like(x_add)  )
    
    if pad_end:

        if len_pad is None:
            padded_len = int(2**np.ceil(np.log2(len(padded_x)) )  )
        else:
            padded_len = len_pad

        total2add = padded_len-len(padded_x)
        add_right =  int(total2add/2)

        if total2add%2==1:
            add_left = int(total2add/2) + 1
        else:
            add_left = int(total2add/2)

        x_add_right = padded_x[-1] + np.arange(1, add_right+1)*cadence
        x_add_left =  padded_x[0] - np.arange(1, add_left+1)[::-1]*cadence
        padded_x = np.concatenate([x_add_left, padded_x, x_add_right ])
        
        y_add_left = padded_y[:add_left]
        y_add_right = padded_y[-add_right:]
        padded_y = np.concatenate([y_add_left[::-1], padded_y, y_add_right[::-1] ])

        truth_array = np.concatenate([np.zeros(add_left),truth_array,np.zeros(add_right)])

    truth_array = np.array(truth_array, dtype=bool)

    return padded_x, padded_y, truth_array



def get_transit_depth_ppm(rp, rs):

    return 84. * rp**2. / rs**2.




def get_p_tdur_t0(tce):
    return tce[1], tce[4], tce[3]


