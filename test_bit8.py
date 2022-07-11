#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:56:20 2022

@author: rfwilso1
"""

import numpy as np
from astropy.io import fits
from trashdump import dump

import sys
import glob
import pandas as pd
from tqdm import tqdm



def mad(x, scale=1.4826):
    return 1.4826 * np.nanmedian(np.abs(x-np.nanmedian(x))) 

def norm(x):
    return x/np.nanmedian(x)
    

def get_eleanor_lightcurve(f):
    
    
    hdulist = fits.open(f, mmap=False)
    lc = hdulist[1].data
    hdr = fits.getheader(f)
    
    metadata = {'tic':hdr['TIC_ID'], 'ccd':hdr['CCD'], 'camera':hdr['CAMERA'],'sector':hdr['SECTOR']}
    
    tesslc = dump.LightCurve(time=lc['time'], flux=lc['CORR_FLUX'],        
                             flux_err=lc['FLUX_ERR'], flags=lc['QUALITY'],
                            qflags=None, mission='TESS', ID=hdr['TIC_ID'] )
    
    hdulist.close()
    
    
    wotan_kw = {'method':'biweight','robust':True, 'window_length':.5,'edge_cutoff':0.,'break_tolerance':0.5,'cval':5}
    
    tesslc.flatten(wotan_kw)
    
    return tesslc, metadata



def make_bitmask(q, qflags=[]):
    
    qflags = [f-1 for f in qflags]
    
    good_guys = np.ones_like(q, dtype=bool)
    quality_mask =  np.sum(2 ** np.array(qflags))
        
    good_guys &= (q & quality_mask) == 0
    
    return good_guys


def test_bit(lc, bit=8):
    
    all_flags = np.arange(1,20)
    test_flags = np.delete(all_flags, bit-1)
    
    bit_all = make_bitmask(lc.flags, all_flags)
    bit_test = make_bitmask(lc.flags, test_flags)
    
    norm_flux_all = norm(lc.flux[bit_all])
    norm_flux_test = norm(lc.flux[bit_test])
    
    std_all = np.nanstd(norm_flux_all)
    std_test = np.nanstd(norm_flux_test)
    
    mad_all = mad(norm_flux_all)
    mad_test = mad(norm_flux_test)
    
    frac_all = sum(bit_all)/len(bit_all)
    frac_test = sum(bit_test)/len(bit_test)
    
    
    result_dic={'frac_all':frac_all, 'frac_test':frac_test,
            'std_all_flags':std_all, 'std_test_flag':std_test,
                'mad_all_flags':mad_all,'mad_test_flag':mad_test}
    
    return result_dic



def merge_dicts(x, y):
    return {**x, **y}


def test_bitflags(f):
    
    lc, meta = get_eleanor_lightcurve(f)
    tests = test_bit(lc)
    
    return merge_dicts(meta,tests)
    

def run_on_all_lc(fnames):
    dlist = [test_bitflags(f) for f in tqdm(fnames)]
    return pd.DataFrame(dlist)



if __name__=='__main__':
    

    
    fnames = glob.glob(sys.argv[1])
    res=run_on_all_lc(fnames)
    
    print(res)
    
    res.to_csv('bit8_results.txt', index=False)
        
    
