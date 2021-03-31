
'''
Removal of Exoplanet Candidates Yearning to be Condoned as Lookalikes and Eclipsing BINaries
'''


import sys
from os import path, getcwd

import numpy as np
from scipy.signal import fftconvolve, resample, medfilt, find_peaks, resample_poly, convolve
from scipy.stats import trim_mean, sigmaclip
import pandas as pd
from astropy.io import fits
import glob

import matplotlib.pyplot as plt

from tqdm import tqdm

import batman
import warnings


from scipy.optimize import curve_fit

from .transit_search import make_transit_mask, calc_mes, calc_var_stat




class RecycleBin(object):

    def __init__(self, Dump, TCEs):


        self.dump = Dump
        self.tces = TCEs

        self.exptime = Dump.lc.exptime
        self.limb_coeff = Dump.limb_darkening[0]

        self.time = Dump.lc.time[Dump.lc.mask]
        self.flux = Dump.lc.flux[Dump.lc.mask]
        self.trend = Dump.lc.trend[Dump.lc.mask]
        self.fluxerr = np.sqrt(calc_var_stat(self.flux-1., 1., exp_time=self.exptime, method='mad')  )


    def _calc_mes(self, tce_num):

        tce = self.tces[tce_num]

        t0 = tce[3]
        width = tce[4]
        P = tce[1]


        
        return 1.


    def cosine_vs_transit_local(self, tce_num, showplot=False, nwidth=4):

        tce = self.tces[tce_num]

        t0 = tce[3]
        width = tce[4]
        P = tce[1]

        chit, chic, nf = compare_cosine_and_transit_model(time=self.time,
                                                          flux=self.flux*self.trend,
                                                          t0=t0, width=width, P=P, 
                                                          exptime=self.exptime,
                                                          limb_dark_coeffs=self.limb_coeff,
                                                          flux_err=self.fluxerr,
                                                          plot=showplot, nwidth=nwidth,
                                                          local_plots=False)

        return chit, chic, nf

    def cosine_vs_transit_global(self, tce_num, showplot=False, nwidth=4):

        tce = self.tces[tce_num]

        t0 = tce[3]
        width = tce[4]
        P = tce[1]


        chit, chic, nf = compare_cosine_and_transit_model(time=self.time,
                                                          flux=self.flux,
                                                          t0=t0, width=width, P=P, 
                                                          exptime=self.exptime,
                                                          limb_dark_coeffs=self.limb_coeff,
                                                          flux_err=self.fluxerr,
                                                          plot=showplot, nwidth=nwidth,
                                                          local_plots=False,global_fit=True)

        return chit, chic, nf




    def mes_metrics(self, tce_num ):

        tce = self.tces[tce_num]

        t0 = tce[3]
        width = tce[4]
        P = tce[1]
        
        
        

        return 1.



    def odd_even_depth_test(self, tec_num):

        

        return 1.




    


    


    


def fit_sinewave(x, y, yerr, p0=None):
    
    sine_func = lambda x,a,p,x0: a*np.cos((2.*np.pi) * (x-x0) / p) 
    
    
    result = curve_fit(sine_func, xdata=x, ydata=y, p0=p0, sigma=yerr, 
                       bounds=[[-np.inf, p0[1]/3.,-np.inf],[np.inf, p0[1]*4.,np.inf]], 
              check_finite=True, method='trf', jac=None, )
    
    resids = y - sine_func(x, *result[0])
    
    chi2 = np.sum(resids**2./yerr**2.)
    
    return chi2, resids, np.sort(x), sine_func(np.sort(x), *result[0])
    
    
    
def fit_transit(x,y,yerr,period,limb_darkening,exptime,p0,t0):
    
    params = batman.TransitParams()
    params.t0 = t0                       #time of inferior conjunction
    params.per = np.pi*(p0[1])/np.arcsin(1./50.)                      #orbital period
    params.rp = 0.1                     #planet radius (in units of stellar radii)
    params.a = 50.                       #semi-major axis (in units of stellar radii)
    params.inc = 90.                     #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = limb_darkening            #limb darkening coefficients [u1, u2]
    params.limb_dark = "nonlinear"       #limb darkening model
              
    m = batman.TransitModel(params, x, exp_time=exptime,)

    def lc_func(t, *par):
        
        rprs, width, t0 = par
        params.per = np.pi*(width)/np.arcsin(1./50.)                      #orbital period
        params.rp = 10.**rprs   
        params.t0 = t0
        return m.light_curve(params)-1.
    
    
    result = curve_fit(lc_func, xdata=x, ydata=y, p0=p0, sigma=yerr, absolute_sigma=True, 
                       bounds=[[-8, p0[1]/2., -p0[1]],[0., p0[1]*2., p0[1]]],
                  check_finite=True, jac=None, )
    
    
    resids = y - lc_func(x, *result[0])
    chi2 = np.sum(resids**2./yerr**2.)
    
    x_fold =(x+period/2.)%period-period/2.
            
    return chi2, resids, np.sort(x_fold), lc_func(x_fold, *result[0] )[np.argsort(x_fold)]
    
    
    


def compare_cosine_and_transit_model(time,flux,t0,width,P,limb_dark_coeffs,exptime,flux_err=None,
                                       nwidth=4,plot=True,local_plots=False, global_fit=False):

    if global_fit:
        foldtime = (time - t0 + P/2.)%P - P/2.
        mask = ~make_transit_mask(time, P, t0, width*nwidth/2)

        transit_time = foldtime[mask]
        cosine_time = foldtime[mask]

        transit_flux = flux[mask]-1.
        cosine_flux = flux[mask]-1.

        fluxerr=flux_err[mask]


    else:
        transit_time,cosine_time,transit_flux,cosine_flux,fluxerr,_ = get_local_cosine_and_transit_fits(time, flux, t0, width, P, limb_dark_coeffs, flux_err=flux_err, nwidth=nwidth, show_plots=local_plots)

    
    trim_to_width = np.abs(transit_time)< nwidth*width/2.
    transit_time = transit_time[trim_to_width]
    transit_flux = transit_flux[trim_to_width]
    
    trim_to_width_cos = np.abs(cosine_time)< nwidth*width/2.
    cosine_time = cosine_time[trim_to_width_cos]
    cosine_flux = cosine_flux[trim_to_width_cos]
    
    sine_chi2s = []
    all_sine_results = []
    for n in [0.5,1,2]:
        results = fit_sinewave(cosine_time, cosine_flux, yerr=fluxerr[trim_to_width_cos],p0=[1e-4, n*width,0.])  
        sine_chi2s.append(results[0])
        all_sine_results.append(results)
        
    
    sine_chi2, sine_resids, sine_x, sine_y = all_sine_results[np.argmin(sine_chi2s)]

    
    tran_chi2, tran_resids, tran_x, tran_y = fit_transit(transit_time, transit_flux, yerr=fluxerr[trim_to_width], 
                                                         p0=[-5,width,0.], t0=0.,
                                                         period=P,limb_darkening=limb_dark_coeffs,
                                                         exptime=exptime)
    

    tran_chi2 = tran_chi2/len(tran_resids)
    sine_chi2 = sine_chi2/len(sine_resids)
    

    if plot:
        
        t_bins = np.arange(min(transit_time), max(transit_time)+width/3, width/3.)
    
        cosine_hist, t_hist = np.histogram(cosine_time, bins=t_bins, weights=cosine_flux) 
        transit_hist, _ = np.histogram(transit_time, bins=t_bins, weights=transit_flux)
        t_transit_count, _ = np.histogram(transit_time, bins=t_bins,)     
        t_cosine_count, _ = np.histogram(cosine_time, bins=t_bins,)     

    
        sine_resid_hist,_ = np.histogram(cosine_time, bins=t_bins, weights=sine_resids)
        tran_resid_hist,_ = np.histogram(transit_time, bins=t_bins, weights=tran_resids)
        
        fig = plt.figure(figsize=(6,4))
        gs = fig.add_gridspec(3, 4, hspace=0, wspace=0)
        
        ax3 = fig.add_subplot(gs[:2, 2:], )
        ax4 = fig.add_subplot(gs[2, 2:], sharex=ax3,)

        ax1 = fig.add_subplot(gs[:2, :2], sharex=ax3,sharey=ax3)
        ax2 = fig.add_subplot(gs[2, :2], sharex=ax3, sharey=ax4)
        
        ax1.plot(cosine_time, cosine_flux, '.', color='0.7',markersize=2)
        ax1.plot((t_hist[1:]+t_hist[:-1])/2., cosine_hist/t_cosine_count, 'o', color='dodgerblue',markeredgecolor='k')
        
        ax3.plot(transit_time, transit_flux, '.', color='0.7',markersize=2)
        ax3.plot((t_hist[1:]+t_hist[:-1])/2., transit_hist/t_transit_count, 'o', color='dodgerblue',markeredgecolor='k')
        
        
        ax1.plot(sine_x, sine_y, '-', color='tomato', lw=2, 
                 label='$\mathregular{\chi^2_{\\nu, cosine}}=$'+'{:.4f}'.format(sine_chi2))
        ax1.legend(framealpha=0., loc='lower center')
        
        ax3.plot(tran_x, tran_y, '-', color='tomato', lw=2, 
                 label='$\mathregular{\chi^2_{\\nu, transit}}=$'+'{:.4f}'.format(tran_chi2))
        ax3.legend(framealpha=0., loc='lower center')
        
        
        ax2.plot(cosine_time, sine_resids, '.', color='0.7',markersize=2)
        ax2.plot((t_hist[1:]+t_hist[:-1])/2., sine_resid_hist/t_cosine_count, 'o', color='dodgerblue',
                 markeredgecolor='k')
        ax2.plot(sine_x, np.zeros_like(sine_y), '-', color='tomato', lw=2)

        ax4.plot(transit_time, tran_resids, '.', color='0.7',markersize=2)
        ax4.plot((t_hist[1:]+t_hist[:-1])/2., tran_resid_hist/t_transit_count, 'o', color='dodgerblue',
                 markeredgecolor='k')
        ax4.plot(tran_x, np.zeros_like(tran_y), '-', color='tomato', lw=2)
        
        ax4.set_xlabel('$\mathregular{\Delta t_0}$ [days]')
        ax2.set_xlabel('$\mathregular{\Delta t_0}$ [days]')
        
        ax1.set_ylabel('$\mathregular{\delta F/F }$')
        ax2.set_ylabel('Resids')
        
        #ax1.set_xticklabels([])
        #ax3.set_xticklabels([])


        plt.tight_layout()
        plt.show()
        
    nfreedom = len(transit_time) - 3.
        
    return sine_chi2, tran_chi2, nfreedom




def get_local_cosine_and_transit_fits(time, flux, t0, width, P, limb_dark_coeffs, flux_err=None, nwidth=1,
                                     show_plots=False):
    
    
    if flux_err is None:
        flux_err = np.ones(len(flux))
        
    i=0
    
    exptime = np.median(time[1:]-time[:-1])

    dt_cosine = np.array([])
    dt_transit = np.array([])
    df_transit = np.array([])
    df_cosine = np.array([])    
    dferr_transit = np.array([])
    dferr_cosine = np.array([])
    
    while t0 < max(time):

        t0 += i*P
        i+=1

        t_mask = np.abs(t0-time) <= nwidth*width
        time_seg = time[t_mask]
        flux_seg = flux[t_mask]
        fluxerr_seg = flux_err[t_mask]

        if len(flux_seg)> 0.05 * (width/exptime):
            c_results=[]
            t_results=[]
            
            try:
                c_result = fit_cosine_local_poly(x=time_seg, y=flux_seg, yerr=fluxerr_seg,
                                               p0=[0,0,0,-0.,width,t0])

                t_result = fit_transit_local_poly(x=time_seg, y=flux_seg,yerr=fluxerr_seg,t0=t0,
                                                      width=width,
                                               p0=[0,0,0,-2,t0], limb_darkening=limb_dark_coeffs)


                if show_plots:
                    plt.plot(time_seg, flux_seg, 'ko')
                    plt.plot(time_seg, t_result[1], 'r-', label='Transit')
                    plt.plot(time_seg, c_result[1], 'b-', label='Cosine')

                    plt.axvline(t0, ls='--',color='0.7')
                    plt.show()

                cosine_par = c_result[2]
                transit_par = t_result[2]

                t0_fit_tr = transit_par[-1]
                t0_fit_cos = cosine_par[-1]

                dt_transit = np.append(dt_transit, time_seg-t0_fit_tr)
                dt_cosine = np.append(dt_cosine, time_seg-t0_fit_cos)
                df_transit = np.append(df_transit, flux_seg/np.poly1d(transit_par[:3][::-1])(time_seg))
                df_cosine = np.append(df_cosine, flux_seg/np.poly1d(cosine_par[:3][::-1])(time_seg))
                dferr_transit = np.append(dferr_transit, fluxerr_seg)
                dferr_cosine = np.append(df_cosine, fluxerr_seg)

            
            except RuntimeError:
                pass
    
    return dt_transit, dt_cosine, df_transit-1., df_cosine-1., dferr_transit, dferr_cosine


def fit_cosine_local_poly(x,y,yerr,p0):
    
    all_chi2 = []
    all_fitpar = []
    for n in [0.5,1]:
        fitfunc = lambda x,a0,a1,a2,amp,per,t0: a0 + a1*x +a2*x**2. + amp*np.cos(2*np.pi*(x-t0)/per)
        a0,a1,a2,amp,per,t0 = p0
        p0_n = a0,a1,a2,amp,per*n,t0
        fit_par, fit_var = curve_fit(fitfunc, xdata=x, ydata=y, sigma=yerr, p0=p0_n,
                                    bounds=[[-np.inf,-np.inf,-np.inf,-np.inf, n*per/2.,t0-per/10.],
                                            [ np.inf, np.inf, np.inf, 0., n*per*2.,t0+per/10.]])
        
        all_chi2.append( sum(( (y[i] - fitfunc(x[i], *fit_par))**2./yerr[i]**2. for i in range(len(y)) )  ) )
        all_fitpar.append(fit_par)
        
    best_fitpar = all_fitpar[np.argmin(all_chi2)]
    best_chi2 = np.min(all_chi2)
    
    yfit = fitfunc(x, *best_fitpar)
            
    return best_chi2, yfit, fit_par


def fit_transit_local_poly(x,y,yerr,p0,t0,width,limb_darkening,exptime=0.0204):
        
    params = batman.TransitParams()
    params.t0 = t0                       #time of inferior conjunction
    params.per = np.pi*(width)/np.arcsin(1./50.)                      #orbital period
    params.rp = 0.1                      #planet radius (in units of stellar radii)
    params.a = 50.                       #semi-major axis (in units of stellar radii)
    params.inc = 90.                     #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = limb_darkening            #limb darkening coefficients [u1, u2]
    params.limb_dark = "nonlinear"       #limb darkening model
         
    m = batman.TransitModel(params, x, exp_time=exptime,)

    def lc_func(t, *par):
        a0,a1,a2,rprs,t_0 = par
        params.rp = 10.**rprs  
        params.t0 = t_0
        return  a0 +a1*t + a2*t**2. + m.light_curve(params)-1.        
            
    fit_par, fit_var = curve_fit(lc_func, xdata=x, ydata=y, sigma=yerr, p0=p0,
                                bounds=[[-np.inf, -np.inf,-np.inf,-8,t0-width/10.],
                                        [np.inf, np.inf,np.inf, 0,t0+width/10.]])
    chi2 = sum(( (y[i] - lc_func(x[i], *fit_par))**2./yerr[i]**2. for i in range(len(y)) )  )
    
    yfit = lc_func(x, *fit_par)
            
    return chi2, yfit, fit_par






def check_minmax_mes(foldtime, mes, tdur, delta_t0=0.):


    
    


    return 1.



    



def odd_even_transit_depths():


    return 1.







