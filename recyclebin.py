
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
import wotan


from scipy.optimize import curve_fit
from lmfit import minimize, Parameters

from .dump import make_transit_mask, calc_mes, calc_var_stat
from .utils import *
from .fitfuncs import *



class RecycleBin(object):

    def __init__(self, Dump, TCEs):


        self.dump = Dump
        self.tces = TCEs

        self.mcmc_tces = TCEs.copy()

        self.exptime = Dump.lc.exptime
        self.limb_coeff = Dump.limb_darkening[0]

        self.time = Dump.lc.time[Dump.lc.mask]
        self.flux = Dump.lc.flux[Dump.lc.mask]
        self.trend = Dump.lc.trend[Dump.lc.mask]
        self.raw_flux = self.trend*self.flux
        self.flux_err = Dump.lc.flux_err[Dump.lc.mask]



    def _update_mcmc_tce(self, tce_num):

        #self.mcmc_tce[tce_num] = [0]

        return 1.

    def _calc_mes(self, tce_num):

        tce = self.tces[tce_num]

        t0 = tce[3]
        P = tce[1]
        width = tce[4]

        foldtime = (self.time - t0 + P/2.)%P 

        width_i = np.argmin(np.abs(width-self.dump.tdurs) )
        print(width_i, self.dump.tdurs[width_i])
        num = self.dump.num[width_i]
        den = self.dump.den[width_i]

        mestime, mes = calc_mes(foldtime, num,den,P,texp=self.dump.lc.exptime)
        
        return mestime-P/2., mes


    def _get_odd_even_mes(self, tce_num):

        tce = self.tces[tce_num]

        t0 = tce[3]
        width = tce[4]
        P = tce[1]*2.

        foldtime = (self.time - t0 + P/4.)%P 

        width_i = np.argmin(np.abs(width-self.dump.tdurs) )
        num = self.dump.num[width_i]
        den = self.dump.den[width_i]

        mestime, mes = calc_mes(foldtime, num,den,P,texp=self.dump.lc.exptime,n_trans=1)

        mestime =  mestime/(P/2.) - 0.5

        odd_mes = mes[mestime<0.5]
        even_mes = mes[mestime>=0.5]

        odd_time= mestime[mestime<0.5]
        even_time= mestime[mestime>=0.5]


        return odd_time, odd_mes, even_time, even_mes, mad(odd_mes), mad(even_mes)


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
                                                          flux_err=self.flux_err,
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
                                                          flux_err=self.flux_err,
                                                          plot=showplot, nwidth=nwidth,
                                                          local_plots=False,global_fit=True)

        return chit, chic, nf




    def _get_mes_metrics(self, tce_num ):

        tce = self.tces[tce_num]

        t0 = tce[3]
        width = tce[4]
        P = tce[1]
        
        mestime, mes = self._calc_mes(tce_num)

        phase = mestime/P - 0.5

        max_mes = np.max(mes)
        min_mes = np.abs(np.min(mes) )
        mad_mes = mad(mes)
        mes_over_mad = max_mes/mad_mes

        out_dict = {'max_mes':max_mes, 'min_mes':min_mes, 'mad_mes':mad_mes,
                    'mes_over_mad':mes_over_mad}
        return out_dict



    def odd_even_depth_test(self, tec_num):

        

        return 1.


    def odd_even_mes_test(self, tce_num):

        odd_phase, odd_mes, even_phase, even_mes, odd_mad, even_mad = self._get_odd_even_mes(tce_num)

        odd_peak_idx = np.argmin(np.abs(odd_phase))
        odd_peak = odd_mes[odd_peak_idx-2:odd_peak_idx+2]
        
        even_peak_idx = np.argmin(np.abs(even_phase-1.))
        even_peak = even_mes[even_peak_idx-2:even_peak_idx+2]

        odd_max = np.max(odd_peak)
        even_max = np.max(even_peak)


        out_dict = {'odd_even_mes_ratio':min(odd_max/even_max, even_max/odd_max),
                    'odd_mes': odd_max, 'even_mes':even_max}
        
        return out_dict



    def local_morphology_test(self, tce_num):

        
        tce = self.tces[tce_num]

        t0 = tce[3]
        width = tce[4]
        P = tce[1]

        t = self.time
        f = self.flux
        ferr = self.flux_err


        morph_test = bic_morphology_test(t,f,ferr,P,t0,width,show_plot=False,
                                         show_progress=True,fit_method='lbfsg',
                                         mask_detrend=False,min_frac=0.5)

        
        return morph_test


    


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
    params.rp = 1e-3                     #planet radius (in units of stellar radii)
    params.a = 50.                       #semi-major axis (in units of stellar radii)
    params.inc = 90.                     #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = limb_darkening            #limb darkening coefficients [u1, u2]
    params.limb_dark = "nonlinear"       #limb darkening model
              
    m = batman.TransitModel(params, x, exp_time=exptime, fac=0.001)

    def lc_func(t, *par):
        
        rprs, width, t0 = par
        params.per = np.pi*(width)/np.arcsin(1./50.)                      #orbital period
        params.rp = 10.**rprs   
        params.t0 = t0
        return m.light_curve(params)-1.
    
    
    result = curve_fit(lc_func, xdata=x, ydata=y, p0=p0, sigma=yerr, absolute_sigma=True, 
                       bounds=[[-8, p0[1]/2., -p0[1]],[0., p0[1]*2., p0[1]]],
                       check_finite=True, jac=None, method='trf')
    
    
    resids = y - lc_func(x, *result[0])
    chi2 = np.sum(resids**2./yerr**2.)
    
    x_fold =(x+period/2.)%period-period/2.

    
    return chi2, resids, np.sort(x_fold), lc_func(x_fold, *result[0] )[np.argsort(x_fold)]
    
    
    


def compare_cosine_and_transit_model(time,flux,t0,width,P,limb_dark_coeffs,exptime,
                                     flux_err=None,nwidth=4,plot=True,local_plots=False,
                                     global_fit=False):

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




def get_local_cosine_and_transit_fits(time, flux, t0, width, P, limb_dark_coeffs,
                                      flux_err=None, nwidth=1,show_plots=False):
    
    
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

                t_result = fit_transit_local_poly(x=time_seg, y=flux_seg,yerr=fluxerr_seg,
                                                  t0=t0,width=width,p0=[0,0,0,-2,t0],
                                                  limb_darkening=limb_dark_coeffs)


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
                                    bounds=[[-np.inf,-np.inf,-np.inf,-np.inf, n*per/2.,
                                             t0-per/10.],
                                            [ np.inf, np.inf, np.inf, 0., n*per*2.,
                                              t0+per/10.]])
        
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
    chi2 = sum( ((y[i]-lc_func(x[i],*fit_par))**2./yerr[i]**2. for i in range(len(y))) )
    
    yfit = lc_func(x, *fit_par)
            
    return chi2, yfit, fit_par






def check_minmax_mes(foldtime, mes, tdur, delta_t0=0.):
    


    return 1.

    



def odd_even_transit_depths():


    return 1.







def bic_morphology_test(t, f, ferr, P, t0, tdur, fit_method='leastsq', ntdur=2,texp=0.0204,
                        show_plot=False,min_frac=0.8,show_progress=True,mask_detrend=True):

    
    boxparams = Parameters()
    boxparams.add('a', value=1e-4, min=0.)
    boxparams.add('t0', value=0., vary=True, min=-2*tdur, max=2*tdur )
    boxparams.add('tdur', value=tdur*0.75, vary=True, min=.5*tdur, max=1.5*tdur)
    boxparams.add('c0', value=np.median(f), vary=True)



    jumpparams = Parameters()
    jumpparams.add('a', value=1e-4, max=1e-3)
    jumpparams.add('t0', value=-tdur/2., vary=True, min=-2*tdur, max=2*tdur  )
    jumpparams.add('c0', value=np.median(f), vary=True)



    spsdparams = Parameters()
    spsdparams.add('a', value=1e-4, max=0.5, min=0. )
    spsdparams.add('d', value=2, min=1, max=100)
    spsdparams.add('t0', value=-tdur/2., vary=True, min=-tdur, max=tdur/2.  )
    spsdparams.add('tdur', value=tdur, vary=True, min=-2*tdur, max=2*tdur  )
    spsdparams.add('c0', value=np.median(f), vary=True)




    sineparams = Parameters()
    sineparams.add('a', value=1e-4, max=0.5, min=0. )
    sineparams.add('t0', value=0., vary=True, min=-tdur, max=tdur  )
    sineparams.add('tdur', value=tdur/2., vary=True, min=tdur/8., max=2*tdur  )
    sineparams.add('c0', value=np.median(f), vary=True)
    


    if mask_detrend:
        tmask = make_transit_mask(t, P, t0, tdur)
        f = wotan.flatten(t,f,method='lowess',window_length=4*tdur,return_trend=False,
                          robust=True,mask=tmask)
        
        sineparams.add('c1', value=0, vary=False)
        sineparams.add('c2', value=0, vary=False)

        spsdparams.add('c1', value=0, vary=False)
        spsdparams.add('c2', value=0, vary=False)

        jumpparams.add('c1', value=0, vary=False)
        jumpparams.add('c2', value=0, vary=False)

        boxparams.add('c1', value=0, vary=False)
        boxparams.add('c2', value=0, vary=False)

    else:
        sineparams.add('c1', value=0, vary=True)
        sineparams.add('c2', value=0, vary=True)

        spsdparams.add('c1', value=0, vary=True)
        spsdparams.add('c2', value=0, vary=True)

        jumpparams.add('c1', value=0, vary=True)
        jumpparams.add('c2', value=0, vary=True)

        boxparams.add('c1', value=0, vary=True)
        boxparams.add('c2', value=0, vary=True)
        


    jump_bic_probs = []
    jump_aic_probs = []

    spsd_bic_probs = []
    spsd_aic_probs = []

    sine_bic_probs = []
    sine_aic_probs = []


    n_transits = int(np.floor( (t[-1]-t[0])/P))

    num_good_transits = 0
    frac_good_transit_points=0

    if show_progress:
        transit_iterable = tqdm(range(int(n_transits)) )
    else:
        transit_iterable=range(int(n_transits))
    
    for N in transit_iterable:
    
        t_min, t_max = t0 + N*P - ntdur*tdur, t0 + N*P + ntdur*tdur

        t_n = t0 + N*P

        tce_cut = np.logical_and(t>t_min, t<t_max)
    
        tce_time = t[tce_cut] - t_n
        tce_flux = f[tce_cut]
        tce_err = ferr[tce_cut]        
    
        if sum(tce_cut)<min_frac*(4*tdur/texp):
            frac_good_transit_points += sum(tce_cut)/(4*tdur/texp)
            continue

        num_good_transits+=1
        frac_good_transit_points += sum(tce_cut)/(4*tdur/texp)
    

        box_out = minimize(box_residual, boxparams, args=(tce_time, tce_flux, tce_err),
                       method=fit_method)
        jump_out = minimize(jump_residual, jumpparams, args=(tce_time, tce_flux, tce_err),
                        method=fit_method)
        spsd_out = minimize(spsd_residual, spsdparams, args=(tce_time, tce_flux, tce_err),
                        method=fit_method)
        sine_out = minimize(sine_residual, sineparams, args=(tce_time, tce_flux, tce_err),
                        method=fit_method)


            
        jump_bic_probs.append( (box_out.bic-jump_out.bic)/2.) 
        jump_aic_probs.append(  (box_out.aic-jump_out.aic)/2.) 
    
        spsd_bic_probs.append( (box_out.bic-spsd_out.bic)/2.) 
        spsd_aic_probs.append(  (box_out.aic-spsd_out.aic)/2.) 
    
        sine_bic_probs.append( (box_out.bic-sine_out.bic)/2.) 
        sine_aic_probs.append(  (box_out.aic-sine_out.aic)/2.) 


        if show_plot:
            plot_time = np.linspace(min(tce_time), max(tce_time), 500)

            plt.errorbar(tce_time, tce_flux, yerr=tce_err, fmt='.', color='k', ecolor='0.5')

            plt.plot(plot_time, box_residual(box_out.params, plot_time), '-' , zorder=2,
                 label='bic: ${:.2f}$\naic: ${:.2f}$'.format(box_out.bic,box_out.aic))
            plt.plot(plot_time, jump_residual(jump_out.params, plot_time), '-' , zorder=2,
                label='bic: ${:.2f}$\naic: ${:.2f}$'.format(jump_out.bic,jump_out.aic))
            plt.plot(plot_time, spsd_residual(spsd_out.params, plot_time), '-' , zorder=2,
                label='bic: ${:.2f}$\naic: ${:.2f}$'.format(spsd_out.bic,spsd_out.aic))
            plt.plot(plot_time, sine_residual(sine_out.params, plot_time), '-' , zorder=2,
                label='bic: ${:.2f}$\naic: ${:.2f}$'.format(spsd_out.bic,spsd_out.aic))
        
            plt.title('transit {}'.format(N))
            plt.legend(ncol=2)
            plt.xlabel('$\mathregular{\Delta t_0}$')
            plt.ylabel('flux')
            plt.show()



    output_dict = {'num_transits': n_transits,
                   'num_good_transits':num_good_transits,
                   'frac_avg_points_in_transit': frac_good_transit_points/n_transits,
                   'spsd_mean_bic_stat':np.mean(spsd_bic_probs),
                   'spsd_mean_aic_stat':np.mean(spsd_aic_probs),
                   'spsd_median_bic_stat':np.median(spsd_bic_probs),
                   'spsd_median_aic_stat':np.median(spsd_aic_probs),
                   'spsd_bic_stat':np.sum(spsd_bic_probs),
                   'spsd_aic_stat':np.sum(spsd_aic_probs),
                   'jump_mean_bic_stat':np.mean(jump_bic_probs),
                   'jump_mean_aic_stat':np.mean(jump_aic_probs),
                   'jump_median_bic_stat':np.median(jump_bic_probs),
                   'jump_median_aic_stat':np.median(jump_aic_probs),
                   'jump_bic_stat':np.sum(jump_bic_probs),
                   'jump_aic_stat':np.sum(jump_aic_probs),
                   'sine_mean_bic_stat':np.mean(sine_bic_probs),
                   'sine_mean_aic_stat':np.mean(sine_aic_probs),
                   'sine_median_bic_stat':np.median(sine_bic_probs),
                   'sine_median_aic_stat':np.median(sine_aic_probs),
                   'sine_bic_stat':np.sum(sine_bic_probs),
                   'sine_aic_stat':np.sum(sine_aic_probs) }

    return output_dict







