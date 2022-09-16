
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
from .ses import get_transit_signal, ocwt, calc_var_stat



class RecycleBin(object):

    def __init__(self, Dump, TCEs):


        self.dump = Dump
        self.tces = TCEs

        self.refined_tces = TCEs.copy()

        self.exptime = Dump.lc.exptime
        self.limb_coeff = Dump.limb_darkening[0]

        self.time = Dump.lc.time[Dump.lc.mask]
        self.flux = Dump.lc.flux[Dump.lc.mask]
        self.trend = Dump.lc.trend[Dump.lc.mask]
        self.raw_flux = self.trend*self.flux
        self.flux_err = Dump.lc.flux_err[Dump.lc.mask]

        self.batman_model, self.batman_params = self._get_batman()


    def _get_tce_mask(self, tce_num):

        P,width,t0 = get_p_tdur_t0(self.tces[tce_num])

        return make_transit_mask(self.time, P, t0, width)



    def _get_batman(self, limb_law='nonlinear'):

        params = batman.TransitParams()       #object to store transit parameters
        params.t0 = self.time[0]              #time of inferior conjunction
        params.per = 1.                       #orbital period
        params.rp = 0.1                       #planet radius (in units of stellar radii)
        params.a = 15.                        #semi-major axis (in units of stellar radii)
        params.inc = 90.                      #orbital inclination (in degrees)
        params.ecc = 0.                       #eccentricity
        params.w = 90.                        #longitude of periastron (in degrees)
        params.limb_dark = limb_law           #limb darkening model
        params.u = self.limb_coeff

        m = batman.TransitModel(params, self.time, supersample_factor=3, exp_time=self.exptime)

        return m, params

    def _get_lightcurve(self):
        return self.batman_model.light_curve(self.batman_params)


    def _transit_resids(self, par):

        vals = par.valuesdict()
        b = vals['b']
        tdur = vals['tdur']
        t0 = vals['t0']
        per = vals['per']
        logdepth = vals['logdepth']
        
        a = 1./np.sin((np.pi*tdur)/per )
        inc = np.arccos(b/a)*(180./np.pi)
    
        self.batman_params.t0 = t0                     
        self.batman_params.per = per
        self.batman_params.rp = np.sqrt( 10.**logdepth )      
        self.batman_params.a =  a
        self.batman_params.inc = inc

        lc = self._get_lightcurve()

        return (self.flux-lc) / self.flux_err**2.

        
    

    def _update_tce(self, tce_num, fit_method='leastsq', return_fit=False):


        P,width,t0 = get_p_tdur_t0(self.tces[tce_num])        
        
        params = Parameters()

        if P>max(self.time)-min(self.time):
            params.add('per', value=max(self.time)-min(self.time), vary=False, min=.99*P, max=1.01*P)
        else:
            params.add('per', value=P, vary=True, min=.99*P, max=1.01*P)
        params.add('t0', value=t0, vary=True, min=t0-width, max=t0+width)
        params.add('tdur', value=width, min=0.25 * width, max=4*width, vary=True)
        params.add('logdepth', value=-3, vary=True, max=0)
        params.add('b', value=0.5, vary=True,max=1.2, min=0)

        in_guess = minimize(self._transit_resids, params,method='LBFGS',nan_policy='omit')

        if fit_method=='emcee':
            emcee_kws = dict(steps=2000, burn=500,
                             progress=True,nan_policy='omit')
            tce_out = minimize(self._transit_resids, in_guess.params, method=fit_method , **emcee_kws)
        else:
            tce_out = minimize(self._transit_resids, in_guess.params, method=fit_method)
        out = tce_out.params.valuesdict()

        if P>max(self.time)-min(self.time):
            self.refined_tces[tce_num,1] = P
        else:
            self.refined_tces[tce_num,1] = out['per']

        self.refined_tces[tce_num,2] = 10.**out['logdepth']
        self.refined_tces[tce_num,3] = out['t0']
        self.refined_tces[tce_num,4] = out['tdur']

        if return_fit:
            return out

        
    def _calc_mes(self, tce_num, mask_tce=False, calc_ses=False,):

        P,width,t0 = get_p_tdur_t0(self.tces[tce_num])        
        width_i = np.argmin(np.abs(width-self.dump.tdurs) )

        mask = self.dump.lc.mask.copy()

        if mask_tce:
            mask &= make_transit_mask(self.dump.lc.time, P, t0, width*1.5)

        if calc_ses:
            self.dump.Calculate_SES_by_Segment(mask=mask, tdurs=[width])

        num = self.dump.num[0]
        den = self.dump.den[0]
        
        foldtime = (self.dump.lc.time.copy() - t0 + P/4.)%P 
        
        mestime, mes = calc_mes(foldtime[mask],num,den,P,texp=self.dump.lc.exptime,n_trans=1., return_nans=True)

        return mestime-P/4., (mestime)/P , mes


    def _get_odd_even_mes(self, tce_num):

        P,width,t0 = get_p_tdur_t0(self.tces[tce_num])
        P_twice =P*2.
        

        foldtime = (self.time - t0 + P_twice/4.)%P_twice 

        width_i = np.argmin(np.abs(width-self.dump.tdurs) )
        num = self.dump.num[width_i]
        den = self.dump.den[width_i]

        mestime, mes = calc_mes(foldtime,num,den,P_twice,texp=self.dump.lc.exptime,n_trans=1)

        mestime =  mestime/(P/2.) - 0.5

        odd_mes = mes[mestime<0.5]
        even_mes = mes[mestime>=0.5]

        odd_time= mestime[mestime<0.5]
        even_time= mestime[mestime>=0.5]


        return odd_time, odd_mes, even_time, even_mes, mad(odd_mes), mad(even_mes)


    def weak_secondary_test(self, tce_num):

        mestime, mesphase, mes = self._calc_mes(tce_num,calc_ses=True,mask_tce=True)

        mes[mes==0] = np.nan
        
        max_secondary_mes = np.nanmax(mes)
        max_secondary_phase = mesphase[np.nanargmax(mes)]
        min_secondary_mes = np.nanmin(mes)
        min_secondary_phase = mesphase[np.nanargmin(mes)]
        mad_secondary_mes = mad(mes)
        

        output = {'max_secondary_mes': max_secondary_mes,
                  'max_secondary_mes_phase': max_secondary_phase,
                  'min_secondary_mes': min_secondary_mes,
                  'min_secondary_mes_phase':min_secondary_phase,
                  'mad_secondary_mes':mad_secondary_mes
                  }
        
        return output
    
    def cosine_vs_transit_local(self, tce_num, showplot=False, nwidth=4,local_plots=False):

        P,width,t0 = get_p_tdur_t0(self.tces[tce_num])
        depth = self.refined_tces[tce_num,2]



        chit, chic, nf = compare_cosine_and_transit_model(time=self.time,
                                                          flux=self.flux*self.trend,
                                                          t0=t0, width=width, P=P, 
                                                          exptime=self.exptime,
                                                          depth=depth**0.5,
                                                          limb_dark_coeffs=self.limb_coeff,
                                                          flux_err=self.flux_err,
                                                          plot=showplot, nwidth=nwidth,
                                                          local_plots=local_plots)

        return chit, chic, nf

    def cosine_vs_transit_global(self, tce_num, showplot=False, nwidth=4, depth=None):

        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])
        depth = self.refined_tces[tce_num,2]

        chic, chit, nf = compare_cosine_and_transit_model(time=self.time,
                                                          flux=self.flux,
                                                          t0=t0, width=width, P=P,
                                                          depth=depth**0.5,
                                                          exptime=self.exptime,
                                                          limb_dark_coeffs=self.limb_coeff,
                                                          flux_err=self.flux_err,
                                                          plot=showplot, nwidth=nwidth,
                                                          local_plots=False,global_fit=True)

        output = {'global_transit_red_chi2':chit, 'global_cosine_red_chi2':chic, 'global_diff_red_chi2':chit-chic, 'global_transit_chi2':chit*nf,  'global_cosine_chi2':chic*nf, 'global_diff_chi2':(chit-chic)*nf,}
        
        return output




    def _get_mes_metrics(self, tce_num , use_mask=False):

        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])

        try:
            mestime, mesphase, mes = self._calc_mes(tce_num, use_mask, calc_ses=True)
        except RuntimeError:
            self.dump.Calculate_SES()
            mestime, mesphase, mes = self._calc_mes(tce_num, use_mask)
            

        phase = mestime/P - 0.5

        max_mes = np.nanmax(mes)
        min_mes = np.nanmin(mes) 
        mad_mes = mad(mes)
        mes_over_mad = max_mes/mad_mes

        mes_secondary = np.max(mes[~np.logical_and(mestime>P/4.-2*width,mestime>P/4.-2*width)])
        
        
        out_dict = {'max_mes':max_mes, 'min_mes':min_mes, 'mad_mes':mad_mes,
                    'mes_over_mad':mes_over_mad, 'max_secondary_mes':mes_secondary}
        return out_dict



    def odd_even_depth_test(self, tce_num, return_dict=False,fit_method='leastsq',):

        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])

        _, both, even, odd = odd_even_transit_depths(t=self.time,f=self.flux,ferr=self.flux_err,
                                             P=P,t0=t0,width=width,fit_method='leastsq',
                                                    initial_fit_method='LBFGS')


        if even.errorbars and odd.errorbars:
            stat = np.abs(odd.params['a']-even.params['a'])/np.sqrt(even.params['a'].stderr**2. + odd.params['a'].stderr**2.)
        else:
            print('Odd-Even Test Failed')
            stat=np.nan

        if return_dict:
            return {'odd_even_depth_stat':stat,
                    'odd_depth':odd.params['a'].value,
                    'even_depth':even.params['a'].value,
                    'odd_depth_stderr':odd.params['a'].stderr,
                    'even_depth_stderr':even.params['a'].stderr,
                    }
        else:
            return both, even, odd, stat


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



    def local_morphology_test(self, tce_num, **test_kw):
        
        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])
        depth = self.refined_tces[tce_num,2]

        t = self.time
        f = self.flux * self.trend
        ferr = self.flux_err
        texp=self.exptime


        morph_test = bic_morphology_test(t,f,ferr,P,t0,width,depth,texp=texp,**test_kw)

        
        return morph_test


    def chi2_tests(self, tce_num, calc_ses=False):

        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])

        time = self.time
        flux = self.flux
        texp=self.exptime

        if calc_ses:
            self.dump.Calculate_SES_by_Segment(mask=None, tdurs=[width])

        num = self.dump.num[0]
        den = self.dump.den[0]
                
        channel_red_chi2, channel_chi2_stat = channel_chi2_statistic(time, flux, t0, P, width, cadence=texp,fill_mode='reflect')
        temporal_red_chi2, temporal_chi2_stat = temporal_chi2_statistic(time, num, den, t0, P , cadence=texp)   
        
        return {'channel_red_chi2':channel_red_chi2, 'channel_chi2_stat':channel_chi2_stat,'temporal_red_chi2': temporal_red_chi2, 'temporal_chi2_stat':temporal_chi2_stat}


    def get_all_vetting_metrics(self, tce_num):

        bestfit = self._update_tce(tce_num, return_fit=True)

        depth = self.refined_tces[tce_num,2]
        P,width,t0 = get_p_tdur_t0(self.refined_tces[tce_num])
        b = bestfit['b']

        result_list =[{'P':P, 'depth_ppt':depth*1e3,'tdur':width,'t0':t0, 'b':b} ,
                     self._get_mes_metrics(tce_num),
                     self.chi2_tests(tce_num),
                     self.weak_secondary_test(tce_num),
                     #self.odd_even_mes_test(tce_num),
                     self.odd_even_depth_test(tce_num, True),
                     self.cosine_vs_transit_global(tce_num,depth=depth),
                     self.local_morphology_test(tce_num)]
                     
        results = {k: v for d in result_list for k, v in d.items()}

        return results


    


def fit_sinewave(x, y, yerr, p0=None):
    
    sine_func = lambda x,c,a,p,x0: c+a*np.cos((2.*np.pi) * (x-x0) / p)

    result = curve_fit(sine_func, xdata=x, ydata=y, p0=p0, sigma=yerr, 
                       bounds=[[-np.inf,-np.inf, p0[2]/3.,-np.inf],[np.inf,np.inf, p0[2]*4.,np.inf]], 
              check_finite=True, method='trf', jac=None, )
    
    resids = y - sine_func(x, *result[0])
    
    chi2 = np.sum(resids**2./yerr**2.)

    plot_x = np.linspace(min(x), max(x), 100)
    
    return chi2, resids, plot_x, sine_func(plot_x, *result[0])
    
    
    
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
        
        rprs, width, t0, c = par
        params.per = np.pi*(width)/np.arcsin(1./50.)                      #orbital period
        params.rp = 10.**rprs   
        params.t0 = t0
        return m.light_curve(params)-1.+c
    

    p0 = np.append(p0,[0])
    result = curve_fit(lc_func, xdata=x, ydata=y, p0=p0, sigma=yerr, absolute_sigma=True, 
                       bounds=[[-8, p0[1]/2., -p0[1],-1],[0., p0[1]*2., p0[1],1]],
                       check_finite=True, jac=None,)
    
    
    resids = y - lc_func(x, *result[0])
    chi2 = np.sum(resids**2./yerr**2.)
    
    x_fold =(x+period/2.)%period-period/2.

    plot_x = np.linspace(min(x_fold), max(x_fold), 100)

    
    m = batman.TransitModel(params, plot_x, exp_time=exptime, fac=0.001, supersample_factor=5)
    
    return chi2, resids, plot_x, lc_func(plot_x,  *result[0] )
    
    
    


def compare_cosine_and_transit_model(time,flux,t0,width,P,limb_dark_coeffs,exptime,
                                     flux_err=None,nwidth=4,plot=True,local_plots=False,
                                     global_fit=False, depth=None):

    if depth is None:
        depth=1e-3

    if global_fit:
        foldtime = (time - t0 + P/2.)%P - P/2.
        mask = ~make_transit_mask(time, P, t0, width*nwidth/2)

        transit_time = foldtime[mask]
        cosine_time = foldtime[mask]

        transit_flux = flux[mask]-1.
        cosine_flux = flux[mask]-1.

        transit_fluxerr=flux_err[mask]
        cosine_fluxerr=flux_err[mask]


    else:
        transit_time,cosine_time,transit_flux,cosine_flux,fluxerr,_ = get_local_cosine_and_transit_fits(time, flux, t0, width, P, limb_dark_coeffs, flux_err=flux_err, nwidth=nwidth, show_plots=local_plots)

    
        trim_to_width = np.abs(transit_time)< nwidth*width/2.
        transit_time = transit_time[trim_to_width]
        transit_flux = transit_flux[trim_to_width]
    
        trim_to_width_cos = np.abs(cosine_time)< nwidth*width/2.
        cosine_time = cosine_time[trim_to_width_cos]
        cosine_flux = cosine_flux[trim_to_width_cos]

        transit_fluxerr = fluxerr[trim_to_width]        
        cosine_fluxerr = fluxerr[trim_to_width_cos]

        
    sine_chi2s = []
    all_sine_results = []
    for n in [0.5,1,2]:
        results = fit_sinewave(cosine_time, cosine_flux, yerr=cosine_fluxerr,p0=[0., depth, n*width,0.])  
        sine_chi2s.append(results[0])
        all_sine_results.append(results)
        
    
    sine_chi2, sine_resids, sine_x, sine_y = all_sine_results[np.argmin(sine_chi2s)]


    tran_chi2, tran_resids, tran_x, tran_y = fit_transit(transit_time, transit_flux, yerr=transit_fluxerr, 
                                                         p0=[np.log10(depth),width,0.], t0=0.,
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
                 label='$\mathregular{\chi^2_{\\nu, sine}}$'+'\n={:.3f}'.format(sine_chi2))
        ax1.legend(framealpha=0., loc='lower right')
        
        ax3.plot(tran_x, tran_y, '-', color='tomato', lw=2, 
                 label='$\mathregular{\chi^2_{\\nu, tran}}$'+'\n={:.3f}'.format(tran_chi2))
        ax3.legend(framealpha=0., loc='lower right')
        
        
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
#        plt.show()
        
    nfreedom = len(transit_time) - 3.
        
    return sine_chi2, tran_chi2, nfreedom




def get_local_cosine_and_transit_fits(time, flux, t0, width, P, limb_dark_coeffs,
                                      flux_err=None, nwidth=1,show_plots=False):
    
    
    if flux_err is None:
        flux_err = np.ones(len(flux))*np.nanstd(flux)
        
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
                                               p0=[1.,0,0,0.,width,t0])

                t_result = fit_transit_local_poly(x=time_seg, y=flux_seg,yerr=fluxerr_seg,
                                                  t0=t0,width=width,p0=[1.,0,0,-3,t0],
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
        fitfunc = lambda x,a0,a1,a2,amp,per,t0: (a0 + a1*x +a2*x**2.) + amp*np.cos(2*np.pi*(x-t0)/per)
        a0,a1,a2,amp,per,t0 = p0
        p0_n = [a0,a1,a2,amp,per*n,t0]
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
        return  (a0 +a1*t + a2*t**2.) *  m.light_curve(params)      
            
    fit_par, fit_var = curve_fit(lc_func, xdata=x, ydata=y, sigma=yerr, p0=p0,
                                bounds=[[-np.inf, -np.inf,-np.inf,-8,t0-width/10.],
                                        [np.inf, np.inf,np.inf, 0,t0+width/10.]])
    chi2 = sum( ((y[i]-lc_func(x[i],*fit_par))**2./yerr[i]**2. for i in range(len(y))) )
    
    yfit = lc_func(x, *fit_par)
            
    return chi2, yfit, fit_par






def odd_even_transit_depths(t,f,ferr,P,t0,width,initial_fit_method='LBFGS',
                            fit_method='leastsq'):


    phase = (t-t0 + P/4)%(2*P) 

    odd = phase<P
    even = phase>=P

    odd_flux = f[odd]
    odd_phase = phase[odd]
    
    even_flux = f[even]
    even_phase = phase[even]-P
    

    
    bothparams = Parameters()
    bothparams.add('t0', value=0.25*P, vary=True,min=0.2*P,max=0.3*P)
    bothparams.add('a', value=1e-3, vary=True, min=1e-8, max=.99)
    bothparams.add('b', value=1e3, vary=True, min=0, max=np.inf)
    bothparams.add('tdur', value=width, min=0.5 * width, max=2*width, vary=True)
    bothparams.add('c0', value=1., vary=True,)
    bothparams.add('c1', value=0., vary=False)
    bothparams.add('c2', value=0., vary=False)

    

    initresult = minimize(trap_residual, bothparams,
                          args=(phase%P, f, ferr), method=initial_fit_method)

    bothresult =  minimize(trap_residual, initresult.params,
                          args=(phase%P, f, ferr),method=fit_method)

    bothresult.params['b'].vary=False
    
    evenresult = minimize(trap_residual, bothresult.params,
                          args=(even_phase, even_flux, ferr[even]),method=fit_method)

    oddresult = minimize(trap_residual, bothresult.params,
                          args=(odd_phase, odd_flux, ferr[odd]),method=fit_method)


    return phase, bothresult, evenresult, oddresult








def bic_morphology_test(t, f, ferr, P, t0, tdur, depth=1e-4, fit_method='LBFGS',
                        ntdur=4,texp=0.0204,show_plot=False,min_frac=0.8,
                        show_progress=True,mask_detrend=False):

    
    boxparams = Parameters()
    boxparams.add('a', value=depth, min=0.)
    boxparams.add('t0', value=0., vary=True, min=-2*tdur, max=2*tdur )
    boxparams.add('tdur', value=tdur*0.75, vary=True, min=.5*tdur, max=2*tdur)
    boxparams.add('c0', value=np.median(f), vary=True)



    jumpparams = Parameters()
    jumpparams.add('a', value=depth, max=1, min=-1.)
    jumpparams.add('t0', value=-0., vary=True, min=-2*tdur, max=2*tdur  )
    jumpparams.add('c0', value=np.median(f), vary=True)



    spsdparams = Parameters()
    spsdparams.add('a', value=depth*2., max=1., min=0. )
    spsdparams.add('d', value=2, min=1, max=100)
    spsdparams.add('t0', value=-tdur/4., vary=True, min=-tdur, max=tdur  )
    spsdparams.add('tdur', value=tdur, vary=True, min=0, max=2*tdur  )
    spsdparams.add('c0', value=np.median(f), vary=True)




    sineparams = Parameters()
    sineparams.add('a', value= depth*2., max=1., min=0. )
    sineparams.add('t0', value=0., vary=True, min=-tdur, max=tdur  )
    sineparams.add('tdur', value=tdur/4., vary=True, min=tdur/8., max=2*tdur  )
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

    spsd_bic_probs = []

    sine_bic_probs = []

    num_good_transits = 0
    frac_good_transit_points=0


    n_transits=0
    tn=t0


    if show_plot:
        transit_labels=[]
        tce_times = []
        tce_fluxes = []
        tce_errs=[]
        
        plot_times=[]
        spsd_fits = []
        box_fits = []
        jump_fits = []
        sine_fits = []
    
    while tn < max(t):

        t_min, t_max = tn - ntdur*tdur, tn + ntdur*tdur
        
        tce_cut = np.logical_and(t>t_min, t<t_max)
    
        tce_time = t[tce_cut] - tn
        tce_flux = f[tce_cut]
        tce_err = ferr[tce_cut]

        n_transits+=1
        tn+=P
    
        if sum(tce_cut)<min_frac*(2*ntdur*tdur/texp):
            frac_good_transit_points += sum(tce_cut)/(2*ntdur*tdur/texp)
            continue


        
        num_good_transits+=1
        frac_good_transit_points += sum(tce_cut)/(2*ntdur*tdur/texp)
    

        box_out = minimize(box_residual, boxparams, args=(tce_time, tce_flux, tce_err),
                           method=fit_method,)
        jump_out = minimize(jump_residual, jumpparams, args=(tce_time, tce_flux, tce_err),
                            method=fit_method,)
        spsd_out = minimize(spsd_residual, spsdparams, args=(tce_time, tce_flux, tce_err),
                            method=fit_method,)
        sine_out = minimize(sine_residual, sineparams, args=(tce_time, tce_flux, tce_err),
                            method=fit_method,)


            
        jump_bic_probs.append( (box_out.bic-jump_out.bic)/2.) 
        #jump_aic_probs.append(  (box_out.aic-jump_out.aic)/2.) 
    
        spsd_bic_probs.append( (box_out.bic-spsd_out.bic)/2.) 
        #spsd_aic_probs.append(  (box_out.aic-spsd_out.aic)/2.) 
    
        sine_bic_probs.append( (box_out.bic-sine_out.bic)/2.) 
        #sine_aic_probs.append(  (box_out.aic-sine_out.aic)/2.) 


        if show_plot:
            plot_time = np.linspace(min(tce_time), max(tce_time), 500)
            plot_times.append(plot_time)
            tce_times.append(list(tce_time))
            tce_fluxes.append(tce_flux)
            tce_errs.append(tce_err)

            box_fits.append( box_residual(box_out.params, plot_time) )
            spsd_fits.append( spsd_residual(spsd_out.params, plot_time) )
            jump_fits.append( jump_residual(jump_out.params, plot_time) )
            sine_fits.append( sine_residual(sine_out.params, plot_time) )

            transit_labels.append(n_transits)

    if show_plot:

        def get_ppt(x):
            return (x-np.median(x))/np.median(x)*1e3


        for i in range(len(transit_labels)):

            snum = int(np.sqrt(len(transit_labels))+1)


            if i==0:
                plt.figure(figsize=(snum*2.5,snum*2.) )
                
            
            ax = plt.subplot(snum,snum,i+1)

            if i%snum==0:
                ax.set_ylabel(r'$\mathregular{\delta F/F }$ [ppt]')

            ax.set_xlabel(r'$\mathregular{\Delta t_0}$ [day]')

            plot_time=plot_times[i]
            ax.errorbar(tce_times[i], get_ppt(tce_fluxes[i]), yerr=1e3*tce_errs[i], fmt='.', color='k', ecolor='0.5')

            l1, = ax.plot(plot_time, get_ppt(box_fits[i]), '-' , zorder=2,
                    label='box')
            l2, = ax.plot(plot_time, get_ppt(jump_fits[i]), '-' , zorder=2,
                    label='jump')
            l3, = ax.plot(plot_time, get_ppt(spsd_fits[i]), '-' , zorder=2,
                    label='spsd')
            l4, = ax.plot(plot_time, get_ppt(sine_fits[i]), '-' , zorder=2,
                    label='sine')

            
            ax.text(0.15,0.1,'{}'.format(transit_labels[i]),ha='center',va='bottom',transform=ax.transAxes)

        ax = plt.subplot(snum,snum,i+2)
        ax.legend([l1,l2,l3,l4], ['box','jump','spsd','sine'], ncol=1, loc='upper left')
        ax.axis('off')
    
            #plt.xlabel('$\mathregular{\Delta t_0}$')

            #plt.ylabel('flux')
        plt.tight_layout()
        #plt.show()

            

    output_dict = {'num_transits': n_transits,
                   'num_good_transits':num_good_transits,
                   'frac_avg_points_in_transit': frac_good_transit_points/n_transits,
                   #'jump_mean_bic_stat':np.mean(jump_bic_probs),
                   #'jump_mean_aic_stat':np.mean(jump_aic_probs),
                   'jump_median_bic_stat':np.median(jump_bic_probs),
                   'jump_min_bic_stat':np.min(jump_bic_probs),
                   'jump_bic_stat':np.sum(jump_bic_probs),
                   #'jump_aic_stat':np.sum(jump_aic_probs),
                   #'spsd_mean_bic_stat':np.mean(spsd_bic_probs),
                   #'spsd_mean_aic_stat':np.mean(spsd_aic_probs),
                   'spsd_median_bic_stat':np.median(spsd_bic_probs),
                   'spsd_min_bic_stat':np.min(spsd_bic_probs),
                   'spsd_bic_stat':np.sum(spsd_bic_probs),
                   #'spsd_aic_stat':np.sum(spsd_aic_probs),
                   #'sine_mean_bic_stat':np.mean(sine_bic_probs),
                   #'sine_mean_aic_stat':np.mean(sine_aic_probs),
                   'sine_median_bic_stat':np.median(sine_bic_probs),
                   'sine_min_bic_stat':np.min(sine_bic_probs),
                   'sine_bic_stat':np.sum(sine_bic_probs),
                   #'sine_aic_stat':np.sum(sine_aic_probs)
                   }

    return output_dict





def temporal_chi2_statistic(t, num, den, t0, P , cadence):    
    
    foldtime = (t-t0 + P/2.)%P - P/2.
    t0_mask = np.abs(foldtime)<=cadence/2.
    n_transits = np.sum(t0_mask)
    
    t0_num, t0_den = num[t0_mask], den[t0_mask]
    
    Z = np.sum(t0_num)/np.sqrt(np.sum(t0_den))
    Zj = t0_num/np.sqrt(np.sum(t0_den))
    Qj = t0_den/np.sum(t0_den)

    delta_Zj = (Zj - Qj*Z)**2.
    
    chi2 = np.sum( delta_Zj / Qj)
    red_chi2 = chi2/(n_transits-1)
    
    chi2_stat = Z /np.sqrt(red_chi2)
        
    return red_chi2, chi2_stat
    

def channel_chi2_statistic(time, flux, t0, P, width, cadence, fill_mode='reflect'):
    
    
    pad_time, pad_flux, pad_boo = pad_time_series(time, flux,in_mode=fill_mode,
                                                      pad_end=True,
                                                      fill_gaps=True,
                                                      cadence=cadence)
    
    template = get_transit_signal(width=width, depth=100., pad=len(pad_flux), b=0.25)
    
    flux_transform = ocwt(pad_flux - np.median(pad_flux))
    template_transform = ocwt(template-1.)
    
    
    window_size=8.*width
    sig2 = [1./calc_var_stat(x, window_size*2**i, method='mad', exp_time=cadence) for i,x in enumerate(flux_transform)]
    sig2 = np.array(sig2, )
        
    n_levels = flux_transform.shape[0]
    levels = np.concatenate([np.arange(1,n_levels), [n_levels-1] ] )[:,np.newaxis]

    N_i=[]
    D_i=[]
    
    for i, l in enumerate(levels):
        
        N_channel = 2.** (-l) * convolve(flux_transform[i]*sig2[i], template_transform[i,::-1], mode='same')
        D_channel = 2.** (-l) * convolve(sig2[i], template_transform[i,::-1]**2., mode='same')  
        
        N_i.append(N_channel)
        D_i.append(D_channel)
        
    
    N_i=np.array(N_i)[:,pad_boo]
    D_i=np.array(D_i)[:,pad_boo]
    
    N = np.sum(N_i, axis=0)
    D = np.sum(D_i, axis=0)


    z = N/np.sqrt(D)
    zi = N_i/np.sqrt(D)
    qi = D_i/D
        
    delta_zi = (zi - qi*z)
    chi2_n = np.sum(delta_zi**2./qi, axis=0)
    
    
    foldtime = (time-t0 + P/2.)%P - P/2.
    t0_mask = np.abs(foldtime)<cadence/2.
    n_transits = np.sum(t0_mask)
    
    Z = np.sum(N[t0_mask])/np.sqrt(np.sum(D[t0_mask]) )
        
    chi2 = np.sum(chi2_n[t0_mask])
    red_chi2 = chi2 / (n_transits*(n_levels-1))
        
    chi2_stat = Z / np.sqrt(red_chi2)
        
    return red_chi2, chi2_stat




