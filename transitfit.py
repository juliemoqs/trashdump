import sys
from os import path, getcwd

from fast_histogram import histogram1d
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import median_filter

from .lightcurves import *
from .utils import *
from .dump import  make_transit_mask


from lmfit import minimize, Parameters

from tqdm import tqdm
import batman

filepath = path.abspath(__file__)
dir_path = path.dirname(filepath)

TRASH_DATA_DIR = dir_path+'/'



class TransitFitter(object):


    def __init__(self, Dump, tces,  use_density=True):

        self.dump=Dump
        self.time = Dump.lc.time[Dump.lc.mask]
        self.flux = Dump.lc.flux[Dump.lc.mask]
        self.flux_err = Dump.lc.flux_err[Dump.lc.mask]

        self.limb_dark = Dump.limb_darkening[0]

        self.tces = tces
        self.lc = Dump.lc
        self.model_params = batman.TransitParams()
        self.model = self._get_transit_model()



    def _get_transit_model(self, tce_mask=None):


        params = batman.TransitParams()       #object to store transit parameters
        params.t0 = 0                        #time of inferior conjunction
        params.per = 10                        #orbital period
        params.rp = 0.1                      #planet radius (in units of stellar radii)
        params.a = 20.                       #semi-major axis (in units of stellar radii)
        params.inc = 89.                      #orbital inclination (in degrees)
        params.ecc = 0.                      #eccentricity
        params.w = 0.                          #longitude of periastron (in degrees)
        params.limb_dark = 'nonlinear'        #limb darkening model
        params.u = self.limb_dark.copy()           #limb darkeing coefficients


        if tce_mask is None:
            m = batman.TransitModel(params, self.time, exp_time=self.lc.exptime) 
        else:
            m = batman.TransitModel(params, self.time[tce_mask], fac=0.01, exp_time=self.lc.exptime) 
            
        return m

        

    def _evaluate_model(self, *par):


        P, t0, depth, width, impact, u1, u2, u3, u4 = par
        
        a_rs = P/(np.pi * width)
        inc = np.rad2deg( np.arccos(impact/a_rs) )
    
        self.model_params.t0 = t0                        #time of inferior conjunction
        self.model_params.per = P                        #orbital period
        self.model_params.rp = np.sqrt(depth/1e6)         #planet radius (in units of stellar radii)
        self.model_params.a = a_rs                       #semi-major axis (in units of stellar radii)
        self.model_params.inc = inc
        self.model_params.ecc = 0.                    
        self.model_params.w = 0.                          
        self.model_params.limb_dark = 'nonlinear'                          
        self.model_params.u = [u1, u2, u3, u4]           #limb darkeing coefficients
                
        
        return self.model.light_curve(self.model_params)



    def residuals(self, par, data=None, err=None):

        vals = par.valuesdict()
        P = vals['P']
        t0 = vals['t0']
        depth = vals['depth']
        width = vals['width']
        impact = vals['impact']
        u1 = vals['u1']
        u2 = vals['u2']
        u3 = vals['u3']
        u4 = vals['u4']
        
        model_flux = self._evaluate_model(P, t0, depth, width, impact, u1, u2, u3, u4)
        
        if data is None:
            return model_flux
        if err is None:
            return data-model_flux
        
        return (data-model_flux)**2. / err**2.
    

    
    def fit_tce(self, tce_num, fit_method='leastsq', use_mask=True):

        tce_P, tce_width, tce_t0 = get_p_tdur_t0(self.tces[tce_num])

        fitparams = Parameters()
        fitparams.add('P', tce_P, min=tce_P-self.lc.exptime, max=tce_P+self.lc.exptime,
                      vary=True)
        fitparams.add('t0',tce_t0-self.lc.exptime, min=tce_t0-2.*tce_width, max=tce_t0+2.*tce_width,
                      vary=True)
        fitparams.add('depth', 100., min=0., max=1e6, vary=True)
        fitparams.add('width', tce_width, min=0.3*tce_width, max=3.*tce_width,vary=True)
        fitparams.add('impact', 0.25, min=0., max=1., vary=True)

        limbs = self.limb_dark
        fitparams.add('u1', limbs[0], min=limbs[0]-0.5, max=limbs[0]+0.5)
        fitparams.add('u2', limbs[1], min=limbs[1]-0.5, max=limbs[1]+0.5)
        fitparams.add('u3', limbs[2], min=limbs[2]-0.5, max=limbs[2]+0.5)
        fitparams.add('u4', limbs[3], min=limbs[3]-0.5, max=limbs[3]+0.5)

        if use_mask:
            tmask = ~make_transit_mask(self.time, tce_P, 2.*tce_width, tce_t0)
            self.model = self._get_transit_model(tce_mask=tmask)
            out = minimize(self.residuals, fitparams,
                           args=(self.flux[tmask],self.flux_err[tmask]),
                           method=fit_method)

        else:
            out = minimize(self.residuals, fitparams, args=(self.flux, self.flux_err),
                        method=fit_method)

        return out




def get_transit_model_implied_density(P, tdur, b, depth, ):

    numerator = (1+np.sqrt(depth))**2. - b**2. * (1.- np.sin(tdur*np.pi/P)**2.)
    denom = np.sin(tdur*np.pi/P)**2.

    return 365.25**2./215**3 * P**2. * (numerator/denom)**(3./2.)






