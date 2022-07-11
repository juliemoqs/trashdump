import sys
from os import path, getcwd

import numpy as np

import batman
import warnings

from .utils import *


def transit_residual(m,params):

    per, t0, dur, depth, b = p    
    
    a = 1./np.sin( (np.pi*dur)/per )
    inc = np.arccos(b/a)*(180./np.pi)   
    
    params.t0 = t0                     
    params.per = per
    params.rp = np.sqrt( 10.**depth )      
    params.a =  a
    params.inc = inc

    
    return m




def odd_even_transit(t, *par):

    t0, a_rs, depth = par


    return 1.

    


def phase_folded_transit(t, *par):

    t0, a_rs, depth = par
    
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = t0                        #time of inferior conjunction
    params.per = 1.                       #orbital period
    params.rp = np.sqrt(depth)            #planet radius (in units of stellar radii)
    params.a = a_rs                       #semi-major axis (in units of stellar radii)
    params.inc = 90.                      #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "nonlinear"        #limb darkening model
    params.u=[0.5, 0.1, 0.1, -0.1]

    m = batman.TransitModel(params,t)
    
    return m.light_curve(params) 


def jump(t, *par):
    a, b, t0 = par
    return a/(1. + np.exp(-b*(t-t0)))

def spsd(t, *par):
    a,b,t0,d,tdur = par
    return jump(t, *[a,b,t0]) * (1.- np.exp(-d*(t-tdur/2.)) )  

def box(t, *par):
    a,b,tdur,t0=par
    return jump(t-tdur/2., *[a,b,t0]) - jump(t+tdur/2., *[a,b,t0]) 

def sine(t, *par):
    a,tdur,t0=par
    return -a*np.cos((t-t0)/(2*tdur) )








def phase_folded_transit_residual(pars,x,data=None,err=None):

    vals = pars.valuesdict()
    width = vals['tdur']
    t0 = vals['t0']
    depth = vals['depth']
    c0 = vals['depth']
    
    model = c0 + phase_folded_transit(x, *[t0,width,depth]) - 1.
    
    if data is None:
        return model
    if err is None:
        return data-model
    return (data-model) / err**2.



def sine_residual(pars, x, data=None, err=None):
    
    vals = pars.valuesdict()
    a = vals['a']
    tdur = vals['tdur']
    t0 = vals['t0']
    c0 = vals['c0']
    c1 = vals['c1']
    c2 = vals['c2']
    
    model = (c1*x + c2*x**2. + c0 ) + sine(x, *[a, tdur,t0])
    if data is None:
        return model
    if err is None:
        return data-model
    return (data-model) / err**2.


def jump_residual(pars, x, data=None, err=None):
    vals = pars.valuesdict()
    a = vals['a']
    #b = vals['b']
    t0 = vals['t0']
    c0 = vals['c0']
    c1 = vals['c1']
    c2 = vals['c2']

    model = (c1*x + c2*x**2. + c0 ) + jump(x, *[a, 9999.,t0])
    if data is None:
        return model
    if err is None:
        return data-model
    return (data-model) / err**2.


def spsd_residual(pars, x, data=None, err=None):
    vals = pars.valuesdict()
    a = vals['a']
    d = vals['d']
    t0 = vals['t0']
    tdur = vals['tdur']
    c0 = vals['c0']
    c1 = vals['c1']
    c2 = vals['c2']

    model = (c1*x + c2*x**2. + c0 ) + spsd(x, *[a,9999.,t0,d,tdur])
    
    if any(np.isnan(model)):
        print('OH NO!!!!!! NANS!!!!!!!')
        print(vals)
    
    if data is None:
        return model
    if err is None:
        return data-model
    return (data-model) / err**2.



def box_residual(pars, x, data=None, err=None):
    vals = pars.valuesdict()
    a = vals['a']
    tdur = vals['tdur']
    t0 = vals['t0']
    c0 = vals['c0']
    c1 = vals['c1']
    c2 = vals['c2']

    model = (c1*x + c2*x**2. + c0 ) + box(x, *[a, 5e3, tdur,t0])
    if data is None:
        return model
    if err is None:
        return data-model
    return (data-model) / err**2.



def trap_residual(pars, x, data=None, err=None):
    
    vals = pars.valuesdict()
    a = vals['a']
    b = vals['b']
    tdur = vals['tdur']
    t0 = vals['t0']
    c0 = vals['c0']
    c1 = vals['c1']
    c2 = vals['c2']

    model = (c1*x + c2*x**2. + c0 ) + box(x, *[a, b, tdur,t0])
    
    if data is None:
        return model
    if err is None:
        return data-model
    return (data-model) / err**2.    
