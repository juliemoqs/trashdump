import sys
from os import path, getcwd

import numpy as np

import batman
import warnings
import batman

from lmfit import minimize, Parameters

from .utils import *


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

    model = (c1*x + c2*x**2. + c0 ) + box(x, *[a,500.,tdur,t0])
    if data is None:
        return model
    if err is None:
        return data-model
    return (data-model) / err**2.



