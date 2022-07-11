import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import lightkurve as lk
import trashdump.dump as dump


ticid = 52368076
rstar = 0.86
mstar = 0.85
teff=5000.
logg= 4.5 #np.log10(mstar/(rstar**2.)) + 4.44
density = 1.99 #mstar/(rstar**3.)


search_result = lk.search_lightcurve('TIC'+str(ticid), mission='TESS',
                                     author='TESS-SPOC',exptime=1800)

lc_collection = search_result.download_all(quality_bitmask='default')
lc = lc_collection.stitch().remove_nans()


tesslc = dump.LightCurve(time=lc['time'].btjd, flux=lc['pdcsap_flux'].value, flux_err=lc['pdcsap_flux_err'].value,
                               flags=lc['quality'],mission='TESS',ID=ticid )


wotan_kw = {'method':'biweight','robust':True, 'window_length':1.,'edge_cutoff':0.5,'break_tolerance':0.25,
           'cval':5.}
tesslc.flatten(wotan_kw)
tesslc.mask_badguys(sig=3)


tessdump = dump.Dump(tesslc, min_transits=2, star_logg=logg, star_teff=teff, star_density=density,)


tce_search_kw = {'print_updates':False, 'progbar':True, 'single_frac':0.7, 'check_vetoes':True,
                'dur_range':(0.3,2.5), 'threshold':7.}

tces = tessdump.Iterative_TCE_Search(niter=3, **tce_search_kw)


print(tces)
