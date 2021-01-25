import sys

from .utils import *
from .transit_search import *
import wotan
import warnings
from .oxjc import KData, JumpFinder, JumpClassifier, correct_jumps

# Object to read in single Quarter light curve files from Kepler
class single_quarter_lc(object):    
    
    def __init__(self, fpath,):
        
        f = fits.open(fpath, mmap=False)
        self.lc_header = f[1].header
        self.kic_header = f[0].header
        self.data = fits.getdata(fpath)
        
        # Light Curve Data

        self.pdcsap = self.data['PDCSAP_FLUX']/np.nanmedian(self.data['PDCSAP_FLUX'])
        self.pdcsap_err = self.data['PDCSAP_FLUX_ERR']/np.nanmedian(self.data['PDCSAP_FLUX_ERR'])

        self.sap = self.data['SAP_FLUX']
        self.sap_err = self.data['SAP_FLUX_ERR']

        self.sap_bkg = self.data['SAP_BKG']
        self.sap_bkg_err = self.data['SAP_BKG_ERR']

        self.time = self.data['TIME']
        self.cadenceno = self.data['CADENCENO']

        self.mom_centr1 = self.data['MOM_CENTR1']
        self.mom_centr1_err = self.data['MOM_CENTR1_ERR']
        self.mom_centr2 = self.data['MOM_CENTR2']
        self.mom_centr2_err = self.data['MOM_CENTR2_ERR']


        self.cdpp3 = self.lc_header['CDPP3_0']
        self.cdpp6 = self.lc_header['CDPP6_0']
        self.cdpp12= self.lc_header['CDPP12_0']

        self.quality = self.data['SAP_QUALITY']


        #KIC Data
        self.kic = self.kic_header['KEPLERID']
        self.quarter = self.kic_header['QUARTER']

        self.ra = self.kic_header['RA_OBJ']
        self.dec = self.kic_header['DEC_OBJ']
        self.kepmag = self.kic_header['KEPMAG']
        
        f.close()




        
# Class to store fits files in kepler lightcurves from each quarter, convert them
class kepler_lc(object):
    
    def __init__(self, lcfiles, q_detrend=True):
        
        lc_list = [single_quarter_lc(f) for f in sorted(lcfiles)]
        self.lc_list = lc_list
        self.lc_files = lcfiles
        
        # Light Curve Data
        self.pdcsap =  np.concatenate([lc.pdcsap for lc in lc_list])
        self.pdcsap_err = np.concatenate([lc.pdcsap_err for lc in lc_list])
        self.sap = np.concatenate([lc.sap for lc in lc_list])
        self.sap_err = np.concatenate([lc.sap_err for lc in lc_list])

        self.time = np.concatenate([lc.time for lc in lc_list])

        self.mom_centr1 = np.concatenate([lc.mom_centr1 for lc in lc_list])
        self.mom_centr1_err = np.concatenate([lc.mom_centr1_err for lc in lc_list])
        self.mom_centr2 = np.concatenate([lc.mom_centr2 for lc in lc_list])
        self.mom_centr2_err = np.concatenate([lc.mom_centr2_err for lc in lc_list])
        
        self.quality = np.concatenate([lc.quality for lc in lc_list])
        
        
        #KIC Data
        self.kic = lc_list[0].kic
        self.quarters = [lc.quarter for lc in lc_list]

        self.ra = lc_list[0].ra
        self.dec = lc_list[0].dec
        self.kepmag = lc_list[0].kepmag


        stlr = pd.read_hdf('data/meta/berger18_gaia_stlr_q1_q17_table.h5', 'stlr')
        stlr_metadata = stlr.loc[stlr['KIC'] == self.kic].to_dict('list')

        self.stlr = stlr_metadata

    
    # remove nans and outliers and organize everything
    def remove_bad_guys(self, qflags=[5,], sig=3.5, detrended=False):
        
        good_guys = self.pdcsap == self.pdcsap
        
        # mark positive outliers as nans
        self.pdcsap[self.pdcsap > 1.+sig*np.nanstd(self.pdcsap) ] = np.nan

        if detrended:
            self.detrended_pdcsap[self.detrended_pdcsap > 1.+sig*np.nanstd(self.detrended_pdcsap) ] = np.nan
            
        
        #remove nans 
        good_guys = np.logical_and(good_guys, ~np.isnan(self.pdcsap))
        good_guys = np.logical_and(good_guys, ~np.isnan(self.time))
        good_guys = np.logical_and(good_guys, ~np.isnan(self.mom_centr1))
        good_guys = np.logical_and(good_guys, ~np.isnan(self.mom_centr2))
        
        #remove infs
        good_guys = np.logical_and(good_guys, ~np.isinf(self.pdcsap))
        good_guys = np.logical_and(good_guys, ~np.isinf(self.time))
        good_guys = np.logical_and(good_guys, ~np.isinf(self.mom_centr1))
        good_guys = np.logical_and(good_guys, ~np.isinf(self.mom_centr2))

        if detrended:
            good_guys = np.logical_and(good_guys, ~np.isnan(self.detrended_pdcsap))
            good_guys = np.logical_and(good_guys, ~np.isinf(self.detrended_pdcsap))
        
        
        #remove points flagged as bad quality
        for flag in qflags:
            good_guys = np.logical_and(good_guys, self.quality!=2**flag)



        self.time = self.time[good_guys]

        self.pdcsap = self.pdcsap[good_guys]+1e-9 # add 1e-9 to keep from dividing by zero
        self.pdcsap_err = self.pdcsap_err[good_guys]
        self.sap = self.sap[good_guys]
        self.sap_err = self.sap_err[good_guys]
                
        self.mom_centr1 = self.mom_centr1[good_guys]
        self.mom_centr1_err = self.mom_centr1_err[good_guys]
        
        self.mom_centr2 = self.mom_centr2[good_guys]
        self.mom_centr2_err = self.mom_centr2_err[good_guys]
        
        self.quality = self.quality[good_guys]

        if detrended:
            self.detrended_pdcsap = self.detrended_pdcsap[good_guys]
            self.pdcsap_trend = self.pdcsap_trend[good_guys]


        #sort everything by time
        time_order = np.argsort(self.time)
        
        self.time = self.time[time_order]
        self.pdcsap = self.pdcsap[time_order]
        self.pdcsap_err = self.pdcsap_err[time_order]

        self.sap = self.sap[time_order]
        self.sap_err = self.sap_err[time_order]
        
        self.mom_centr1 = self.mom_centr1[time_order]
        self.mom_centr2 = self.mom_centr2[time_order]
        
        self.mom_centr1_err = self.mom_centr1_err[time_order]
        self.mom_centr2_err = self.mom_centr2_err[time_order]
        
        self.quality = self.quality[time_order]

        if detrended:
            self.detrended_pdcsap = self.detrended_pdcsap[time_order]
            self.pdcsap_trend = self.pdcsap_trend[time_order]
        
        return self


    
    def flatten(self, window, method='biweight'):

        self.detrended_pdcsap, self.pdcsap_trend = flatten_lc(self.time, self.pdcsap, window=window, method=method, )

        return self

  
    def remove_gap_edges(self, min_dif=0.5, sig=2.):

        self.time, self.detrended_pdcsap, edge_cut = flag_gap_edges(self.time, self.detrended_pdcsap, min_dif, sig)

        self.pdcsap = self.pdcsap[edge_cut]
        self.pdcsap_err = self.pdcsap_err[edge_cut]

        self.pdcsap_trend = self.pdcsap_trend[edge_cut]

        self.sap = self.sap[edge_cut]
        self.sap_err = self.sap_err[edge_cut]
        
        self.mom_centr1 = self.mom_centr1[edge_cut]
        self.mom_centr2 = self.mom_centr2[edge_cut]
        
        self.mom_centr1_err = self.mom_centr1_err[edge_cut]
        self.mom_centr2_err = self.mom_centr2_err[edge_cut]
        
        self.quality = self.quality[edge_cut]

        return self


    
    def save_as_h5(self, fname):

        lc_df = create_df_from_lc(self)
        metadata = {'kic':self.kic, 'tdur_sigma':self.tdur_sigma, 'tdurs':self.tdurs, 'stlr':self.stlr }
        
        h5store(fname, lc_df, **metadata)
        
        return self



class KepCBV(object):

    def __init__(self, quarter, channel, cbv_dir='data/meta/CBV/'):

        cbvfiles = ['kplr2009131105131-q00-d25_lcbv.fits',
                    'kplr2009166043257-q01-d25_lcbv.fits',
                    'kplr2009259160929-q02-d25_lcbv.fits',
                    'kplr2009350155506-q03-d25_lcbv.fits',
                    'kplr2010078095331-q04-d25_lcbv.fits',
                    'kplr2010174085026-q05-d25_lcbv.fits',
                    'kplr2010265121752-q06-d25_lcbv.fits',
                    'kplr2010355172524-q07-d25_lcbv.fits',
                    'kplr2011073133259-q08-d25_lcbv.fits',
                    'kplr2011177032512-q09-d25_lcbv.fits',
                    'kplr2011271113734-q10-d25_lcbv.fits',
                    'kplr2012004120508-q11-d25_lcbv.fits',
                    'kplr2012088054726-q12-d25_lcbv.fits',
                    'kplr2012179063303-q13-d25_lcbv.fits',
                    'kplr2012277125453-q14-d25_lcbv.fits',
                    'kplr2013011073258-q15-d25_lcbv.fits',
                    'kplr2013098041711-q16-d25_lcbv.fits',
                    'kplr2013131215648-q17-d25_lcbv.fits']

        open_file = fits.open(cbv_dir+cbvfiles[quarter])
        self.table = open_file[channel].data
        self.vectors = np.array([self.table['VECTOR_{}'.format(i)] for i in range(1,16)])
        open_file.close()
        
    def get_vectors(self, n=5):

        vectors = self.vectors[:n]        
        vectors_plus_constant = np.insert( vectors, 0, np.ones_like(vectors[0]), axis=0 )

        return vectors_plus_constant


    
    

class KepData( object ):

    def __init__(self, lcfiles, ncbvs=5):

        self.lcfiles = sorted(lcfiles)

        self.lightcurves =  [single_quarter_lc(f) for f in sorted(lcfiles)]
        self.quarters = [lc.kic_header['QUARTER'] for lc in self.lightcurves]
        self.channels = [lc.kic_header['CHANNEL'] for lc in self.lightcurves]
        self.modules = [lc.kic_header['MODULE'] for lc in self.lightcurves]

        self.kic = self.lightcurves[0].kic
        self.tric = None

        self.X = self._get_cbv_matrices(n=ncbvs)


    def _get_cbv_matrices(self, n=5):

        matrices=[]
        for i in range(len(self.lightcurves)):
            q = self.quarters[i]
            c = self.channels[i]

            matrices.append( KepCBV(q,c).get_vectors(n) )

        return matrices



    def _get_TrIC_Data(self, TrIC_file):

        data=1.
        return data


    # Taken from lightkurve's regression corrector
    # (https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/correctors/
    # regressioncorrector.py) and modified for use here 
    def _fit_coefficients(self, i, cadence_mask=None, prior_mu=None,
                          prior_sigma=None, propagate_errors=False):
        """Fit the linear regression coefficients.
        This function will solve a linear regression with Gaussian priors
        on the coefficients.
        Parameters
        ----------
        cadence_mask : np.ndarray of bool
            Mask, where True indicates a cadence that should be used.
        Returns
        -------
        coefficients : np.ndarray
            The best fit model coefficients to the data.
        """
        if prior_mu is not None:
            if len(prior_mu) != len(self.X[i].T):
                raise ValueError('`prior_mu` must have shape {}'
                                 ''.format(len(self.X[i].T)))
        if prior_sigma is not None:
            if len(prior_sigma) != len(self.X[i].T):
                raise ValueError('`prior_sigma` must have shape {}'
                                 ''.format(len(self.X[i].T)))
            if np.any(prior_sigma <= 0):
                raise ValueError('`prior_sigma` values cannot be smaller than '
                                 'or equal to zero')

        # If prior_mu is specified, prior_sigma must be specified
        if not ((prior_mu is None) & (prior_sigma is None)) | \
                    ((prior_mu is not None) & (prior_sigma is not None)):
            raise ValueError("Please specify both `prior_mu` and `prior_sigma`")

        # Default cadence mask
        if cadence_mask is None:
            cadence_mask = ~np.isnan(self.lightcurves[i].sap)
            
        # If flux errors are not all finite numbers, then default to array of ones
        if np.all(~np.isfinite(self.lightcurves[i].sap_err)):
            flux_err = np.ones(cadence_mask.sum())
        else:
            flux_err = self.lightcurves[i].sap_err[cadence_mask]

        # Retrieve the CBV matrix (X) as a numpy array
        X = self.X[i].T[cadence_mask]        

        # Compute `X^T cov^-1 X + 1/prior_sigma^2`
        sigma_w_inv = np.dot(X.T, X / flux_err[:, None]**2)
        if prior_sigma is not None:
            sigma_w_inv += np.diag(1. / prior_sigma**2)

        # Compute `X^T cov^-1 y + prior_mu/prior_sigma^2`
        norm_sap_flux = lambda sap: sap/np.nanmedian(sap) -1.        
        B = np.dot(X.T,  norm_sap_flux(self.lightcurves[i].sap[cadence_mask] - self.lightcurves[i].sap_bkg[cadence_mask]) / flux_err**2)
        if prior_sigma is not None:
            B += (prior_mu / prior_sigma**2)

        # Solve for weights w
        w = np.linalg.solve(sigma_w_inv, B).T
        if propagate_errors:
            w_err = np.linalg.inv(sigma_w_inv)
        else:
            w_err = np.zeros(len(w)) * np.nan

        return w, w_err


    # Shamelessly taken from lightkurve's regression corrector and modified for use here
    # https://github.com/KeplerGO/lightkurve/blob/master/lightkurve/correctors/regressioncorrector.py
    def cbv_correct_lc(self, i, cadence_mask=None, sigma=3,
                niters=5, propagate_errors=False):
        """Find the best fit correction for the light curve.
        Parameters
        ----------
        CBV_Matrices: CBV.
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        sigma : int (default 5)
            Standard deviation at which to remove outliers from fitting
        niters : int (default 5)
            Number of iterations to fit and remove outliers
        propagate_errors : bool (default False)
            Whether to propagate the uncertainties from the regression. Default is False.
            Setting to True will increase run time, but will sample from multivariate normal
            distribution of weights.
        Returns
        -------
        `.LightCurve`
            Corrected light curve, with noise removed.
        """

        #if isinstance(design_matrix_collection, DesignMatrix):
        #    design_matrix_collection = DesignMatrixCollection([design_matrix_collection])
        #design_matrix_collection._validate()
        #self.design_matrix_collection = design_matrix_collection

        if cadence_mask is None:
            cadence_mask = ~np.isnan(self.lightcurves[i].sap)
        else:
            cadence_mask = np.copy(cadence_mask)

        # Prepare for iterative masking of residuals
        clean_cadences = np.ones_like(cadence_mask)
        
        # Iterative sigma clipping
        for count in range(niters):
            coefficients, coefficients_err = \
                self._fit_coefficients(i, cadence_mask=cadence_mask & clean_cadences,
                                       propagate_errors=propagate_errors)
            model = np.ma.masked_array(data=np.dot(self.X[i].T, coefficients),
                                       mask=~(cadence_mask & clean_cadences))

            norm = lambda x: x/np.nanmedian(x) -1.
            residuals = norm(self.lightcurves[i].sap) - model
            _,_,clean_cadences = sigma_clip(cadence_mask, residuals, upper_sig=sigma, lower_sig=sigma)
            #log.debug("correct(): iteration {}: clipped {} cadences"
            #          "".format(count, (~clean_cadences).sum()))

        #self.cadence_mask = cadence_mask & clean_cadences
        #self.coefficients = coefficients
        #self.coefficients_err = coefficients_err

        model_flux = np.dot(self.X[i].T, coefficients)
        #model_flux -= np.median(model_flux)
        if propagate_errors:
            with warnings.catch_warnings():
                # ignore "RuntimeWarning: covariance is not symmetric positive-semidefinite."
                warnings.simplefilter("ignore", RuntimeWarning)
                samples = np.asarray(
                    [np.dot(self.X[i].T,
                            np.random.multivariate_normal(coefficients, coefficients_err))
                     for idx in range(100)]).T
            model_err = np.abs(np.percentile(samples, [16, 84], axis=1) - np.median(samples, axis=1)[:, None].T).mean(axis=0)
        else:
            model_err = np.zeros(len(model_flux))
            
        #self.model_lc = LightCurve(self.lc.time, model_flux, model_err)
        #self.corrected_lc = self.lc.copy()
        #self.corrected_lc.flux = self.lc.flux - self.model_lc.flux
        #self.corrected_lc.flux_err = (self.lc.flux_err**2 + model_err**2)**0.5
        #self.diagnostic_lightcurves = self._create_diagnostic_lightcurves()

        corrected_flux =  norm(self.lightcurves[i].sap) - model_flux

        if propagate_errors:
            sap_err = self.lightcurves[i].sap_err / np.nanmedian(self.lightcurves[i].sap)
            corrected_flux_errs =  (self.lightcurves[i].sap_err**2 + model_err**2)**0.5

        else:
            sap_err = self.lightcurves[i].sap_err
            corrected_flux_errs =   w_err = np.zeros(len(sap_err)) * np.nan
             
        return corrected_flux, corrected_flux_errs



    def fix_jumps(self, i):


        lc = self.lightcurves[i]

        jc_kdata = KData(cadence = lc.cadenceno, time = lc.time, flux=lc.sap, quality=lc.quality)
        
        jf = JumpFinder(jc_kdata,)
        jumps = jf.find_jumps()
        jc = JumpClassifier(jc_kdata, jf.hp)
        jc.classify(jumps)

        cdata = correct_jumps(jc_kdata, jumps)

        self.lightcurves[i].sap = cdata._flux
        
        return cdata




    def make_LightCurve(self, cotrend=True):

        time = np.concatenate([self.lightcurves[i].time for i in range(len(self.lightcurves))] )
        nan_mask = np.concatenate([~np.isnan(self.lightcurves[i].sap) for i in range(len(self.lightcurves))] )

        flags = np.concatenate([self.lightcurves[i].quality for i in range(len(self.lightcurves))] )

        lc_fluxes = []
        lc_flux_errs = []
        lc_trends = []
        
        for i,lc in enumerate(self.lightcurves):

            if cotrend:
                print('Cotrending Quarter {}......'.format(self.quarters[i]) )
                flux, flux_err = self.cbv_correct_lc(i, propagate_errors=True)
            else:
                flux, flux_err = self.lightcurves[i].sap, self.lightcurves[i].sap_err
                
            
            lc_fluxes.append(flux)
            lc_flux_errs.append(flux_err)
        
        flux_err = np.concatenate(lc_flux_errs)
        flux = np.concatenate(lc_fluxes)
        lightcurve = LightCurve(time = time, flux = flux,
                                flux_err = flux_err,mask=nan_mask,
                                flags = flags,
                                mission='KEPLER', ID=self.kic)

        return lightcurve
    

    def make_Transitsearch_LightCurve(self, use_pdcsap=False, **wotan_kw):

        time = np.concatenate([self.lightcurves[i].time for i in range(len(self.lightcurves))] )
        nan_mask = np.concatenate([~np.isnan(self.lightcurves[i].sap) for i in range(len(self.lightcurves))] )

        flags = np.concatenate([self.lightcurves[i].quality for i in range(len(self.lightcurves))] )

        lc_fluxes = []
        lc_flux_errs = []
        lc_trends = []

        
        for i,lc in enumerate(self.lightcurves):


            if use_pdcsap:
                corr_flux, corr_flux_err = lc.pdcsap, lc.sap_err/lc.sap

            else:
                self.fix_jumps(i)
                print('Cotrending Quarter {}......'.format(self.quarters[i]) )
                corr_flux, corr_flux_err = self.cbv_correct_lc(i, propagate_errors=True)
            
            #mask = ~np.isnan(self.lightcurves[i].sap)
            print('Detrending Quarter {}......'.format(self.quarters[i]) )
            detrend_flux, trend = wotan.flatten(lc.time, corr_flux+1., return_trend=True, **wotan_kw)

            lc_fluxes.append(detrend_flux)
            lc_flux_errs.append(corr_flux_err)
            lc_trends.append(trend)

        
        flux_err = np.concatenate(lc_flux_errs)
        flux = np.concatenate(lc_fluxes)
        trend = np.concatenate(lc_trends)
        transitsearch_lightcurve = LightCurve(time = time, flux = flux,
                                              flux_err = flux_err,
                                              trend = trend, mask=nan_mask,flags = flags,
                                              mission='KEPLER', ID=self.kic)

        return transitsearch_lightcurve



    def make_TransitInjection_LightCurve(self, rp_range=(0.5, 20.), per_range=(1.,500.), **wotan_kw):

        '''
        TODO: MAKE THIS FUNCTION
        '''

        
        return 1.

    

    def pkl_dump(self, DATA_DIR, fname=None):
        
        '''
        TODO: MAKE THIS FUNCTION
        '''


        
        if fname is None:
            fname = 'kic{:9}_kepdata.pkl'.format(self.kic)
        

        return 1.





class TESS_2minData(object):

    def __init__(self, lcfiles, ncbvs=5):

        self.lcfiles = sorted(lcfiles)

        self.lightcurves =  [single_quarter_lc(f) for f in sorted(lcfiles)]
        self.quarters = [lc.kic_header['QUARTER'] for lc in self.lightcurves]
        self.channels = [lc.kic_header['CHANNEL'] for lc in self.lightcurves]
        self.modules = [lc.kic_header['MODULE'] for lc in self.lightcurves]

        self.kic = self.lightcurves[0].kic
        self.tric = None


class TESS_30minData(object):

    '''
    Designed to work with Eleanor FFI Lightcurves. Input a list of TargetData objects 
    for each sector, and go from there. 
    '''

    def __init__(self, AllSectorData, tric_data=None, tricfile=None):

        self.AllSectorData = AllSectorData

        self.sectors = [data.source_info.sector for data in self.AllSectorData]
        self.cameras = [data.source_info.camera for data in self.AllSectorData]
        self.chips = [data.source_info.chip for data in self.AllSectorData]
            
        self.tic = self.AllSectorData[0].source_info.tic
        self.tess_mag = self.AllSectorData[0].source_info.tess_mag

        if tric_data is None:
            self.tric = self._get_TrIC_Data(tricfile)
        else:
            self.tric = tric_data

        self.header = AllSectorData[0].header




    def _get_TrIC_Data(self, TrIC_file):

        data=1.
        return data



    def make_Transitsearch_LightCurve(self, **wotan_kw):

        time = np.concatenate([self.AllSectorData[i].time for i in range(len(self.AllSectorData))] )
        nan_mask = np.concatenate([~np.isnan(self.AllSectorData[i].corr_flux) for i in range(len(self.AllSectorData))] )

        flags = np.concatenate([self.AllSectorData[i].quality for i in range(len(self.AllSectorData))] ).astype(int)

        sector_dates = [self.AllSectorData[i].time[0] for i in range(len(self.AllSectorData))]

        lc_fluxes = []
        lc_flux_errs = []
        lc_trends = []

        
        for i,lc in enumerate(self.AllSectorData):
            
            #mask = ~np.isnan(self.lightcurves[i].sap)
            print('Detrending Sector {}......'.format(self.sectors[i]) )
            detrend_flux, trend = wotan.flatten(lc.time, lc.corr_flux, return_trend=True, **wotan_kw)

            lc_fluxes.append(detrend_flux)
            lc_flux_errs.append(lc.flux_err)
            lc_trends.append(trend)

        
        flux_err = np.concatenate(lc_flux_errs)
        flux = np.concatenate(lc_fluxes)
        trend = np.concatenate(lc_trends)


        
        transitsearch_lightcurve = LightCurve(time = time, flux = flux,
                                              flux_err = flux_err,
                                              trend = trend, mask=nan_mask, flags = flags,
                                              mission='TESS', ID=self.tic,
                                              segment_dates=sector_dates)


        return transitsearch_lightcurve
    
    
    
def flatten_lc(time, pdcsap, window, method='biweight'):

    return flatten(time, pdcsap, window_length=window, method=method, return_trend=True)







