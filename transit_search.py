import sys
from os import path, getcwd

from fast_histogram import histogram1d
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import median_filter

from .ffa import *
from .lightcurves import *
from .utils import *
from tqdm import tqdm

filepath = path.abspath(__file__)
dir_path = path.dirname(filepath)

TRASH_DATA_DIR = dir_path+'/'


class Dump(object):

    def __init__(self, LightCurve, tdurs=None, star_logg=None, star_teff=None,
                 star_density=None, min_transits=3.):
        
        self.lc = LightCurve

        if star_logg is None:
            star_logg = 4.4
        if star_teff is None:
            star_teff=5877.
        if star_density is None:
            star_density=1.
            
        self.star_logg = star_logg
        self.star_teff = star_teff
        self.star_density = star_density

        if tdurs is None:
            self.tdurs = self._calc_best_tdurs(n_trans=min_transits)
        else:
            self.tdurs = tdurs
            
        self.tdur_sigma = [[]]*len(self.tdurs)
        
        self.limb_darkening = self._get_limb_darkening()
        
        self.ses = None
        self.num = None
        self.den = None

        self.TCEs = None
        self.tce_mask = self.lc.mask.copy()
        self.tdur_cdpp = []
        self.search_periods = kep_period_sampling(time=self.lc.time[self.lc.mask],
                                                  P_min=None, n_tr=min_transits,
                                                  rho_star=star_density)
        self.min_transits = min_transits



    def _add_tce_to_mask(self, TCEs):

        for tce in TCEs:
            self.tce_mask &= make_transit_mask(self.lc.time, P=tce[1], t0=tce[3], dur=tce[4])
        self.tce_mask &= self.lc.mask.copy()

    def _calc_best_tdurs(self, n_trans=2., maxtdur = 1.5):

        if self.star_density is None:
            rho_star = 1.
        else:
            rho_star = self.star_density

        exptime = self.lc.exptime

        baseline = max(self.lc.time[self.lc.mask]) - min(self.lc.time[self.lc.mask])
        max_period = baseline/n_trans
        max_dur = min( (13./24.) * (max_period/365.)**(1./3.) * rho_star**(-1./3.) * 1.5, maxtdur)
        
        tdurs = np.unique( exptime  * (1.1)**np.arange(2,100) // exptime ) * exptime
    
        cut = tdurs<max_dur
        cut &= tdurs>exptime*2.5
    
        return tdurs[cut]

        

    def _get_limb_darkening(self, interpfile=TRASH_DATA_DIR+'data/meta/limb_coeffs.h5'):
        
        mission = self.lc.mission

        if mission=='KEPLER':
            limb_grid = KEPLER_LIMB_DARKENING_GRID
        elif mission=='TESS':
            limb_grid = TESS_LIMB_DARKENING_GRID
        else:
            limb_grid = get_limb_darkening_grid(mission, grid_file=interpfile)
            
        limb_dark = griddata(points=limb_grid[['teff','logg']].to_numpy(), 
                             values=limb_grid[['a1','a2','a3','a4']].to_numpy(),
                             xi=[self.star_teff, self.star_logg], rescale=True,
                             method='nearest')
        
        return limb_dark


    def get_transit_model(self, depth, width, pad=None, b=0.15):

        if pad is None:
            pad = len(self.lc.flux)*2

        tr_model = get_transit_signal(width, depth, limb_dark=self.limb_darkening,
                                      exp_time=self.lc.exptime, pad=pad, b=b)
        
        return tr_model-1.



    def _get_whitening_coeffs(self):

        for sd in self.lc.Segment_Dates:

            print(sd)

        whitening_coeff = 1.
        return whitening_coeff

    
    

    def Calculate_SES(self, mask=None, var_calc='mad',fill_mode='reflect',tdurs=None, ses_weights=None, window_width=9., tdepth=84.):

        if mask is None:
            mask = self.lc.mask.copy()

        gap_time, gap_flux, gap_array =  pad_time_series(self.lc.time.copy()[mask],
                                                         self.lc.flux.copy()[mask],
                                                         in_mode=fill_mode,
                                    pad_end=False, fill_gaps=True )

        pad_time, pad_flux, pad_array =  pad_time_series(gap_time, gap_flux,
                                    pad_end=True, fill_gaps=False)

        FluxTransform = ocwt( pad_flux-1., )
        FluxTransform = FluxTransform[:, pad_array]

        if tdurs is None:
            dur_iter = self.tdurs
        else:
            dur_iter = tdurs

        ses_alldur = [[]]*len(dur_iter)
        num_alldur = [[]]*len(dur_iter)
        den_alldur = [[]]*len(dur_iter)

        for i,dur in enumerate(dur_iter):

            #print( 'Calculating SES for duration = {:.2f} days'.format(dur) )
            
            Signal = self.get_transit_model(depth=tdepth,width=dur, pad=len(pad_array)  )
            SignalTransform = ocwt(Signal)

            ses, num, den, sig2 = calculate_SES(FluxTransform, SignalTransform,
                                                texp=self.lc.exptime,
                                                window_size=window_width*dur,
                                                var_calc=var_calc )
            
            ses_alldur[i] = ses[gap_array]
            num_alldur[i] = num[gap_array]
            den_alldur[i] = den[gap_array]

            self.tdur_cdpp.append(np.median(np.sqrt(1./sig2))*1e6 )

        if ses_weights is None:
            ses_alldur = np.array(ses_alldur)
            num_alldur = np.array(num_alldur)
            
        else:
            ses_alldur = np.array(ses_alldur)/ses_weights
            num_alldur = np.array(num_alldur)/ses_weights
            
        den_alldur = np.array(den_alldur)
        
        self.ses = ses_alldur
        self.num = num_alldur
        self.den = den_alldur
        
        return ses_alldur, num_alldur, den_alldur



    def Search_for_TCEs(self, ses_cut=20., threshold=7.,remove_hises=True,calc_ses=True,return_df=False,use_tce_mask=False, **tce_search_kw):

        if use_tce_mask:
            mask = self.tce_mask.copy()
        else:
            mask = self.lc.mask.copy()
        #tce_ses_mask = np.ix_(np.ones(len(self.tdurs), dtype=bool), self.tce_mask[mask])

        if self.ses is None or calc_ses:
            ses, num, den = self.Calculate_SES(mask=mask)
        else:
            ses, num, den = self.ses.copy(), self.num.copy(), self.den.copy()
        

        periods = self.search_periods
        
        # Create a mask for Bad points in the Light Curve that can cause FPs
        time = self.lc.time.copy()
        mask_time = time[mask]
        
        ses_mask = np.ones(len(ses[0]), dtype=bool )
        
        for s in ses:
            ses_mask = np.logical_and(ses_mask, s<ses_cut)

        if np.sum(~ses_mask)>150:
            print('Not removing {0} points with SES>{1}'.format(np.sum(~ses_mask), int(ses_cut) ) )
            ses_mask =  np.ones(len(ses[0]), dtype=bool )
        else:
            print('Removing {0} points with SES>{1}'.format(np.sum(~ses_mask), int(ses_cut) ) )

        if remove_hises:
            # Cut highest SES:
            hises_ind = np.argmax(ses[-1])
            hises_mask = np.abs(mask_time -  mask_time[hises_ind]) > 0.75
            ses_mask &= hises_mask
            
        # Use for weird multi-dimensional python indexing reasons... 
        nd_ses_mask = np.ix_(np.ones(len(self.tdurs), dtype=bool), ses_mask)

        
        TCEs = Search_for_TCEs_in_all_Tdur_models(time=mask_time[ses_mask],
                                                  num=num[nd_ses_mask],
                                                  den=den[nd_ses_mask],
                                                  ses=ses[nd_ses_mask],
                                                  period_sampling=periods,
                                                  kicid=self.lc.ID,
                                                  t_durs=self.tdurs,
                                                  rho_star=self.star_density,
                                                  outputfile='tce_detection_test.h5',
                                                  texp=self.lc.exptime,
                                                  num_transits=self.min_transits,
                                                  return_df=False,
                                                  threshold=threshold,
                                                  **tce_search_kw)

        
        TCEs_checked = mask_highest_TCE_and_check_others(TCEs=TCEs,
                                                         time=mask_time.copy(),
                            num=self.num.copy(), den=self.den.copy(), 
                                tdurs=self.tdurs, threshold=threshold)

        if return_df:
            return pd.DataFrame(TCEs_checked,
                                columns=['star_id','period','mes','t0_bkjd','tdur'] )

        return TCEs_checked





    def Search_TCEs_FFA(self,super_sample=3, dur_range=[0.25,1.5], **tce_search_kw):

        
        all_tces = pd.DataFrame({'star_id':[],'period':[],'mes':[],'t0_bkjd':[],'tdur':[]}) 

        ses_mask = self.lc.mask.copy()
        ses, num, den = self.Calculate_SES(mask=ses_mask, fill_mode='reflect',
                                           tdurs=self.tdurs)

        masktime = self.lc.time.copy()[self.lc.mask]
        
        for i_dur in range(len(self.tdurs)):


            dur = self.tdurs[i_dur]
            
            print('Star {1}: Searching tdur = {0:.3f}'.format(dur,self.lc.ID) )

            ses_i = self.ses[i_dur].copy()
            num_i = self.num[i_dur].copy()
            den_i = self.den[i_dur].copy()
            
    
            P_min = max(365. * (dur * 24./13.)**3. * (1.) * dur_range[0]**3.,  dur*8.)
            P_max = min(365. * (dur * 24./13.)**3. * (1.) * dur_range[1]**3.,  max(self.search_periods) )

            mid_time, num_hist, den_hist = FFA_Search_Duration_Downsample(masktime, num_i, den_i, dur=dur, exptime=self.lc.exptime, super_sample=super_sample)
            
            TCEs = FFA_TCE_Search(time=mid_time, num=num_hist, den=den_hist,
                                  cadence=dur/super_sample, kicid=self.lc.ID,
                                  P0_range=(P_min,P_max), dur=dur,
                                  **tce_search_kw)


            if len(TCEs.dropna())>0:
                tces_noharm = remove_TCE_harmonics(TCEs.to_numpy(), known_TCEs=None,
                                                      tolerance=0.0001)
                for tce in tces_noharm:
            
                    tce_per, tce_mes, tce_t0, tce_width = find_best_params_for_TCE(time=masktime, num=num, den=den,t0=tce[3], tdurs=self.tdurs, P=tce[1], texp=self.lc.exptime,)


                    TCE_append = [tce[0], tce_per, tce_mes, tce_t0, tce_width]
                    all_tces = pd.concat([all_tces, pd.DataFrame([TCE_append],columns=['star_id','period','mes','t0_bkjd','tdur'])])


        tces_noharm = remove_TCE_harmonics(all_tces.to_numpy(), 
                                           tolerance=0.0005)
        TCEs_checked = mask_highest_TCE_and_check_others(TCEs=tces_noharm,
                            time=masktime, num=num, den=den, tdurs=self.tdurs)

        return TCEs_checked

    


    def Iterative_TCE_Search(self, niter=3, ses_cut=999999., threshold=7.0, check_vetoes=False, remove_hises=False, **tce_search_kw):

        '''
        IN DEVELOPMENT. 
        '''
        
        n=0
        ntces = 1
        candidates = np.array([])

        _ = self.Calculate_SES( )
    
        while n<niter and ntces>0:
            n+=1
            print('\nStarting TCE Search Number {}'.format(n))

            if n>1:
                ses_cut=20
                remove_hises=remove_hises
                check_vetoes=check_vetoes

            else:
                ses_cut=ses_cut
                remove_hises=False
                check_vetoes=False
                
            TCEs = self.Search_for_TCEs(calc_ses=False, threshold=threshold, ses_cut=ses_cut,remove_hises=remove_hises, return_df=False, use_tce_mask=True, check_vetoes=check_vetoes, **tce_search_kw )


            if np.sum(np.isnan(TCEs[:,1]))>0:
                ntces=0.
            else:
                if n>1:
                    TCEs = remove_TCE_harmonics(TCEs, candidates.reshape(-1,5)[:,1])
                ntces = len(TCEs)

                for tce in TCEs:
                    self.tce_mask &= make_transit_mask(self.lc.time,P=tce[1],
                                                       t0=tce[3],dur=2.5*tce[4] )
                   
                candidates = np.append(candidates,TCEs)
            print('\n{0} TCEs added on Iteration {1}/{2}'.format(ntces,n,niter))
            print('Current Candidates:')
            print(candidates.reshape(-1,5))

            
            if len(candidates)>10*5:
                ntces=0

            if ntces > 0:
                if np.sum(self.tce_mask)>np.sum(self.lc.mask)/2.:
                    ses, num, den = self.Calculate_SES(mask = self.tce_mask.copy() )
                else:
                    ses, num, den = self.Calculate_SES(mask = self.tce_mask.copy() )

        self.TCEs = candidates.reshape(-1,5)
        
        return pd.DataFrame(candidates.reshape(-1,5), columns=['star_id','period','mes','t0_bkjd','tdur'])



    def plot_mes(self, t0, P, tdur, norm=False, ses_calc='mad', zoom=True, calc_ses=False, tce_mask=False, use_mask=None, plot_binflux=True):

        mask = self.lc.mask.copy()

        if tce_mask:
            mask &= self.tce_mask.copy()

        if not(use_mask is None):
            mask &= use_mask

        if calc_ses:
            ses, num, den = self.Calculate_SES(mask=mask, var_calc=ses_calc)
        else:
            ses, num, den = self.ses.copy(), self.num.copy(), self.den.copy()

        f, axes = plt.subplots(len(self.tdurs)+1, 1, sharex=True, figsize=(8, len(self.tdurs)*1.5), )

        fold_time = ((self.lc.time.copy()[mask] - t0 + P/2.) % P )

        mes_ylim = 8.
        mes_ymin = -3.

        max_mes = np.array([])
        
        for i,dur in enumerate(self.tdurs):

            t_mes,mes = calc_mes(fold_time, num=num[i], den=den[i], P=P,
                           texp=self.lc.exptime, norm=norm)

            max_mes = np.append(max_mes, np.max(mes) )
            
            axes[i+1].plot(t_mes, mes, lw=0.5, label='{:.2f} days'.format(dur) )

            if max(mes-np.nanmedian(mes)) > mes_ylim:
                mes_ylim = max(mes)
                
            if min(mes-np.nanmedian(mes)) < mes_ymin:
                mes_ymin = min(mes)

            print('{0:.2f} duration max MES:{1:.2f}'.format(dur, np.max( mes )) )


        axes[0].plot(fold_time, self.lc.flux[mask]-1., 'k.', markersize=0.5,
                    )
        axes[0].set_ylabel('$\mathregular{\delta F/F}$')

        for i in range(1,len(axes)):
            axes[i].set_ylim(mes_ymin, mes_ylim)
            axes[i].axhline(7.1, color='k', ls='--')
            axes[i].axhline(0., color='k', ls='-')
            axes[i].set_ylabel('MES')
            axes[i].legend(loc='upper right', framealpha=1.)

        if plot_binflux:
            t_bin, f_bin = make_binned_flux(t=fold_time, f=self.lc.flux[mask]-1.,
                                            texp=tdur/3.)

            axes[0].plot(t_bin,f_bin, 'o', color='tomato', markersize=2,)
            

        if P>4 and zoom:

            t_range = self.tdurs[np.argmax(max_mes)]
            axes[0].set_xlim(P/2.-3*t_range, P/2.+3*t_range)

        axes[-1].set_xlabel('Phase [days]')

        return axes




    
class LightCurve(object):

    def __init__(self, time, flux, flux_err, ID, flags=None, mask=None, trend=None, mission='KEPLER', segment_dates=None):
        
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.ID = ID

        if type(flags) == None:
            self.flags = np.zeros(len(self.time) )
        else:
            self.flags = flags

        if mask is None:
            self.mask = np.ones(len(self.time), dtype=bool)
        else:
            self.mask = mask

        if type(trend) == None:
            self.trend = np.ones(len(self.time), dtype=float)
        else:
            self.trend = trend
            
        self.mission = mission
        self.exptime = np.nanmedian(time[1:]-time[:-1])

        if mission=='KEPLER':
            self.qflags = [1,2,3,4,6,7,8,9,11,12,13,15,16,17,]
            
        elif mission=='TESS':
            self.qflags = [1,2,3,4,5,6,12]

        self.segment_dates = segment_dates

        
    def mask_bad_gap_edges(self, sig=3.0, edge_cut=1., min_dif=0.2, npoints=10):

        mask = self.mask.copy()
        _, _, edge_mask = flag_gap_edges(self.time[mask], self.flux[mask], min_dif=min_dif, sig=sig,gap_size=edge_cut, npoints=npoints )
        
        self.mask[mask] = edge_mask

    def mask_badguys(self, sig=4., qflags=None):

        if qflags is None:
            qflags = self.qflags
        
        good_guys = self.mask
        
        quality_mask =  np.sum(2 ** np.array(qflags))
        good_guys &= (self.flags & quality_mask) == 0

        good_guys &= ~np.isnan(self.flux)
        
        self._mask_outliers(nsigma=sig)
        
        self.mask &= good_guys

    def flatten(self, wotan_kw):

        self.flux, self.trend = flatten(self.time, self.flux, return_trend=True, **wotan_kw)
        self.flux_err = self.flux_err/self.trend
        
        
    def _mask_outliers(self, window=2., nsigma=5., n_mad_window=51):

        zero_flux = self.flux.copy() - np.median(self.flux[self.mask])
        zero_flux[~self.mask] = 0.

        mad_estimate = np.sqrt(calc_var_stat(zero_flux,window_size=window, method='mad',
                                             n_mad_window=n_mad_window))

        pos_outliers = zero_flux > nsigma * mad_estimate
        neg_outliers = zero_flux < -nsigma * mad_estimate
        bad_flux = zero_flux<-1.

        
        neg_out_ind = np.arange(len(zero_flux))[neg_outliers]
        negout_times = self.time[neg_out_ind]

        for i in neg_out_ind:

            n_close_points = np.sum( np.abs(negout_times-self.time[i])<5*self.exptime )
            
            if n_close_points>1 or np.abs(zero_flux[i]-np.nanmedian(zero_flux[i-4:i+4]))<2*mad_estimate[i]:
                neg_outliers[i] = False


        self.mask &= ~pos_outliers
        self.mask &= ~neg_outliers
        self.mask &= ~bad_flux
        self.mask &= ~np.isnan(zero_flux)



class Injection_Test(object):


    def __init__(self, LightCurve, P_range=[1.,1000.], Rp_range=[-0.5,1.5]):

        self.lc = LightCurve
        self.ts = TransitSearch(LightCurve)




    def inject_transit_signal():


        return 1.



    

def make_transit_mask(time, P, t0, dur):

    fold_time = (time - t0 + P/2.)%P - P/2.
    mask = np.abs(fold_time) > 1.*dur
    
    return mask


def get_limb_darkening_grid(mission, grid_file='data/meta/limb_coeffs.h5'):

    ld_file = TRASH_DATA_DIR + grid_file
    limbs = pd.read_hdf(ld_file, mission)
    
    return limbs


KEPLER_LIMB_DARKENING_GRID = get_limb_darkening_grid('KEPLER')
TESS_LIMB_DARKENING_GRID = get_limb_darkening_grid('TESS')


def get_whitening_coeffs(t, x, window_size, seg_dates, exp_time=0.020417, method='mad',
                  slat_frac=0.2, n_mad_window=51):


    for i,seg in seg_dates:

        if i<len(seg_dates)-1:
            x_seg = x[np.logical_and(t>=seg and t<seg_dates[i+1])]
        else:
            x_seg = x[np.logical_and(t>=seg and t<seg_dates[i+1])]

        x_seg_pad = pad_time_series()

    return 1.




def calc_var_stat(x, window_size, exp_time=0.020417, method='mad',
                  slat_frac=0.2, n_mad_window=51):

    if method=='mean':
        window_points = 2*int(window_size/exp_time)
        sig2 = moving_average( x**2., n=window_points+1)

        return sig2
    
    elif method=='mad':
        
        window_points = 2*int(window_size/exp_time)

        if window_points<n_mad_window:
            sig = 1.4826*median_filter( np.abs(x), window_points+1 , mode='reflect')
        else:
            nskip = int(np.floor(window_points/n_mad_window))
            x_skipped = x[::nskip] 
            sig_decimated = 1.4826*median_filter( np.abs(x_skipped), n_mad_window , mode='reflect')
            sig = np.repeat(sig_decimated, nskip)[:len(x)]

        #if any(np.isnan(sig)):
        #    mask = np.isnan(sig)
        #    sig[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), sig[~mask])
        return sig**2.

    elif method=='gap_mean':
        window_points = 2*int(window_size/exp_time)

        slat_points = int(window_points/2)
        delta_slat = int((slat_frac)*window_points/2)

        delta_slat = int(delta_slat/2)*2
        slat_points = int(slat_points/2)*2
        
        sig2_left = moving_average( np.roll(x**2., -delta_slat), n=slat_points-1)
        sig2_right = moving_average(np.roll(x**2., delta_slat), n=slat_points-1)
        
        return 0.5*(sig2_left+sig2_right)






def calculate_ses_by_segment(time, flux, seg_dates, var_calc='mad'):


    '''
    UNDER CONSTRUCTION
    '''

    for i,segdate in enumerate(seg_dates):

        if i < len(seg_dates)-1:
            segmask = np.logical_and( time >= segdate, time<seg_dates[i+1] )
        else:
            segmask = time >= segdate


            segtime = time[segmask]
            segflux = flux[segmask]



            gap_time, gap_flux, gap_array =  pad_time_series(segtime, segflux,
                                    pad_end=False, fill_gaps=True )
        
            pad_time, pad_flux, pad_array =  pad_time_series(gap_time, gap_flux,
                                    pad_end=True, fill_gaps=False)


    
            seg_FluxTransform = ocwt(pad_flux-1.)
            seg_FluxTransform = seg_FluxTransform[:, pad_array]

            


    return 1.

    


def calculate_SES(flux_transform, signal_transform, window_size, var_calc='mad', texp=0.0204, seg_dates=None):

    # Calculate the Doubly-Whitened Coefficients
    sig2 = [1./calc_var_stat(x, window_size, method=var_calc, exp_time=texp) for i,x in enumerate(flux_transform)]

    n_levels = len(flux_transform)
    sig2 = np.array(sig2, )

    # Trim SignalTransform to speed up computation
    midpoint = signal_transform[0].size // 2
    signal_transform_trimmed = signal_transform[:,midpoint-250:midpoint+251 ]
        
    N = np.zeros(len(sig2[0]) )
    D = np.zeros(len(sig2[0]) )

    levels = np.concatenate([np.arange(1,n_levels), [n_levels-1] ] )
    
    for i, l in enumerate(levels):

        # Use direct convolution to avoid rounding errors from high dynamic range
        dN =  2.** (-l) * convolve(sig2[i]*flux_transform[i], signal_transform_trimmed[i,::-1], mode='same',method='direct')
        N +=dN

        dD =   2.** (-l) * convolve(sig2[i], signal_transform_trimmed[i,::-1]**2., mode='same',method='direct')
        D +=  dD 
    
    num = np.sum(N, axis=0)
    den = np.sum(D, axis=0)
    
    ses = N / np.sqrt(D)
    
    return ses, N, D, sig2[0]




def get_optimal_period_sampling(time, A = 5.217e-5, OS=1., rho=1., ntransits=3.):

    '''
    Calculates the optimal period spacing to search for transit signals (from Ofir+2014)
    time: time of each observation
    A: Constant related to M, R of star
    OS: Oversampling factor; recommended to be 2-5
    '''

    A = A * rho**(-1./3.) / OS
    
    f_max = 2*np.pi*1.661  #Set by Roche lobe of Sun-like star, can change this later
    f_min = ntransits*2.*np.pi/(max(time) - min(time)) # Minimum frequency calculated from observing time
    N_samp = np.ceil( ( f_max**(1./3.) - f_min**(1./3.) + A/3.) * (3./A) ) 
        
    freq_sampling = (A*np.arange(1., N_samp, 1)/3. + f_min**(1./3.)  - A/3.)**3.

    # convert frequency to period
    period_sampling = 2.*np.pi/freq_sampling[::-1]
    
    return period_sampling



def kep_period_sampling(time, P_min=None, n_tr=3., rho_star=1., d_leastsq=0.125):

    if P_min is None:
        P_min = 0.5 / np.sqrt(rho_star)
    
    Pt = P_min
    periods = [P_min]

    baseline = max(time) - min(time)
    
    while Pt < baseline/n_tr:
        
        d_dur = d_leastsq * (13./24.) * (Pt/365.)**(1./3.) * rho_star**(-1./3.)
        Pt += 4. * d_dur * (Pt/baseline)
        periods.append(Pt)
    
    return np.array(periods)





'''
def Detrend_and_Calculate_SES_to_df(time, flux, limb_dark, calc_depth=True, 
                                    tdurs=np.array([1.,1.5,2,2.5,3,3.5,4.5,5,6,7,8,9,10,11,12,13.,14.,15.,16.5,18.,19.5,21,23.,25.])/24.,  method='biweight', w=5.,var_calc='gap_mean', var_calc_window=10., detrend_model=False, texp=0.0204):

    
    ses_models = []
    num_models = []
    den_models = []

    detrend_masks = []
    detrend_flux = []

    ses_keys = ['ses_{0:.2f}d'.format(dur) for dur in tdurs]
    num_keys = ['num_{0:.2f}d'.format(dur) for dur in tdurs]
    den_keys = ['den_{0:.2f}d'.format(dur) for dur in tdurs]
    
    detrend_mask_keys = ['detrend_mask_{0:.2f}d'.format(dur) for dur in tdurs]
    detrend_flux_keys = ['detrend_flux_{0:.2f}d'.format(dur) for dur in tdurs]
    
    tdur_sigma = []


    for dur in tdurs:

        window_length = max(w*dur, 0.75)

        flux_detrended, flux_trend = flatten(time, flux, window_length=window_length, method=method, return_trend=True, edge_cutoff=window_length/2., break_tolerance=0.5*window_length)

        _, _, sig_clip = sigma_clip(time, flux_detrended, use_mad=False)

        flux_detrended[~sig_clip]=np.nan
        
        detrend_flux.append(flux_detrended )
        
        mask =  ~np.isnan(flux_detrended)

        detrend_masks.append(mask)
        pad_time, pad_flux, pad_truth = pad_time_series(x=time[mask], y=flux_detrended[mask],pad_end=True )

        
        flux_transform = ocwt(pad_flux-1.)

        if calc_depth:
            depth = mad(pad_flux)*1e6
        else:
            depth = calc_depth

        if detrend_model:
            mod_f = get_transit_signal(dur, depth, limb_dark=limb_dark, )
            mod_t = np.arange(0,len(mod_f))*texp
            model = flatten(mod_t, mod_f, window_length=w*dur, method=method)-1.
        else:
            model =  get_transit_signal(dur, depth, limb_dark=limb_dark, ) - 1.
            
        model_transform = ocwt( model )
                
        ses,num,den = calculate_SES(flux_transform, model_transform, window_size=var_calc_window*dur, var_calc=var_calc)
        
        ses = ses[pad_truth]
        num = num[pad_truth]
        den = den[pad_truth]

        tdur_sigma.append(np.median(den) )

        ses_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        ses_zero_padded[mask] = ses

        num_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        num_zero_padded[mask] = num

        den_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        den_zero_padded[mask] = den
        
        ses_models.append( ses_zero_padded )
        num_models.append( num_zero_padded )
        den_models.append( den_zero_padded )

    keys = np.concatenate([detrend_flux_keys,detrend_mask_keys,ses_keys,num_keys,den_keys])
    data = np.concatenate([detrend_flux,detrend_masks,ses_models,num_models,den_models])

    keys = np.append(['time', 'pdcsap'], keys )
    data = np.concatenate([[time, flux], data ])
                     

    #print(np.shape(data), len(data) )
    
    df = pd.DataFrame( data=data.T, columns=keys,  )

    #lc.tdur_sigma = tdur_sigma
    
    return df
'''


def Detrend_and_Calculate_SES_for_all_Tdur_models(lc, tdurs=np.array([1.,1.5,2,2.5,3,3.5,4.5,5,6,7,8,9,10,11,12,13.,14.,15.,16.5,18.,19.5,21,23.,25.])/24., method='biweight', w=5.,var_calc='slat_mean',  var_calc_window=10.):
    
    lc.tdurs = tdurs
    
    ses_models = []
    num_models = []
    den_models = []

    detrend_masks = []
    detrend_flux = []

    ses_keys = ['ses_{0:.2f}d'.format(dur) for dur in tdurs]
    num_keys = ['num_{0:.2f}d'.format(dur) for dur in tdurs]
    den_keys = ['den_{0:.2f}d'.format(dur) for dur in tdurs]
    
    detrend_mask_keys = ['detrend_mask_{0:.2f}d'.format(dur) for dur in tdurs]
    detrend_flux_keys = ['detrend_flux_{0:.2f}d'.format(dur) for dur in tdurs]
    tdur_sigma = []


    limb_dark = np.concatenate([lc.stlr['limb1'],lc.stlr['limb2'],lc.stlr['limb3'],lc.stlr['limb4']])
    
    for dur in lc.tdurs:

        window_length = max(w*dur, 0.75)

        flux_detrended, flux_detrend = flatten(lc.time, lc.pdcsap, window_length=window_length, method=method, return_trend=True, edge_cutoff=window_length/2., break_tolerance=0.5*window_length)
        
        _, _, sig_clip = sigma_clip(lc.time, flux_detrended, use_mad=False)
        flux_detrended[~sig_clip]=np.nan

        detrend_flux.append(flux_detrended )
        
        mask = ~np.isnan(flux_detrended)
        detrend_masks.append(mask)
        
        pad_time, pad_flux, pad_truth = pad_time_series(x=lc.time[mask], y=flux_detrended[mask])
        
        flux_transform = ocwt(pad_flux-1.)
        depth = 250.
        model_transform = ocwt( get_transit_signal(dur, depth, limb_dark=limb_dark )-1. )
                
        ses,num,den = calculate_SES(flux_transform, model_transform, window_size=var_calc_window*dur, var_calc=var_calc)
        
        ses = ses[pad_truth]
        num = num[pad_truth]
        den = den[pad_truth]

        tdur_sigma.append (np.median(den) )

        ses_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        ses_zero_padded[mask] = ses

        num_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        num_zero_padded[mask] = num

        den_zero_padded = np.zeros(len(mask), dtype=np.float32 )
        den_zero_padded[mask] = den
        
        ses_models.append( ses_zero_padded )
        num_models.append( num_zero_padded )
        den_models.append( den_zero_padded )

    lc.ses = ses_models
    lc.num = num_models
    lc.den = den_models
    
    lc.detrend_masks = detrend_masks
    lc.detrend_flux = detrend_flux

    lc.ses_keys = ses_keys
    lc.num_keys = num_keys
    lc.den_keys = den_keys

    lc.detrend_mask_keys = detrend_mask_keys
    lc.detrend_flux_keys = detrend_flux_keys
    lc.tdur_sigma = tdur_sigma
    
    return lc




def FFA_Search_Duration_Downsample(time, num, den, dur, exptime, super_sample=3.):    

    norm_factor = dur/(super_sample * exptime)
    time_bins = np.arange(min(time), max(time)+dur/super_sample, dur/super_sample)
    
    num_hist,_ = np.histogram(time, bins = time_bins, weights = num/norm_factor)
    den_hist, time_bins = np.histogram(time, bins = time_bins, weights = den/norm_factor)
    
    mid_time = (time_bins[:-1] + time_bins[1:])/2.

    return mid_time, num_hist, den_hist





def FFA_TCE_Search(time, num, den, cadence, dur, P0_range, kicid=99, progbar=True,
                   fill_gaps=False, threshold=7.1, check_vetoes=False, single_frac=0.7,
                   print_updates=True,return_df=True):

    
    time_cad = (time / cadence).astype(int)
    t0 = time[0]

    if fill_gaps:

        _,num_padded,_ = pad_time_series(x=time_cad, y=num, min_dif=1.01, cadence=1,
                                           in_mode='constant', 
                                            pad_end=False, fill_gaps=True, constant_val=0.)
    
        _,den_padded,_ = pad_time_series(x=time_cad, y=den, min_dif=1.01, cadence=1,
                                           in_mode='constant', 
                                            pad_end=False, fill_gaps=True, constant_val=0.)
    
    else:
        num_padded = num
        den_padded = den
    
    ses = num / np.sqrt(den)
    
    numlength = len(num_padded)
    max_periods = []
    max_mes = []

    get_Npad= lambda XX: int( XX * 2**np.ceil(np.log2(numlength/XX) )  - numlength )

    if progbar:
        Pcad0 = np.arange(P0_range[0]//cadence, np.ceil(P0_range[1]/cadence), dtype=int)
        period_iterable = tqdm(Pcad0)
    else:
        period_iterable = np.arange(P0_range[0]//cadence, np.ceil(P0_range[1]/cadence), dtype=int)
    
    
    detections = {'star_id':[], 'period':[], 'mes':[],'t0_bkjd':[],'tdur':[]}

    
    
    for P0 in period_iterable:

        N_pad = get_Npad(P0)

        num_endpad = np.pad(num_padded, (0,N_pad), mode='constant').reshape((-1, P0) )
        den_endpad = np.pad(den_padded, (0,N_pad), mode='constant').reshape((-1, P0) )

        mes =  FFA(num_endpad) / np.sqrt( FFA( den_endpad) ) 

        if np.amax(mes)>threshold:
            
            dt0 = np.arange(mes.shape[1]) * cadence + cadence/2.
            
            split_indices = np.arange(0,numlength+N_pad,P0)
            Pt = (P0 + (split_indices/P0) / (len(split_indices)  ) ) * cadence
            
            max_args = np.unravel_index(np.argmax(mes,) , dims=mes.shape)
            max_mes = mes[max_args[0],max_args[1]]
                        
            best_period = Pt[max_args[0]]
            best_dt0 = dt0[max_args[1]]
            detected_mes = mes[max_args[0],:]
                        
            # Check TCE Vetoes
            if check_vetoes:
                
                one_tce_kwargs = {'time_bins':detected_time_bins, 'mes':detected_mes,
                                  'time_bin_t0':best_dt0,
                                  'dur':detected_duration, 'n_dur':5.}
                single_event_kwargs = {'fold_time':time%best_period, 'max_mes':max_mes,
                                       'ses':ses, 'peak_loc':best_dt0+time[0],
                                       'dur':dur, 'P':best_period,'frac':single_frac}
                gap_edge_kwargs = {'time':time, 't0':detected_peak_loc+t0,
                                   'P':P, 't_dur':dur}
                not_vetoed = ts.check_tce_vetoes(one_tce_kwargs,single_event_kwargs,
                                              gap_edge_kwargs)
                
            else:

                single_event_kwargs = {'fold_time':time%best_period, 'max_mes':max_mes,
                                       'ses':ses, 'peak_loc':best_dt0+time[0],
                                       'dur':dur, 'P':best_period, 'frac':single_frac}
                
                not_vetoed = not(check_tce_caused_by_single_event(**single_event_kwargs))
            
            
            if not_vetoed:

                if print_updates:
                    print('\rKIC{4:09d}: TCE at P={0:.7f} days,    MES={1:.2f},    Tdur={2:.3f} days,    t0_bkjd={3:.2f}'.format(best_period,max_mes,dur,best_dt0+t0, kicid))

                    #plot_detection(mes, dt0, Pt)
                    #plt.show()
                    
                detections['star_id'].append(kicid)
                detections['period'].append(best_period)
                detections['mes'].append(max_mes)
                detections['t0_bkjd'].append(best_dt0+t0)
                detections['tdur'].append(dur)
            
                           
    df = pd.DataFrame( detections )
    # save final output
    if len(df)>0:
        
        cleaned_df = clean_tces_of_harmonics(df)

        if print_updates:
            print(cleaned_df)

        if return_df:
            return cleaned_df
        return cleaned_df.to_numpy()
    else:
        print('\nNo Reliable TCEs found in KIC {0}'.format(kicid))
        if return_df:
            return df
        return np.array([[kicid, np.nan, np.nan, np.nan, np.nan]])





'''
def Search_for_TCEs_from_h5_file(lc_file_object, tcefile='results/tce_detection_test.h5', print_updates=True, dur_range=(.5, 2.)):

    data, metadata = lc_file_object
    
    time = data['time'].values.astype(np.float32)

    data_cols = data.columns

    num_keys = sorted([i for i in data_cols if i.startswith('num')])
    den_keys = sorted([i for i in data_cols if i.startswith('den')])
    mask_keys = sorted([i for i in data_cols if i.startswith('detrend_mask')])

    all_masks = data[mask_keys].values.T.astype('bool')
    all_num = data[num_keys].values.T.astype(np.float32)
    all_den = data[den_keys].values.T.astype(np.float32)

    rho_star = float(metadata['stlr']['b18_density'][0] )
    
    psamp = get_optimal_period_sampling(time, ntransits=3., OS=1., rho=rho_star)

    TCEs = Search_for_TCEs_in_all_Tdur_models(time=time, num=all_num, den=all_den, period_sampling=psamp, kicid=metadata['kic'], t_durs=metadata['tdurs'], ses_masks=all_masks, rho_star=rho_star, dur_range=dur_range, outputfile=tcefile, print_updates=print_updates)
    
    return TCEs
'''




def calc_mes(fold_time, num, den, P, n_trans=2., texp=0.0204, norm=False):

    nbins = int(P//texp)+1
    bin_range = (0,P)
    bin_edge = np.linspace(bin_range[0], bin_range[1], nbins)

    num_binned = histogram1d(fold_time, weights=num, bins=nbins, range=bin_range)
    den_binned = histogram1d(fold_time, weights=den, bins=nbins, range=bin_range)
    n_transits = histogram1d(fold_time, bins=nbins, range=bin_range)
    
    transit_cut = n_transits>=n_trans
    
    mes = num_binned / np.sqrt(den_binned)

    if norm:
        mes -= np.nanmedian(mes)
        mes/=mad(mes[~np.isnan(mes)])
    
    mes[~transit_cut] = 0.
    
    return bin_edge, mes



def calc_mes_fast(timefold_bins, num, den):

    numbinned = np.bincount(timefold_bins, weights=num) 
    denbinned = np.bincount(timefold_bins, weights=den) 
    
    mes = numbinned/np.sqrt(denbinned)

    return mes



def calc_mes_loop(timefold_bins, num, den):
        
    mes_all = [calc_mes_fast(timefold_bins, num[i], den[i]) for i in range(num.shape[0])]
    return np.array(mes_all)










def Search_for_TCEs_in_all_Tdur_models_2(time,num,den,ses,period_sampling,kicid,t_durs,rho_star=1.,dur_range=(.25, 1.5), outputfile='tce_detection_test.h5', print_updates=True, texp=0.0204, return_df=False, progbar=True, num_transits=2, check_vetoes=False, single_frac=0.9, norm_mes=False, threshold=7.1):



    '''
    Time: Time series data
    ses: the single event statistic from the previous step
    period_sampling: given from above
    transit duration: the width of the transit model used
    kicid: the KIC of the star used to identify it in files
    '''

    t0 = np.min(time)
    t0_time = time-t0
    
    detections = {'star_id':[], 'period':[], 'mes':[],'t0_bkjd':[],'tdur':[]}

    buffer_count=0
    
    min_dur_buff = (13./24.) * (1./365.)**(1./3.) * rho_star**(-1./3.) * dur_range[0]
    max_dur_buff = (13./24.) * (1./365.)**(1./3.) * rho_star**(-1./3.) * dur_range[1]

    
    #calculate the MES for each period in period sampling
    if progbar:
        iterable =  tqdm(period_sampling[::-1])
    else:
        iterable = period_sampling[::-1]
        
    for P in iterable:
        
        mes_time = np.arange(0, P+texp, texp)
        fold_time = np.mod(t0_time, P)
        
        foldtime_bins = np.digitize(fold_time, bins=mes_time )
        number_transits = np.bincount(foldtime_bins)
        
        
        min_dur = P**(1./3.) * min_dur_buff
        max_dur = P**(1./3.) * max_dur_buff


        tdur_mask = t_durs <= max_dur
        tdur_mask &= t_durs >= min_dur
        tdur_mask &= t_durs < P/7.

        #print(tdur_mask)

        num_dur = num[tdur_mask]
        den_dur = den[tdur_mask]
        
        mes = calc_mes_loop(foldtime_bins, num_dur, den_dur)

        #print(np.argmax(  ) ) 

        if np.nanmax(mes) > threshold:

            all_maxmes = np.nanmax(mes, axis=1)
            dur_i = np.nanargmax(all_maxmes)
            mes_i = np.nanargmax(mes[dur_i])

            mes[np.isnan(mes)]  = 0.
            max_mes = all_maxmes[dur_i]

            detected_duration = t_durs[dur_i]
            detected_mes = mes[dur_i]
            detected_time_bins = mes_time
            detected_peak_loc = mes_time[mes_i]
            detected_ses = ses[tdur_mask][dur_i]
        
        
        
        #for i,dur in enumerate(t_durs):
                 
       #     if dur<=max_dur and dur>=min_dur and dur<P/10.:
                
                #bin_edge, mes = calc_mes(fold_time=fold_time,num=num[i], den=den[i],
                #                         norm=norm_mes, texp=texp,P=P, n_trans=num_transits)

                #if P<1000.:
                #    mes-=np.nanmedian(mes)
                #if norm_mes:
                #    mes /= mad(mes)

                #if np.max(mes) > max_mes:
                    
                #    max_mes=np.max(mes)

                #    detected_duration = dur
                #    detected_mes = mes
                #    detected_time_bins = bin_edge
                #    detected_peak_loc = bin_edge[np.argmax(mes)]
                #    detected_ses = ses[i]

    
        #if max_mes>7.1:

            # Check TCE Vetoes
            if check_vetoes:
                one_tce_kwargs = {'time_bins':detected_time_bins, 'mes':detected_mes,
                                  'time_bin_t0':detected_peak_loc,
                                  'dur':detected_duration, 'n_dur':5.}
                single_event_kwargs = {'fold_time':fold_time, 'max_mes':max_mes,
                                       'ses':detected_ses, 'peak_loc':detected_peak_loc,
                                       'dur':detected_duration, 'P':P,'frac':0.8}
                gap_edge_kwargs = {'time':time, 't0':detected_peak_loc+t0,
                                   'P':P, 't_dur':detected_duration}
                not_vetoed = check_tce_vetoes(one_tce_kwargs,single_event_kwargs,
                                              gap_edge_kwargs)
                #mad_mes_ratio_test = check_mad_mes_ratio(detected_mes, P)
                
            else:
                single_event_kwargs = {'fold_time':fold_time, 'max_mes':max_mes,
                                       'ses':detected_ses, 'peak_loc':detected_peak_loc,
                                       'dur':detected_duration, 'P':P, 'frac':single_frac}
                
                not_vetoed = not(check_tce_caused_by_single_event(**single_event_kwargs))
            
            if not_vetoed:

                if print_updates:
                    print('\rKIC{4:09d}: TCE at P={0:.7f} days,    MES={1:.2f},    Tdur={2:.3f} days,    t0_bkjd={3:.2f}'.format(P,max_mes,detected_duration,detected_peak_loc+t0, kicid))

                # write detections to buffer
                buffer_count +=1
                detections['star_id'].append(kicid)
                detections['period'].append(P)
                detections['mes'].append(max_mes)
                detections['t0_bkjd'].append(detected_peak_loc+t0)
                detections['tdur'].append(detected_duration)
                
    
    df = pd.DataFrame( detections )
    # save final output
    if buffer_count>0:
        
        cleaned_df = clean_tces_of_harmonics(df)

        if print_updates:
            print(cleaned_df)

        if return_df:
            return cleaned_df
        return cleaned_df.to_numpy()
    else:
        print('\nNo Reliable TCEs found in KIC {0}'.format(kicid))
        return np.array([[kicid, np.nan, np.nan, np.nan, np.nan]])

    return TCEs





def Search_for_TCEs_in_all_Tdur_models(time,num,den,ses,period_sampling,kicid,t_durs,rho_star=1.,dur_range=(.25, 1.5), outputfile='tce_detection_test.h5', print_updates=True, texp=0.0204, return_df=False, progbar=True, num_transits=2, check_vetoes=False, single_frac=0.9, norm_mes=False,threshold=7.):
    
    '''
    Time: Time series data
    ses: the single event statistic from the previous step
    period_sampling: given from above
    transit duration: the width of the transit model used
    kicid: the KIC of the star used to identify it in files
    '''

    t0 = np.min(time)
    t0_time = time-t0
    
    detections = {'star_id':[], 'period':[], 'mes':[],'t0_bkjd':[],'tdur':[]}

    buffer_count=0
    
    min_dur_buff = (13./24.) * (1./365.)**(1./3.) * rho_star**(-1./3.) * dur_range[0]
    max_dur_buff = (13./24.) * (1./365.)**(1./3.) * rho_star**(-1./3.) * dur_range[1]


    #calculate the MES for each period in period sampling
    if progbar:
        iterable =  tqdm(period_sampling[::-1])
    else:
        iterable = period_sampling[::-1]
    for P in iterable:
        
        max_mes=threshold
        
        fold_time = np.mod(t0_time, P)
        min_dur = P**(1./3.) * min_dur_buff
        max_dur = P**(1./3.) * max_dur_buff
        
        for i,dur in enumerate(t_durs):
                 
            if dur<=max_dur and dur>=min_dur and dur<P/8.:
                
                bin_edge, mes = calc_mes(fold_time=fold_time,num=num[i], den=den[i],
                                         norm=norm_mes, texp=texp,P=P, n_trans=num_transits)

                #if P<1000.:
                #    mes-=np.nanmedian(mes)
                #if norm_mes:
                #    mes /= mad(mes)

                if np.max(mes) > max_mes:
                    
                    max_mes=np.max(mes)

                    detected_duration = dur
                    detected_mes = mes
                    detected_time_bins = bin_edge
                    detected_peak_loc = bin_edge[np.argmax(mes)]
                    detected_ses = ses[i]

        
        if max_mes>threshold:

            # Check TCE Vetoes
            if check_vetoes:
                one_tce_kwargs = {'time_bins':detected_time_bins, 'mes':detected_mes,
                                  'time_bin_t0':detected_peak_loc,
                                  'dur':detected_duration, 'n_dur':5.}
                single_event_kwargs = {'fold_time':fold_time, 'max_mes':max_mes,
                                       'ses':detected_ses, 'peak_loc':detected_peak_loc,
                                       'dur':detected_duration, 'P':P,'frac':0.8}
                gap_edge_kwargs = {'time':time, 't0':detected_peak_loc+t0,
                                   'P':P, 't_dur':detected_duration}
                not_vetoed = check_tce_vetoes(one_tce_kwargs,single_event_kwargs,
                                              gap_edge_kwargs)
                #mad_mes_ratio_test = check_mad_mes_ratio(detected_mes, P)
                
            else:
                single_event_kwargs = {'fold_time':fold_time, 'max_mes':max_mes,
                                       'ses':detected_ses, 'peak_loc':detected_peak_loc,
                                       'dur':detected_duration, 'P':P, 'frac':single_frac}
                
                not_vetoed = not(check_tce_caused_by_single_event(**single_event_kwargs))
            
            if not_vetoed:

                if print_updates:
                    print('\rKIC{4:09d}: TCE at P={0:.7f} days,    MES={1:.2f},    Tdur={2:.3f} days,    t0_bkjd={3:.2f}'.format(P,max_mes,detected_duration,detected_peak_loc+t0, kicid))

                # write detections to buffer
                buffer_count +=1
                detections['star_id'].append(kicid)
                detections['period'].append(P)
                detections['mes'].append(max_mes)
                detections['t0_bkjd'].append(detected_peak_loc+t0)
                detections['tdur'].append(detected_duration)
                
    
    df = pd.DataFrame( detections )
    # save final output
    if buffer_count>0:
        
        cleaned_df = clean_tces_of_harmonics(df)

        if print_updates:
            print(cleaned_df)

        if return_df:
            return cleaned_df
        return cleaned_df.to_numpy()
    else:
        print('\nNo Reliable TCEs found in KIC {0}'.format(kicid))
        return np.array([[kicid, np.nan, np.nan, np.nan, np.nan]])
        



    
def remove_TCE_harmonics(test_TCEs, known_TCEs=None, tolerance=0.0001):


    test_TCEs = test_TCEs[ test_TCEs[:,1]>5.*test_TCEs[:,4] ]
    
    sorted_mes_index = np.argsort(-test_TCEs[:,2])
    
    if known_TCEs is None:
        test_TCEs_sorted = test_TCEs[sorted_mes_index[1:]]
        is_harmonic = np.zeros(len(test_TCEs)-1, dtype=bool)
        known_periods =  [test_TCEs[sorted_mes_index[0],1]]
        known_t0s =  [test_TCEs[sorted_mes_index[0],3]]
        add=True

    else:
        test_TCEs_sorted = test_TCEs[sorted_mes_index]
        is_harmonic = np.zeros(len(test_TCEs),dtype=bool)
        known_periods = known_TCEs[:,1]
        known_t0s = known_TCEs[:,3]
        add=False

    
    tce_periods = test_TCEs_sorted[:,1]
    tce_t0s = test_TCEs_sorted[:,3]
    tce_durs = test_TCEs_sorted[:,4]
    
    for i,p in enumerate( tce_periods ):


        harm_test = np.concatenate([(known_periods/p)%1. , (p/known_periods)%1.])


        if any(harm_test<tolerance) or any((1.-harm_test) < tolerance):
            if any( np.abs(known_t0s - tce_t0s[i]) < 2.*tce_durs[i]):
                is_harmonic[i] = True
            
        else:
            known_periods = np.append(known_periods, p)

    if add:
        test_TCEs_sorted = test_TCEs[(-test_TCEs[:,2]).argsort()]
        is_harmonic = np.append(False, is_harmonic)

        
    test_TCEs_noharm = test_TCEs_sorted[~is_harmonic]
    same_t0 = np.zeros(len(test_TCEs_noharm), dtype=bool)

    
    for i, tce in enumerate(test_TCEs_noharm[1:]):
        
        if any(np.abs(test_TCEs_noharm[:i+1,3]-tce[3])<2*tce[4]):
            
            same_t0[i+1] = True
    
    
    return  test_TCEs_noharm[~same_t0]




    
def clean_tces_of_harmonics(tces,):
                
    periods = tces['period'].values
    t0s =  tces['t0_bkjd'].values
    good_mes = []
    good_mes2 = []

    for i,p in enumerate(periods[::-1]):
        highest_mes = 0.
        same_t0s = np.abs( tces['t0_bkjd'].values - t0s[i] ) < 0.03
        same_periods =  np.abs( (periods - p )/p) < 0.02
                
        highest_mes = np.max( tces['mes'].values[ same_periods ])
        good_mes.append(highest_mes )

    
    cut  =  np.isin(tces['mes'].values, np.unique( good_mes ) )
    
    cleaned_tces ={'star_id':tces['star_id'].values[cut],
                   'period':tces['period'].values[cut], 
                   'mes':tces['mes'].values[cut],
                   't0_bkjd':tces['t0_bkjd'].values[cut],
                   'tdur':tces['tdur'].values[cut],
}
    
    tce_count = 0
    
    super_clean_period = [ ]
        
    for i,p in enumerate(cleaned_tces['period']):
            
        add=True
            
        for n in [1.,3./2., 4./3.,5./3., 7./3., 2., 5./2., 7/2., 6./5.,17./3., 3., 4., 5., 6., 7., 8., 9., 10.,]:
                
            condition1 = np.abs( (cleaned_tces['period']-n*p)/cleaned_tces['period'] )
            condition2 = np.abs( (cleaned_tces['period']-(p/n))/cleaned_tces['period'] )

            is_harm1 = np.logical_and(condition1<0.01, condition1>0.) 
            is_harm2 = np.logical_and(condition2<0.01, condition2>0.)
            
            condition = np.logical_or(is_harm1, is_harm2)
            
            if condition.any() and cleaned_tces['mes'][i] < np.max(cleaned_tces['mes'][condition]):
                                
                add=False
            
        if add:
            super_clean_period.append( cleaned_tces['period'][i] )
                

    cut2 = np.isin( cleaned_tces['period'], np.unique( super_clean_period ) )
    
    final_tces = pd.DataFrame({'star_id':cleaned_tces['star_id'][cut2],
                               'period':cleaned_tces['period'][cut2],
                               'mes':cleaned_tces['mes'][cut2],
                               't0_bkjd':cleaned_tces['t0_bkjd'][cut2],
                               'tdur':cleaned_tces['tdur'][cut2],
    } )
            
    noharm_tces = final_tces.drop_duplicates(keep='last')
    
    return noharm_tces






'''
def mask_TCEs_and_run_transit_search_again(TCEs, lc_file_object):

    
    data, meta = lc_file_object

    time = data['time'].values.astype(np.float32)
    flux = data['pdcsap'].values.astype(np.float32)
    
    tr_mask = np.ones_like(time, dtype=bool)

    for i,row in TCEs.iterrows():

        P, t0, tdur = row['period'], row['t0_bkjd'], row['tdur']*1.5
        tr_mask &= ~transit_mask(time, period=P, duration=tdur, T0=t0, )
        

    limbs = np.concatenate([meta['stlr']['limb1'],meta['stlr']['limb2'],meta['stlr']['limb3'],meta['stlr']['limb4']])


    print('Detrending Light Curves ... ')

    pad_time, pad_flux, pad_mask = pad_time_series(time[tr_mask], flux[tr_mask], )
    
    
    lc_df = Detrend_and_Calculate_SES_to_df(pad_time, pad_flux, limb_dark=limbs, calc_depth=250., tdurs=meta['tdurs'])
    

    data_cols = data.columns

    num_keys = sorted([i for i in data_cols if i.startswith('num')])
    den_keys = sorted([i for i in data_cols if i.startswith('den')])
    mask_keys = sorted([i for i in data_cols if i.startswith('detrend_mask')])

    
    all_masks = lc_df[mask_keys].values.T.astype('bool')
    
    all_num = lc_df[num_keys].values.T.astype(np.float32)
    all_den = lc_df[den_keys].values.T.astype(np.float32)

    rho_star = float(meta['stlr']['b18_density'][0] )
    
    psamp = get_optimal_period_sampling(time, ntransits=3., OS=1., rho=rho_star)

    print('Searching for new TCEs ... ')
    
    new_TCEs = Search_for_TCEs_in_all_Tdur_models(time=time[tr_mask], num=all_num, den=all_den, period_sampling=psamp, kicid=meta['kic'], t_durs=meta['tdurs'], ses_masks=all_masks, rho_star=rho_star, )

    return new_TCEs
'''

    


def check_tce_happens_once(time_bins, mes, time_bin_t0, dur, n_dur=4.5, P_lim=10. ):


    if np.max(time_bins)<P_lim:
        width_tolerance = np.max(time_bins)/4.
    else:
        width_tolerance = min(dur*n_dur, np.max(time_bins)/10.)

    out_of_transit_mask = np.abs(time_bins - time_bin_t0) > width_tolerance

    if dur > np.max(time_bins)/3.:
        return False
    
    if np.max(mes[out_of_transit_mask]) > 7.1:
        return False
    else:
        return True



def choose_highest_mes_not_caused_by_one_event(fold_time, time_bins, mes, ses, dur, P, texp=0.0204, P_lim=50., frac=0.9):

    dominated_by_one_event=True
    max_mes = 10

    while max_mes >= 7.1 and dominated_by_one_event:

        max_mes = np.max(mes)
        peak_loc = time_bins[np.argmax(mes)]

        dominated_by_one_event =  check_tce_caused_by_single_event(fold_time, max_mes, ses, peak_loc, dur, P, texp=texp, P_lim=P_lim, frac=frac)

        if dominated_by_one_event:
            near_event = np.abs(time_bins - peak_loc) < dur
            mes[near_event] = -99.


    return peak_loc, max_mes





def check_tce_caused_by_single_event(fold_time, max_mes, ses, peak_loc, dur, P, texp=0.0204, P_lim=50., frac=0.7):

    fold_time_shift = (fold_time - peak_loc + P/2)%P - P/2.
    
    in_transit = np.abs(fold_time_shift) < dur
    ses_in_transit = ses[in_transit]
    
    if P<P_lim:
        if np.max(ses_in_transit) > max_mes:
            return True
        return False

    if np.max(ses_in_transit)/max_mes>frac:
        return True
    else:
        return False



def check_if_TCE_at_gap_edge(time, t0, P, t_dur, texp=0.0204):

    dt = time[1:]-time[:-1]
    gaps =  dt>0.5*t_dur
    gaps_bool = np.logical_or( np.concatenate([gaps, [True]]), np.concatenate([[True], gaps]))

    in_transit = make_transit_mask(time, P=P, dur=t_dur, t0=t0 )

    transit_points_near_gaps = np.logical_and(gaps_bool, in_transit)

    N_transits_in_gaps = np.sum( transit_points_near_gaps)
    N_transits = np.sum( in_transit )

    fraction_near_gaps =  N_transits_in_gaps/N_transits
    
    if fraction_near_gaps>0.33:
        return True
    else:
        return False


def check_tce_vetoes(one_tce_kwargs, single_event_kwargs, gap_edge_kwargs):

    if check_tce_happens_once(**one_tce_kwargs):
        if not(check_tce_caused_by_single_event(**single_event_kwargs) ):
               if not( check_if_TCE_at_gap_edge(**gap_edge_kwargs) ):
                   return True

    return False



def check_mad_mes_ratio(mes, P, thresh=7.1):

    if P>10.:
        return np.max(mes)/mad(mes[np.abs(mes)>1e-9]) > thresh
    else:
        return True
    


def plot_mes(lc_file_object, period, t0, mes_ls='-', texp=0.0204, mes_norm=False ):
    
    P=period

    data, metadata = lc_file_object

    time = data['time'].values.astype(float)

    tdur = metadata['tdurs']
    rho_star = metadata['stlr']['b18_density'][0]

    data_cols = data.columns.values.astype(str)
    ses_cols = sorted([i for i in data_cols if i.startswith('ses')])
    num_cols = sorted([i for i in data_cols if i.startswith('num')])
    den_cols = sorted([i for i in data_cols if i.startswith('den')])

    mask_cols = sorted([i for i in data_cols if i.startswith('detrend_mask')])
    flux_cols = sorted([i for i in data_cols if i.startswith('detrend_flux')])
    
    ses = data[ses_cols].values.T.astype(float)
    num = data[num_cols].values.T.astype(float)
    den = data[den_cols].values.T.astype(float)

    mask = data[mask_cols].values.T.astype(bool)
        
    f, axes = plt.subplots(len(tdur), 1, sharex=True, figsize=(10, len(tdur)*1.3), )
    
    fold_time = (time - t0 + P/2.)%P
    max_mes = 7.09
    min_mes = 0.

    i_max = 0
    
    for i in range(1,len(axes)):
        
        detrend_mask = mask[i]
        
        ses_masked = ses[i][detrend_mask]
        time_masked = fold_time[detrend_mask]
        
        num_masked = num[i][detrend_mask]
        den_masked = den[i][detrend_mask]

        
        bin_edge, mes = calc_mes_from_dn(fold_time=time_masked,
                                    num=num_masked, den=den_masked,
                                         P=period, norm=mes_norm, texp=texp)

        if period<10.:

            if period<3.:
                mes-=np.nanpercentile(mes, [25])[0]
            else:
                mes-=np.nanmedian(mes)

        if np.min(mes)<min_mes:
            min_mes = np.min(mes)

        if np.max(mes)>max_mes:
            max_mes = np.max(mes)
            i_max = i

        
        # find peaks in the MES distribution
        detect_width = np.array([.5, 2.])*tdur[i]/texp
        peak_ind, peak_props = find_peaks(mes, width=detect_width, height=5, prominence=5)

        #print(peak_props['widths'])

        # return number and location of peaks
        num_peaks = len(peak_ind)
        peak_loc = bin_edge[np.argmax(mes)]

        max_dur = (13./24.) * (period/365.)**(1./3.) * rho_star**(-1./3.) * 2.0
        min_dur = (13./24.) * (period/365.)**(1./3.) * rho_star**(-1./3.) * 0.5

        if tdur[i] < max_dur and tdur[i] > min_dur:
            axes[i].plot(bin_edge-P/2., mes, ls=mes_ls, label='Tdur={0:.2f} days'.format(tdur[i]), color='dodgerblue',  )

        else:
            axes[i].plot(bin_edge-P/2., mes,  ls=mes_ls, label='Tdur={0:.2f} days'.format(tdur[i]), color='lightslategrey' )
        
        if num_peaks>0:
            axes[i].plot(bin_edge[peak_ind]-P/2., mes[peak_ind], 'ro')
        
        axes[i].set_ylabel('MES[$\sigma_{mad}$]', fontsize=12)
        axes[i].axhline(7.1, ls='--', color='k')    
        axes[i].axhline(0., ls='-', color='k')    
        axes[i].legend(framealpha=0.)


    flux_to_use=i_max
    flux = data[flux_cols].values.T.astype(float)[flux_to_use]
    flux_mask = mask[flux_to_use]
    
    mean_flux = histogram1d(fold_time[flux_mask], weights=flux[flux_mask], bins=750, range=(0,period))/histogram1d(fold_time[flux_mask], bins=750, range=(0,period))
    flux_bins = np.linspace( 0, period, 750  )

        
    axes[0].plot(fold_time-P/2., 1e3*(flux-1.), '.', markersize=0.3, color='dodgerblue')
    axes[0].plot(flux_bins-P/2., 1e3*(mean_flux-1.), '-', color='salmon', label='P = {0:.6f} days'.format(period) )

    axes[0].set_ylim(np.nanmin( 1e3*(flux-1.)), np.nanmax( 1e3*(flux-1.)) )
    axes[0].legend(framealpha=0.)

    axes[0].set_ylabel('Flux [ppt]', fontsize=12)
    axes[-1].set_xlabel('Phase [Days]')
    
    for ax in axes:
        ax.set_xlim(-0.03*P-P/2.,P/2.+0.03*P)
        
    for ax in axes[1:]:
        ax.set_ylim(min_mes, max_mes + 0.1*max_mes)
        
    plt.tight_layout()
    f.subplots_adjust(hspace=0)

    print(max_mes)
            


def run_transit_search(kicid, outputfile='results/tce_detections.h5',  data_dir = 'data', overwrite_lcfile=False):

    h5_ses_file = 'kic'+kicid+'ses_lc.h5'

    if path.exists(data_dir+'/ses_timeseries/'+h5_ses_file ) and not(overwrite_lcfile):
            
        lc_file = loadh5file('data/ses_timeseries/'+h5_ses_file)

        print('Beginning Transit Search ...')
        TCEs = Search_for_TCEs_from_h5_file(lc_file,print_updates=True,tcefile=outputfile)

        df_TCEs = pd.DataFrame(TCEs, columns=['kicid', 'period', 'mes', 't0_bkjd','tdur'])
        cleaned_TCEs =  mask_highest_TCE_and_check_others(df_TCEs, lc_file)
        
        print('Finished Search! See your TCEs above. ')
        return cleaned_TCEs.to_numpy()

    elif path.exists(data_dir+'/lightcurves/'+kicid ):

        print('Detrending Light Curve...')
        lc = kepler_lc( get_lc_files(kicid, directory=data_dir+'/lightcurves')).remove_bad_guys()
        lc_ses = Detrend_and_Calculate_SES_for_all_Tdur_models(lc)
        lc_ses.save_as_h5(fname=data_dir+'/ses_timeseries/'+h5_ses_file )

        lc_file = loadh5file(data_dir+'/ses_timeseries/'+h5_ses_file)

        print('Beginning Transit Search ... ')
        TCEs = Search_for_TCEs_from_h5_file(lc_file,print_updates=True,tcefile=outputfile)

        df_TCEs = pd.DataFrame(TCEs, columns=['kicid', 'period', 'mes', 't0_bkjd','tdur'])
        cleaned_TCEs =  mask_highest_TCE_and_check_others(df_TCEs, lc_file)
        
        print('Finished Search! See your TCEs above. ')
        return cleaned_TCEs.to_numpy()
     
    else:
        print('\n\nERROR: NO AVAILABLE FILE FOR KIC '+kicid+'.\n\n\n')
        return np.array([[kicid, np.nan, np.nan, np.nan, np.nan]])


def run_transit_search_list(kicids,outputfile='results/tce_detections.h5',data_dir='data'):

    TCE_list = [run_transit_search(kic, outputfile=outputfile, data_dir=data_dir) for kic in kicids]
    return pd.concat(TCE_list)



def estimate_transit_depth(fold_time, flux, t0, width,):

    in_transit = np.abs(fold_time-t0) <= 0.25*width
    best_depth = (1.-np.median(flux[in_transit]))*1.0e6
    
    return best_depth





def find_best_params_for_TCE(time, num, den, tdurs, P, texp, t0, harmonics = [1., 1.5, 2., 3.]):
    
    best_period = P
    best_tdur = 0.
    max_mes = 0.
    best_t0 = 0.

    min_time = min(time)
    
    t0_time = time-min_time
    
    for n in harmonics:
        
        mes_time = np.arange(0, n*P+texp, texp)
        fold_time = np.mod(t0_time, n*P)
        
        foldtime_bins = np.digitize(fold_time, bins=mes_time )
        #number_transits = np.bincount(foldtime_bins)
    
        mes = calc_mes_loop(foldtime_bins, num, den)
        
        if np.nanmax(mes)>max_mes:
            
            all_maxmes = np.nanmax(mes, axis=1)
            dur_i = np.nanargmax(all_maxmes)
            mes_i = np.nanargmax(mes[dur_i])
            
            best_tdur = tdurs[dur_i]
            max_mes = all_maxmes[dur_i]

            #mes_dur = mes[dur_i]
            best_t0 = mes_time[mes_i] + texp/2. + min_time
            
            best_period = n*P

    if np.abs(best_t0-t0)<1.:
        return best_period, max_mes, best_t0, best_tdur
    else:
        return  best_period, 0., best_t0, best_tdur




def mask_highest_TCE_and_check_others(TCEs, time, num, den, tdurs, threshold=7.,mes_norm=False, texp=0.0204):
    
    
    tces_sorted = TCEs[TCEs[:,2].argsort()][::-1]
    top_tce = tces_sorted[0]
    
    # Assume highest S/N TCE is real, and mask it
    tr_mask = make_transit_mask(time, P=top_tce[1],t0=top_tce[3],dur=top_tce[4] )
    true_TCE_list = np.ones(len(tces_sorted), dtype=bool)
    
    # Check if TCEs remain after masking the higher S/N TCEs. 
    for i in range(1, len(tces_sorted)):

        _, period, mes_orig, t0, tdur = tces_sorted[i]

        if num.ndim==2:
            tdur_index = np.argmin(np.abs(tdur - tdurs) )
            num_masked = num[tdur_index][tr_mask]
            den_masked = den[tdur_index][tr_mask]
        else:
            num_masked = num[tr_mask]
            den_masked = den[tr_mask]

        fold_time = ((time - t0 + period/2.) % period )
        mes_time, mes = calc_mes(fold_time[tr_mask],num_masked,den_masked,period,norm=mes_norm)
        
        max_mes = np.nanmax(mes)
        num_points, _ = np.histogram(fold_time[tr_mask], bins=np.arange(0,period,texp))
        ntransits = num_points[np.argmax(mes)]
        new_t0 = mes_time[np.nanargmax(mes)]

        t0_diff = np.abs(period/2.-new_t0)

        if max_mes>threshold and max_mes > 0.5*mes_orig and ntransits>3 and t0_diff<tdur:
            tr_mask &= make_transit_mask(time=time, P=period, dur=tdur, t0=t0)
            print('tce at {:.5f}: Modified MES = {:.2f}, t0 diff={:.2f}, REAL?'.format(period, max_mes, t0_diff))
        else:
            print('tce at {:.5f}: Modified MES = {:.2f}, t0 diff={:.2f}, FAKE!'.format(period, max_mes, t0_diff))
            true_TCE_list[i]=False
            
    return tces_sorted[true_TCE_list]

    
        


