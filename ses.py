
from fast_histogram import histogram1d
from .utils import *
from scipy.ndimage import median_filter

from scipy.fftpack import rfft, irfft
import pywt


'''

def make_dec_filters(wavelet_name, levels):
    
    all_dec_lo = []
    all_dec_hi = []
    
    wavelet = pywt.Wavelet(wavelet_name)
    h0, h1, _, _ = wavelet.filter_bank

    #H0 = rfft(h0)
    #H1 = rfft(h1)
    
    
    all_dec_lo = [h0]
    all_dec_hi = [h1]
    
    
    for i in range(1,levels):
                
        h0_old = all_dec_lo[i-1]
        h1_old = all_dec_hi[i-1]


        all_dec_lo.append(np.repeat(h0_old[1::2],2))
        all_dec_hi.append(np.repeat(h1_old[1::2],2))
        
        #all_dec_lo.append(np.concatenate([h0_old[0::2],h0_old[1::2][::-1]]) )
        #all_dec_hi.append(np.concatenate([h1_old[0::2],h1_old[1::2][::-1]]) )
        
    return all_dec_lo, all_dec_hi


#ocwt_filters = make_dec_filters('db12', 16)



def ocwt_WRONG(arr, dec_filters=ocwt_filters,norm=True):
    
    all_dec_lo, all_dec_hi = dec_filters
    levels = min(int(np.log2(len(arr))), len(all_dec_lo))
    cD = []

    a=np.copy(arr)
    
    for i in range(levels-1):
        
        d = convolve(arr, all_dec_hi[i], mode='same', )
        a = convolve(arr, all_dec_lo[i], mode='same', )

        cD = np.append(cD, d)
    
    cD = np.append(cD, a).reshape(levels, -1)

    if norm:
        return cD / np.vstack(np.median(np.abs(cD),axis=1))
    else:
        return cD


'''


    


def ocwt(x, max_level=14):
    level = min(max_level, np.log2(len(x)) )
    return np.array(pywt.swt(x, 'haar', trim_approx=True, norm=False, level=level) )[::-1]


    

def get_transit_signal(width, depth, limb_dark=[0.4012, 0.5318, -0.2411,0.0194], exp_time=0.020417, pad=65536, b=0.):

    if len(limb_dark) != 4:
        limb_dark = [0.4012, 0.5318, -0.2411, 0.0194]
    
    params = batman.TransitParams()
    params.a = 20.                        
    params.t0 = 0.                        
    params.per = np.pi*(width)/np.arcsin(1./params.a)
    params.inc = np.arccos(b/params.a)*(180./np.pi)                      
    params.ecc = 0.                       
    params.w = 90.                        
    params.limb_dark = "nonlinear"       
    params.u = limb_dark 
    
    n_points = 3.*width/exp_time
    n_pad = int(pad - 2*n_points)
    
    #t = np.linspace(-3.*width , 3.*width, int(2*np.ceil(n_points)) )

    t = np.concatenate([(-np.arange(exp_time, 3*width, exp_time))[::-1], np.arange(0., 3*width, exp_time)])
    
    params.rp = np.sqrt(depth/1.0e6)
    m = batman.TransitModel(params, t, exp_time=exp_time, fac=0.001, supersample_factor=5)
    lc = m.light_curve(params, )

    ends = np.array_split(np.ones( pad - len(lc) ), 2)
    
    return np.concatenate( [ends[0], lc, ends[1] ])




def calc_var_stat(x, window_size, exp_time=0.020417, method='mad',
                  slat_frac=0.2, n_mad_window=251):

    if method=='mean':
        window_points = 2*int(window_size/exp_time)
        sig2 = moving_average( x**2., n=window_points+1)

        return sig2
    
    elif method=='mad':
        
        window_points = 2*int(window_size/exp_time)

        if window_points<n_mad_window:
            sig = 1.4826*median_filter( np.abs(x), window_points+1 , mode='reflect')
            #sig = 1.4826 * running_median_insort(np.abs(x), window_points)
        else:
            nskip = int(np.floor(window_points/n_mad_window))
            x_skipped = x[::nskip] 
            #sig_decimated = 1.4826*running_median_insort( np.abs(x_skipped), n_mad_window , mode='reflect')
            sig_decimated = 1.4826*median_filter( np.abs(x_skipped), n_mad_window, mode='reflect')
            
            sig = np.repeat(sig_decimated, nskip)[:len(x)]
            sig = (np.roll(sig,-1)+np.roll(sig,1)+sig)/3.


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



    
def get_whitening_coeffs(t, x, window_size, seg_dates, exp_time=0.020417, method='mad',
                  slat_frac=0.2, n_mad_window=51):


    for i,seg in seg_dates:

        if i<len(seg_dates)-1:
            x_seg = x[np.logical_and(t>=seg and t<seg_dates[i+1])]
        else:
            x_seg = x[np.logical_and(t>=seg and t<seg_dates[i+1])]

        x_seg_pad = pad_time_series()

    return 1.




def calculate_SES(flux_transform, signal_transform, window_size, var_calc='mad', texp=0.0204, seg_dates=None):

    # Calculate the Doubly-Whitened Coefficients
    sig2 = [1./calc_var_stat(x, window_size*2**i, method=var_calc, exp_time=texp) for i,x in enumerate(flux_transform)]

    n_levels = len(flux_transform)
    sig2 = np.array(sig2, )
        
    N = np.zeros(len(sig2[0]) )
    D = np.zeros(len(sig2[0]) )

    #signal_transform /=  np.vstack(np.max(np.abs(signal_transform),axis=1))
    
    levels = np.concatenate([np.arange(1,n_levels), [n_levels-1] ] )
    
    for i, l in enumerate(levels):
        N +=  2.** (-l) * convolve(sig2[i]*flux_transform[i], signal_transform[i,::-1], mode='same')

        D +=  2.** (-l) * convolve(sig2[i], signal_transform[i,::-1]**2., mode='same')    
    
    ses = N / np.sqrt(D)
    
    return ses, N, D, sig2[0]





def calc_mes(fold_time, num, den, P, n_trans=2, texp=0.0204, norm=False):

    nbins = int(P//texp)+1
    bin_range = (0, P)
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




