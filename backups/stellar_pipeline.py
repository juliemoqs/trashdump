import sys
from os import path, getcwd

import numpy as np
import pandas as pd

from isochrones.mist.bc import MISTBolometricCorrectionGrid
from isochrones.mist import MIST_EvolutionTrack, MISTEvolutionTrackGrid

from uncertainties import ufloat

import emcee


class Star(object):


    def __init__(self, obs, ModelObject):

        models = ModelObject.models

        self.obs = obs # A Dictionary of Value, Err to fit to the models. 
        self.obs_keys = list( self.obs.keys() )
        self.mag_keys = [k for k in self.obs_keys if k[-3:]=='mag']
        self.all_models = models
        self.good_models = self._cut_badfit_models()

        self.StarModels = ModelObject

        self.mcmc_posterior = None
        self.physical_posterior = None
        self.posterior_models = None


    def _cut_badfit_models(self, sig=5.):

        model_cut  = np.ones(len(self.all_models), dtype=bool )
        models = self.all_models

        # First cut non-reddened quantities
        if self._in_obs('Teff'):

            teff_max = self.obs['Teff'][0] + sig * self.obs['Teff'][1]
            teff_min = self.obs['Teff'][0] - sig * self.obs['Teff'][1]
            model_cut &= models['Teff'] > teff_min
            model_cut &= models['Teff'] < teff_max

            print('{} Models remaining after Teff Cuts'.format(np.sum(model_cut)))

        
        if self._in_obs('feh'):
            feh_max = self.obs['feh'][0] + sig * self.obs['feh'][1]
            feh_min = self.obs['feh'][0] - sig * self.obs['feh'][1]
            model_cut &= models['feh'] > feh_min
            model_cut &= models['feh'] < feh_max

            print('{} Models remaining after feh cuts'.format(np.sum(model_cut)))

            
        if self._in_obs('logg'):

            logg_max = self.obs['logg'][0] + sig * self.obs['logg'][1]
            logg_min = self.obs['logg'][0] - sig * self.obs['logg'][1]
            model_cut &= models['logg'] > logg_min
            model_cut &= models['logg'] < logg_max

            print('{} Models remaining after logg Cuts'.format(np.sum(model_cut)))


        models = models[model_cut]
        model_cut = np.ones(len(models), dtype=bool)

        
        # Cut on Magnitudes/Distance:
        if self._in_obs('plax'):

            plax_min = self.obs['plax'][0] - sig * self.obs['plax'][1]
            plax_max = self.obs['plax'][0] + sig * self.obs['plax'][1]
            
            d_mod_min = 5. * np.log10(1000./plax_max) - 5.
            d_mod_max = 5. * np.log10(1000./plax_min) - 5.
            
            if  np.isin(['ebv'], self.obs_keys)[0]:
                ebv, ebv_err = self.obs['ebv']
            else:
                ebv, ebv_err = 0., 0.

            for magkey in self.mag_keys:
                
                mag, magerr = self.obs[magkey]

                dered = extinction(magkey, ebv)

                dered_mag = mag - dered
                dered_magerr = (magerr**2. + (dered*ebv_err)**2. )**0.5

                abs_mag_min = dered_mag - sig*dered_magerr - d_mod_min 
                abs_mag_max = dered_mag + sig*dered_magerr - d_mod_max
                
                model_cut &= models[magkey] > abs_mag_min
                model_cut &= models[magkey] < abs_mag_max

        print('{} Models remaining after magnitude and distance cuts'.format(np.sum(model_cut)) )
        return models[model_cut]
        

    def _in_obs(self, k):
        return  np.isin([k], self.obs_keys)[0]

    def _in_mags(self, k):
        return  np.isin([k], self.mag_keys)[0]



    def get_implied_model_parallax(self, magkey='K_mag' ):

        mag, magerr = self.obs[magkey]

        if self._in_obs('ebv'):

            ebv, ebv_err = self.obs['ebv']
            dered = extinction(magkey, ebv)
            dered_mag = mag - dered

        else:
            dered_mag = mag

        model_mags = self.good_models[magkey]
        model_plax = 10. ** (-0.2 * (dered_mag - model_mags + 5.) + 3. )

        return model_plax

        
    def mod_prior(self):

        parallax = self.get_implied_model_parallax()
        
        if self._in_obs('feh'):
            return distance_prior(1000./parallax)
        else:
            feh = self.good_models['feh']
            return distance_prior(1000./parallax) * gauss_prior(feh, 0., 0.2)

        
    def mod_likelihood(self):

        obs=self.obs

        tot_likelh = np.ones(len(self.good_models))
        
        for k in self.obs_keys:

            if self._in_mags(k) and self._in_obs('plax'):
                
                plax, plax_err = obs['plax']
                mag, magerr = obs[k]

                if self._in_obs('ebv'):

                    ebv, ebv_err = obs['ebv']
                    dered = extinction(k, ebv)
                    dered_mag = mag - dered
                    dered_magerr = np.sqrt(magerr**2. + (dered*ebv_err)**2. )

                else:
                    dered_mag = mag
                    dered_magerr = magerr
                
                abs_mag = -5.0 * np.log10(1000. / plax) + dered_mag + 5.0
                abs_mag_err = np.sqrt(
                    (-5.0 / (plax * np.log(10)))**2 * plax_err**2 + dered_magerr**2  )
                
                tot_likelh *= gauss_prior(self.good_models[k], abs_mag, abs_mag_err)
                
            elif k=='ebv' or k=='plax':
                tot_likelh * 1.

            else:
                x, xerr = obs[k]
                tot_likelh *= gauss_prior(self.good_models[k], x, xerr)


        return tot_likelh
        

    def mod_prob(self):

        likelihood = self.mod_likelihood()
        prior = self.mod_prior()

        return prior * likelihood

    
    def return_best_model(self):

        prob = self.mod_prob()
        return self.good_models.iloc[np.argmax(np.array(prob))]

    

    def _evaluate_model(self, params):
        # params has shape mass, eep, feh, parallax, ebv

        if len(np.shape(params))==2:
            return self.StarModels.interpolate(params.T)

        return self.StarModels.interpolate(params)

        
    def mcmc_prior(self, model_param):

        mass, eep, feh, parallax, ebv = model_param

        if ebv<0. or parallax<0.:
            return -np.inf
        
        if self._in_obs('feh'):
            return ln_distance_prior(1000./parallax)
        else:
            return ln_distance_prior(1000./parallax) + ln_gauss_prior(feh, 0., 0.2)
        

    def mcmc_likelihood(self, model_param):
        
        mass, eep, feh, parallax, ebv = model_param

        model = self._evaluate_model(model_param)
        
        ln_lh = 0.

        for k in self.obs_keys:

            if k == 'ebv':
                x, xerr = self.obs[k]
                ln_lh += ln_gauss_prior(ebv, x, xerr )
            elif k== 'plax':
                x, xerr = self.obs[k]
                ln_lh += ln_gauss_prior(parallax, x, xerr )

            else:
                try:
                    x, xerr = self.obs[k]
                    ln_lh += ln_gauss_prior(float(model[k]), x, xerr )

                except KeyError:
                    print(k+' NOT IN MODEL DICTIONARY. ')

        return ln_lh


    def mcmc_lnpost(self, model_param):

        likelihood = self.mcmc_likelihood(model_param)
        prior = self.mcmc_prior(model_param)

        if np.isnan(likelihood):
            return -np.inf 

        return likelihood + prior



    def run_mcmc_fit(self, nwalkers=50, nsteps=300, ndiscard=100, progress=False):

        best_model = self.return_best_model()

        mass, eep, feh = best_model[['mass', 'eep', 'feh']].to_numpy()
        plax, ebv = self.obs['plax'][0], self.obs['ebv'][0]

        init_samples = [mass, eep, feh, plax, ebv]
        ndim=len(init_samples)
        
        init_samples = np.array([s + np.random.normal(0, 1e-5, size=nwalkers) for s in init_samples]).T

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.mcmc_lnpost )
        sampler.run_mcmc(init_samples, nsteps, progress=progress, )

        posterior = sampler.get_chain(discard=ndiscard, flat=True)

        post_df = pd.DataFrame(posterior, columns=['mass', 'eep', 'feh', 'plax','ebv'])

        self.mcmc_posterior = post_df
        self.posterior_models = self._evaluate_model(posterior)
        
        return post_df



    def get_physical_posterior(self, props=['Teff','logg','feh','mass', 'radius','logL','density','age']):

        post =  self.posterior_models[props]

        post.loc[:,'distance'] = 1000./self.mcmc_posterior['plax']
        post.loc[:,'ebv'] = self.mcmc_posterior['ebv']
        
        self.physical_posterior = post

        return post



    def return_physical_medians(self):

        post = self.physical_posterior
        params = dict()
        for k in self.physical_posterior.columns:

            med, lo, hi = np.percentile(post[k], [50,16,84] )

            params[k] = [med]
            params[k+'_e'] = [lo-med]
            params[k+'_E'] = [hi-med]

        return pd.DataFrame(params, )



class MIST_Track_Models(object):

    def __init__(self, models=None):

        if models==None:
            self.models = get_model_grid(1, 1, 1, make_new=False)
        else:
            self.models = models

        self.keys = self.models.columns
        self._interp = MIST_EvolutionTrack()


    def interpolate(self, pars):

        mass, eep, feh, parallax, ebv = pars
        
        return self._interp(mass, eep, feh, 1000./parallax, 3.1*ebv)



def gauss_prior(a, mu, sig):
    return np.exp(-0.5*(a-mu)**2/sig**2)


def distance_prior(d, lscale=1350.):
    return ((d**2/(2.0*lscale**3.)) *np.exp(d/lscale))


def ln_gauss_prior(x, mu, sig):
    return -0.5 * (x-mu)**2. / sig**2.

def ln_distance_prior(d, lscale=1350.):
    return np.log(distance_prior(d, lscale))



def extinction( mag_key, ebv ):

    if mag_key == 'K_mag':
        ebv_corr = ebv * 0.35666104 # From Cardelli

    else:
        print('NO EXTINCTION COEFFICIENT FOR '+mag_key+'. ASSUMING NO EXTINCTION.')
        return 0.

    return ebv_corr

    

            

def get_radius_uncertainty(teff, Mk, BC):
    
    Mbol = Mk + BC
    L_sun = 10.**(-0.4 * 4.74)
    Lbol = 10.**(-0.4 * Mbol) / L_sun
    radius = Lbol**(0.5) * (5778./teff)**2.
    
    return radius.n, radius.s


def get_model_grid(masses, eep, feh,  make_new=False,  key='mist', 
                   fname='data/meta/stellar_pipeline_models.h5'):
    
    file_exists = path.exists(fname)
    
    if make_new or not(file_exists):
        
        mass_grid, eep_grid, feh_grid = np.meshgrid(masses, eep, feh)

        all_mass = mass_grid.reshape((-1))
        all_eep = eep_grid.reshape((-1))
        all_feh = feh_grid.reshape((-1))

        mist_track = MIST_EvolutionTrack()

        model_tracks = mist_track( all_mass, all_eep, all_feh, distance=10., AV=0.).dropna()
        phase_cut = model_tracks['phase']<3.
        phase_cut &= model_tracks['phase']>=0.
        phase_cut &= model_tracks['star_age']<2.0e10
        phase_cut &= model_tracks['star_age']>5.0e7
        model_tracks = model_tracks[phase_cut]
        
        model_tracks.to_hdf(fname, key)
        
        return model_tracks
        
    else:
        
        model_tracks = pd.read_hdf(fname, key)
        
        return model_tracks



def get_stlr_observed(stlr, i):
    
    stlr_row = stlr.iloc[i]
    
    abs_mags = stlr_row[['M_Bp','M_G','M_Rp','M_J','M_H','M_K']].values.astype(np.float64)
    abs_mag_errs = stlr_row[['M_Bp_err','M_G_err','M_Rp_err','M_J_err','M_H_err','M_K_err']].values.astype(np.float64)

    return abs_mags, abs_mag_errs, stlr_row
    
    

def get_bc(pars, bands=['K']):
    
    #teff, logg, feh, av = par
    bc_grid = MISTBolometricCorrectionGrid(['BP', 'RP','G','J', 'H', 'K'])
    
    BC = bc_grid.interp(pars, bands)
    BC_med, BC_err = np.median(BC), np.std(BC)
    
    return BC_med, BC_err




def get_likely_models(mags, mag_errs, model_tracks, N=10, teff_prior=None, feh_prior=None, radius_prior=None, mag_list=['BP_mag','G_mag','RP_mag','J_mag','H_mag','K_mag'], sig=5.):
    
    original_tracks = model_tracks.copy()
    
    if teff_prior == None:
        prior_cut = np.ones( len(model_tracks), dtype=bool )
    else:   
        if teff_prior[0]<4000.:
            teff_prior==None
            prior_cut = np.ones( len(model_tracks), dtype=bool )
        else:
            pho_teff, teff_err = teff_prior
            prior_cut = np.abs(pho_teff - model_tracks['Teff'].values) < sig*teff_err
            
    for i,band in enumerate(mag_list):
        prior_cut &= np.abs(mags[i] - model_tracks[band].values) < sig*mag_errs[i]
    
    model_tracks = model_tracks.iloc[prior_cut]
    
    if len(model_tracks)==0.:
        
        print('NO GOOD MODELS. BAD. BAD. WHATS WRONG WITH THIS ONE?')
        print('Is it bad Errors?:\n  mag_err:{}'.format(mag_errs))
        
        prior_cut2 = np.ones( len(original_tracks), dtype=bool )
        for i,band in enumerate(mag_list):
            prior_cut2 &= np.abs(mags[i] - original_tracks[band].values) < 1.
        
        model_tracks = original_tracks.iloc[prior_cut2]
    
    model_mags = model_tracks[mag_list].values
    n_mods = len(model_mags)
    n_mags = len(mags)
    
    sim_mags = np.reshape(np.tile(mags, N)+np.random.normal(loc=0.,scale=np.tile(mag_errs, N)), (N,n_mags), )
    sim_mags_modelshape = np.tile( sim_mags, n_mods ).reshape(N, n_mods, n_mags)
    
    model_mags_tiled = np.tile( np.concatenate(model_mags), N ).reshape(N, n_mods, n_mags,)
    mag_errs_tiled = np.tile(mag_errs, n_mods*N ).reshape( N, n_mods, n_mags) 
    
    
    n_dim = n_mags
        
    chi2 = np.sum( (model_mags_tiled - sim_mags_modelshape)**2./mag_errs_tiled, axis=2 )

    if feh_prior != None:
        
        n_dim+=1

        sim_feh = feh_prior[0] + np.random.normal(0, feh_prior[1], N )
        sim_feh_modelshape = np.tile( sim_feh, n_mods ).reshape(n_mods, N).T
        
        chi2 += ( (sim_feh_modelshape - model_tracks['feh'].values)**2./feh_prior[1] )
        
    if teff_prior != None and teff_prior[0]>=4000.:
        
        n_dim+=1    
        
        sim_teff = pho_teff + np.random.normal(0, teff_err, N )
        sim_teff_modelshape = np.tile( sim_teff, n_mods ).reshape(n_mods, N).T
        
        chi2 += ( (sim_teff_modelshape - model_tracks['Teff'].values)**2./teff_err )
        
        
    if radius_prior != None:
        
        n_dim+=1    
        
        sim_radius = radius_prior[0] + np.random.normal(0, radius_prior[1], N )
        sim_radius_modelshape = np.tile( sim_radius, n_mods ).reshape(n_mods, N).T
        
        chi2 += ( (sim_radius_modelshape - model_tracks['radius'].values)**2./radius_prior[1] )
        
    chi2/=n_dim
    
    best_model_indices = np.argmin(chi2, axis=1)
    best_models = model_tracks.iloc[best_model_indices]
    
    #best_models['chi2'] = chi2[best_model_indices]
    
    return best_models



def stellar_parameter_pipeline(mags, mag_errs, obs, model_tracks, bc_grid=MISTBolometricCorrectionGrid(['K']), num1=250, num2=250, sig=5., use_teff_prior=True):

    pho_teff = (obs['pho_teff'], obs['pho_teff_err'])
    obs_feh = (obs['kic_feh_cal'], 0.3)
    
    good_models = get_likely_models(mags, mag_errs, model_tracks, N=num1, 
                                    teff_prior=pho_teff, feh_prior=obs_feh,
                                    radius_prior=None,sig=sig, 
                                     mag_list=['BP_mag','G_mag','RP_mag','J_mag','H_mag','K_mag'])
    
    bc_par = [good_models['Teff'].values, good_models['logg'].values, good_models['feh'].values, np.zeros(num1) ]
    
    BC = np.mean( bc_grid.interp(bc_par, ['K']) )
    BC_err = np.std( bc_grid.interp(bc_par, ['K']) )
    
    teff_unc = ufloat(pho_teff[0], pho_teff[1])
    mag_unc = ufloat(mags[-1], mag_errs[-1])
    BC_unc = ufloat(BC, BC_err)
    
    radius_prior = get_radius_uncertainty( teff_unc, mag_unc, BC_unc)
    
    better_models = get_likely_models(mags, mag_errs, model_tracks, N=num2, 
                                    teff_prior=pho_teff, feh_prior=obs_feh,
                                      radius_prior=radius_prior, sig=sig ,
                                      mag_list=['BP_mag','G_mag','RP_mag','J_mag','H_mag','K_mag'])
    
    
    return better_models[['feh', 'logg', 'eep', 'Teff', 'radius', 'density',
    'mass', 'logL', 'delta_nu', 'Mbol', 'phase', 'age', 'dt_deep']]



def progbar(i, n, full_progbar=30, loaded_txt='#', loading_txt='.'):
    frac = float(i+1)/float(n)
    filled_progbar = int(frac*full_progbar)
    print('Progress: '+loaded_txt*filled_progbar + loading_txt*int(full_progbar-filled_progbar) + ' {:>7.1%}'.format(frac),
         end='\r') 



def create_uniform_catalog(stlr, kicids, model_tracks, num_to_do=999999, cache_size=100,catalog_file='data/meta/stellar_pipeline_catalog.h5', print_prog=False, save_cat=False, return_whole_catalog=False):
    
    
    fit_columns = ['iso_teff', 'iso_radius', 'iso_mass', 'iso_density','iso_logL', 'iso_age', 'iso_eep']
    model_columns=['Teff', 'radius', 'mass', 'density', 'logL', 'age', 'eep']
    all_columns =['kic', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2', 'iso_radius', 
                    'iso_radius_err1','iso_radius_err2', 'iso_mass', 'iso_mass_err1', 
                    'iso_mass_err2', 'iso_density','iso_density_err1', 'iso_density_err2', 
                    'iso_logL', 'iso_logL_err1', 'iso_logL_err2', 'iso_age','iso_age_err1', 
                    'iso_age_err2', 'iso_eep','iso_eep_err1', 'iso_eep_err2']
    
    file_exists = path.exists(catalog_file)
    
    if file_exists:
        
        catalog = pd.read_hdf(catalog_file, 'mist', )[all_columns]
        kicids = kicids[~np.isin(kicids, catalog['kic'].values, assume_unique=True) ]
            
    else:
        catalog = pd.DataFrame(columns=all_columns)
        
    count=0
    catalog_dict = {key: [] for key in all_columns}
    catalog_dict['kic'] = []
    
    print( 'Catalog Length: {0}\nNumber Left to Complete: {1}\n'  .format(len(catalog), len(stlr)-len(catalog)) )


    while count<=num_to_do and count<len(kicids):
        
        count+=1
        
        kic_index = np.where( stlr['KIC'] == kicids[count-1] )[0]
        mags, mag_errs, obs = get_stlr_observed( stlr, int(kic_index))
        posterior = stellar_parameter_pipeline(mags, mag_errs, obs, model_tracks)

        if not(np.isfinite([mags, mag_errs]).all() ): 
            continue
        
        catalog_dict['kic'].append( stlr['KIC'].values[int(kic_index)] )
        
        for i,col in enumerate(fit_columns):
            
            col_err1 = col+'_err1'
            col_err2 = col+'_err2'
            
            post = posterior[ model_columns[i]]
            
            med,(err1,err2)=np.median(post),np.median(post)-np.percentile(post, [16,84])

            catalog_dict[col].append(med)
            catalog_dict[col_err1].append(err1)
            catalog_dict[col_err2].append(err2)
        
            
        if print_prog:
            progbar(count-1, min(num_to_do+1, len(kicids)))

    
    print('{} Stars added to Uniform Catalog'.format( min(num_to_do, len(kicids) )) )

            
    df_catalog_dict = pd.DataFrame( catalog_dict )

    if save_cat:
        df_catalog_dict.to_hdf(catalog_file, 'mist', index=False)

    if return_whole_catalog:
        new_catalog = pd.concat([catalog, df_catalog_dict], axis=0, ignore_index=True)
        return new_catalog
    else:
        return df_catalog_dict


def run_stellar_pipeline(kic, stlr, model_tracks, catalog_file='data/meta/stellar_pipeline_catalog.h5',
    fit_columns = ['iso_teff', 'iso_radius', 'iso_mass', 'iso_density','iso_logL',
                   'iso_age', 'iso_eep'], 
    model_columns=['Teff', 'radius', 'mass', 'density', 'logL', 'age', 'eep'], 
    all_columns =['kic', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2', 'iso_radius', 
                    'iso_radius_err1','iso_radius_err2', 'iso_mass', 'iso_mass_err1', 
                    'iso_mass_err2', 'iso_density','iso_density_err1', 'iso_density_err2', 
                    'iso_logL', 'iso_logL_err1', 'iso_logL_err2', 'iso_age','iso_age_err1', 
                    'iso_age_err2', 'iso_eep','iso_eep_err1', 'iso_eep_err2'] ):

    
    kic_index = np.where( stlr['KIC'] == kic )[0]
    mags, mag_errs, obs = get_stlr_observed( stlr, int(kic_index))
    
    if np.isfinite([mags, mag_errs]).all(): 

        posterior = stellar_parameter_pipeline(mags, mag_errs, obs, model_tracks)

        post_med = np.mean(posterior[model_columns].values, axis=0)
        post_err1,post_err2 = np.mean(posterior[model_columns].values, axis=0)-np.percentile(posterior[model_columns], [16,84], axis=0)

        return np.concatenate(np.array([post_med, post_err1, post_err2]).T )

    else:
        print('KIC {0}: is not finite\n     mags:{1}'.format(kic, mags) )
        return np.array([np.nan]*21)



if __name__ == "__main__":

    arg1 = int(sys.argv[1])-1

    masses = np.arange(0.3, 8., 0.025)
    eep = np.arange(0, 2000, 2.)
    feh = np.arange(-1.5, 0.8, 0.05)

    model_tracks = get_model_grid(masses, eep, feh, make_new=False)

    stlr = pd.read_hdf('data/meta/FINAL_stlr_q1_q17_table.h5', 'stlr_cut')

    uni_cat=create_uniform_catalog(stlr,stlr['KIC'].values,model_tracks,num_to_do=arg1,cache_size=100, print_prog=True)


