#base on output of UWLCM in ICMW24_CC configuration, TODO: how to deal with different outputs...

import numpy as np

def calc_all(ds):
    return calc_qc(ds) \
    .pipe(calc_qr) \
    .pipe(calc_r_mean) \
    .pipe(calc_r_sigma) \
    .pipe(calc_na) \
    .pipe(calc_nc) \
    .pipe(calc_nr) \
    .pipe(calc_r_m6)

# cloud mixing ratio [kg/kg]
def calc_rc(ds):
    return ds.assign(rc=lambda x: x.cloud_rw_mom3 * 4./3. * np.pi * 1e3)
    
# rain mixing ratio [kg/kg]
def calc_rr(ds):
    return ds.assign(rr=lambda x: x.rain_rw_mom3 * 4./3. * np.pi * 1e3)
    
# cloud mixing ratio [kg/m3]
def calc_qc(ds):
    return ds.assign(qc=lambda x: x.cloud_rw_mom3 * 4./3. * np.pi * 1e3 * x.rhod)
    
# rain mixing ratio [kg/m3]
def calc_qr(ds):
    return ds.assign(qr=lambda x: x.rain_rw_mom3 * 4./3. * np.pi * 1e3 * x.rhod)
    

# number concentration per mass of air of all hydrometeors [1/kg]
def calc_n_m(ds):
    return ds.cloud_rw_mom0 + ds.aerosol_rw_mom0 + ds.rain_rw_mom0

# mean radius [m]
def calc_r_mean(ds):
    return ds.assign(r_mean=lambda x: x.all_rw_mom1 / calc_n_m(x))
    
# std dev of radius [m]
def calc_r_sigma(ds):
    return ds.assign(r_sigma=lambda x: np.sqrt(x.all_rw_mom2 / calc_n_m(x) - np.power(x.all_rw_mom1 / calc_n_m(x), 2)))
    
# number concentration of aerosols [1/m3]
def calc_na(ds):
    return ds.assign(na=lambda x: ds.aerosol_rw_mom0 *ds.rhod)
    
# number concentration of cloud droplets [1/m3]
def calc_nc(ds):
    return ds.assign(nc=lambda x: ds.cloud_rw_mom0 *ds.rhod)
    
# number concentration of rain drops [1/m3]
def calc_nr(ds):
    return ds.assign(nr=lambda x: ds.rain_rw_mom0 *ds.rhod)
    
# 6th mooment of droplet radius[m^6/m^3]
def calc_r_m6(ds):
    return ds.assign(r_m6=lambda x: ds.all_rw_mom6 *ds.rhod)