#base on output of UWLCM in ICMW24_CC configuration, TODO: how to deal with different outputs...
import xarray as xr

import numpy as np

L_evap = 2264.76e3 # latent heat of evaporation [J/kg]

def calc_all(ds):
    return calc_qc(ds) \
    .pipe(calc_qr) \
    .pipe(calc_qt) \
    .pipe(calc_r_mean) \
    .pipe(calc_r_sigma) \
    .pipe(calc_na) \
    .pipe(calc_nc) \
    .pipe(calc_nr) \
    .pipe(calc_r_m6) \
    .pipe(calc_dv) \
    .pipe(calc_surface_area) \
    .pipe(calc_precip_flux) 

# aerosol mixing ratio [kg/kg]
def calc_ra(ds):
    return ds.assign(rc=lambda x: x.aerosol_rw_mom3 * 4./3. * np.pi * 1e3)
    
# cloud mixing ratio [kg/kg]
def calc_rc(ds):
    return ds.assign(rc=lambda x: x.cloud_rw_mom3 * 4./3. * np.pi * 1e3)
    
# rain mixing ratio [kg/kg]
def calc_rr(ds):
    return ds.assign(rr=lambda x: x.rain_rw_mom3 * 4./3. * np.pi * 1e3)

# aerosol mixing ratio [kg/m3]
def calc_qa(ds):
    return ds.assign(qa=lambda x: x.aerosol_rw_mom3 * 4./3. * np.pi * 1e3 * x.rhod)

# cloud mixing ratio [kg/m3]
def calc_qc(ds):
    return ds.assign(qc=lambda x: x.cloud_rw_mom3 * 4./3. * np.pi * 1e3 * x.rhod)
    
# rain mixing ratio [kg/m3]
def calc_qr(ds):
    return ds.assign(qr=lambda x: x.rain_rw_mom3 * 4./3. * np.pi * 1e3 * x.rhod)
    
# total water mixing ratio [kg/m3]
def calc_qt(ds):
    return ds.assign(qt=lambda x: (x.rain_rw_mom3 + x.aerosol_rw_mom3 + x.cloud_rw_mom3) * 4./3. * np.pi * 1e3 * x.rhod)
    

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
    return ds.assign(na=lambda x: x.aerosol_rw_mom0 *x.rhod)
    
# number concentration of cloud droplets [1/m3]
def calc_nc(ds):
    return ds.assign(nc=lambda x: x.cloud_rw_mom0 *x.rhod)
    
# number concentration of rain drops [1/m3]
def calc_nr(ds):
    return ds.assign(nr=lambda x: x.rain_rw_mom0 *x.rhod)
    
# 6th mooment of droplet radius[m^6/m^3]
def calc_r_m6(ds):
    return ds.assign(r_m6=lambda x: ds.all_rw_mom6 *x.rhod)

def calc_cloud_base(ds, cond):
    return ds.assign(cb_z=lambda x: x.z.where(cond).idxmin(dim='z'))
    
def calc_cloud_top(ds, cond):
    return ds.assign(ct_z=lambda x: x.z.where(cond).idxmax(dim='z'))
    
def calc_surface_area(ds):
    return ds.assign(surf_area=(ds.x1-ds.x0)*(ds.y1-ds.y0)*(ds.z1-ds.z0))

def calc_dv(ds): # TODO: rewrite in a more consistent way
    dx=xr.zeros_like(ds.G)# just to get the shape..
    dx[1:-1,:,:]=ds.di
    dx[0,:,:]=ds.di/2.
    dx[-1,:,:]=ds.di/2.
    
    dy=xr.zeros_like(ds.G)# just to get the shape..
    dy[:,1:-1,:]=ds.dj
    dy[:,0,:]=ds.dj/2.
    dy[:,-1,:]=ds.dj/2.
    
    dz=xr.zeros_like(ds.G)# just to get the shape..
    dz[:,:,1:-1]=ds.dk
    dz[:,:,0]=ds.dk/2.
    dz[:,:,-1]=ds.dk/2.
    
    return ds.assign(dv=dx*dy*dz)

def calc_precip_flux(ds):
    return ds.assign(prflux=lambda x: x.precip_rate * 4./3. * np.pi * 1e3 / x.dv * L_evap)