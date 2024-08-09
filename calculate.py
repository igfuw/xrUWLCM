#base on output of UWLCM in ICMW24_CC configuration, TODO: how to deal with different outputs...
import xarray as xr

import numpy as np

L_evap = 2264.76e3 # latent heat of evaporation [J/kg]

def calc_all(ds):
    return calc_rc(ds) \
    .pipe(calc_rr) \
    .pipe(calc_rl) \
    .pipe(calc_rt) \
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
    if ds.microphysics == "super-droplets":
        return ds.assign(ra=lambda x: x.aerosol_rw_mom3 * 4./3. * np.pi * 1e3)
    else:
        return ds.assign(ra=np.NaN)
    
# cloud mixing ratio [kg/kg]
def calc_rc(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(rc=lambda x: x.cloud_rw_mom3 * 4./3. * np.pi * 1e3)
    else:
        return ds # with bulk micro, ds already contains rc
    
# rain mixing ratio [kg/kg]
def calc_rr(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(rr=lambda x: x.rain_rw_mom3 * 4./3. * np.pi * 1e3)
    else:
        return ds # with bulk micro, ds already contains rr
    
# liquid water mixing ratio [kg/kg]
def calc_rl(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(rl=lambda x: (x.rain_rw_mom3 + x.aerosol_rw_mom3 + x.cloud_rw_mom3) * 4./3. * np.pi * 1e3)
    else:
        return ds.assign(rl=lambda x: x.rc + x.rr)
        
# total water mixing ratio [kg/kg]
def calc_rt(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(rt=lambda x: x.rv + (x.rain_rw_mom3 + x.aerosol_rw_mom3 + x.cloud_rw_mom3) * 4./3. * np.pi * 1e3)
    else:
        return ds.assign(rt=lambda x: x.rv + x.rc + x.rr)
    
# number concentration per mass of air of all hydrometeors [1/kg]
def calc_nt(ds):
    if ds.microphysics == "super-droplets":
        return ds.cloud_rw_mom0 + ds.aerosol_rw_mom0 + ds.rain_rw_mom0
    elif ds.microphysics == 'single-moment bulk':
        return np.NaN
    elif ds.microphysics == 'double-moment bulk':
        return ds.nc + ds.nr
    
# mean radius [m]
def calc_r_mean(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(r_mean=lambda x: x.all_rw_mom1 / calc_nt(x))
    else:
        return ds.assign(r_mean=np.NaN)
    
# std dev of radius [m]
def calc_r_sigma(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(r_sigma=lambda x: np.sqrt(x.all_rw_mom2 / calc_nt(x) - np.power(x.all_rw_mom1 / calc_nt(x), 2)))
    else:
        return ds.assign(r_sigma=np.NaN)
    
# number concentration of aerosols [1/kg]
def calc_na(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(na=lambda x: x.aerosol_rw_mom0)
    elif ds.microphysics == 'single-moment bulk':
        return ds.assign(na=np.NaN)
    elif ds.microphysics == 'double-moment bulk':
        return ds.assign(na=np.NaN)
    
# number concentration of cloud droplets [1/kg]
def calc_nc(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(nc=lambda x: x.cloud_rw_mom0)
    elif ds.microphysics == 'single-moment bulk':
        return ds.assign(nc=np.NaN)
    elif ds.microphysics == 'double-moment bulk':
        return ds
    
# number concentration of rain drops [1/kg]
def calc_nr(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(nr=lambda x: x.rain_rw_mom0)
    elif ds.microphysics == 'single-moment bulk':
        return ds.assign(nr=np.NaN)
    elif ds.microphysics == 'double-moment bulk':
        return ds
    
# 6th mooment of droplet radius[m^6/m^3]
def calc_r_m6(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(r_m6=lambda x: x.all_rw_mom6 *x.rhod)
    else:
        return ds.assign(r_m6=np.NaN)

def calc_cloud_base(ds, cond):
    return ds.assign(cb_z=lambda x: x.z.where(cond).idxmin(dim='z'))
    
def calc_cloud_top(ds, cond):
    return ds.assign(ct_z=lambda x: x.z.where(cond).idxmax(dim='z'))
    
def calc_surface_area(ds):
    return ds.assign(surf_area = (ds.dims["x"]-1) * ds.di * (ds.dims["y"]-1) * ds.dj)

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
    if ds.microphysics == "super-droplets":
        return ds.assign(prflux=lambda x: x.precip_rate * 4./3. * np.pi * 1e3 / x.dv * L_evap)
    elif ds.microphysics == "single-moment bulk":
        # in bulk micro, precip_rate is the difference between influx and outflux
        prflux = ds.precip_rate.copy()
        prflux[:,:,-1] *= ds.rhod[-1]
        for k in np.arange(ds.dims["z"]-2, 0, -1):
            prflux[k] = prflux[k+1] + prflux[k] * ds.rhod[k]
        return ds.assign(prflux = prflux * -1 * ds.dk * L_evap)
