#base on output of UWLCM in ICMW24_CC configuration, TODO: how to deal with different outputs...
import xarray as xr
import numpy as np

# could use formulas from libcloud via python bindings, but this would introduce dependency issues
L_evap = 2264.76e3 # latent heat of evaporation [J/kg]
R_d = 287.052874 # specific gas constant for dry air [J/kg/K]
c_pd = 1005 # specific heat capacity [J/kg/K]

def calc_all(ds):
    return calc_rc(ds) \
    .pipe(calc_temp) \
    .pipe(calc_RH) \
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
    .pipe(calc_lwp) \
    .pipe(calc_rwp)
    

# aerosol mixing ratio [kg/kg]
def calc_ra(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(ra=lambda x: x.aerosol_rw_mom3 * 4./3. * np.pi * 1e3)
    else:
        return ds.assign(ra=np.nan)
    
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
        return np.nan
    elif ds.microphysics == 'double-moment bulk':
        return ds.nc + ds.nr
    
# mean radius [m]
def calc_r_mean(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(r_mean=lambda x: x.all_rw_mom1 / calc_nt(x))
    else:
        return ds.assign(r_mean=np.nan)
    
# std dev of radius [m]
def calc_r_sigma(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(r_sigma=lambda x: np.sqrt(x.all_rw_mom2 / calc_nt(x) - np.power(x.all_rw_mom1 / calc_nt(x), 2)))
    else:
        return ds.assign(r_sigma=np.nan)
    
# number concentration of aerosols [1/kg]
def calc_na(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(na=lambda x: x.aerosol_rw_mom0)
    elif ds.microphysics == 'single-moment bulk':
        return ds.assign(na=np.nan)
    elif ds.microphysics == 'double-moment bulk':
        return ds.assign(na=np.nan)
    
# number concentration of cloud droplets [1/kg]
def calc_nc(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(nc=lambda x: x.cloud_rw_mom0)
    elif ds.microphysics == 'single-moment bulk':
        return ds.assign(nc=np.nan)
    elif ds.microphysics == 'double-moment bulk':
        return ds
    
# number concentration of rain drops [1/kg]
def calc_nr(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(nr=lambda x: x.rain_rw_mom0)
    elif ds.microphysics == 'single-moment bulk':
        return ds.assign(nr=np.nan)
    elif ds.microphysics == 'double-moment bulk':
        return ds
    
# 6th mooment of droplet radius[m^6/m^3]
def calc_r_m6(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(r_m6=lambda x: x.all_rw_mom6 *x.rhod)
    else:
        return ds.assign(r_m6=np.nan)

def calc_cloud_base(ds, cond):
    return ds.assign(cb_z=lambda x: x.z.where(cond).idxmin(dim='z'))
    
def calc_cloud_top(ds, cond):
    return ds.assign(ct_z=lambda x: x.z.where(cond).idxmax(dim='z'))
    
def calc_surface_area(ds):
    return ds.assign(surf_area = (ds.sizes["x"]-1) * ds.di * (ds.sizes["y"]-1) * ds.dj)

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
        prflux = prflux.chunk({'t': '1'})
        prflux *= ds.rhod
        for k in np.arange(ds.sizes["z"]-2, 0, -1):
            prflux[dict(z=k)] = prflux.isel(z=k+1) + prflux.isel(z=k)
        return ds.assign(prflux = prflux * -1 * ds.dk * L_evap)

#liquid water path in columns [kg/m2]
def calc_lwp(ds):
    return ds.assign(lwp = (ds.rl * ds['rhod']).sum(["z"]) * ds.dz)
    
#liquid water path in columns [kg/m2]
def calc_cwp(ds):
    return ds.assign(cwp = (ds.rc * ds['rhod']).sum(["z"]) * ds.dz)
    
#liquid water path in columns [kg/m2]
def calc_rwp(ds):
    return ds.assign(rwp = (ds.rr * ds['rhod']).sum(["z"]) * ds.dz)

# inversion height [m]
def calc_zi(ds, cond):
    return ds.assign(zi=lambda x: x.z.where(cond).idxmin(dim='z'))

# temperature [K]
def calc_temp(ds):
    return ds.assign(temp = ds.th * pow(ds.p_e / 1e5, R_d / c_pd)) # could use formulas from libcloud via python bindings, but this would introduce dependency issues

# relative humidity
def calc_RH(ds):
    if ds.microphysics == "super-droplets":
        return ds # in lgrngn microphysics RH is stored, calculated based on the RH_formula option
    else:
        # TODO: could use libcloud's functions, but for now we hardcode Tetens formulas
        T_C = ds.temp - 273.15
        rv_s_tet = 380 / (ds.p_e * np.exp(-17.2693882 * T_C  / (ds.temp - 35.86)) - 610.9)
        return ds.assign(RH = ds.rv / rv_s_tet)