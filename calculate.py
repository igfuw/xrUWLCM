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
    .pipe(calc_na) \
    .pipe(calc_nc) \
    .pipe(calc_nr) \
    .pipe(calc_rr) \
    .pipe(calc_cloud_r_mean) \
    .pipe(calc_cloud_r_sigma) \
    .pipe(calc_rain_r_mean) \
    .pipe(calc_rain_r_sigma) \
    .pipe(calc_dv) \
    .pipe(calc_surface_area) \
    .pipe(calc_ra) \
    .pipe(calc_rl) \
    .pipe(calc_rt) \
    .pipe(calc_all_r_mean) \
    .pipe(calc_all_r_sigma) \
    .pipe(calc_all_r_m6) \
    .pipe(calc_lwp) \
    .pipe(calc_rwp)

    
    

def calc_ra(ds):
    if ds.microphysics == "super-droplets" and "aerosol_rw_mom3" in ds:
        ra=lambda x: x.aerosol_rw_mom3 * 4./3. * np.pi * 1e3
    else:
        ra=np.nan
    dsn = ds.assign(ra=ra)
    dsn.ra.attrs["units"] = "kg/kg"
    dsn.ra.attrs["long_name"] = "humidified aerosol mixing ratio"
    return dsn

def calc_rc(ds):
    if ds.microphysics == "super-droplets": 
        if "cloud_rw_mom3" in ds:
            rc=lambda x: x.cloud_rw_mom3 * 4./3. * np.pi * 1e3
        else:
            rc=np.nan
    else:
        rc=ds.rc
    dsn = ds.assign(rc=rc)
    dsn.rc.attrs["units"] = "kg/kg"
    dsn.rc.attrs["long_name"] = "cloud water mixing ratio"
    return dsn


def calc_rr(ds):
    if ds.microphysics == "super-droplets":
        if "rain_rw_mom3" in ds:
            rr=lambda x: x.rain_rw_mom3 * 4./3. * np.pi * 1e3
        else:
            rr=np.nan
    else:
        rr=ds.rr
    dsn = ds.assign(rr=rr)
    dsn.rr.attrs["units"] = "kg/kg"
    dsn.rr.attrs["long_name"] = "rain water mixing ratio"
    return dsn
    
    
# liquid water mixing ratio [kg/kg]
def calc_rl(ds):
    if ds.microphysics == "super-droplets":
        if "rain_rw_mom3" in ds and "cloud_rw_mom3" in ds and "aerosol_rw_mom3" in ds:
            rl=lambda x: (x.rain_rw_mom3 + x.aerosol_rw_mom3 + x.cloud_rw_mom3) * 4./3. * np.pi * 1e3
        else:
            rl=np.nan
    else:
        rl=ds.rr + ds.rc
    dsn = ds.assign(rl=rl)
    dsn.rl.attrs["units"] = "kg/kg"
    dsn.rl.attrs["long_name"] = "liquid water mixing ratio"
    return dsn
        
# total water mixing ratio [kg/kg]
def calc_rt(ds):
    if ds.microphysics == "super-droplets":
        if "rain_rw_mom3" in ds and "cloud_rw_mom3" in ds and "aerosol_rw_mom3" in ds:
            rt=lambda x: x.rv + (x.rain_rw_mom3 + x.aerosol_rw_mom3 + x.cloud_rw_mom3) * 4./3. * np.pi * 1e3
        else:
            rt=np.nan
    else:
        rt=ds.rr + ds.rc + ds.rv
    dsn = ds.assign(rt=rt)
    dsn.rt.attrs["units"] = "kg/kg"
    dsn.rt.attrs["long_name"] = "total water mixing ratio"
    return dsn
    
# number concentration per mass of air of all hydrometeors [1/kg]
def calc_nt(ds):
    if ds.microphysics == "super-droplets":
        if "rain_rw_mom0" in ds and "cloud_rw_mom0" in ds and "aerosol_rw_mom0" in ds:
            nt=lambda x: x.rain_rw_mom0 + x.aerosol_rw_mom0 + x.cloud_rw_mom0
        else:
            nt=np.nan    
    elif ds.microphysics == 'single-moment bulk':
        nt=np.nan
    elif ds.microphysics == 'double-moment bulk':
        nt=ds.nc + ds.nr
        
    dsn = ds.assign(nt=nt)
    dsn.nt.attrs["units"] = "1/kg"
    dsn.nt.attrs["long_name"] = "concentration of hydrometeors"
    return dsn
    
# mean radius [m]
def calc_all_r_mean(ds):
    if ds.microphysics == "super-droplets" and "all_rw_mom1" in ds:
        if "nt" not in ds:
            ds = calc_nt(ds)        
        all_r_mean=lambda x: x.all_rw_mom1 / x.nt
    else:
        all_r_mean=np.nan
    dsn = ds.assign(all_r_mean=all_r_mean)
    dsn.all_r_mean.attrs["units"] = "m"
    dsn.all_r_mean.attrs["long_name"] = "mean radius of hydrometeors"
    return dsn
    
# mean radius [m]
def calc_cloud_r_mean(ds):
    if ds.microphysics == "super-droplets" and "cloud_rw_mom1" in ds:
        if "nc" not in ds:
            ds = calc_nc(ds)
        cloud_r_mean=lambda x: x.cloud_rw_mom1 / x.nc
    else:
        cloud_r_mean=np.nan
    dsn = ds.assign(cloud_r_mean=cloud_r_mean)
    dsn.cloud_r_mean.attrs["units"] = "m"
    dsn.cloud_r_mean.attrs["long_name"] = "mean radius of cloud droplets"
    return dsn
    
# mean radius [m]
def calc_rain_r_mean(ds):
    if ds.microphysics == "super-droplets" and "rain_rw_mom1" in ds:
        if "nr" not in ds:
            ds = calc_nr(ds)
        rain_r_mean=lambda x: x.rain_rw_mom1 / x.nr
    else:
        rain_r_mean=np.nan
    dsn = ds.assign(rain_r_mean=rain_r_mean)
    dsn.rain_r_mean.attrs["units"] = "m"
    dsn.rain_r_mean.attrs["long_name"] = "mean radius of rain drops"
    return dsn
        
# std dev of radius [m]
def calc_all_r_sigma(ds):
    if ds.microphysics == "super-droplets" and "all_rw_mom1" in ds and "all_rw_mom2" in ds:
        if "nr" not in ds:
            ds = calc_nr(ds)
        all_r_sigma=lambda x: np.sqrt(x.all_rw_mom2 / calc_nt(x) - np.power(x.all_rw_mom1 / calc_nt(x), 2))
    else:
        all_r_sigma=np.nan
    dsn = ds.assign(all_r_sigma=all_r_sigma)
    dsn.all_r_sigma.attrs["units"] = "m"
    dsn.all_r_sigma.attrs["long_name"] = "standard deviation of radius of hydrometeors"
    return dsn
    
# std dev of radius [m]
def calc_cloud_r_sigma(ds):
    if ds.microphysics == "super-droplets":
        if "nc" not in ds:
            ds = calc_nc(ds)
        cloud_r_sigma=lambda x: np.sqrt(x.cloud_rw_mom2 / x.nc - np.power(x.cloud_rw_mom1 / x.nc, 2))
    else:
        cloud_r_sigma=np.nan
    dsn = ds.assign(cloud_r_sigma=cloud_r_sigma)
    dsn.cloud_r_sigma.attrs["units"] = "m"
    dsn.cloud_r_sigma.attrs["long_name"] = "standard deviation of radius of hydrometeors"
    return dsn
    
# std dev of radius [m]
def calc_rain_r_sigma(ds):
    if ds.microphysics == "super-droplets":
        if "nr" not in ds:
            ds = calc_nr(ds)
        rain_r_sigma=lambda x: np.sqrt(x.rain_rw_mom2 / x.nr - np.power(x.rain_rw_mom1 / x.nr, 2))
    else:
        rain_r_sigma=np.nan
    dsn = ds.assign(rain_r_sigma=rain_r_sigma)
    dsn.rain_r_sigma.attrs["units"] = "m"
    dsn.rain_r_sigma.attrs["long_name"] = "standard deviation of radius of hydrometeors"
    return dsn
    
# number concentration of aerosols [1/kg]
def calc_na(ds):
    if ds.microphysics == "super-droplets" and "aerosl_rw_mom0" in ds:
        na=lambda x: x.aerosol_rw_mom0
    else:
        na=np.nan
    dsn = ds.assign(na=na)
    dsn.na.attrs["units"] = "1/kg"
    dsn.na.attrs["long_name"] = "number concentration of aerosols"
    return dsn
        
# number concentration of cloud droplets [1/kg]
def calc_nc(ds):
    dsn = ds
    if ds.microphysics != 'double-moment bulk':    
        if ds.microphysics == "super-droplets" and "cloud_rw_mom0" in ds:
            nc=lambda x: x.cloud_rw_mom0
        else:
            nc=np.nan
        dsn = ds.assign(nc=nc)
    dsn.nc.attrs["units"] = "1/kg"
    dsn.nc.attrs["long_name"] = "number concentration of cloud droplets"
    return dsn

# number concentration of rain drops [1/kg]
def calc_nr(ds):
    dsn = ds
    if ds.microphysics != 'double-moment bulk':    
        if ds.microphysics == "super-droplets" and "rain_rw_mom0" in ds:
            nr=lambda x: x.rain_rw_mom0
        else:
            nr=np.nan
        dsn = ds.assign(nr=nr)
    dsn.nr.attrs["units"] = "1/kg"
    dsn.nr.attrs["long_name"] = "number concentration of rain drops"
    return dsn    
   
# 6th moment of droplet radius[m^6/m^3]
def calc_all_r_m6(ds):
    if ds.microphysics == "super-droplets" and "all_rw_mom6" in ds:
        all_r_m6=lambda x: x.all_rw_mom6 *x.rhod
    else:
        all_r_m6=np.nan
    dsn = ds.assign(all_r_m6=all_r_m6)
    dsn.nr.attrs["units"] = "m$^6$/m$^3$"
    dsn.nr.attrs["long_name"] = "6th moment of hydrometeor radius per unit volume"
    return dsn    

def calc_cloud_base(ds, cond):
    return ds.assign(cb_z=lambda x: x.z.where(cond).idxmin(dim='z'))
    
def calc_cloud_top(ds, cond):
    return ds.assign(ct_z=lambda x: x.z.where(cond).idxmax(dim='z'))
    
def calc_surface_area(ds):
    if len(ds.G.dims)==3: #3D
        return ds.assign(surf_area = (ds.sizes["x"]-1) * ds.di * (ds.sizes["y"]-1) * ds.dj)
    elif len(ds.G.dims)==2: #2D
        return ds.assign(surf_area = (ds.sizes["x"]-1) * ds.di)

def calc_dv(ds): # TODO: rewrite in a more consistent way
    if len(ds.G.dims)==3: #3D
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
        
    elif len(ds.G.dims)==2: #2D
        dx=xr.zeros_like(ds.G)# just to get the shape..
        dx[1:-1,:]=ds.di
        dx[0,:]=ds.di/2.
        dx[-1,:]=ds.di/2.
    
        dz=xr.zeros_like(ds.G)# just to get the shape..
        dz[:,1:-1]=ds.dj
        dz[:,0]=ds.dj/2.
        dz[:,-1]=ds.dj/2.
    
        return ds.assign(dv=dx*dz)

def calc_precip_flux(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(prflux=lambda x: x.precip_rate * 4./3. * np.pi * 1e3 / x.dv * L_evap)
    elif ds.microphysics == "single-moment bulk":
        # in bulk micro, precip_rate is the difference between influx and outflux
        prflux = ds.precip_rate.copy()
        prflux = prflux.chunk({'t': '1'})
        prflux *= ds.rhod
        prflux = prflux.isel(z=slice(None, None, -1))   # reverse z
        prflux = prflux.cumsum(dim="z")                 # integrate downward
        prflux = prflux.isel(z=slice(None, None, -1))   # reverse z back
        return ds.assign(prflux = prflux * -1 * ds.dk * L_evap)

#liquid water path in columns [kg/m2]
def calc_lwp(ds):
    if "rl" not in ds:
        ds = calc_rl(ds)
    lwp = (ds.rl * ds['rhod']).sum(["z"]) * ds.dz    
    lwp.attrs["units"] = "kg/m$^2$"
    lwp.attrs["long_name"] = "liquid water path"
    return ds.assign(lwp = lwp)
    
#liquid water path in columns [kg/m2]
def calc_cwp(ds):
    if "rc" not in ds:
        ds = calc_rc(ds)
    cwp = (ds.rc * ds['rhod']).sum(["z"]) * ds.dz    
    cwp.attrs["units"] = "kg/m$^2$"
    cwp.attrs["long_name"] = "cloud water path"
    return ds.assign(cwp = cwp)
    
#liquid water path in columns [kg/m2]
def calc_rwp(ds):
    if "rr" not in ds:
        ds = calc_rr(ds)
    rwp = (ds.rr * ds['rhod']).sum(["z"]) * ds.dz    
    rwp.attrs["units"] = "kg/m$^2$"
    rwp.attrs["long_name"] = "rain water path"
    return ds.assign(rwp = rwp)

# inversion height [m]
def calc_zi(ds, cond):
    zi=ds.z.where(cond).idxmin(dim='z')
    zi.attrs["units"] = "m"
    zi.attrs["long_name"] = "inversion height"
    return ds.assign(zi = zi)
#    return ds.assign(zi=lambda x: x.z.where(cond).idxmin(dim='z'))

# temperature [K]
def calc_temp(ds):
    temp = ds.th * pow(ds.p_e / 1e5, R_d / c_pd) # could use formulas from libcloud via python bindings, but this would introduce dependency issues
    temp.attrs["units"] = "K"
    temp.attrs["long_name"] = "temperature"
    return ds.assign(temp = temp)

# relative humidity
def calc_RH(ds):
    if ds.microphysics == "super-droplets":
        ds.RH.attrs["units"] = "1"
        ds.RH.attrs["long name"] = "relative humidity"
        return ds # in lgrngn microphysics RH is stored, calculated based on the RH_formula option
    else:
        # TODO: could use libcloud's functions, but for now we hardcode Tetens formulas
        T_C = ds.temp - 273.15
        rv_s_tet = 380 / (ds.p_e * np.exp(-17.2693882 * T_C  / (ds.temp - 35.86)) - 610.9)
        RH = ds.rv / rv_s_tet
        RH.attrs["units"] = "1"
        RH.attrs["long name"] = "relative humidity"
        return ds.assign(RH = RH)