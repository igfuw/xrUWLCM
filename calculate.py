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

    
    #.pipe(calc_ra) \
    #.pipe(calc_rl) \
    #.pipe(calc_rt) \
    #.pipe(calc_all_r_mean) \
    #.pipe(calc_all_rain_r_sigma) \
    #.pipe(calc_r_m6) \
    #.pipe(calc_lwp) \
    #.pipe(calc_rwp)
    

def calc_ra(ds):
    if ds.microphysics == "super-droplets":
        ra=lambda x: x.aerosol_rw_mom3 * 4./3. * np.pi * 1e3
    else:
        ra=np.nan
    dsn = ds.assign(ra=ra)
    dsn.ra.attrs["units"] = "kg/kg"
    dsn.ra.attrs["long_name"] = "humidified aerosol mixing ratio"
    return dsn

def calc_rc(ds):
    if ds.microphysics == "super-droplets":
        rc=lambda x: x.cloud_rw_mom3 * 4./3. * np.pi * 1e3
    else:
        rc=ds.rc
    dsn = ds.assign(rc=rc)
    dsn.rc.attrs["units"] = "kg/kg"
    dsn.rc.attrs["long_name"] = "cloud water mixing ratio"
    return dsn


def calc_rr(ds):
    if ds.microphysics == "super-droplets":
        rr=lambda x: x.rain_rw_mom3 * 4./3. * np.pi * 1e3
    else:
        rr=ds.rr
    dsn = ds.assign(rr=rr)
    dsn.rr.attrs["units"] = "kg/kg"
    dsn.rr.attrs["long_name"] = "rain water mixing ratio"
    return dsn
    
    
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
def calc_all_r_mean(ds):
    if ds.microphysics == "super-droplets":
        try:
            ds.nt
        except:
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
    if ds.microphysics == "super-droplets":
        try:
            ds.nc
        except:
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
    if ds.microphysics == "super-droplets":
        try:
            ds.nr
        except:
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
    if ds.microphysics == "super-droplets":
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
        try:
            ds.nc
        except:
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
        try:
            ds.nr
        except:
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
    if ds.microphysics == "super-droplets":
        na=lambda x: x.aerosol_rw_mom0
    elif ds.microphysics == 'single-moment bulk':
        na=np.nan
    elif ds.microphysics == 'double-moment bulk':
        na=np.nan
    dsn = ds.assign(na=na)
    dsn.na.attrs["units"] = "1/kg"
    dsn.na.attrs["long_name"] = "number concentration of aerosols"
    return dsn
        
# number concentration of cloud droplets [1/kg]
def calc_nc(ds):
    if ds.microphysics == "super-droplets":
        nc=lambda x: x.cloud_rw_mom0
    elif ds.microphysics == 'single-moment bulk':
        nc=np.nan
    elif ds.microphysics == 'double-moment bulk':
        nc=np.nan
    dsn = ds.assign(nc=nc)
    dsn.nc.attrs["units"] = "1/kg"
    dsn.nc.attrs["long_name"] = "number concentration of cloud droplets"
    return dsn
    
# number concentration of rain drops [1/kg]
def calc_nr(ds):
    if ds.microphysics == "super-droplets":
        nr=lambda x: x.rain_rw_mom0
    elif ds.microphysics == 'single-moment bulk':
        nr=np.nan
    elif ds.microphysics == 'double-moment bulk':
        nr=np.nan
    dsn = ds.assign(nr=nr)
    dsn.nr.attrs["units"] = "1/kg"
    dsn.nr.attrs["long_name"] = "number concentration of rain drops"
    return dsn
   
# 6th mooment of droplet radius[m^6/m^3]
def calc_all_r_m6(ds):
    if ds.microphysics == "super-droplets":
        return ds.assign(all_r_m6=lambda x: x.all_rw_mom6 *x.rhod)
    else:
        return ds.assign(all_r_m6=np.nan)

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
        for k in np.arange(ds.sizes["z"]-2, 0, -1):
            prflux[dict(z=k)] = prflux.isel(z=k+1) + prflux.isel(z=k)
        return ds.assign(prflux = prflux * -1 * ds.dk * L_evap)

#liquid water path in columns [kg/m2]
def calc_lwp(ds):
    lwp = (ds.rl * ds['rhod']).sum(["z"]) * ds.dz    
    lwp.attrs["units"] = "kg/m$^2$"
    lwp.attrs["long_name"] = "liquid water path"
    return ds.assign(lwp = lwp)
    
#liquid water path in columns [kg/m2]
def calc_cwp(ds):
    cwp = (ds.rc * ds['rhod']).sum(["z"]) * ds.dz    
    cwp.attrs["units"] = "kg/m$^2$"
    cwp.attrs["long_name"] = "cloud water path"
    return ds.assign(cwp = cwp)
    
#liquid water path in columns [kg/m2]
def calc_rwp(ds):
    rwp = (ds.rr * ds['rhod']).sum(["z"]) * ds.dz    
    rwp.attrs["units"] = "kg/m$^2$"
    rwp.attrs["long_name"] = "rain water path"
    return ds.assign(rwp = rwp)

# inversion height [m]
def calc_zi(ds, cond):
    zi=ds.z.where(cond).idxmin(dim='z')
    z_i.attrs["units"] = "m"
    z_i.attrs["long_name"] = "inversion height"
    return ds.assign(z_i = z_i)
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