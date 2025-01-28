import xarray as xr
from functools import partial
import numpy as np

# returns:
# data - constants and all timesteps with DSD vars dropped
# data_DSD - only timesteps that have DSD vars, from constants only rhod to facilitate calculations of derived variables
def load_outdir(datadir, engine=None):
    const = load_const(datadir, engine)
    data = xr.merge([const, load_timesteps(datadir, const, engine)], combine_attrs="no_conflicts").chunk({"t" : 1}) #set chunk for dask, each task executes takes one file
    data_DSD = xr.merge([const, load_DSD(datadir, const, engine)]).chunk({"t" : 1}) #set chunk for dask, each task executes takes one file
    # hardcoded for now, TODO: output it in const in UWLCM
    data.attrs["aerosol definition"] = "rw < 0.5 microns"
    data.attrs["cloud droplet definition"] = "0.5 microns < rw < 25 microns"
    data.attrs["rain drop definition"] = "25 microns < rw"
    #data_DSD = load_DSD(datadir, const).assign(rhod=const.rhod) \
    #                                   .assign(out_wet_lft_edges=const.out_wet_lft_edges) \
    #                                   .assign(out_wet_rgt_edges=const.out_wet_rgt_edges) \
    #                                   .chunk({"t" : 1}) 
    return data, data_DSD

def load_const(datadir, engine=None):
    open_dataset_kwargs = {'phony_dims': 'sort'} if engine=='h5netcdf' else {} # h5netcdf-specific option
    const = xr.open_dataset(datadir + "const.h5", engine=engine, **open_dataset_kwargs)

    if len(const.G.dims)==3: # 3D
        const = const.rename({const.G.dims[0] : "x", const.G.dims[1] : "y", const.G.dims[2] : "z"})

        # may need this renaming if sizes of some coordinates are equal
        const["X"] = const.X.rename({const.X.dims[0] : "xe", const.X.dims[1] : "ye", const.X.dims[2] : "ze"})
        const["Y"] = const.Y.rename({const.Y.dims[0] : "xe", const.Y.dims[1] : "ye", const.Y.dims[2] : "ze"})
        const["Z"] = const.Z.rename({const.Z.dims[0] : "xe", const.Z.dims[1] : "ye", const.Z.dims[2] : "ze"})
    
        #coordinates of cell edges
        const = const.assign_coords({"xe" : const.X[:,0,0], "ye" : const.Y[0,:,0], "ze" : const.Z[0,0,:]})#, "Y", "Z", "T"])

        #coordinates of cell centers
        X = const.rolling(xe=2).mean().X.dropna(dim="xe").drop_isel(ye=[-1], ze=[-1]).rename({"xe": "x", "ye": "y", "ze": "z"}).compute()
        Y = const.rolling(ye=2).mean().Y.dropna(dim="ye").drop_isel(xe=[-1], ze=[-1]).rename({"xe": "x", "ye": "y", "ze": "z"}).compute()
        Z = const.rolling(ze=2).mean().Z.dropna(dim="ze").drop_isel(xe=[-1], ye=[-1]).rename({"xe": "x", "ye": "y", "ze": "z"}).compute()
        const = const.assign_coords({"x" : X[:,0,0], "y" : Y[0,:,0], "z" : Z[0,0,:]})#, "Y", "Z", "T"])
        
    elif len(const.G.dims)==2: # 2D                                     
        const = const.rename({const.G.dims[0] : "x", const.G.dims[1] : "z", "Y" : "Z"})
        
        const["X"] = const.X.rename({const.X.dims[0] : "xe", const.X.dims[1] : "ze"})
        const["Z"] = const.Z.rename({const.Z.dims[0] : "xe", const.Z.dims[1] : "ze"})

        #coordinates of cell edges
        const = const.assign_coords({"xe" : const.X[:,0], "ze" : const.Z[0,:]})
        
        #coordinates of cell centers
        X = const.rolling(xe=2).mean().X.dropna(dim="xe").drop_isel(ze=[-1]).rename({"xe": "x", "ze": "z"}).compute()
        Z = const.rolling(ze=2).mean().Z.dropna(dim="ze").drop_isel(xe=[-1]).rename({"xe": "x", "ze": "z"}).compute()
        const = const.assign_coords({"x" : X[:,0], "z" : Z[0,:]})#, "Y", "Z", "T"])
    else:
        raise Exception("load_const: data needs to be either 2D or 3D") 

    # time coordinate in case it has the same size as some other coord
    const["T"] = const.T.rename({const.T.dims[0] : "t"})
                
    #output bins if applicable
    if const.microphysics == "super-droplets":
        try:
            const["out_wet_lft_edges"] = const.out_wet_lft_edges.rename({const.out_wet_lft_edges.dims[0] : "outbins_wet"})         
            const["out_wet_rgt_edges"] = const.out_wet_rgt_edges.rename({const.out_wet_rgt_edges.dims[0] : "outbins_wet"})         
        except:
            pass
                
    #time coordinates
    const = const.assign_coords(t=(const.T.values))
    const.t.attrs["units"] = "s"
    const.t.attrs["long_name"] = "time"
  
    const = const.assign_attrs(datadir=datadir)
    const.z.attrs["units"] = "m"
    const.z.attrs["long_name"] = "height"

    #merge all groups into a single dataset
    for grname in ["rt_params", "ForceParameters", "MPI details", "advection", "git_revisions", "lgrngn", "misc", "piggy", "prs", "rhs", "sgs", "user_params", "vip"]:
        try:
            const = const.merge(xr.open_dataset(datadir + "const.h5", group="/"+grname+"/", engine=engine, **open_dataset_kwargs), combine_attrs="no_conflicts")
        except:
            print("Group " + grname + " not found in " + datadir + "const.h5")
    return const


def load_timesteps(datadir, const, engine=None):
    open_dataset_kwargs = {'phony_dims': 'sort'} if engine=='h5netcdf' else {} # h5netcdf-specific option
    filenames = []
    for t in const.T.values:
        filename=datadir + "timestep"+str(int(t / (const.dt))).zfill(10)+".h5"
        try:
            xr.open_dataset(filename, engine=engine, **open_dataset_kwargs)
            filenames.append(filename)
        except:
            continue
    _squeeze_and_set_time = partial(squeeze_and_set_time, const=const, drop_DSD=True, engine=engine)
    return xr.open_mfdataset(filenames, parallel=False, preprocess=_squeeze_and_set_time, engine=engine, **open_dataset_kwargs)


#load size spectra
def load_DSD(datadir, const, engine=None):
    open_dataset_kwargs = {'phony_dims': 'sort'} if engine=='h5netcdf' else {} # h5netcdf-specific option
    filenames = []
    for t in const.T.values:
        filename=datadir + "timestep"+str(int(t / (const.dt))).zfill(10)+".h5"
        try:
            ds = xr.open_dataset(filename, engine=engine, **open_dataset_kwargs)
            if 'rw_rng000_mom0' in ds.variables:
                filenames.append(filename)
        except:
            continue
    _squeeze_and_set_time = partial(squeeze_and_set_time, const=const, drop_DSD=False, engine=engine)
    if len(filenames)>0:
      return xr.open_mfdataset(filenames, parallel=False, preprocess=_squeeze_and_set_time, engine=engine, **open_dataset_kwargs)
    else:
      return xr.Dataset()
    #_squeeze_and_set_time = partial(squeeze_and_set_time, const=const)
    #return xr.open_mfdataset(filenames, parallel=False, preprocess=_squeeze_and_set_time)


def squeeze_and_set_time(ds, const, drop_DSD, engine=None):
    open_dataset_kwargs = {'phony_dims': 'sort'} if engine=='h5netcdf' else {} # h5netcdf-specific option
    
    ds = ds.expand_dims("t")
    #get time from filename
    t = np.float32(ds.encoding["source"][-13:-3]) * const.dt 
    
    if len(const.G.dims)==3: # 3D
        ds = ds.rename({"phony_dim_0" : "x", "phony_dim_1" : "y"})
        #order of phony dims can depend on micro used, e.g. in blk_1m latent_heat_flux is the first array, hence phony_dim_2 has size 1 and z is phony_dim_3
        if ds.phony_dim_2.size == 1:
          ds = ds.rename({"phony_dim_3" : "z"})
          try:
            ds = ds.squeeze("phony_dim_2", drop=True) # surface fluxes are 3D arrays with len(z)=1, convert to 2D arrays
          except:
            pass
            
        elif ds.phony_dim_3.size == 1:
          ds = ds.rename({"phony_dim_2" : "z"})
          try:
            ds = ds.squeeze("phony_dim_3", drop=True) # surface fluxes are 3D arrays with len(z)=1, convert to 2D arrays
          except:
            pass       
        ds = ds.assign_coords({"x" : const.x, "y" : const.y, "z" : const.z, "t" : [t]})#, "Y", "Z", "T"])
    elif len(const.G.dims)==2: # 2D
        ds = ds.rename({"phony_dim_0" : "x"})
        #order of phony dims can depend on micro used, e.g. in blk_1m latent_heat_flux is the first array, hence phony_dim_1 has size 1 and z is phony_dim_2
        if ds.phony_dim_1.size == 1:
          ds = ds.rename({"phony_dim_2" : "z"})
          try:
            ds = ds.squeeze("phony_dim_1", drop=True) # surface fluxes are 2D arrays with len(z)=1, convert to 1D arrays
          except:
            pass
            
        elif ds.phony_dim_2.size == 1:
          ds = ds.rename({"phony_dim_1" : "z"})
          try:
            ds = ds.squeeze("phony_dim_2", drop=True) # surface fluxes are 2D arrays with len(z)=1, convert to 1D arrays
          except:
            pass       
        ds = ds.assign_coords({"x" : const.x, "z" : const.z, "t" : [t]})#, "Y", "Z", "T"])
    else:
        raise Exception("squeeze_and_set_time: data needs to be either 2D or 3D") 
    
    ds.t.attrs["units"] = "s"
    ds.t.attrs["long_name"] = "time"

    ds.z.attrs["units"] = "m"
    ds.z.attrs["long_name"] = "height"
    
    #read puddle
    ds_puddle = xr.open_dataset(ds.encoding["source"], group="/puddle/", engine=engine, **open_dataset_kwargs)
    ds_puddle = ds_puddle.assign(ds_puddle.attrs) # convert data stored in attributes to variables
    for name in ds_puddle.attrs:
        ds_puddle = ds_puddle.rename({name : "puddle_"+name}) # rename variables to indicate that this is puddle
    ds = ds.merge(ds_puddle) # ds_puddle attributes are dropped on merge (thats good)
    
    #drop size spectrum data, because it may not be available at all timesteps making merging difficult; to load the spectrum, use load_spectra
    if(drop_DSD and "outbins_wet" in const):
        out_spec_names = []
        for i in np.arange(len(const.outbins_wet)):
            out_spec_names.append('rw_rng'+str(i).zfill(3)+'_mom0')
        ds = ds.drop_vars(out_spec_names, errors='ignore')
        
    return ds
