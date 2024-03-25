import xarray as xr
from functools import partial

def load_outdir(datadir):
    const = load_const(datadir)
    data = xr.merge([const, load_timesteps(datadir, const)])
    data = data.chunk({"t" : 1}) #for dask, each task executes takes one file
    return data

def load_const(datadir):
    const = xr.open_dataset(datadir + "const.h5")
    const = const.rename({"phony_dim_0" : "x", "phony_dim_1" : "y", "phony_dim_2" : "z", "phony_dim_3" : "t", \
                          "phony_dim_4" : "xe", "phony_dim_5" : "ye", "phony_dim_6" : "ze"}) # positions of cell edges
    #time coordinates
    const = const.assign_coords(t=("t",const.T.values))
    #coordinates of cell edges
    const = const.assign_coords({"xe" : const.X[:,0,0], "ye" : const.Y[0,:,0], "ze" : const.Z[0,0,:]})#, "Y", "Z", "T"])
    #coordinates of cell centers
    X = const.rolling(xe=2).mean().X.dropna(dim="xe").drop_isel(ye=[-1], ze=[-1]).rename({"xe": "x", "ye": "y", "ze": "z"}).compute()
    Y = const.rolling(ye=2).mean().Y.dropna(dim="ye").drop_isel(xe=[-1], ze=[-1]).rename({"xe": "x", "ye": "y", "ze": "z"}).compute()
    Z = const.rolling(ze=2).mean().Z.dropna(dim="ze").drop_isel(xe=[-1], ye=[-1]).rename({"xe": "x", "ye": "y", "ze": "z"}).compute()
    const = const.assign_coords({"x" : X[:,0,0], "y" : Y[0,:,0], "z" : Z[0,0,:]})#, "Y", "Z", "T"])
  
    const = const.assign_attrs(datadir=datadir)

    #merge all groups into a single dataset
    for grname in ["rt_params", "ForceParameters", "MPI details", "advection", "git_revisions", "lgrngn", "misc", "piggy", "prs", "rhs", "sgs", "user_params", "vip"]:
        try:
            const = const.merge(xr.open_dataset(datadir + "const.h5", group="/"+grname+"/"), combine_attrs="no_conflicts")
        except:
            print("Group " + granme + " not found in " + datadir + "const.h5")
    return const


def load_timesteps(datadir, const):
    filenames = []
    for t in const.T.values:
        filename=datadir + "timestep"+str(int(t / (const.dt))).zfill(10)+".h5"
        try:
            xr.open_dataset(filename)
            filenames.append(filename)
        except:
            continue
    _squeeze_and_set_time = partial(squeeze_and_set_time, const=const)
    return xr.open_mfdataset(filenames, parallel=False, preprocess=_squeeze_and_set_time)

def squeeze_and_set_time(ds, const):
    ds = ds.rename({"phony_dim_0" : "x", "phony_dim_1" : "y", "phony_dim_2" : "z"})
    ds = ds.squeeze("phony_dim_3") # surface fluxes are 3D arrays with len(z)=1, convert to 2D arrays
    ds = ds.expand_dims("t")
    #get time from filename
    t = int(ds.encoding["source"][-13:-3]) * const.dt 
    ds = ds.assign_coords({"x" : const.x, "y" : const.y, "z" : const.z, "t" : [t]})#, "Y", "Z", "T"])
    #TODO: read puddle
    #ds_puddle = xr.open_dataset(datadir + "timestep0000012000.h5", group="/puddle/")
    return ds
