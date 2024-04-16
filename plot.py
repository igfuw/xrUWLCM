from xhistogram.xarray import histogram
import numpy as np
import matplotlib.pyplot as plt

def plot_CFAD(data, varname, cond=True, xbins=None, **kwargs):
    ds = data[varname].where(cond)
    try:
        if xbins==None:
            xbins = np.linspace(ds.min(), ds.max(), 200)
    except:
        pass
    zbins = np.arange(0, data.z[-1], data.dz)
    hist = histogram( data.z, ds, bins=[zbins, xbins])#, density=True)
    hist = hist / np.sum(hist).values * 100
    plt.imshow(hist, cmap='hot_r', origin='lower', aspect='auto', extent=(xbins[0], xbins[-1], zbins[0], zbins[-1]), **kwargs)
    ds.mean(["x","y","t"]).plot(y="z")
    ds.chunk(dict(t=-1)).quantile([0.1,0.9],["x","y","t"]).plot.line(y="z")
    #plt.contour(hist, extent=(0,1000,0,nc.z[-1]), levels=5)
    #plt.contour(centers, nc.z.values, hist)
    cbar = plt.colorbar(orientation='vertical')
    cbar.set_label('fraction of cells [%]', rotation=270, labelpad=20)
    plt.ylabel('$z$ [m]')
    #plt.show()

def plot_DSD(ds, cond=True, **kwargs):
    bin_centers = np.exp((np.log(ds.out_wet_lft_edges) + np.log(ds.out_wet_rgt_edges))/2.) * 1e6 # [um]
    log_bin_width = np.log(ds.out_wet_rgt_edges * 1e6) - np.log(ds.out_wet_lft_edges * 1e6) # [um]
    bin_mom0 = [] # list of pairs: (bin center, 0-th mom of wer radius in the bin)
    for i,le in enumerate(bin_centers):
        bin_mom0.append(( \
            (bin_centers[i]), \
            (ds['rw_rng'+str(i).zfill(3)+'_mom0']*ds.rhod/log_bin_width/1e6).where(cond).mean() \
        )) # /1e6 to get [1/cm3]
    r, mom0 = zip(*bin_mom0) # unpack a list of pairs into two tuples
    plt.plot(r, mom0, **kwargs)#, label='z='+str(z))
    #plt.show()