from xhistogram.xarray import histogram
import numpy as np
import matplotlib.pyplot as plt

def plot_CFAD(data, ds, xbins=None, ylim=(0,5000)):
    try:
        if xbins==None:
            xbins = np.linspace(0, ds.max(), 100)
    except:
        pass
    zbins = np.arange(0, data.z[-1], data.dz)
    hist = histogram( data.z, ds, bins=[zbins, xbins])#, density=True)
    hist = hist / np.sum(hist).values * 100
    plt.imshow(hist, cmap='hot_r', origin='lower', extent=(0, xbins[-1],0, zbins[-1]), aspect='auto')
    ds.mean(["x","y","t"]).plot(y="z")
    ds.chunk(dict(t=-1)).quantile([0.1,0.9],["x","y","t"]).plot.line(y="z")
    #plt.contour(hist, extent=(0,1000,0,nc.z[-1]), levels=5)
    #plt.contour(centers, nc.z.values, hist)
    plt.colorbar(orientation='vertical')
    plt.ylim(ylim)
    plt.show()