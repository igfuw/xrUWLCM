# based on UWLCM_plotting/Energy_spectrum/spectrum_refined.py
# but without grid refinement

import xarray as xr
import numpy as np
import scipy as sp

# 1d - calculate spetra along x and y and average
def calc_spectrum(ds, axs, res, exp): # dataset, axes over which fft is done, grid resolution, amplitude (exp=1) or power (exp=2) spectrum
    assert(len(axs)==2) # spectrum is calculated over a horizontal slice
    E, lmbd, K = [], [], []
    for ax, ax2 in zip(axs, reversed(axs)): # axis along which we calculate fft and the perpendicular axis
        K.append(np.fft.rfftfreq(ds.shape[ax]) / res)
        lmbd.append(1 / K[-1])
        wk = np.fft.rfft(ds, axis = ax, norm='forward')
        E.append(np.mean(np.abs(wk) ** exp, axis = ax2))
    return E, K, lmbd

def spectrum_average(E, K, lmbd):  # arguments are the output of calc_spectrum
    lmbd = lmbd[0] # assume nx=ny...
    K = K[0]
    Exy = 0.5 * (E[0] + E[1])
    if(E.ndim == 4):
        Exy_avg = Exy.mean(axis=0).mean(axis=1) # time and height average
    elif(E.ndim == 3):
        Exy_avg = Exy.mean(axis=0) # time average only
    else:
        raise Exception("wrong number of dimensions") 
    return Exy_avg, lmbd


# 2d Fourier spectrum over a xy plane
def calc_spectrum2d(ds, axs, res, exp): # dataset, axes over which fft is done, grid resolution, amplitude (exp=1) or power (exp=2) spectrum
    assert(len(axs)==2) # spectrum is calculated over a horizontal slice   
    shift = np.fft.fftshift
    wk = shift(np.fft.fft2(ds, axes = axs))#, norm='backward'))
    E = np.abs(wk) ** exp
    ky = shift(np.fft.fftfreq(ds.shape[axs[0]], res)) #* 2 * np.pi
    #kx = np.fft.rfftfreq(ds.shape[axs[1]]) / res #* 2 * np.pi # in 2d fft, real transform is done over the last axis
    kx = shift(np.fft.fftfreq(ds.shape[axs[1]], res)) #* 2 * np.pi # in 2d fft, real transform is done over the last axis
    
    # get (len(kx), len(ky)) arrays with kx and ky values repeated
    kyx = np.atleast_2d(ky).T # turn into column vector
    kxy = np.broadcast_to(kx, (len(ky),len(kx)))
    kyx = np.broadcast_to(kyx, (len(ky),len(kx)))
    
    #theta = np.arctan(kyx / kxy)
    ##theta = np.arctan2(kyx, kxy)
    K = np.sqrt(np.power(np.abs(kxy), 2) + np.power(np.abs(kyx), 2))
    #E[:,0,0,:]=0
    # c = plt.pcolormesh(K)
    # plt.colorbar(c)
    # plt.show()
    return E, K, kx, ky

# angular integration of a 2d spectrum using spline interpolation following
# https://dsp.stackexchange.com/questions/36902/calculate-1d-power-spectrum-from-2d-images
# TODO: add time and height averaging
def spectrum2d_rad_spline_integration(kx, ky, E): # x and y wavenumbers, amplitude (or power); comes from calc_spectrum2d
    assert(E.ndim == 2) # for now the function works only with horizontal slices (e.g. a given time and height)
    E_interp = sp.interpolate.RectBivariateSpline(kx, ky, E)
    k_int = np.logspace(np.log10(max(kx[kx > 0.].min(), ky[ky > 0.].min())), np.log10(max(kx.max(), ky.max())), 256)
    
    ## integrate only over some cone around theta_center. 
    ## Around 0deg the spectrum has similar shape as the 1D spectrum along x
    ## Around 90deg the spectrum has similar shape as the 1D spectrum along y
    #theta_center = 90 # in deg
    #theta = np.linspace(np.radians(theta_center - 10) , np.radians(theta_center + 10), 11, False)
    ## integrate over entire range of theta (we need -pi to pi, beacause we use fft (not rfft) and we want both positive and negative kx and ky
    theta = np.linspace(-np.pi, np.pi, 100, False)
    E_int = np.empty_like(k_int)
    for i in range(k_int.size):
        _kx = np.sin(theta) * k_int[i]
        _ky = np.cos(theta) * k_int[i]
        E_int[i] = np.mean(E_interp.ev(_kx, _ky) * 4 * np.pi)
    return E_int, 1./k_int


# angular integration of a 2d spectrum using averaging assuming equal weights for all points within some range (bin) of K
def spectrum2d_rad_mean_integration(K, S, kx, ky):  # wavenumber, amplitude (or power); comes from calc_spectrum2d
    # flatten along x and y to simplify integration, 
    # and create a list of such flattened horizontal slices 
    # (each list element is @ different time and height)
    K = K.flatten()
    if(S.ndim == 4):
        S = S.reshape(S.shape[0], S.shape[1] * S.shape[2], S.shape[3]) # 0 - time, 1 - x, 2 - y, 3 - z
        Slist = [S[i, :, j] for i in range(S.shape[0]) for j in range(S.shape[2])]
    elif(S.ndim == 3):
        S = S.reshape(S.shape[0], S.shape[1] * S.shape[2]) # 0 - time, 1 - x, 2 - y
        Slist = [s for s in S[:]]
    elif(S.ndim == 2):
        S = S.reshape(S.shape[0] * S.shape[1]) # 0 - x, 1 - y
        Slist = [S]
    else:
        raise Exception("wrong number of dimensions") 
    lmbd = 1/K

    lmbd_bins = np.logspace(np.log10(1./max(kx.max(), ky.max())), np.log10(1./max(kx[kx > 0.].min(), ky[ky > 0.].min())), 101)
    S_mean = sp.stats.binned_statistic(lmbd, Slist, statistic='mean', bins=lmbd_bins).statistic
    S_mean = S_mean.mean(axis=0) # average over the Slist results (time or time and height average depending on the case)
    S_mean *= 4* np.pi #integrated over -pi,pi range of theta
    # replace nan with 0 in bins that have no points
    S_mean = np.nan_to_num(S_mean, nan=0.0)
    lmbd_cntr = (lmbd_bins[1:] + lmbd_bins[:-1])/2.
    return S_mean, lmbd_cntr
