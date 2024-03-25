# xrUWLCM
Loading University of Warsaw Lagrangian Cloud Model output as Xarray.

```
import xrUWLCM as xrU
data = xrU.load_outdir(outdir)
```
'outdir' is an output directory from UWLCM.
'data' is a xarray.Dataset.

Plot time series of the 3rd moment of cloud water:

```
data.cloud_rw_mom3.mean(["x","y","z"]).plot()
```

Plot vertical profile of the 3rd moment of cloud water in cloudy cells:

```
data.cloud_rw_mom3.where(data.cloud_rw_mom3>1e-7).mean(["x","y","t"]).plot(y="z")
```
