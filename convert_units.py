def convert_units(ds):
    multiplier = {
        "qc" : 1e3, # to [g/m3]
        "qr" : 1e3, # to [g/m3]
        "qa" : 1e3, # to [g/m3]
        "qt" : 1e3, # to [g/m3]
        "rc" : 1e3, # to [g/kg]
        "rr" : 1e3, # to [g/kg]
        "r_mean" : 1e6, # to [um]
        "r_sigma" : 1e6, # to [um]
        "na" : 1e-6, # to [1/cc]
        "nc" : 1e-6, # to [1/cc]
        "nr" : 1e-6, # to [1/cc]
        "r_m6" : 1e18, # to [mm^6/m^3]
    }

    for m in multiplier:
        try:
            ds[m] *= multiplier[m]
        except:
            pass
    return ds