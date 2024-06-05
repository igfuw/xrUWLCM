def convert_units(ds):
    multiplier = {
        "ra" : 1e3, # to [g/kg]
        "rl" : 1e3, # to [g/kg]
        "rt" : 1e3, # to [g/kg]
        "rc" : 1e3, # to [g/kg]
        "rr" : 1e3, # to [g/kg]
        "r_mean" : 1e6, # to [um]
        "r_sigma" : 1e6, # to [um]
        "na" : 1e-6, # to [1/mg]
        "nc" : 1e-6, # to [1/mg]
        "nr" : 1e-6, # to [1/mgres100m_rt_i_clean_series.pngres100m_rt_i_clean_series.png]
        "r_m6" : 1e18, # to [mm^6/m^3]
    }

    for m in multiplier:
        try:
            ds[m] *= multiplier[m]
        except:
            pass
    return ds