# cloudiness mask
def is_cloudy(ds, type):
    match type:
        case "dycoms":
            if ds.microphysics == 'single-moment bulk':
                raise Exception("DataSet does not contain a \"nc\" DataArray, because it comes from single moment bulk microphysics. Can't calculate cloudiness mask using the dycoms expression: nc > 20 [1/cm^3]")
            else:
                return ds.nc * ds.rhod > 20, # assuming nc in 1/mg
        case "rico":
            return ds.rc > 1e-2
        case _:
            raise Exception("unrecognized type")

# inversion height
def zi(ds, type):
    match type:
        case "dycoms":
            return ds.rt < 8
        case _:
            raise Exception("unrecognized type")