'''
Provides functionality to process xarray datasets and data arrays.
'''


def get_missing_value(da):
    '''
    Given an xarray data array, return its missing_value attribute when available;
    otherwise fall back to its _FillValue attribute.
    '''
    for metadata, key in (
        (da.attrs, 'missing_value'),
        (da.encoding, 'missing_value'),
        (da.encoding, '_FillValue'),
        (da.attrs, '_FillValue'),
    ):
        value = metadata.get(key)

        if value is not None:
            return value

    variable_name = da.name if da.name is not None else '<unnamed>'

    raise ValueError(f'Variable {variable_name} has neither a missing_value nor a _FillValue defined.')


def mask_with_missing_value(da, mask, missing_value):
    '''
    Retains values in an xarray data array where mask is True
    and sets them to a given missing_value where mask is False.
    Retains attibutes and encoding of the data array.
    '''
    attrs = da.attrs.copy()
    encoding = da.encoding.copy()

    # Set values failing the mask to the declared missing value.
    da = da.where(mask, other=missing_value)

    da.attrs = attrs
    da.encoding = encoding

    return da
