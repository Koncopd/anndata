from .. import h5py
from ..base import AnnData

def postprocess_reading(key, value):
    # record arrays should stay record arrays and not become scalars
    if value.ndim == 1 and len(value) == 1 and value.dtype.names is None:
        value = value[0]
    if value.dtype.kind == 'S':
        value = value.astype(str, copy=False)
        # backwards compat:
        # recover a dictionary that has been stored as a string
        if len(value) > 0:
            if value[0] == '{' and value[-1] == '}': value = eval(value)
    # transform byte strings in recarrays to unicode strings
    # TODO: come up with a better way of solving this, see also below
    if (key not in AnnData._H5_ALIASES['obs']
        and key not in AnnData._H5_ALIASES['var']
        and key != 'raw.var'
        and not isinstance(value, dict) and value.dtype.names is not None):
        new_dtype = [((dt[0], 'U{}'.format(int(int(dt[1][2:])/4)))
                      if dt[1][1] == 'S' else dt) for dt in value.dtype.descr]
        value = value.astype(new_dtype, copy=False)
    return key, value

def read_alt(filename):
    d = {}
    ignore = []
    f = h5py.File(filename)
    def tour(name, object):
        dic = d
        sparse = 'h5sparse_format' in object.attrs
        ds = isinstance(object, h5py.Dataset)
        if sparse or ds:
            keys = name.split('/')
            if len(keys) > 1 and keys[-2] in ignore:
                return None
            for key in keys[:-1]:
                dic = dic.setdefault(key, {})
            if sparse:
                object = h5py.SparseDataset(object)
                ignore.append(keys[-1])
            key, value = postprocess_reading(keys[-1], object[()])
            dic[key] = value
    f.h5py_group.visititems(tour)
    return AnnData(d)
