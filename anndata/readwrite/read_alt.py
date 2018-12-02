from .. import h5py
from ..base import AnnData
import numpy as np
import scipy.sparse as ss
from collections import OrderedDict

from ..h5py.h5sparse import get_format_class

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
                format_class = get_format_class(object.attrs['h5sparse_format'])
                shape = tuple(object.attrs['h5sparse_shape'])
                data_array = format_class(shape, dtype=object['data'].dtype)
                data_array.data = np.empty(object['data'].shape, object['data'].dtype)
                data_array.indices = np.empty(object['indices'].shape, object['indices'].dtype)
                data_array.indptr = np.empty(object['indptr'].shape, object['indptr'].dtype)
                object['data'].read_direct(data_array.data)
                object['indices'].read_direct(data_array.indices)
                object['indptr'].read_direct(data_array.indptr)
                ignore.append(keys[-1])
            else:
                data_array = np.empty(object.shape, object.dtype)
                object.read_direct(data_array)
            key, value = postprocess_reading(keys[-1], data_array)
            dic[key] = value
    f.h5py_group.visititems(tour)
    return AnnData(d)

def read_ds_direct(ds):
    if isinstance(ds, h5py.Dataset):
        data_array = np.empty(ds.shape, ds.dtype)
        ds.read_direct(data_array)
    elif isinstance(ds, h5py.SparseDataset):
        data_array = ds.value(True)
    else:
        data_array = ds[()]
    return data_array

def read_tree(group, d):
    ignore = []
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
            key, value = postprocess_reading(keys[-1], read_ds_direct(object))
            dic[key] = value
    group.visititems(tour)

def read_no_recursion(filename):
    d = {}
    d['uns'] = OrderedDict()
    f = h5py.File(filename)
    f_keys = set(f.keys())
    raw = {'raw.X', 'raw.var', 'raw.varm', 'raw.cat'}

    for k, v in AnnData._H5_ALIASES.items():
        sv = set(v)
        key = (sv & f_keys)
        if len(key) > 0:
            key = key.pop()
        else:
            continue
        if key in AnnData._H5_ALIASES['layers']:
            d['layers'] = OrderedDict()
            for l in f[key].keys():
                _, d['layers'][l] = postprocess_reading(l, read_ds_direct(f[key][l]))
        elif key in AnnData._H5_ALIASES['uns']:
            read_tree(f[key].h5py_group, d['uns'])
        else:
            _, d[k] = postprocess_reading(k, read_ds_direct(f[key]))
        f_keys = f_keys - sv

    for k in raw:
        if k in f_keys:
            _, d[k] = postprocess_reading(k, read_ds_direct(f[k]))
    f_keys = f_keys - raw

    for k in f_keys:
        if isninstance(f[k], (h5py.Dataset, h5py.SparseDataset)):
            _, d['uns'][k] = postprocess_reading(k, read_ds_direct(f[k]))
        else:
            read_tree(f[k].h5py_group, d['uns'])
    return AnnData(d)
