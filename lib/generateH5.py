#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:53:01 2021

@author: tianyu
"""


import h5py
import scipy as sp
import numpy as np
import scanpy.api as sc
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'lib/')
import utilsdata 

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)



def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

def create_h5file(filename, data, labels):
    data = np.asarray(data).astype(np.int)
    gene_names = np.array([i+1 for i in range(data.shape[1])])

    data_shape = (data.shape[0], data.shape[1]) 
    row = np.where(data != 0)[0]
    col = np.where(data != 0)[1]
    spdata = sp.sparse.csr_matrix((data[np.where(data!=0)].reshape(-1), (row, col)), shape=data_shape)  
    
    with h5py.File(filename, "w") as f:
        g1 = f.create_group("exprs")
        g1["data"] = spdata.data
        g1["indices"] = spdata.indices
        g1["indptr"] = spdata.indptr
        g1["shape"] = data_shape
        
        g1 = f.create_group("var")
        g1["var"] = gene_names
        g1 = f.create_group("obs")
        g1["cell_type1"] = labels    
        
        g1=f.create_group("uns")
        g1["expressed_genes"]= np.arange(1)
        g1["scmap_genes"] = np.arange(1)
        g1["seurat_genes"] = np.arange(1)
    
  

def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]))
        var = pd.DataFrame(dict_from_group(f["var"]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns



dataset = 'Zhengsorted'
if dataset == 'usoskin':
    filepath = '/Users/tianyu/google drive/fasttext/imputation/'  
    _, features_all,labels_all = utilsdata.load_usoskin(path = filepath, dataset='usoskin')
else:
    filepath = '/Users/tianyu/Desktop/scRNAseq_Benchmark_datasets/Intra-dataset/'
    _, features_all,labels_all,shuffle_index = utilsdata.load_largesc(path = filepath, dataset=dataset, net='String')
print('******************************', features_all.shape)

filename = "/Users/tianyu/Documents/scziDesk/dataset/"+dataset+"/data.h5"
create_h5file(filename, features_all.T, labels_all)

sparsify = False
skip_exprs = False

mat, obs, var, uns = read_data(filename, sparsify = sparsify, skip_exprs = skip_exprs)
print(mat.shape)




















