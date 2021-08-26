import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

class load_data(Dataset):
    def __init__(self, dataset, labels):
        self.x = dataset
        self.y = labels

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def high_var_npdata(data, num, gene = None, ind=False): #data: gene*cell
    dat = np.asarray(data)
    datavar = np.var(dat, axis = 1)*(-1)
    ind_maxvar = np.argsort(datavar)
    gene_ind = ind_maxvar[:num]
    if gene is None and ind is False:
        return data[gene_ind]
    if ind:
        return data[gene_ind],gene_ind
    return data[gene_ind],gene.iloc[gene_ind]




def getSubset(data, labels, partDataset, testSize=0.2):
    indexes = [i for i in range(len(labels)) if labels[i] in partDataset]
    labels = labels[indexes]
    
    le = preprocessing.LabelEncoder()
    encodedLabels = le.fit_transform(labels)
    data = data[indexes]
    train_X, test_X, train_y, test_y = train_test_split(data, encodedLabels, test_size=testSize, random_state=42, stratify = encodedLabels)
    return train_X, test_X, train_y, test_y

def getTorchLoader(train_data, test_data, train_labels, test_labels, batchsize):
    train_loader, test_loader = None, None
    if train_data is not None:
        train_labels = train_labels.astype(np.int64)
        train_data = torch.FloatTensor(train_data)
        train_labels = torch.LongTensor(train_labels)
        dset_train = Data.TensorDataset(train_data, train_labels)
        train_loader = Data.DataLoader(dset_train, batch_size = batchsize, shuffle = True)
    if test_data is not None:
        test_labels = test_labels.astype(np.int64)
        test_data = torch.FloatTensor(test_data)
        test_labels = torch.LongTensor(test_labels)       
        dset_test = Data.TensorDataset(test_data, test_labels)
        test_loader = Data.DataLoader(dset_test, shuffle = False)    
    
    return train_loader, test_loader

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.utils.linear_assignment_ import linear_assignment

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

from sklearn import metrics
def print_res(true_labels, pred_labels, tag):
    ARI = metrics.adjusted_rand_score(true_labels, pred_labels)
    NMI = metrics.normalized_mutual_info_score(true_labels, pred_labels)
    FMI = metrics.cluster.fowlkes_mallows_score(true_labels, pred_labels)
    ACC = cluster_acc(true_labels, pred_labels)
    print("ACC-ARI-NMI-"+tag, ACC, ARI, NMI, FMI) 
    return ACC, ARI, NMI, FMI
    
def getResults(x_train_latent, train_labels, p_labels,q_labels, z_labels):
    results = pd.DataFrame()

    acc, ari, nmi, fmi = print_res(train_labels, z_labels, "gcn-z")
    results = pd.concat([results, pd.DataFrame([acc, ari, nmi, fmi]).T], 1)
    acc, ari, nmi, fmi = print_res(train_labels, p_labels, "gcn-p")
    results = pd.concat([results, pd.DataFrame([acc, ari, nmi, fmi]).T], 1)
    acc, ari, nmi, fmi = print_res(train_labels, q_labels, "gcn-q")
    results = pd.concat([results, pd.DataFrame([acc, ari, nmi, fmi]).T], 1)
    results.columns = ["z_acc", "z_ari", "z_nmi", "z_fmi", "p_acc", "p_ari", "p_nmi", "p_fmi", "q_acc", "q_ari", "q_nmi", "q_fmi"]
      
    return results