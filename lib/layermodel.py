#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:15:43 2020

@author: tianyu
"""
import torch
#from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, 'lib/')

import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph, sparse_mx_to_torch_sparse_tensor
from GNN import GNNLayer
from evaluation import eva
from collections import Counter
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

from coarsening import lmax_L
from coarsening import rescale_L

class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input )
        return grad_input_dL_dW, grad_input_dL_dx


#########################################################################################################    

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = torch.relu(self.enc_1(x))
        enc_h2 = torch.relu(self.enc_2(enc_h1))
        enc_h3 = torch.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = torch.relu(self.dec_1(z))
        dec_h2 = torch.relu(self.dec_2(dec_h1))
        dec_h3 = torch.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z

 
    
class DGC(nn.Module):

    def __init__(self,pretrain_path, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(DGC, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)
        self.fc1 = nn.Linear(n_z, n_input)
#        self.fc2 = nn.Linear(n_input//2, n_input)
 
        
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
            
        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x) 
        
        sigma = 0.3
                
        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)

        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj, active=False)
        h5 = self.gnn_5((1-sigma)* F.relu(h) + sigma*z, adj, active=False) # n_z --> n_cluster, F.relu(z) does not work
        predict = F.softmax(h5, dim=1)
        
        deco = self.fc1(torch.relu(h))
#        deco = self.fc2(torch.relu(deco))
        

        x_bar = torch.relu(deco)
        
        adj_bar = 0 #F.relu(torch.mm(h, h.t()))
        
        #print(adj_bar)
        #print(predict.shape, h.shape, h.unsqueeze(1).shape, self.cluster_layer.shape) 
        ''' 
        predict.shape:            torch.Size([622, 4]), 622 nodes x 4 clusters
        h.shape:                  torch.Size([622, 10]) , bottle neck layer, embeddings of each node
        h.unsqueeze(1).shape:     torch.Size([622, 1, 10]), each node is a vector of 1x10
        self.cluster_layer.shape: torch.Size([4, 10]), 4 clusters, each cluster has a centroid of vector 1x10
        '''
        
        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(h.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)     
        q = q.pow((self.v + 1.0) / 2.0) # torch.Size([622, 4])  
        q = (q.t() / torch.sum(q, 1)).t()  # torch.Size([622, 4])

        return x_bar, q, predict, z, adj_bar, 0, [h, tra1, tra2, tra3]


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x)+np.inf, x)

#def _nelem(x):
#    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
#    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)
#
#def _reduce_mean(x):
#    nelem = _nelem(x)
#    x = _nan2zero(x)
#    return tf.divide(tf.reduce_sum(x), nelem)

def NB(theta, y_true, y_pred, mask = False, debug = False, mean = True):
    eps = 1e-10
    scale_factor = 1.0

    t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + (y_true * (torch.log(theta + eps) - torch.log(y_pred + eps)))

    final = t1 + t2
    final = _nan2inf(final)
    if mean:
        final = torch.mean(final)
    else:
        final = torch.sum(final)
    
    return final

def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mask = False, debug = False, mean = True):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_true, y_pred, mean=True, debug=debug) - torch.log(1.0 - pi + eps)

    zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(torch.le(y_true, 1e-8), zero_case, nb_case)
    ridge = ridge_lambda * torch.pow(pi, 2)
    result += ridge
    if mean:
        result = torch.mean(result)
    else:
        result = torch.sum(result)

    result = _nan2inf(result)
    
    return result



class DGCNb(nn.Module):

    def __init__(self,pretrain_path, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters,distribution="NB", v=1):
        super(DGCNb, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        self.distribution = distribution
        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)
        self.fc1 = nn.Linear(n_z, n_input//2)
        self.fc_pi = nn.Linear(n_input//2, n_input)
        self.fc_disp = nn.Linear(n_input//2, n_input)
        self.fc_mu = nn.Linear(n_input//2, n_input)
        
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
            
        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x) 
        
        sigma = 0.3
        
        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj, active=False)
        h5 = self.gnn_5((1-sigma)* F.relu(h) + sigma*z, adj, active=False) # n_z --> n_cluster
        predict = F.softmax(h5, dim=1)
        
        deco = self.fc1(F.relu(h))
        deco = F.relu(deco)
        
        if self.distribution == "ZINB":            
            pi = self.fc_pi(deco)
            self.pi = torch.sigmoid(pi)
            disp = self.fc_disp(deco)
            self.disp = torch.clamp(F.softplus(disp), min=0, max=1e4)
            mean = self.fc_mu(deco)
            self.mean = torch.clamp(F.relu(mean), min=0, max=1e6)
            self.output = self.mean
            self.likelihood_loss = ZINB(self.pi, self.disp, x, self.output, ridge_lambda=1.0, mean=True)
            
        elif self.distribution == "NB":
            disp = self.fc_disp(deco)
            disp = torch.clamp(F.softplus(disp), min=0, max=1e4)
            mean = self.fc_mu(deco)
            self.output = F.relu(mean)
            self.likelihood_loss = NB(disp, x, self.output, mask=False, debug=False, mean=True)

        
        adj_bar = 0 #F.relu(torch.mm(h, h.t()))
        
        ''' 
        predict.shape:            torch.Size([622, 4]), 622 nodes x 4 clusters
        h.shape:                  torch.Size([622, 10]) , bottle neck layer, embeddings of each node
        h.unsqueeze(1).shape:     torch.Size([622, 1, 10]), each node is a vector of 1x10
        self.cluster_layer.shape: torch.Size([4, 10]), 4 clusters, each cluster has a centroid of vector 1x10
        '''
        
        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(h.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)     
        q = q.pow((self.v + 1.0) / 2.0) # torch.Size([622, 4])  
        q = (q.t() / torch.sum(q, 1)).t()  # torch.Size([622, 4])

        return self.output, q, predict, z, adj_bar, self.likelihood_loss, [h, tra1, tra2, tra3]
    
