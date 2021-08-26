#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 17:26:33 2021

@author: tianyu
"""
import collections, copy, time
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy import sparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import torch.utils.data as Data

import sys
sys.path.insert(0, 'lib/')


from GNN import GNNLayer
from evaluation import eva
from collections import Counter
from layermodel import *
import utils

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise

def buildGraphNN(X, neighborK):
    #nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
    #distances, indices = nbrs.kneighbors(X)
    A = kneighbors_graph(X, neighborK, mode='connectivity', metric='cosine', include_self=True)
    return A


def buildGammaNN(adj, X, beta):
    '''
    @input adj: adj dense matrix, in numpy format
    @X: data, sample*feature
    @beta: hyperparameter
    '''
    degree = np.array(adj.sum(0).tolist()[0])
    n = len(degree)
    epsilon = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            epsilon[i, j] = (degree[i]*degree[j])/(degree[i]+degree[j]) - adj[i, j]
    zeta = pairwise.cosine_similarity(X)
    zeta_norm = zeta/np.sum(zeta, 1).reshape(-1, 1)
    gamma = beta*epsilon + (1-beta)*zeta_norm
    return gamma, degree

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
#            torch.nn.init.xavier_uniform_(m.bias)
            m.bias.data.fill_(0.0)
            
def pretrain_ae(dataloader, lr, epochs, aeepochs, lambda1, pretrain_path, layer1, layer2, layer3, neighborK, n_input,n_z,n_clusters, distribution, tag, device):
    
    model_ae = AE(layer1, layer2, layer3, layer3, layer2, layer1,
                n_input=n_input,
                n_z=n_z,
                ).to(device)
    print(model_ae)
    optimizer = Adam(model_ae.parameters(), lr= lr)
    

    for epoch in range(aeepochs):  
    
        # reset time
        t_start = time.time()
    
        # extract batches
        epoch_loss = 0.0
        count = 0
        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    
            x_bar, enc_h1, enc_h2, enc_h3, z = model_ae(batch_x)       
            loss_batch = F.mse_loss(x_bar, batch_x)
            
            optimizer.zero_grad()  
            loss_batch.backward()
            optimizer.step() 
            
            
            count += 1
            epoch_loss += loss_batch.item()
            
            # print
            if count % 1000 == 0: # print every x mini-batches
                print('epoch= %d, i= %4d, loss(batch)= %.4f' % (epoch + 1, count, loss_batch.item()))
    
        epoch_loss /= count
        t_stop = time.time() - t_start
        if epoch % 10 == 0:
            print('epoch= %d, loss(train)= %.3f,  time= %.3f' %
                  (epoch + 1, epoch_loss,  t_stop))
            #print('training_time:',t_stop)
        
    torch.save(model_ae.state_dict(), 'lib/ae_state_dict_model.pt')
    return model_ae

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()



def train_dgc(dataset, lr, epochs, aeepochs, lambda1, pretrain_path, layer1, layer2, layer3, neighborK, n_input,n_z,n_clusters, distribution, tag, device):

    if tag == "DGC":
        model = DGC(pretrain_path, layer1, layer2, layer3, layer3, layer2, layer1,
                    n_input=n_input,
                    n_z=n_z,
                    n_clusters=n_clusters,
                    v=1.0).to(device)
    elif tag == "DGCNb":
        model = DGCNb(pretrain_path, layer1, layer2, layer3, layer3, layer2, layer1,
                    n_input=n_input,
                    n_z=n_z,
                    n_clusters=n_clusters,
                    distribution=distribution,
                    v=1.0).to(device)     
    
 
    print(model)
    # instantiate the object net of the class
    #model.apply(weight_init)


    optimizer = Adam(model.parameters(), lr=lr)

    ###### KNN Graph

    adj = buildGraphNN(dataset.x, neighborK)

#    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#    adj = utils.normalize(adj + sparse.eye(adj.shape[0]))
    print(type(adj))
    adj = utils.normalize(adj)
    adjdense = torch.FloatTensor(adj.todense())   
    adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
    
    
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    
    y = dataset.y
    
    if tag == "DGC" or tag == "DGCNb":
        with torch.no_grad():
            _, _, _, _, z = model.ae(data)    
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        y_pred_last = y_pred
        
        ######model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
        eva(y, y_pred, 'kmeans')


            
    for epoch in range(epochs):
        if epoch % 30 == 0:
        # update_interval
            _, tmp_q, pred, _,_, likelihood, _ = model(data, adj)

            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')

        x_bar, q, pred, _, adj_bar, likelihood, tempvals = model(data, adj)
        eps=1e-6
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div((pred+eps).log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = lambda1 * kl_loss + lambda1 * ce_loss + 1*re_loss + 0.001*likelihood
        
        if epoch % 100 == 0:
            print('----',kl_loss, ce_loss, re_loss, likelihood)
            print(loss.detach().numpy())
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    
    return p, q, pred, likelihood, x_bar, tempvals

