#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:06:03 2021

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

import utils

from layermodel import *
import preprocessH5

import train
# torch.cuda.set_device(1)
parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--user', type=str, default='personal', help="personal or hpc")
parser.add_argument('--datagroup', type=str, default='new')
parser.add_argument('--dataset', type=str, default='usoskin') #Segerstolpe
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--lr', type=float, default= 0.001 )
parser.add_argument('--testSize', default=0.2, type=float)
parser.add_argument('--n_clusters', default=4, type=int)
parser.add_argument('--n_z', default=10, type=int)
parser.add_argument('--layer1', default=128, type=int)
parser.add_argument('--layer2', default=64, type=int)
parser.add_argument('--layer3', default=32, type=int)
parser.add_argument('--neighborK', default=5, type=int)
parser.add_argument('--lambda1', type=float, default= 0.01 )
parser.add_argument('--tag', type=str, default = 'DGCNb', help='method')
parser.add_argument('--distribution', type=str, default='NB')
parser.add_argument('--num_gene', type=int, default = 500, help='# of genes')
parser.add_argument('--aeepochs', type=int, default = 100, help='# of epoch')
parser.add_argument('--epochs', type=int, default = 300, help='# of epoch')
parser.add_argument('--batchsize', type=int, default = 64, help='# of genes')
parser.add_argument('--pretrain_path', type=str, default = 'lib/ae_state_dict_model.pt', help='pretrain path')
parser.add_argument('--dataPath', type=str, default = '/Users/tianyu/Documents/scziDesk/dataset', help='dataset path')
parser.add_argument('--partDataset', type=list, default = list([]), help='pretrain path')

#parser.add_argument('--pretrain_path', type=str, default='pkl')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")
    


print("------ start -------")

def getEncodeLabels(obs):
    labels = obs['cell_type1'].tolist()
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels) 
    return labels
def getTrainData(data):
    features_all = data.T
    features = np.log1p(features_all) 
    features = features/np.max(features)
    features, geneind = utils.high_var_npdata(features, num=args.num_gene, ind=1)
    print("--------- percent of zeros in subset:",np.sum(features == 0)/features.shape[0]/features.shape[1])
    return features.T

def load(dataPath, dataset):
    filename = dataPath + "/" + dataset + "/data.h5"
    if dataset == "usoskin":
        data, obs, var, uns = preprocessH5.preprocess(filename, normalize=False, sparsify = False, skip_exprs = False)
    else:
        data, obs, var, uns = preprocessH5.preprocess(filename, sparsify = False, skip_exprs = False)
        
    train_data = getTrainData(data)
    labels = getEncodeLabels(obs)
       
    
    if dataset == "Zhengsorted":
        train_X, test_X, train_y, test_y = utils.getSubset(train_data, labels, args.partDataset, args.testSize)
    else:
        test_X, test_y = train_data, labels
    
    data = utils.load_data(test_X, test_y)
    trainLoader_preTrainAE, testLoader_preTrainAE = utils.getTorchLoader(data.x, data.x, data.y, data.y, args.batchsize)    

    return data, trainLoader_preTrainAE
            



def testing(dataPath, dataset, save, **kwargs):
#    print("kwargs.items:")
#    print(kwargs.items())
    
    data, trainLoader_preTrainAE= load(dataPath, dataset)
    
    start_time = time.time()
    if args.tag == "DGC" or args.tag == "DGCNb":
        if args.tag == "DGC" or args.tag == "DGCNb":
            model_ae = train.pretrain_ae(trainLoader_preTrainAE, **kwargs)        
        p,q,z,likelihood, x_bar,tempvals = train.train_dgc(data, **kwargs)
        
    end_time = time.time()
    running_time = end_time - start_time
    
    pred_p = torch.argmax(p, 1).detach().numpy()
    pred_q = torch.argmax(q, 1).detach().numpy()
    pred_z = torch.argmax(z, 1).detach().numpy()
    embbeding = tempvals[0].detach().numpy()
    
    p = p.detach().numpy()
    q = q.detach().numpy()
    z = z.detach().numpy()
    
    results = utils.getResults(data.x, data.y, pred_p,pred_q,pred_z)

    if save:
        np.savetxt("../figures/sigDGCNb/"+ args.dataset +"_time_sigDGCNb"+str(args.num_gene)+"g_"+str(args.epochs)+"ep.txt", np.array([running_time]))
        pd.DataFrame(p).to_csv("../figures/sigDGCNb/"+args.dataset+"_sigDGCNb_pred_labelsP"+str(args.num_gene)+"g_"+str(args.epochs)+"ep.csv")
        pd.DataFrame(q).to_csv("../figures/sigDGCNb/"+args.dataset+"_sigDGCNb_pred_labelsQ"+str(args.num_gene)+"g_"+str(args.epochs)+"ep.csv")
        pd.DataFrame(z).to_csv("../figures/sigDGCNb/"+args.dataset+"_sigDGCNb_pred_labelsZ"+str(args.num_gene)+"g_"+str(args.epochs)+"ep.csv")
        pd.DataFrame(embbeding).to_csv("../figures/sigDGCNb/"+args.dataset+"_sigDGCNb_embedding"+str(args.num_gene)+"g_"+str(args.epochs)+"ep.csv")
        results.to_csv("../figures/sigDGCNb/"+args.dataset+"_sigDGCNb_res"+str(args.num_gene)+"g_"+str(args.epochs)+"ep.csv")
    
    return results

def init():
    device = torch.device("cuda" if args.cuda else "cpu")

    data, trainLoader_preTrainAE= load(args.dataPath, args.dataset)    
    args.n_clusters = len(np.unique(data.y))
    args.n_input = args.num_gene  
    
    kwargs = {"lr": args.lr,
          "epochs": args.epochs,
          "aeepochs": args.aeepochs,
          "lambda1": args.lambda1,
          "pretrain_path": args.pretrain_path, 
          "layer1": args.layer1, 
          "layer2": args.layer2, 
          "layer3": args.layer3, 
          "neighborK": args.neighborK,
          "n_input": args.num_gene,
          "n_z": args.n_z,
          "n_clusters":args.n_clusters,
          "distribution": args.distribution,
          "tag": args.tag,
          "device": device
          } 
    return kwargs

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    np.random.seed(seed)    
    random.seed(seed)    
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    
    print(args) 
    seeds = [0, 20, 200]
    resultsAll = pd.DataFrame()    
    for seed in seeds:
        setup_seed(seed)
        kwargs = init()
        results = testing(args.dataPath, args.dataset, save = False, **kwargs)
        resultsAll = pd.concat((resultsAll, results), 0)
    
    resultsAll.index = seeds    
    print(resultsAll.mean(0))
    print(resultsAll.std(0))
    
#    resultsAll.to_csv(args.dataPath + "/seeds_" + args.dataset +  "_500g.csv")





 
