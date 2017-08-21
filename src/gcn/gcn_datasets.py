import sys
'''
Graph Convolution Network Datasets
'''
import pickle
import numpy as np
PY3 = sys.version_info[0] == 3

class GCNDataset(object):

    def __init__(self,dataname):
        self.name=dataname
        self.X=None  #Correspond F_n^0 ; node features
        self.E=None  #Correspond F_e^0 Edge Feature Matrix E[:,0] input_node, E[:,1] output_node, the remaining columns are edge features
        self.A=None  #Correspond to N in JL notation, Adjacency Matrix
        self.Y=None  #Labels for the node in 1 sall format ie is a matrix n_node time n_label

    def load_pickle(self,pickle_fname):
        if PY3:
            f=open(pickle_fname, 'rb')
            L = pickle.load(f, encoding='latin1')
        else:
            f=open(pickle_fname,'rb')
            L=pickle.load(f)

        self.X=L[0]
        self.Y=L[1]
        self.A=L[2]
        self.E = L[3]

    def print_stats(self):
        print('X:',self.X.shape)
        print('Y:',self.Y.shape,' class distrib:',np.bincount(np.argmax(self.Y,axis=1)))
        print('A:',self.A.shape)
        print('E:',self.E.shape)




