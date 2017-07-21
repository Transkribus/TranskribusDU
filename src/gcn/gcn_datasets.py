
'''
Graph Convolution Network Datasets
'''
import pickle
import numpy as np
class GCNDataset(object):

    def __init__(self,dataname):
        self.name=dataname
        self.X=None  #Correspond F_n^0
        self.E=None  #Correspond F_e^0 Edge Feature Matrix
        self.A=None       #Correspond to N in JL notation, Adjacency Matrix
        self.Y=None

    def load_pickle(self,pickle_fname):
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




