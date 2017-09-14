import sys
'''
Graph Convolution Network Datasets
'''
import pickle
from sklearn.preprocessing import LabelBinarizer,Normalizer
import numpy as np
PY3 = sys.version_info[0] == 3
import gzip
import scipy.sparse as sp
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

    def compute_EA(self):
        edge_dim= self.E.shape[1]-2
        nb_node=self.X.shape[0]
        EA =np.zeros((edge_dim,(nb_node*nb_node)),dtype=np.float32)

        i_list =[]
        j_list=[]
        for x,y in zip(self.E[:,0],self.E[:,1]):
            i_list.append(int(x))
            j_list.append(int(y))

        for i in range(edge_dim):
            idim_mat =sp.coo_matrix((self.E[:,i+2],(i_list,j_list)), shape=(nb_node, nb_node))
            D= np.asarray(idim_mat.todense()).squeeze()
            EA[i,:]=np.reshape(D,-1)
        self.EA=EA

    def compute_N(self):
        #Fixe
        Dinv_ = np.diag(np.power(1.0+self.A.sum(axis=1),-0.5))

        #Dinv  =tf.constant(Dinv_)
        #A     =tf.constant(self.dataset.A)

        #N=tf.constant(np.dot(Dinv_,self.dataset.A+np.identity(self.dataset.A.shape[0]).dot(Dinv_)),dtype=np.float32)
        #N=np.dot(Dinv_,self.A+np.identity(self.A.shape[0]).dot(Dinv_))
        self.NA=Dinv_

    def normalize(self):
        l2_normalizer =Normalizer()
        self.X=l2_normalizer.fit_transform(self.X)

        edge_normalizer=Normalizer()
        #Normalize EA
        self.X=l2_normalizer.fit_transform(self.X)




    @staticmethod
    def load_transkribus_pickle(pickle_fname):
        gcn_list=[]

        f=gzip.open(pickle_fname,'rb')
        Z=pickle.load(f)
        lX=Z[0]
        lY=Z[1]
        graph_id=0

        lb=LabelBinarizer()
        lys=[]

        for _,ly in zip(lX,lY):
            lys.extend(list(ly))

        lb.fit(lys)

        for lx,ly in zip(lX,lY):
            nf=lx[0]
            edge=lx[1]
            ef =lx[2]

            graph = GCNDataset(str(graph_id))
            graph.X=nf
            graph.Y=lb.transform(ly)
            #We are making the adacency matrix here
            nb_node=nf.shape[0]
            #Correct this edge should be swap ..
            A=sp.coo_matrix((np.ones(edge.shape[0]),(edge[:,0],edge[:,1])), shape=(nb_node, nb_node))
            graph.A=A

            edge_normalizer=Normalizer()
            #Normalize EA
            efn=edge_normalizer.fit_transform(ef)
            #Duplicate Edge
            edge_swap=np.array(edge)
            edge_swap[:,0]=edge[:,1]
            edge_swap[:,1]=edge[:,0]

            E0=np.hstack([edge,ef])#check order
            E1=np.hstack([edge_swap,ef])#check order

            graph.E=np.vstack([E0,E1])#check order
            gcn_list.append(graph)
            graph.compute_EA()
            graph.compute_N()
            #graph.normalize()
        return gcn_list



