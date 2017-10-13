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

    def compute_NA(self):
        '''
        Compute a normalized adjacency matrix as in the original GCN Model
        :return:
        '''
        # Should Compute that once for each graph
        # Here we add 1.0 and the identity matrix to account for the self-loop
        degree_vect=np.asarray(self.A.sum(axis=1)).squeeze()
        #print(degree_vect)
        Dinv_ = np.diag(np.power(1.0+degree_vect,-0.5))
        self.Dinv=Dinv_
        #print(Dinv_.shape)
        #print(self.A.shape)
        #Check how this dot deals with the matrix multiplication with sparse matrix
        #TODO
        N = np.dot(Dinv_, self.A + np.identity(self.A.shape[0]).dot(Dinv_))
        self.NA=N

    def normalize(self):
        l2_normalizer =Normalizer()
        self.X=l2_normalizer.fit_transform(self.X)

        edge_normalizer=Normalizer()
        #Normalize EA
        self.X=l2_normalizer.fit_transform(self.X)




    @staticmethod
    def load_transkribus_pickle(pickle_fname,is_zipped=True,sym_edge=True):
        gcn_list=[]

        if is_zipped:
            f=gzip.open(pickle_fname,'rb')
        else:
            f = open(pickle_fname, 'rb')
        if PY3:
            Z = pickle.load(f, encoding='latin1')
        else:
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
            #This is not correct for edges, we should add edge swap
            #A=sp.coo_matrix((np.ones(edge.shape[0]),(edge[:,0],edge[:,1])), shape=(nb_node, nb_node))
            #TODO Check this then
            if sym_edge:
                A1 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])), shape=(nb_node, nb_node))
                A2 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 1], edge[:, 0])), shape=(nb_node, nb_node))
                graph.A=A1+A2
            else:
                A1 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])), shape=(nb_node, nb_node))
                graph.A = A1

            edge_normalizer=Normalizer()
            #Normalize EA
            efn=edge_normalizer.fit_transform(ef)
            #Duplicate Edge
            edge_swap=np.array(edge)
            edge_swap[:,0]=edge[:,1]
            edge_swap[:,1]=edge[:,0]

            E0=np.hstack([edge,ef])#check order
            E1=np.hstack([edge_swap,ef])#check order

            if sym_edge:
                graph.E=np.vstack([E0,E1])#check order
            else:
                graph.E = E0
            gcn_list.append(graph)
            graph.compute_EA()
            graph.compute_NA()
            #graph.normalize()
        return gcn_list

    @staticmethod
    def load_snake_pickle(pickle_fname):

        #return GCNDataset.load_transkribus_pickle(pickle_fname,is_zipped=False)
        return GCNDataset.load_transkribus_pickle(pickle_fname, is_zipped=False,sym_edge=False)
        gcn_list = []


        f = open(pickle_fname, 'rb')
        if PY3:
            Z = pickle.load(f, encoding='latin1')
        else:
            Z = pickle.load(f)

        lX = Z[0]
        lY = Z[1]
        graph_id = 0

        lb = LabelBinarizer()
        lys = []

        for _, ly in zip(lX, lY):
            lys.extend(list(ly))

        lb.fit(lys)

        for lx, ly in zip(lX, lY):
            nf = lx[0]
            edge = lx[1]
            ef = lx[2]

            graph = GCNDataset(str(graph_id))
            graph.X = nf
            graph.Y = lb.transform(ly)
            # We are making the adacency matrix here
            nb_node = nf.shape[0]
            # Correct this edge should be swap ..
            # This is not correct for edges, we should add edge swap
            # A=sp.coo_matrix((np.ones(edge.shape[0]),(edge[:,0],edge[:,1])), shape=(nb_node, nb_node))


            #Now build the edge feature matrix
            #
            #
            #Tmp
            '''
            EF=[]
            A_list_i=[]
            A_list_j = []
            for node_s,node_t,edge_feat in zip(edge[:,0],edge[:,1],ef):
                A_list_i.append(node_s)
                A_list_j.append(node_t)

                #Right Edge
                if edge_feat[0]==1:
                    #EF.append([1,0])
                    EF.append([1,0,0,0])
                    #Now add the reverse edge

                    A_list_i.append(node_t)
                    A_list_j.append(node_s)

                    EF.append([0, 1, 0, 0])

                elif edge_feat[1]==1:
                    #EF.append([0, 1])
                    EF.append([0, 0, 1, 0])

                    A_list_i.append(node_t)
                    A_list_j.append(node_s)

                    EF.append([0, 0, 0, 1])
                else:
                    print(edge_feat)
                    raise ValueError('Invalid Edge Feature')

            A = sp.coo_matrix((np.ones(len(EF)), (A_list_i, A_list_j)), shape=(nb_node, nb_node))
            graph.A = A
            '''

            graph.E = np.array(EF,dtype='f')  # check order
            print(graph.E)
            gcn_list.append(graph)
            graph.compute_EA()
            graph.compute_NA()
            # graph.normalize()


        return gcn_list






