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
import pdb

class GCNDataset(object):
    '''
    The graph object needed for Edge GCN
    '''

    def __init__(self,dataname):
        self.name=dataname
        self.X=None  #Node Features Matrix shape (n_node,nb_feat)
        self.E=None  #Edge Matrix E[:,0] input_node, E[:,1] output_node, the remaining columns are edge features
        self.A=None  #Adjanceny Matrix
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

    #TODO Remove
    def compute_EA(self):
        '''
        Compute the Edge Adjanceny Matrix
        :return:
        '''
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


    def compute_NodeEdgeMat(self):
        '''
        initialize matrix S and T where S indicates the source node for edges and T indicates the target node for edges
        :return:
        '''
        nb_edge = self.E.shape[0]
        nb_node = self.X.shape[0]

        edge_list = range(nb_edge)
        #TODO speed up
        S = sp.coo_matrix( (np.ones(nb_edge),([ int(i) for i in self.E[:,0]],edge_list)), shape=(nb_node, nb_edge))
        T = sp.coo_matrix( (np.ones(nb_edge),([ int(i) for i in self.E[:,1]],edge_list)), shape=(nb_node, nb_edge))

        Sindices = np.array([[i, j] for i, j in zip(self.E[:,0],edge_list)], dtype='int64')
        Tindices = np.array([[j, i] for i, j in zip(self.E[:,1], edge_list)],dtype='int64')
        self.S=S
        self.Sind=Sindices
        self.T=T
        self.Tind=Tindices
        self.F=self.E[:,2:]
        return S,T



    def compute_NA(self):
        '''
        Compute a normalized adjacency matrix as in the original GCN Model
        :return:
        '''
        # Here we add 1.0 and the identity matrix to account for the self-loop
        degree_vect=np.asarray(self.A.sum(axis=1)).squeeze()
        #print(degree_vect)
        Dinv_ = np.diag(np.power(1.0+degree_vect,-0.5))
        self.Dinv=Dinv_

        #TODO Check how this dot deals with the matrix multiplication with sparse matrix
        N = np.dot(Dinv_, self.A + np.identity(self.A.shape[0]).dot(Dinv_))
        self.NA=N

    def normalize(self):
        '''
        Normalize the node feature matrix
        :return:
        '''
        l2_normalizer =Normalizer()
        self.X=l2_normalizer.fit_transform(self.X)




    @staticmethod
    def load_transkribus_pickle(pickle_fname,is_zipped=True,sym_edge=True):
        '''
        Loas existing pickle file used with CRF in the Transkribus project
        :param pickle_fname:
        :param is_zipped:
        :param sym_edge:
        :return:
        '''
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
            graph.compute_NodeEdgeMat()
        return gcn_list

    @staticmethod
    def load_snake_pickle(pickle_fname):
        '''
        Load the snake pickle
        :param pickle_fname:
        :return:
        '''
        #return GCNDataset.load_transkribus_pickle(pickle_fname,is_zipped=False)
        #return GCNDataset.load_transkribus_pickle(pickle_fname, is_zipped=False,sym_edge=False)

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

            EF=[]
            A_list_i=[]
            A_list_j = []
            for node_s,node_t,edge_feat in zip(edge[:,0],edge[:,1],ef):
                A_list_i.append(node_s)
                A_list_j.append(node_t)

                #Right Edge
                if edge_feat[0]==1:
                    #EF.append([1,0])
                    EF.append([node_s,node_t,1,0,0,0])
                    #Now add the reverse edge

                    A_list_i.append(node_t)
                    A_list_j.append(node_s)

                    EF.append([node_t,node_s,0, 1, 0, 0])

                elif edge_feat[1]==1:
                    #EF.append([0, 1])
                    EF.append([node_s,node_t,0, 0, 1, 0])

                    A_list_i.append(node_t)
                    A_list_j.append(node_s)

                    EF.append([node_t,node_s,0, 0, 0, 1])
                else:
                    print(edge_feat)
                    raise ValueError('Invalid Edge Feature')

            A = sp.coo_matrix((np.ones(len(EF)), (A_list_i, A_list_j)), shape=(nb_node, nb_node))
            graph.A = A


            graph.E = np.array(EF,dtype='f')  # check order
            #print(graph.E)
            gcn_list.append(graph)
            graph.compute_EA()
            graph.compute_NA()
            #print('Edge Features ,shape',graph.E.shape)
            # graph.normalize()


        return gcn_list

    @staticmethod
    def merge_graph(graph_a,graph_b):
        '''
        Merge two graph
        :param graph_a:
        :param graph_b:
        :return:
        '''
        graph = GCNDataset('Union_'+graph_a.name+':'+graph_b.name)

        graph.X = np.vstack((graph_a.X,graph_b.X))

        #Change Index of node in the Edge matrix
        Ea=np.array(graph_a.E)
        nb_node_a=graph_a.X.shape[0]
        Eb=np.array(graph_b.E)
        Eb[:,0] = Eb[:,0] +nb_node_a
        Eb[:,1] = Eb[:,1] +nb_node_a

        #print(Ea.shape)
        #print(Eb.shape)
        graph.E =np.vstack((Ea,Eb))

        nb_node_total = graph.X.shape[0]

        graph.compute_EA()
        graph.A=np.diag(np.ones(nb_node_total))
        graph.NA =np.diag(np.ones(nb_node_total))
        #print('Warning ....')
        #print('Normalized Adcency not set')
        #print('Adjacency not set')


        graph.Y=np.vstack((graph_a.Y,graph_b.Y))

        return graph




    @staticmethod
    def load_test_pickle(pickle_fname,nb_classes, is_zipped=True, sym_edge=True):
        '''
        Load a test pickle file, a list of X
        :param pickle_fname:
        :param is_zipped:
        :param sym_edge:
        :return:
        '''
        gcn_list = []

        if is_zipped:
            f = gzip.open(pickle_fname, 'rb')
        else:
            f = open(pickle_fname, 'rb')
        if PY3:
            Z = pickle.load(f, encoding='latin1')
        else:
            Z = pickle.load(f)

        lX = Z

        graph_id = 0

        #Aie
        #I have not storred the LabelBinarizer
        for lx in lX:
            nf = lx[0]
            edge = lx[1]
            ef = lx[2]

            graph = GCNDataset(str(graph_id))
            nb_node = nf.shape[0]
            graph.X = nf
            graph.Y = -np.ones((nb_node,nb_classes),dtype='i')
            # We are making the adacency matrix here

            # Correct this edge should be swap ..
            # This is not correct for edges, we should add edge swap
            # A=sp.coo_matrix((np.ones(edge.shape[0]),(edge[:,0],edge[:,1])), shape=(nb_node, nb_node))
            # TODO Check this then
            if sym_edge:
                A1 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])), shape=(nb_node, nb_node))
                A2 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 1], edge[:, 0])), shape=(nb_node, nb_node))
                graph.A = A1 + A2
            else:
                A1 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])), shape=(nb_node, nb_node))
                graph.A = A1

            edge_normalizer = Normalizer()
            # Normalize EA
            efn = edge_normalizer.fit_transform(ef)
            # Duplicate Edge
            edge_swap = np.array(edge)
            edge_swap[:, 0] = edge[:, 1]
            edge_swap[:, 1] = edge[:, 0]

            E0 = np.hstack([edge, ef])  # check order
            E1 = np.hstack([edge_swap, ef])  # check order

            if sym_edge:
                graph.E = np.vstack([E0, E1])  # check order
            else:
                graph.E = E0
            gcn_list.append(graph)
            graph.compute_EA()
            graph.compute_NA()
            # graph.normalize()
            graph.compute_NodeEdgeMat()
        return gcn_list

    def load_transkribus_reverse_arcs_pickle(pickle_fname,pickle_ra_fname, is_zipped=True):
        '''
        Loas existing pickle file used with CRF in the Transkribus project
        :param pickle_fname:
        :param is_zipped:
        :param sym_edge:
        :return:
        '''
        gcn_list = []

        if is_zipped:
            f = gzip.open(pickle_fname, 'rb')
            g = gzip.open(pickle_ra_fname, 'rb')
        else:
            f = open(pickle_fname, 'rb')
            g = open(pickle_ra_fname, 'rb')

        if PY3:
            Z = pickle.load(f, encoding='latin1')
            Zr = pickle.load(g, encoding='latin1')
        else:
            Z = pickle.load(f)
            Zr = pickle.load(g)

        lX = Z[0]
        lY = Z[1]

        lX_reversed = Zr[0]
        lY_reversed = Zr[1]

        graph_id = 0

        lb = LabelBinarizer()
        lys = []

        for _, ly in zip(lX, lY):
            lys.extend(list(ly))

        lb.fit(lys)

        for lx, ly ,lxr,lyr in zip(lX, lY,lX_reversed,lY_reversed):
            nf = lx[0]
            edge = lx[1]
            ef = lx[2]

            nfr   = lxr[0]
            edger = lxr[1]
            efr    = lxr[2]

            diff_node_features = (nf-nfr).sum()
            assert(diff_node_features<1e-5)

            #assert edge swap on node source -target
            edge_test1 = np.sum(edge[:,1] ==edger[:,0]) == ef.shape[0]
            edge_test2 = np.sum( edge[:, 0] == edger[:, 1]) == ef.shape[0]

            assert(edge_test1)
            assert(edge_test2)

            graph = GCNDataset(str(graph_id))
            graph.X = nf
            graph.Y = lb.transform(ly)
            # We are making the adacency matrix here
            nb_node = nf.shape[0]

            A1 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])), shape=(nb_node, nb_node))
            A2 = sp.coo_matrix((np.ones(edger.shape[0]), (edger[:, 0], edger[:, 1])), shape=(nb_node, nb_node))
            graph.A = A1 + A2


            edge_normalizer = Normalizer()
            # Normalize EA
            efn = edge_normalizer.fit_transform(ef)

            E0 = np.hstack([edge, ef])  # check order
            E1 = np.hstack([edger, efr])  # check order


            graph.E = np.vstack([E0, E1])  # check order

            gcn_list.append(graph)
            graph.compute_EA()
            graph.compute_NA()
            # graph.normalize()
            graph.compute_NodeEdgeMat()

        f.close()
        g.close()
        return gcn_list