# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import random
__author__ = 'sclincha'
import operator


import sys,os
sys.path.append('../..')

import numpy as np
import unittest
import pickle
import tensorflow as tf
from tensorflow.python.client import timeline
import scipy.sparse as sp

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from IPython import embed
import sklearn.metrics
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer,Normalizer
from sklearn.linear_model import LogisticRegression
from gcn.gcn_datasets import GCNDataset

from gcn.gcn_models import DummyGCNModel,GCNModelGraphList

def make_fake_gcn_dataset():
    '''
    Build a fake graph dataset from the iris dataset
    
    Consider each point as a node, and for each node add an edge  with a point of the same class
    and an edge for a point of different class
    
    :return: 
    '''
    #TODO Fix this this is not compatible with version 3.0
    X,Y=load_iris(return_X_y=True)


    #Path + edge random edge
    lb=LabelBinarizer()
    Yel=lb.fit_transform(Y)

    nb_items =X.shape[0]
    print(Y)
    D=sklearn.metrics.pairwise_distances(X)


    #Now find closest point with same labels and different labels
    #Square matrix indicating if two elements i,j have the same labels
    Yelel = np.dot(Yel,Yel.T)
    assert(Yelel.max()==1.0)
    DsameLabel = D*Yelel
    DdiffLabel = D*(1.0-Yelel)

    #print(D) # Prendre le plus petit element qui ne soit pas 0 # FIXME
    neareast_same_class = np.argsort(DsameLabel,axis=1)[:,-2] #Neareast Point is itself so take the snd nearest point

    neareast_diff_class = np.argsort(DdiffLabel,axis=1)[:,-1]

    DsL = np.zeros(nb_items)
    DdL = np.zeros(nb_items)

    for i in xrange(nb_items):
        DsL[i]=DsameLabel[i,neareast_same_class[i]]
        DdL[i]=DdiffLabel[i,neareast_diff_class[i]]

    #AdjancyMatrix
    A = np.zeros((nb_items,nb_items))
    edge_dim =3
    EdgeMatrix = np.zeros((2*nb_items,2+edge_dim))
    print(DsL)
    print(DdL)

    for i in xrange(nb_items):
        A[i,neareast_same_class[i]]=1.0
        A[i,neareast_diff_class[i]]=1.0

        EdgeMatrix[2*i,0] =i
        EdgeMatrix[2*i,1] =neareast_same_class[i]

        EdgeMatrix[2*i,2+0] =1.0
        EdgeMatrix[2*i,2+1] =0.0
        EdgeMatrix[2*i,2+2] =np.random.randn()

        EdgeMatrix[2*i+1,0] =i
        EdgeMatrix[2*i+1,1] =neareast_diff_class[i]
        EdgeMatrix[2*i+1,2+0]=0.0
        EdgeMatrix[2*i+1,2+1]=1.0
        EdgeMatrix[2*i+1,2+2]=np.random.randn()


    Z=[X,Yel,A,EdgeMatrix]

    f=open('iris_graph.pickle','wb')
    pickle.dump(Z,f)
    f.close()
    print('Dataset saved')
    embed()



class UT_gcn(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(UT_gcn, self).__init__(*args, **kwargs)

    def test_01_load(self):
        dataset=GCNDataset('UT_iris_0')
        dataset.load_pickle('iris_graph.pickle')
        dataset.print_stats()
        return True

    def test_02_createmodel(self):
        dataset=GCNDataset('UT_iris_0')
        dataset.load_pickle('iris_graph.pickle')
        dataset.print_stats()

        gcn_model = DummyGCNModel(dataset, num_layers=1, learning_rate=0.1)
        #gcn_model.activation=tf.nn.softmax
        gcn_model.activation=tf.nn.relu
        #gcn_model.activation=tf.nn.sigmoid
        gcn_model.create_model()

        print(dataset.X)
        print(dataset.Y)
        logit_model = DummyGCNModel(dataset, num_layers=0, learning_rate=0.5, mu=0.0)
        logit_model.activation=tf.nn.sigmoid
        logit_model.create_model()

        with tf.Session() as session:
            session.run([gcn_model.init,logit_model.init])
            gcn_model.train(session,n_iter=200)
            print('Logit Model')
            logit_model.train(session,n_iter=200)
        #Bug was due to two activation function of the logit ....


        lr=LogisticRegression()
        lr.fit(dataset.X,np.argmax(dataset.Y,axis=1))
        acc=lr.score(dataset.X,np.argmax(dataset.Y,axis=1))
        print('Accuracy LR',acc)

    def test_03_buildEdgeMat(self):
        #Test of building the EdgeMatrix representation needed for learning the edge feature
        dataset=GCNDataset('UT_iris_0')
        dataset.load_pickle('iris_graph.pickle')
        dataset.print_stats()

        A=dataset.A
        E=dataset.E
        print(A.shape)
        print(E.shape)
        nb_node = A.shape[0]
        edge_dim= dataset.E.shape[1]-2 #Preprocess That

        EA =np.zeros((edge_dim,(nb_node*nb_node)),dtype=np.float32)
        #edge_idx=list(zip(E[:,0],E[:,1]))
        #edge_idx = [(int(x[0]),int(x[1])) for x in edge_idx]
        i_list =[]
        j_list=[]
        for x,y in zip(E[:,0],E[:,1]):
            i_list.append(int(x))
            j_list.append(int(y))

        for i in range(edge_dim):
            #Build a adjecency sparse matrix for the i_dim of the edge
            #pdb.set_trace()
            idim_mat =sp.coo_matrix((E[:,i+2],(i_list,j_list)), shape=(nb_node, nb_node))

            D= np.asarray(idim_mat.todense()).squeeze()
            EA[i,:]=np.reshape(D,-1)

        print(EA.shape)
        print(E[0:4,:])

        #Print rep of first edge
        print('Representation of first node')
        print(EA[:,int(E[0,1])])
        print(EA[:,int(E[1,1])])
        #Print rep of first edge
        print('Representation of second node')
        print(EA[:,nb_node+int(E[2,1])])
        print(EA[:,nb_node+int(E[3,1])])


        Wedge  = tf.Variable(tf.ones([1,edge_dim], dtype=np.float32, name='Wedge'))
        tf_EA=tf.constant(EA)

        Em =(tf.matmul(Wedge,tf_EA))
        Z=tf.reshape(Em,(nb_node,nb_node))

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            Em_=session.run(Em)
            Z_=session.run(Z)
            print(Em_.shape)
            print(Z_.shape)
            #TODO check somestuff on Z

    def test_04_learn_edge(self):
        dataset=GCNDataset('UT_iris_0')
        dataset.load_pickle('iris_graph.pickle')
        dataset.print_stats()

        gcn_model = DummyGCNModel(dataset, num_layers=1, learning_rate=0.1)
        #gcn_model.activation=tf.nn.softmax
        gcn_model.activation=tf.nn.relu
        #gcn_model.activation=tf.nn.sigmoid
        gcn_model.create_model()


        edge_model =DummyGCNModel(dataset, num_layers=1, learning_rate=0.1)
        #gcn_model.activation=tf.nn.softmax
        edge_model.activation=tf.nn.relu
        edge_model.learn_edge=True
        edge_model.create_model()


        nb_iter=300
        with tf.Session() as session:
            session.run([gcn_model.init,edge_model.init])
            gcn_model.train(session,n_iter=nb_iter)
            print('Edge model')
            edge_model.train(session,n_iter=nb_iter)
            we =session.run(edge_model.Wedge)
            print(we)
        #Bug was due to two activation function of the logit ....

    def test_05_load_jl_pickle(self):

        pickle_fname='/opt/project/read/testJL/TABLE/abp_models/abp_CV10_fold_10_tlXlY_trn.pkl'
        gcn_graph= GCNDataset.load_transkribus_pickle(pickle_fname)
        print(len(gcn_graph),'loaded graph')

    def test_06_create_graphlist_model(self):
        pickle_fname='/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph= GCNDataset.load_transkribus_pickle(pickle_fname)


        gcn_graph_test=[gcn_graph[0],gcn_graph[10],gcn_graph[53]]
        node_dim=gcn_graph[0].X.shape[1]
        edge_dim=gcn_graph[0].E.shape[1] -2.0
        nb_class=gcn_graph[0].Y.shape[1]

        #__init__(self,node_dim,edge_dim,nb_classes,num_layers=1,learning_rate=0.1,mu=0.1):
        gcn_model =GCNModelGraphList(node_dim,edge_dim,nb_class,num_layers=1,learning_rate=0.001,mu=0.0)

        gcn_model.create_model()


        #Make Big Graph ....
        total_node=sum([g.X.shape[0] for g in gcn_graph])
        total_nf = np.vstack([g.X for g in gcn_graph])
        total_y  = np.vstack([g.Y for g in gcn_graph])
        total_EA  =np.hstack([g.EA for g in gcn_graph])

        #pdb.set_trace()

        nb_iter=300
        with tf.Session() as session:
            session.run([gcn_model.init])
            #Sample each graph
            #random
            for i in range(1000):
                #random.shuffle(gcn_graph_test)

                if i%10==0:
                    print('Epoch',i)
                    mean_acc=[]
                    for g in gcn_graph_test:
                        print('G Stats #node,#edge',g.X.shape[0],g.E.shape[0])
                        acc=gcn_model.test(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA)
                        mean_acc.append(acc)
                    print('     Mean Accuracy',np.mean(mean_acc))
                else:
                    for g in gcn_graph_test:
                        gcn_model.train(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA,n_iter=1)


            mean_acc=[]
            for g in gcn_graph_test:
                acc=gcn_model.test(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA)
                mean_acc.append(acc)
            print('Mean Accuracy',np.mean(mean_acc))

    def test_07_deep_model(self):
        pickle_fname='/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph= GCNDataset.load_transkribus_pickle(pickle_fname)


        gcn_graph_test=[gcn_graph[7],gcn_graph[13],gcn_graph[21]]
        node_dim=gcn_graph[0].X.shape[1]
        edge_dim=gcn_graph[0].E.shape[1] -2.0
        nb_class=gcn_graph[0].Y.shape[1]

        #__init__(self,node_dim,edge_dim,nb_classes,num_layers=1,learning_rate=0.1,mu=0.1):
        gcn_model =GCNModelGraphList(node_dim,edge_dim,nb_class,num_layers=2,learning_rate=0.001,mu=0.0,node_indim=5)
        #gcn_model =GCNModelGraphList(node_dim,edge_dim,nb_class,num_layers=1,learning_rate=0.001,mu=0.0,node_indim=-1)

        gcn_model.create_model()

        #pdb.set_trace()

        nb_iter=300
        with tf.Session() as session:
            session.run([gcn_model.init])
            #Sample each graph
            #random
            for i in range(2000):
                #random.shuffle(gcn_graph_test)

                if i%10==0:
                    print('Epoch',i)
                    mean_acc=[]
                    for g in gcn_graph_test:
                        print('G Stats #node,#edge',g.X.shape[0],g.E.shape[0])
                        acc=gcn_model.test(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA)
                        mean_acc.append(acc)
                    print('     Mean Accuracy',np.mean(mean_acc))
                else:
                    for g in gcn_graph_test:
                        gcn_model.train(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA,n_iter=1)


            mean_acc=[]
            for g in gcn_graph_test:
                acc=gcn_model.test(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA)
                mean_acc.append(acc)
            print('Mean Accuracy',np.mean(mean_acc))

    def test_08_deep_stack(self):
        pickle_fname='/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph= GCNDataset.load_transkribus_pickle(pickle_fname)


        gcn_graph_test=[gcn_graph[7],gcn_graph[13],gcn_graph[21]]
        node_dim=gcn_graph[0].X.shape[1]
        edge_dim=gcn_graph[0].E.shape[1] -2.0
        nb_class=gcn_graph[0].Y.shape[1]

        #__init__(self,node_dim,edge_dim,nb_classes,num_layers=1,learning_rate=0.1,mu=0.1):
        #gcn_model =GCNModelGraphList(node_dim,edge_dim,nb_class,num_layers=2,learning_rate=0.01,mu=0.0,node_indim=5)
        gcn_model =GCNModelGraphList(node_dim,edge_dim,nb_class,num_layers=3,learning_rate=0.01,mu=0.0,node_indim=-1)
        #gcn_model =GCNModelGraphList(node_dim,edge_dim,nb_class,num_layers=1,learning_rate=0.001,mu=0.0,node_indim=-1)
        gcn_model.stack_instead_add=True
        gcn_model.create_model()

        #pdb.set_trace()

        nb_iter=300
        with tf.Session() as session:
            session.run([gcn_model.init])
            #Sample each graph
            #random
            for i in range(2000):
                #random.shuffle(gcn_graph_test)

                if i%10==0:
                    print('Epoch',i)
                    mean_acc=[]
                    for g in gcn_graph_test:
#                        print('G Stats #node,#edge',g.X.shape[0],g.E.shape[0])
                        acc=gcn_model.test(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA)
                        mean_acc.append(acc)
                    print('     Mean Accuracy',np.mean(mean_acc))
                else:
                    for g in gcn_graph_test:
                        gcn_model.train(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA,n_iter=1)


            mean_acc=[]
            for g in gcn_graph_test:
                acc=gcn_model.test(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA)
                mean_acc.append(acc)
            print('Mean Accuracy',np.mean(mean_acc))

    def test_09_conv_reshape(self):

        dataset=GCNDataset('UT_iris_0')
        dataset.load_pickle('iris_graph.pickle')
        dataset.print_stats()

        A=dataset.A
        E=dataset.E
        print(A.shape)
        print(E.shape)
        nb_node = A.shape[0]
        edge_dim= dataset.E.shape[1]-2 #Preprocess That

        EA =np.zeros((edge_dim,(nb_node*nb_node)),dtype=np.float32)
        #edge_idx=list(zip(E[:,0],E[:,1]))
        #edge_idx = [(int(x[0]),int(x[1])) for x in edge_idx]
        i_list =[]
        j_list=[]
        for x,y in zip(E[:,0],E[:,1]):
            i_list.append(int(x))
            j_list.append(int(y))

        for i in range(edge_dim):
            #Build a adjecency sparse matrix for the i_dim of the edge
            #pdb.set_trace()
            idim_mat =sp.coo_matrix((E[:,i+2],(i_list,j_list)), shape=(nb_node, nb_node))

            D= np.asarray(idim_mat.todense()).squeeze()
            EA[i,:]=np.reshape(D,-1)

        print(EA.shape)
        print(E[0:4,:])

        #Print rep of first edge
        print('Representation of first node')
        print(EA[:,int(E[0,1])])
        print(EA[:,int(E[1,1])])
        #Print rep of first edge
        print('Representation of second node')
        print(EA[:,nb_node+int(E[2,1])])
        print(EA[:,nb_node+int(E[3,1])])


        nconv=2
        w_edge_conv=np.ones((nconv,edge_dim))
        #w_edge_conv[1,:]=-1.0

        Wedge  = tf.Variable(w_edge_conv, dtype=np.float32, name='Wedge')
        Wn     = tf.Variable(tf.eye(dataset.X.shape[1],dtype=tf.float32), dtype=np.float32, name='Wnode')
        tf_EA=tf.constant(EA)

        Em =(tf.matmul(Wedge,tf_EA))

        #Looks ok but not noure
        Z=tf.reshape(Em,(nconv,nb_node,nb_node))

        H0= tf.matmul(tf.constant(dataset.X,dtype=tf.float32),Wn)
        #H=tf.matmul(Z,H0)
        #H=tf.matmul(H0,Z,transpose_a=True)
        #Safe way for loop
        Cops=[]
        for i in range(nconv):
            Hi=tf.matmul(Z[i],H0)
            Cops.append(Hi)

        H=tf.concat(Cops,1)

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            Em_=session.run(Em)
            Z_=session.run(Z)
            print(Em_.shape)
            print(Z_.shape)
            print(Z_[0,0,:])
            print(Z_[1,0,:])
            H_ = session.run(H)
            print(H_.shape)
            print(H_[0,:])
            print(H_[1,:])

    def test_10_nconv(self):
        pickle_fname='/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph= GCNDataset.load_transkribus_pickle(pickle_fname)


        gcn_graph_test=[gcn_graph[7],gcn_graph[13],gcn_graph[21]]
        node_dim=gcn_graph[0].X.shape[1]
        edge_dim=gcn_graph[0].E.shape[1] -2.0
        nb_class=gcn_graph[0].Y.shape[1]

        #__init__(self,node_dim,edge_dim,nb_classes,num_layers=1,learning_rate=0.1,mu=0.1):
        #gcn_model =GCNModelGraphList(node_dim,edge_dim,nb_class,num_layers=2,learning_rate=0.01,mu=0.0,node_indim=5)
        gcn_model =GCNModelGraphList(node_dim,edge_dim,nb_class,num_layers=1,learning_rate=0.01,mu=0.0,node_indim=-1,nconv_edge=10)
        #gcn_model =GCNModelGraphList(node_dim,edge_dim,nb_class,num_layers=1,learning_rate=0.001,mu=0.0,node_indim=-1)
        gcn_model.stack_instead_add=True
        gcn_model.create_model()

        #pdb.set_trace()

        nb_iter=300
        with tf.Session() as session:
            session.run([gcn_model.init])
            #Sample each graph
            #random
            for i in range(2000):
                #random.shuffle(gcn_graph_test)

                if i%10==0:
                    print('Epoch',i)
                    mean_acc=[]
                    for g in gcn_graph_test:
#                        print('G Stats #node,#edge',g.X.shape[0],g.E.shape[0])
                        acc=gcn_model.test(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA)
                        mean_acc.append(acc)
                    print('     Mean Accuracy',np.mean(mean_acc))
                else:
                    for g in gcn_graph_test:
                        gcn_model.train(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA,n_iter=1)


            mean_acc=[]
            for g in gcn_graph_test:
                acc=gcn_model.test(session,g.X.shape[0],g.X,g.EA,g.Y,g.NA)
                mean_acc.append(acc)
            print('Mean Accuracy',np.mean(mean_acc))

    def test_11_2layers(self):
            pickle_fname = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_1_tlXlY_trn.pkl'
            gcn_graph = GCNDataset.load_transkribus_pickle(pickle_fname)

            gcn_graph_test = [gcn_graph[8], gcn_graph[18], gcn_graph[29]]
            node_dim = gcn_graph[0].X.shape[1]
            edge_dim = gcn_graph[0].E.shape[1] - 2.0
            nb_class = gcn_graph[0].Y.shape[1]


            print('Node Dim',node_dim)
            print('Edge Dim',edge_dim)

            gcn_model = GCNModelGraphList(node_dim, edge_dim, nb_class, num_layers=2, learning_rate=0.01, mu=0.0,
                                          node_indim=-1, nconv_edge=5)
            # gcn_model =GCNModelGraphList(node_dim,edge_dim,nb_class,num_layers=1,learning_rate=0.001,mu=0.0,node_indim=-1)
            gcn_model.stack_instead_add = True
            gcn_model.create_model()

            # pdb.set_trace()

            nb_iter = 50
            with tf.Session() as session:
                session.run([gcn_model.init])
                # Sample each graph
                # random
                for i in range(500):
                    # random.shuffle(gcn_graph_test)

                    if i % 10 == 0:
                        print('Epoch', i)
                        mean_acc = []
                        for g in gcn_graph_test:
                            #                        print('G Stats #node,#edge',g.X.shape[0],g.E.shape[0])
                            acc = gcn_model.test(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA)
                            mean_acc.append(acc)
                        print('     Mean Accuracy', np.mean(mean_acc))
                    else:
                        for g in gcn_graph_test:
                            gcn_model.train(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA, n_iter=1)

                mean_acc = []
                for g in gcn_graph_test:
                    acc = gcn_model.test(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA)
                    mean_acc.append(acc)
                print('Mean Accuracy', np.mean(mean_acc))

    def test_12_merge_graph(self):
        #3 nodes
        Xa=np.array([ [1.0,2.0],[6.3,1.0],[4.3,-2.0]])
        Ea=np.array([ [0,1,1,0.5],[1,2,0,0.2]  ])

        Xb = np.array([[6.3, 1.0], [1.3, -2.0]])
        Eb = np.array([[0, 1, 1, 0.5] ])


        gA = GCNDataset('GA')
        gA.X = Xa
        gA.E = Ea

        gB = GCNDataset('GB')
        gB.X = Xb
        gB.E = Eb

        print('Graph A')
        print(gA.X,gA.E.shape)

        print('Graph B')
        print(gB.X, gB.E.shape)

        gc = GCNDataset.merge_graph(gA,gB)

        print(gc.X)
        print(gc.E)

        print(gc.EA)

    def test_fasterconvolve(self):

        #S,T=GCNDataset.compute_NodeEdgeMat(self)
        #Check that S times T ==A
        #Do the Unittest Here finally to debug ..
        #Do the convolutions
        pickle_fname = '/opt/project/read/testJL/TABLE/abp_quantile_models/abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph = GCNDataset.load_transkribus_pickle(pickle_fname)

        Xa = np.array([[1.0, 2.0], [6.3, 1.0], [4.3, -2.0]])
        Ea = np.array([[1, 2, 1.0, 0.5], [0, 1, 0.1, 0.2],[2, 1, 1.0, 0.5], [1, 0, 0.1, 0.2]])
        Y  = np.array([[0,1],[1,0],[1,0]])

        gA = GCNDataset('GA')
        gA.X = Xa
        gA.E = Ea
        gA.Y=Y

        gA.compute_NodeEdgeMat()
        gA.compute_EA()
        #gA.compute_NA()



        gcn_graph_test = [gcn_graph[8], gcn_graph[18], gcn_graph[29]]
        #gcn_graph=[gA]
        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]

        print('Node Dim', node_dim)
        print('Edge Dim', edge_dim)

        gcn_model = GCNModelGraphList(node_dim, edge_dim, nb_class, num_layers=1, learning_rate=0.01, mu=0.0,node_indim=-1, nconv_edge=1)
        gcn_model.init_fixed=True
        gcn_model.activation = lambda  x : x
        gcn_model.stack_instead_add = True

        fast_gcn  = GCNModelGraphList(node_dim, edge_dim, nb_class, num_layers=1, learning_rate=0.01, mu=0.0,node_indim=-1, nconv_edge=1)
        fast_gcn.fast_convolve=True
        fast_gcn.init_fixed = True
        fast_gcn.activation = lambda x: x
        fast_gcn.stack_instead_add = True

        graph = gcn_graph[0]
        Sindices =[[i,j] for i,j in zip(graph.S.row,graph.S.col) ]
        #Sindices_sorted = sorted(Sindices_list,)
        print('Sindices',Sindices)
        #Sindices.sort(key=operator.itemgetter(0, 1))
        #Sindices=[[x[0],x[1]] for x in Sindices]
        print('F',graph.F.shape)

        print(Sindices)





        gcn_model.create_model()
        fast_gcn.create_model()

        #It is not the same initialization

        with tf.Session() as session:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            #sess.run(res, options=run_options, run_metadata=run_metadata)

            # Create the Timeline object, and write it to a json

            session.run([gcn_model.init,fast_gcn.init])

            #Taking the graph 4

            graph = gcn_graph[3]

            fb={
                gcn_model.nb_node: graph.X.shape[0],
                gcn_model.node_input: graph.X,
                gcn_model.EA_input: graph.EA,
                gcn_model.y_input: graph.Y,
                #gcn_model.NA_input: graph.NA,
                gcn_model.dropout_p: gcn_model.dropout_rate
            }

            print('F',graph.F.shape)

            #Sindices=np.array([list(graph.S.row),list(graph.S.col)],dtype='int64')

            Sindices = np.array([[i, j] for i, j in zip(graph.S.row, graph.S.col)], dtype='int64')
            #Sindince=[]

            #print(Sindices)


            #Transpose directly T
            Tindices =[[j,i] for i,j in zip(graph.T.row,graph.T.col) ]
            #Tindices.sort(key=operator.itemgetter(0, 1))
            #Tindices = [[x[0], x[1]] for x in Tindices] #reconvert to list
            fb2={
                fast_gcn.nb_node: graph.X.shape[0],
                fast_gcn.nb_edge: graph.F.shape[0],
                fast_gcn.node_input: graph.X,
                fast_gcn.S: np.asarray(graph.S.todense()).squeeze(),
                #fast_gcn.Ssparse: np.vstack([graph.S.row,graph.S.col]),
                fast_gcn.Ssparse: np.array(Sindices,dtype='int64'),
                fast_gcn.Sshape: np.array([ graph.X.shape[0],graph.F.shape[0] ],dtype='int64'),
                fast_gcn.Tsparse: np.array(Tindices, dtype='int64'),
                fast_gcn.T: np.asarray(graph.T.todense()).squeeze(),
                fast_gcn.F: graph.F,
                #fast_model.EA_input: graph.EA,
                fast_gcn.y_input: graph.Y,
                #fast_model.NA_input: graph.NA,
                fast_gcn.dropout_p: fast_gcn.dropout_rate
            }

            #print("Size,S,T",graph.S.shape,graph.T.shape)
            #Wel1 = session.run(gcn_model.Wel0, feed_dict=fb)
            #Wel2 = session.run(fast_gcn.Wel0, feed_dict=fb)
            #print(Wel1)
            #print(Wel2)


            #A=session.run(tf.shape(fast_gcn.conv[0]),feed_dict=fb2)
            #print('conv_shape',A)
            #B = session.run(tf.shape(fast_gcn.SD), feed_dict=fb2)
            #print('SD shape', B)
            #print(Sindices.shape)



            [Ho,Zo]=session.run([gcn_model.hidden_layers[1],gcn_model.Z],feed_dict=fb,options=run_options, run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline_old.json', 'w') as f:
                f.write(ctf)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            [Hn,Zn,conv] = session.run([fast_gcn.hidden_layers[1],fast_gcn.Z,fast_gcn.conv], feed_dict=fb2,options=run_options, run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline_new_sparse.json', 'w') as f:
                f.write(ctf)
            #type about:tracing in chromium url toolbar and load the file

            print(Ho.shape)
            print(Hn.shape)


            #print(Ho[2])
            #print(Hn[2])
            #print(graph.X[2])
            #print(Ho[2,:])
            #print(graph.S[2,:])

            #print(np.array(Sindices, dtype='int64')[15, :])

            #print(graph.F[71,:],)

            diff =  (np.abs(Ho-Hn))
            print(diff)
            print(np.nonzero(diff))
            #print(np.array(Sindices, dtype='int64')[[2,15,18,27,35,41], :])

            #print(Sindices)
            #To make it correct, I should index edges and the graph correctly
            print(diff.sum())
            print(np.abs(Ho - Hn).sum())

            print(graph.X[:,-1])

            print(Zo)
            print(Zn)

            #pdb.set_trace()
            diff_test = self.assertAlmostEqual(0, (diff<1e-3).sum())


if __name__ == '__main__':
    unittest.main()
