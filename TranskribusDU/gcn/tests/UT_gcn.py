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

from gcn.gcn_models import EdgeConvNet, GraphAttNet, init_glorot, EnsembleGraphNN
from gcn.DU_Model_ECN import DU_Ensemble_ECN
import sklearn
import sklearn.metrics

from tasks import _checkFindColDir, DU_ABPTable
from tensorflow.python import pywrap_tensorflow



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

def get_graph_test():
    #For graph att net
    X=np.array([ [1.0,0.5],[0.5,0.5],[0.0,1.0] ],dtype='float32')
    E=np.array([ [0,1,1.0],[1,0,1.0],[2,1,1.0],[1,2,1.0]],dtype='float32')
    Y=np.array([ [1,0],[0,1],[0,1]],dtype='int32')

    gcn =GCNDataset('UT_test_1')
    gcn.X=X
    gcn.E=E
    gcn.Y=Y
    gcn.compute_NodeEdgeMat()
    return gcn



class UT_gcn(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(UT_gcn, self).__init__(*args, **kwargs)
        if os.path.exists('UT_MODELS') is False:
            os.mkdir('UT_MODELS')

    def test_01_load(self):
        dataset=GCNDataset('UT_iris_0')
        dataset.load_pickle('iris_graph.pickle')
        dataset.print_stats()
        return True

    def test_05_load_jl_pickle(self):

        pickle_fname='abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph= GCNDataset.load_transkribus_pickle(pickle_fname)
        print(len(gcn_graph),'loaded graph')


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


        #TODO Test on Y too
        self.assertEquals(5,gc.X.shape[0])
        self.assertEquals(3,gc.E.shape[0])


    def test_predict(self):
        pickle_fname = 'abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph = GCNDataset.load_transkribus_pickle(pickle_fname)

        gcn_graph_train = [gcn_graph[8], gcn_graph[18], gcn_graph[29]]
        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]

        gcn_model = EdgeConvNet(node_dim, edge_dim, nb_class, num_layers=2, learning_rate=0.01, mu=0.0,
                                      node_indim=-1, nconv_edge=5)
        # gcn_model =EdgeConvNet(node_dim,edge_dim,nb_class,num_layers=1,learning_rate=0.001,mu=0.0,node_indim=-1)
        gcn_model.stack_instead_add = True
        gcn_model.fast_convolve=True

        gcn_model.create_model()

        # pdb.set_trace()

        nb_iter = 50
        with tf.Session() as session:
            session.run([gcn_model.init])
            # Sample each graph
            # random
            for i in range(nb_iter):
                gcn_model.train_lG(session,gcn_graph_train)

            #Get the Test Prediction
            g_acc, node_acc = gcn_model.test_lG(session,gcn_graph_train)
            print('Mean Accuracy', g_acc,node_acc)
            # Get the Test Prediction

            lY_pred = gcn_model.predict_lG(session,gcn_graph_train,verbose=False)

        tp=0
        nb_node=0

        Ytrue_l=[]
        lY_l=[]
        for lY,graph in zip(lY_pred,gcn_graph_train):
            Ytrue = np.argmax(graph.Y,axis=1)
            Ytrue_l.extend(Ytrue)
            lY_l.extend(lY)
            tp += sum (Ytrue==lY)
            #pdb.set_trace()
            nb_node += Ytrue.shape[0]

        print('Final Accuracy',tp/nb_node)
        print('Accuracy_score',sklearn.metrics.accuracy_score(Ytrue_l,lY_l))
        print(sklearn.metrics.classification_report(Ytrue_l,lY_l))
        self.assertAlmostEqual(tp/nb_node,node_acc)

        Z=[lY_pred,[np.argmax(graph.Y,axis=1) for graph in gcn_graph_train] ]
        f=open('debug.pickle','wb')
        pickle.dump(Z,f,protocol=1,fix_imports=True)
        f.close()

        #tstRep = TestReport('UT_test',lY_pred,[np.argmax(graph.Y,axis=1) for graph in gcn_graph_train],None)
        #print(tstRep.getClassificationReport())

    def test_logit_convolve(self):
        # 3 nodes a;b,c  a<->b and c<->b   a->b<c>
        X = np.array([[1.0, 2.0], [6.3, 1.0], [4.3, -2.0]])
        Y = np.array([[1, 0], [0, 1.0], [1.0, 0.0]])
        E = np.array([[0, 1, 1.0, 1, 0], #edge a->b
                      [1, 0, 1.0, 0, 1], #edge b->a
                      [2, 1, 1.0, 0.0, 1.0]
                      ])

        nb_node=3
        gA = GCNDataset('GLogitConvolve')
        gA.X = X
        gA.Y = Y
        gA.E = E
        gA.A = sp.coo_matrix((np.ones(E.shape[0]), (E[:, 0], E[:, 1])), shape=(nb_node, nb_node))

        gA.compute_NodeEdgeMat()
        gA.compute_NA()

        #Test in degree out_degree
        print(gA.in_degree,gA.out_degree)
        self.assertAlmostEqual(2,gA.in_degree[1])
        print(gA.NA_indegree)

        self.assertAlmostEqual(0.5,gA.NA_indegree[1,0])
        #self.assertAlmostEqual(2, gA.indegree[1])
        #now assuming P(Y|a)=[1,0] P(Y|c)=[1,0] and current P(Y|b)=[0.5,0.5]
        pY = np.array([[1, 0], [0.5, 0.5], [0.8, 0.2]])
        #Node b  has two edges
        # Yt=[0 1;1 0]

        Yt = np.array([[0.0,1.0],[1.0, 0.0]])


        pY_Yt = tf.matmul(pY, Yt, transpose_b=True)

        Yt_sum = EdgeConvNet.logitconvolve_fixed(pY,Yt,gA.NA_indegree)


        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            Ytt = session.run(pY_Yt)
            print(Ytt)
            Res = session.run(Yt_sum)
            print(Res)

        #Still buggy for nodes which no incoming edges ...
        #this is not our case ?
        #the formulation is false is this case adding zeros
        # we shoudl add a uniform distribution
        #we should mask it then ...
        # Add a Yt_sum which is zero everywhere except for zero indegree node for which it is uniform

    def test_graphattnet_attnlayer(self):

        gcn_graph = get_graph_test()
        node_dim = gcn_graph.X.shape[1]
        edge_dim = gcn_graph.E.shape[1] - 2.0
        nb_class = gcn_graph.Y.shape[1]

        gcn_model = GraphAttNet(node_dim, nb_class, num_layers=1, learning_rate=0.01, node_indim=8, nb_attention=1)
        gcn_model.create_model()

        Wa = tf.eye(node_dim)
        va = tf.ones([2,node_dim])
        # elf.Ssparse, self.Tspars
        alphas,nH = gcn_model.simple_graph_attention_layer(gcn_model.node_input, Wa, va, gcn_model.Ssparse,
                                                        gcn_model.Tsparse, gcn_model.Aind, gcn_model.Sshape,
                                                        gcn_model.nb_edge, gcn_model.dropout_p_attn,gcn_model.dropout_p_node)
        alphas_shape = tf.shape(alphas)

        init = tf.global_variables_initializer()

        graph=gcn_graph
        with tf.Session() as session:
            session.run([init])

            print('### Graph', graph.X.shape, graph.F.shape[0])
            # print(graph.Sind)
            # print(graph.Tind)
            nb_node =graph.X.shape[0]
            Aind = np.array(np.stack([graph.Sind[:, 0], graph.Tind[:, 1]], axis=-1), dtype='int64')
            print("Adjacency Indices:", Aind.shape, Aind)
            feed_batch = {
                gcn_model.nb_node: graph.X.shape[0],
                gcn_model.nb_edge: graph.F.shape[0],
                gcn_model.node_input: graph.X,
                gcn_model.Ssparse: np.array(graph.Sind, dtype='int64'),
                gcn_model.Sshape: np.array([graph.X.shape[0], graph.F.shape[0]], dtype='int64'),
                gcn_model.Tsparse: np.array(graph.Tind, dtype='int64'),
                gcn_model.Aind: Aind,
                # self.F: graph.F,
                gcn_model.y_input: graph.Y,
                # self.dropout_p_H: self.dropout_rate_H,
                gcn_model.dropout_p_node: 0.0,
                gcn_model.dropout_p_attn: 0.0,

            }
            [c_alphas,c_nH, c_alphas_shape] = session.run([alphas,nH, alphas_shape], feed_dict=feed_batch)
            print('alphas',c_alphas,c_alphas_shape)

            sp_mat = sp.coo_matrix((c_alphas.values, (c_alphas.indices[:,0],c_alphas.indices[:,1])), shape=(nb_node, nb_node))
            Att_dense =sp_mat.todense()
            print(Att_dense)
            self.assertTrue(c_alphas_shape[0]==3)
            self.assertTrue(c_alphas_shape[1]==3)

            self.assertTrue(Att_dense[0,2]==0)
            self.assertAlmostEqual(Att_dense[1,0], np.exp(2.5)/(np.exp(2.5)+np.exp(2)))
            self.assertAlmostEqual(Att_dense[0, 1],1.0)
            self.assertAlmostEqual(Att_dense[2, 1],1.0)

    def test_graphattnet_attnlayer_selfloop(self):

        gcn_graph = get_graph_test()
        node_dim = gcn_graph.X.shape[1]
        edge_dim = gcn_graph.E.shape[1] - 2.0
        nb_class = gcn_graph.Y.shape[1]

        gcn_model = GraphAttNet(node_dim, nb_class, num_layers=1, learning_rate=0.01, node_indim=8, nb_attention=1)
        gcn_model.create_model()

        Wa = tf.eye(node_dim)
        va = tf.ones([2,node_dim])
        # elf.Ssparse, self.Tspars
        alphas,nH = gcn_model.simple_graph_attention_layer(gcn_model.node_input, Wa, va, gcn_model.Ssparse,
                                                        gcn_model.Tsparse, gcn_model.Aind, gcn_model.Sshape,
                                                        gcn_model.nb_edge, gcn_model.dropout_p_attn,gcn_model.dropout_p_node,
                                                        add_self_loop=True)
        alphas_shape = tf.shape(alphas)

        node_indices = tf.range(gcn_model.Sshape[0])
        # Sparse Idendity
        # Debug
        id_indices = tf.stack([node_indices, node_indices], axis=1)
        val = tf.squeeze(tf.matmul(gcn_model.node_input, va, transpose_b=True))
        spI = tf.SparseTensor(indices=id_indices,values=val,dense_shape=[gcn_model.Sshape[0], gcn_model.Sshape[0]])


        init = tf.global_variables_initializer()
        #AI=tf.sparse_add(alphas,spI)

        graph=gcn_graph
        with tf.Session() as session:
            session.run([init])

            print('### Graph', graph.X.shape, graph.F.shape[0])
            # print(graph.Sind)
            # print(graph.Tind)
            nb_node =graph.X.shape[0]
            Aind = np.array(np.stack([graph.Sind[:, 0], graph.Tind[:, 1]], axis=-1), dtype='int64')
            print("Adjacency Indices:", Aind.shape, Aind)
            feed_batch = {
                gcn_model.nb_node: graph.X.shape[0],
                gcn_model.nb_edge: graph.F.shape[0],
                gcn_model.node_input: graph.X,
                gcn_model.Ssparse: np.array(graph.Sind, dtype='int64'),
                gcn_model.Sshape: np.array([graph.X.shape[0], graph.F.shape[0]], dtype='int64'),
                gcn_model.Tsparse: np.array(graph.Tind, dtype='int64'),
                gcn_model.Aind: Aind,
                # self.F: graph.F,
                gcn_model.y_input: graph.Y,
                # self.dropout_p_H: self.dropout_rate_H,
                gcn_model.dropout_p_node: 0.0,
                gcn_model.dropout_p_attn: 0.0,

            }
            [c_alphas,c_nH, c_alphas_shape,spI] = session.run([alphas,nH, alphas_shape,spI], feed_dict=feed_batch)
            print('alphas',c_alphas,c_alphas_shape)
            print('spI',spI)
            #print('AI',AI)
            sp_mat = sp.coo_matrix((c_alphas.values, (c_alphas.indices[:,0],c_alphas.indices[:,1])), shape=(nb_node, nb_node))
            Att_dense =sp_mat.todense()
            print(Att_dense)
            self.assertTrue(c_alphas_shape[0]==3)
            self.assertTrue(c_alphas_shape[1]==3)

            self.assertTrue(Att_dense[0,2]==0)
            #self.assertAlmostEqual(Att_dense[1,0], np.exp(2.5)/(np.exp(2.5)+np.exp(2)))
            #self.assertAlmostEqual(Att_dense[0, 1],1.0)
            #self.assertAlmostEqual(Att_dense[2, 1],1.0)


    def test_graphattnet_train(self):

        pickle_fname = 'abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph = GCNDataset.load_transkribus_pickle(pickle_fname)

        gcn_graph_train = [gcn_graph[8], gcn_graph[18], gcn_graph[29]]
        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]

        gcn_model = GraphAttNet(node_dim, nb_class, num_layers=1, learning_rate=0.01, node_indim=-1,nb_attention=1)
        gcn_model.create_model()

        with tf.Session() as session:
            session.run([gcn_model.init])
            # Get the Test Prediction
            g_acc, node_acc = gcn_model.test_lG(session, gcn_graph_train)
            print('Mean Accuracy', g_acc, node_acc)
            gcn_model.train_lG(session,gcn_graph)
            g_acc, node_acc = gcn_model.test_lG(session, gcn_graph_train)
            print('Mean Accuracy', g_acc, node_acc)

    def test_graphattnet_train_dropout(self):
        pickle_fname = 'abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph = GCNDataset.load_transkribus_pickle(pickle_fname)

        gcn_graph_train = [gcn_graph[8], gcn_graph[18], gcn_graph[29]]
        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]

        gcn_model = GraphAttNet(node_dim, nb_class, num_layers=1, learning_rate=0.01, node_indim=-1,nb_attention=3)
        gcn_model.dropout_rate_node=0.2
        gcn_model.dropout_rate_attention = 0.2
        gcn_model.create_model()

        with tf.Session() as session:
            session.run([gcn_model.init])
            # Get the Test Prediction
            g_acc, node_acc = gcn_model.test_lG(session, gcn_graph_train)
            print('Mean Accuracy', g_acc, node_acc)
            gcn_model.train_lG(session,gcn_graph)
            g_acc, node_acc = gcn_model.test_lG(session, gcn_graph_train)
            print('Mean Accuracy', g_acc, node_acc)

    def test_dense_attn_layer(self):
        gcn_graph = get_graph_test()
        node_dim = gcn_graph.X.shape[1]
        edge_dim = gcn_graph.E.shape[1] - 2.0
        nb_class = gcn_graph.Y.shape[1]

        gcn_model = GraphAttNet(node_dim, nb_class, num_layers=1, learning_rate=0.01, node_indim=8, nb_attention=1)
        gcn_model.create_model()

        Wa = tf.eye(node_dim)
        va = tf.ones([2, node_dim])
        # elf.Ssparse, self.Tspars
        alphas, nH = gcn_model.dense_graph_attention_layer(gcn_model.node_input, Wa, va, gcn_model.nb_node,
                                                            gcn_model.dropout_p_attn,
                                                            gcn_model.dropout_p_node)
        alphas_shape = tf.shape(alphas)

        init = tf.global_variables_initializer()

        graph = gcn_graph
        with tf.Session() as session:
            session.run([init])

            print('### Graph', graph.X.shape, graph.F.shape[0])
            # print(graph.Sind)
            # print(graph.Tind)
            nb_node = graph.X.shape[0]
            Aind = np.array(np.stack([graph.Sind[:, 0], graph.Tind[:, 1]], axis=-1), dtype='int64')
            print("Adjacency Indices:", Aind.shape, Aind)
            feed_batch = {
                gcn_model.nb_node: graph.X.shape[0],
                gcn_model.nb_edge: graph.F.shape[0],
                gcn_model.node_input: graph.X,
                gcn_model.Ssparse: np.array(graph.Sind, dtype='int64'),
                gcn_model.Sshape: np.array([graph.X.shape[0], graph.F.shape[0]], dtype='int64'),
                gcn_model.Tsparse: np.array(graph.Tind, dtype='int64'),
                gcn_model.Aind: Aind,
                # self.F: graph.F,
                gcn_model.y_input: graph.Y,
                # self.dropout_p_H: self.dropout_rate_H,
                gcn_model.dropout_p_node: 0.0,
                gcn_model.dropout_p_attn: 0.0,

            }
            [c_alphas, c_nH, c_alphas_shape] = session.run([alphas, nH, alphas_shape], feed_dict=feed_batch)
            print('alphas', c_alphas, c_alphas_shape)

            #sp_mat = sp.coo_matrix((c_alphas.data, (c_alphas.indices[:, 0], c_alphas.indices[:, 1])),
            #                       shape=(nb_node, nb_node))
            Att_dense = c_alphas
            print(Att_dense)
            #TODO Update True Value
            self.assertTrue(c_alphas_shape[0] == 3)
            self.assertTrue(c_alphas_shape[1] == 3)

            #self.assertTrue(Att_dense[0, 2] == 0)
            #self.assertAlmostEqual(Att_dense[1, 0], np.exp(2.5) / (np.exp(2.5) + np.exp(2)))
            #self.assertAlmostEqual(Att_dense[0, 1], 1.0)
            #self.assertAlmostEqual(Att_dense[2, 1], 1.0)

    def test_diff_in_tf_graph(self):
        g1 = tf.Graph()
        with g1.as_default() as g:
            with g.name_scope("g1") as g1_scope:
                matrix1 = tf.constant([[3., 3.]])
                matrix2 = tf.constant([[2.], [2.]])
                product = tf.matmul(matrix1, matrix2, name="product")
                first_graph=str(g.as_graph_def())

        g2 = tf.Graph()
        with g2.as_default() as g:
            with g.name_scope("g1") as g1_scope:
                matrix1 = tf.constant([[3., 3.]])
                matrix2 = tf.constant([[2.], [2.]])
                #product = tf.matmul(matrix1, tf.matmul(matrix2,matrix1), name="product")
                product = tf.matmul(matrix1, matrix2, name="product")
                second_graph =str(g.as_graph_def())

        print(first_graph)
        print(second_graph)
        diff = pywrap_tensorflow.EqualGraphDefWrapper(g1.as_graph_def().SerializeToString(),
                                                      g2.as_graph_def().SerializeToString())
        print('diff',diff)
        assert not diff

    def test_refactoring_ecn_stack(self):
        #Test the refactoring of ECN

        node_dim = 29
        edge_dim = 42
        nb_class = 18

        nb_layers =3
        lr=0.001
        nb_conv=10

        g1 = tf.Graph()
        with g1.as_default() as g:
            with g.name_scope("g1") as g1_scope:
                print('Old')
                gcn_model = EdgeConvNet(node_dim, edge_dim, nb_class,
                                        num_layers=nb_layers, learning_rate=lr, mu=0.0,
                                        node_indim=-1, nconv_edge=nb_conv,
                                        )
                gcn_model.stack_instead_add=True
                gcn_model.create_model_old()
                #first_graph = str(g.as_graph_def().SerializeToString())
                #f=open('first_graph.txt','w')
                #f.write(first_graph)
                #f.close()

        g2 = tf.Graph()
        with g2.as_default() as g:
            with g.name_scope("g1") as g1_scope:
                print('New')
                gcn_model = EdgeConvNet(node_dim, edge_dim, nb_class,
                                        num_layers=nb_layers, learning_rate=lr, mu=0.0,
                                        node_indim=-1, nconv_edge=nb_conv,
                                        )
                gcn_model.stack_instead_add = True
                gcn_model.create_model()
                second_graph = str(g.as_graph_def().SerializeToString())
                #f = open('second_graph.txt', 'w')
                #f.write(second_graph)
                #f.close()

        #print(first_graph)
        #print(second_graph)
        '''
        for i in range(len(first_graph)):
            if first_graph[i] != second_graph[i]:
                print(i, first_graph[i], second_graph[i])
        '''
        diff = pywrap_tensorflow.EqualGraphDefWrapper(g1.as_graph_def().SerializeToString(),
                                                      g2.as_graph_def().SerializeToString())
        print('diff', diff)
        print('Failure due to change of nonlinearty in the first layer. old model apply non lineartiye on nodes and then convolves')

        assert not diff

    def test_refactoring_ecn_sum(self):
        # Test the refactoring of ECN

        node_dim = 29
        edge_dim = 42
        nb_class = 18

        nb_layers = 3
        lr = 0.001
        nb_conv = 10

        g1 = tf.Graph()
        with g1.as_default() as g:
            with g.name_scope("g1") as g1_scope:
                print('Old')
                gcn_model = EdgeConvNet(node_dim, edge_dim, nb_class,
                                        num_layers=nb_layers, learning_rate=lr, mu=0.0,
                                        node_indim=-1, nconv_edge=nb_conv,
                                        )
                gcn_model.stack_instead_add = False
                gcn_model.create_model_old()
                first_graph = str(g.as_graph_def())

        g2 = tf.Graph()
        with g2.as_default() as g:
            with g.name_scope("g1") as g1_scope:
                print('New')
                gcn_model = EdgeConvNet(node_dim, edge_dim, nb_class,
                                        num_layers=nb_layers, learning_rate=lr, mu=0.0,
                                        node_indim=-1, nconv_edge=nb_conv,
                                        )
                gcn_model.stack_instead_add = False
                gcn_model.create_model()
                second_graph = str(g.as_graph_def())

        diff = pywrap_tensorflow.EqualGraphDefWrapper(g1.as_graph_def().SerializeToString(),
                                                      g2.as_graph_def().SerializeToString())
        print('diff', diff)
        print('Failure due to change of nonlinearty in the first layer. old model apply non lineartiye on nodes and then convolves')

        assert not diff

    def test_train_ensemble(self):
        #TODO Make a proper synthetic dataset for test Purpose
        pickle_fname = 'abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph = GCNDataset.load_transkribus_pickle(pickle_fname)

        gcn_graph_train = [gcn_graph[8], gcn_graph[18], gcn_graph[29]]
        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]

        gat_model = GraphAttNet(node_dim, nb_class, num_layers=1, learning_rate=0.01, node_indim=-1, nb_attention=3)
        gat_model.dropout_rate_node = 0.2
        gat_model.dropout_rate_attention = 0.2
        gat_model.create_model()

        nb_layers = 3
        lr = 0.001
        nb_conv = 2

        ecn_model = EdgeConvNet(node_dim, edge_dim, nb_class,
                                        num_layers=nb_layers, learning_rate=lr, mu=0.0,
                                        node_indim=-1, nconv_edge=nb_conv,
                                        )
        ecn_model.create_model()

        #Check Graphs
        #Are we recopying the models and graph definition implicitly ?
        ensemble =EnsembleGraphNN([ecn_model,gat_model])


        with tf.Session() as session:
            session.run([ensemble.models[0].init])

            for iter in range(500):
                ensemble.train_lG(session,gcn_graph_train)
            prediction = ensemble.predict_lG(session,gcn_graph_train)
            print(prediction)
            self.assertTrue(len(prediction)==len(gcn_graph_train))

            print('Ensemble Prediction')
            accs = ensemble.test_lG(session,gcn_graph_train)
            print('Base Predictions')
            for m in ensemble.models:
                accs = m.test_lG(session,gcn_graph_train)
                print(accs)

            print(accs)
        #C'est pas vraiment bien testé

    def test_train_DU_ECN_Task(self):
        model_dir='UT_MODELS'
        model_name='UT_mod1'

        lTrn = _checkFindColDir('./abp_test')

        dLearnerConfig = {'nb_iter': 50,
                          'lr': 0.001,
                          'num_layers': 3,
                          'nconv_edge': 10,
                          'stack_convolutions': True,
                          'node_indim': -1,
                          'mu': 0.0,
                          'dropout_rate_edge': 0.0,
                          'dropout_rate_edge_feat': 0.0,
                          'dropout_rate_node': 0.0,
                          'ratio_train_val': 0.15,
                          # 'activation': tf.nn.tanh, Problem I can not serialize function HERE
                          }

        doer = DU_ABPTable.DU_ABPTable_ECN(model_name, model_dir, dLearnerConfigArg=dLearnerConfig)

        tstReport=doer.train_save_test(lTrn, lTrn, False, False)
        acc,_=tstReport.getClassificationReport()
        print(acc)
        self.assertTrue(acc>0.5)
    def test_test_DU_ECN_Task(self):
        fname=os.path.join('UT_MODELS','UT_mod1_bestmodel.ckpt.index')

        model_dir = 'UT_MODELS'
        model_name = 'UT_mod1'

        lTrn = _checkFindColDir('./abp_test')

        if os.path.exists(fname):
            #Do the test
            doer = DU_ABPTable.DU_ABPTable_ECN(model_name, model_dir)
            doer.load()
            tstReport = doer.test(lTrn)

            acc, _ = tstReport.getClassificationReport()
            print(acc)
            self.assertTrue(acc > 0.5)
        else:
            self.fail('UT_mod1 was not trained')

    def test_predict_proba(self):
        pickle_fname = 'abp_CV_fold_1_tlXlY_trn.pkl'
        gcn_graph = GCNDataset.load_transkribus_pickle(pickle_fname)

        gcn_graph_train = [gcn_graph[8], gcn_graph[18], gcn_graph[29]]
        node_dim = gcn_graph[0].X.shape[1]
        edge_dim = gcn_graph[0].E.shape[1] - 2.0
        nb_class = gcn_graph[0].Y.shape[1]

        gcn_model = GraphAttNet(node_dim, nb_class, num_layers=1, learning_rate=0.01, node_indim=-1, nb_attention=3)
        gcn_model.dropout_rate_node = 0.2
        gcn_model.dropout_rate_attention = 0.2
        gcn_model.create_model()

        with tf.Session() as session:
            session.run([gcn_model.init])
            # Get the Test Prediction
            gcn_model.train_lG(session, gcn_graph)

            g_proba = gcn_model.prediction_prob(session, gcn_graph_train[1])
            print(g_proba.shape)
            print(type(g_proba))
            print(gcn_graph_train[1].X.shape)
            self.assertTrue(g_proba.shape==(gcn_graph_train[1].X.shape[0],5))

    def test_average_prediction(self):
        model1 = [np.array([[0.9,0.1]]),np.array([[0.3,0.7]])]
        model2 = [np.array([[0.1,0.9]]),np.array([[0.3,0.7]])]

        lY_proba=[model1,model2]

        pred_proba,avgP = DU_Ensemble_ECN.average_prediction(lY_proba)
        print(pred_proba,)

        self.assertAlmostEqual(avgP[0][0][0],0.5)
        self.assertAlmostEqual(avgP[0][0][1], 0.5)

        #Test whether the function fails if one graph prediction is missing
        model1 = [np.array([[0.9, 0.1]]), np.array([[0.3, 0.7]])]
        model2 = [np.array([[0.1, 0.9]])]

        lY_proba = [model1, model2]

        with self.assertRaises(IndexError):
            pred_proba,avgP = DU_Ensemble_ECN.average_prediction(lY_proba)
            print(pred_proba)

    def test_train_ensemble(self):
        model_dir='UT_MODELS'
        model_name='UT_mod2_ensemble'

        lTrn = _checkFindColDir('./abp_test')

        dLearnerConfig = \
            {
                "ecn_ensemble": [
                    {
                        "type": "ecn",
                        "name": "UT_mod2_m1",
                        "dropout_rate_edge": 0.0,
                        "dropout_rate_edge_feat": 0.0,
                        "dropout_rate_node": 0.0,
                        "lr": 0.001,
                        "mu": 0.0,
                        "nb_iter": 50,
                        "nconv_edge": 1,
                        "node_indim": -1,
                        "num_layers": 8,
                        "ratio_train_val": 0.1,
                        "patience": 500,
                        "activation_name": "tanh",
                        "stack_convolutions": True

                    }
                    ,
                    {
                        "type": "ecn",
                        "name": "UT_mod2_m2",
                        "dropout_rate_edge": 0.0,
                        "dropout_rate_edge_feat": 0.0,
                        "dropout_rate_node": 0.0,
                        "lr": 0.001,
                        "mu": 0.0,
                        "nb_iter": 50,
                        "nconv_edge": 1,
                        "node_indim": -1,
                        "num_layers": 8,
                        "ratio_train_val": 0.1,
                        "patience": 50,
                        "activation_name": "relu",
                        "stack_convolutions": True
                    }
                ]
            }


        doer = DU_ABPTable.DU_ABPTable_ECN(model_name, model_dir, dLearnerConfigArg=dLearnerConfig)

        tstReport=doer.train_save_test(lTrn, lTrn, False, False)
        acc,_=tstReport.getClassificationReport()
        print(acc)
        self.assertTrue(acc>0.45)

    def test_test_ensemble(self):
        fname=os.path.join('UT_MODELS','UT_mod2_m2_bestmodel.ckpt.index')

        model_dir = 'UT_MODELS'
        model_name = 'UT_mod2_ensemble'

        lTrn = _checkFindColDir('./abp_test')

        dLearnerConfig = \
            {
                "ecn_ensemble": [
                    {
                        "type": "ecn",
                        "name": "UT_mod2_m1",
                        "dropout_rate_edge": 0.0,
                        "dropout_rate_edge_feat": 0.0,
                        "dropout_rate_node": 0.0,
                        "lr": 0.001,
                        "mu": 0.0,
                        "nb_iter": 50,
                        "nconv_edge": 1,
                        "node_indim": -1,
                        "num_layers": 8,
                        "ratio_train_val": 0.1,
                        "patience": 500,
                        "activation_name": "tanh",
                        "stack_convolutions": True

                    }
                    ,
                    {
                        "type": "ecn",
                        "name": "UT_mod2_m2",
                        "dropout_rate_edge": 0.0,
                        "dropout_rate_edge_feat": 0.0,
                        "dropout_rate_node": 0.0,
                        "lr": 0.001,
                        "mu": 0.0,
                        "nb_iter": 50,
                        "nconv_edge": 1,
                        "node_indim": -1,
                        "num_layers": 8,
                        "ratio_train_val": 0.1,
                        "patience": 50,
                        "activation_name": "relu",
                        "stack_convolutions": True
                    }
                ]
            }

        if os.path.exists(fname):
            #Do the test
            doer = DU_ABPTable.DU_ABPTable_ECN(model_name, model_dir,dLearnerConfigArg=dLearnerConfig)
            doer.load()
            tstReport = doer.test(lTrn)

            acc, _ = tstReport.getClassificationReport()
            print(acc)
            self.assertTrue(acc > 0.5)
        else:
            self.fail('UT_mod1 was not trained')

    #TODO Test files Too
    #Does not work yet Predict does not work


if __name__ == '__main__':
    unittest.main()
