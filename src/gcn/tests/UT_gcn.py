# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pdb
__author__ = 'sclincha'



import sys,os
sys.path.append('../..')

import numpy as np
import unittest
import pickle
import tensorflow as tf
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

from gcn.gcn_models import GCNModel

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

        gcn_model = GCNModel(dataset,num_layers=1,learning_rate=0.1)
        #gcn_model.activation=tf.nn.softmax
        gcn_model.activation=tf.nn.relu
        #gcn_model.activation=tf.nn.sigmoid
        gcn_model.create_model()

        print(dataset.X)
        print(dataset.Y)
        logit_model = GCNModel(dataset,num_layers=0,learning_rate=0.5,mu=0.0)
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

        gcn_model = GCNModel(dataset,num_layers=1,learning_rate=0.1)
        #gcn_model.activation=tf.nn.softmax
        gcn_model.activation=tf.nn.relu
        #gcn_model.activation=tf.nn.sigmoid
        gcn_model.create_model()


        edge_model =GCNModel(dataset,num_layers=1,learning_rate=0.1)
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














if __name__ == '__main__':
    unittest.main()
