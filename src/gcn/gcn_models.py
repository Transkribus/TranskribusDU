# -*- coding: utf-8 -*-
"""
@author: Stéphane Clinchant
"""
try:
    import tensorflow as tf
except:
    print('warning: could not load tensorflow')
import math
import numpy as np
import pprint
import json
import itertools
import random
import scipy.sparse as sp

class GCNModel(object):

    def __init__(self,gcn_dataset,num_layers=1,learning_rate=0.1,mu=0.1):

        self.dataset =gcn_dataset
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.activation=tf.nn.relu
        self.mu=mu
        self.learn_edge=False

    def create_model(self):
        self.node_dim= self.dataset.X.shape[1]
        self.edge_dim= self.dataset.E.shape[1]-2 #Preprocess That
        self.n_classes =self.dataset.Y.shape[1]

        self.node_input = tf.placeholder(tf.float32, [None, self.node_dim], name='X_')
        self.edge_input = tf.placeholder(tf.float32, [None, self.edge_dim], name='E')
        self.y_input = tf.placeholder(tf.float32, [None, self.n_classes], name='Y')

        std_dev_in=float(1.0/ float(self.node_dim))
        self.Wnode  = tf.Variable(np.identity(self.node_dim, dtype=np.float32), name='Wnode')
        #self.Wnode= tf.Variable(tf.random_normal([self.node_dim,self.node_dim],mean=0.0,stddev=std_dev_in, dtype=np.float32), name='Wnode')
        self.Bnode = tf.Variable(tf.zeros([self.node_dim]), name='Bnode',dtype=np.float32)

        edge_dim=float(1.0/float(self.edge_dim))
        #self.Wedge  = tf.Variable(tf.random_normal([self.edge_dim],mean=0.0,stddev=edge_dim, dtype=np.float32, name='Wedge'))
        self.Wedge  = tf.Variable(tf.ones([1,self.edge_dim], dtype=np.float32, name='Wedge'))

        self.Bedge = tf.Variable(tf.zeros([self.edge_dim]), name='Bedge',dtype=np.float32)



        nb_node = self.dataset.A.shape[0]
        if self.learn_edge:

            edge_dim= self.edge_dim
            EA =np.zeros((edge_dim,(nb_node*nb_node)),dtype=np.float32)

            #Build a adjecency sparse matrix for the i_dim of the edge
            i_list =[]
            j_list=[]
            for x,y in zip(self.dataset.E[:,0],self.dataset.E[:,1]):
                i_list.append(int(x))
                j_list.append(int(y))

            for i in range(edge_dim):
                idim_mat =sp.coo_matrix((self.dataset.E[:,i+2],(i_list,j_list)), shape=(nb_node, nb_node))
                D= np.asarray(idim_mat.todense()).squeeze()
                EA[i,:]=np.reshape(D,-1)
            self.EA=EA
            self.tf_EA=tf.constant(EA)


        #print('X:',self.X.shape)
        #print('Y:',self.Y.shape,' class distrib:',np.bincount(self.Y))
        #print('A:',self.A.shape)
        #print('E:',self.E.shape)

        #TODO Do we project the firt layer or not ?
        #TODO Dropout
        #TODO L2 Regularization
        # Initialize the weights and biases for a simple one full connected network
        self.W_classif = tf.Variable(tf.random_uniform((self.node_dim, self.n_classes),
                                                       -1.0 / math.sqrt(self.node_dim),
                                                       1.0 / math.sqrt(self.node_dim)),
                                     name="W_classif",dtype=np.float32)
        self.B_classif = tf.Variable(tf.zeros([self.n_classes]), name='B_classif',dtype=np.float32)


        self.H = self.activation(tf.add(tf.matmul(self.node_input,self.Wnode),self.Bnode))
        #self.H = self.activation(tf.nn.dropout(tf.add(tf.matmul(self.node_input,self.Wnode),self.Bnode),keep_prob=0.8))
        self.hidden_layers=[self.H]

        #Fixe
        Dinv_ = np.diag(np.power(self.dataset.A.sum(axis=1),-0.5))
        #Dinv  =tf.constant(Dinv_)
        #A     =tf.constant(self.dataset.A)

        N=tf.constant(np.dot(Dinv_,self.dataset.A+np.identity(self.dataset.A.shape[0]).dot(Dinv_)),dtype=np.float32)
        print(N)


        if self.learn_edge:
            Hi_=tf.matmul(self.hidden_layers[-1],self.Wnode)
            Em =(tf.matmul(self.Wedge,self.tf_EA))
            Z=tf.reshape(Em,(nb_node,nb_node))
            Hi =self.activation(tf.matmul(Z,Hi_))
            self.hidden_layers.append(Hi)
        else:
            for i in range(self.num_layers):

                Hi_ = tf.matmul(self.hidden_layers[-1],self.Wnode)
                Hi = self.activation(tf.matmul(N,Hi_))
                self.hidden_layers.append(Hi)

        #Wrong Here ....
        #self.logits = self.activation(tf.add(tf.matmul(self.hidden_layers[-1],self.W_classif),self.B_classif))
        self.logits =tf.add(tf.matmul(self.hidden_layers[-1],self.W_classif),self.B_classif)
        #Le code rajoute du Dropout aussi

        cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y_input)
        #cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
        #TODO Add L2 Regularization  for Wedge and Node ...
        self.loss = tf.reduce_mean(cross_entropy_source)+self.mu*tf.nn.l2_loss(self.W_classif) +self.mu*tf.nn.l2_loss(self.Wedge)


        self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


        self.optalg = tf.train.AdagradOptimizer(self.learning_rate)
        #self.optalg = tf.train.AdamOptimizer(self.learning_rate)
        #self.optalg = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.grads_and_vars = self.optalg.compute_gradients(self.loss)
        self.train_step = self.optalg.apply_gradients(self.grads_and_vars)


        # Add ops to save and restore all the variables.
        self.init = tf.global_variables_initializer()

    #We do not use the edge features here
    #todo add a distance on the edge
    def train(self,session,n_iter=10):
        #TrainEvalSet Here
        for i in range(n_iter):
            feed_batch={self.node_input:self.dataset.X,
                        self.y_input:self.dataset.Y
            }
            Ops =session.run([self.train_step,self.loss,self.accuracy], feed_dict=feed_batch)
            print('Training Loss',Ops[1],'Accuracy:',Ops[2])
    #TODO Predict
    #Next Steps Make dataset with relevant links
    #Code Edge operation
    #Residual Connection 0,1,2
    #Initialize with logistic Regression

    #Compute the Tensor EdgeMatrix ...
    #Ensuite ,c'est juste un tensor dot product ....



#TODO Class EdgeGCNModels
#PICKLE version 3.4 to get the data ...



