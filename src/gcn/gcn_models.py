# -*- coding: utf-8 -*-
"""
@author: Stéphane Clinchant

Code for Original GCN is in /opt/MLS_db/usr/sclincha/GraphConvolutionNets/gcn

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
        self.y_input    = tf.placeholder(tf.float32, [None, self.n_classes], name='Y')


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
            #TODO Add Multiple Layers
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

#TWO POssible Update
# I give the full graph anbd learn from it
# Or I pass one graph at a time ...


class GCNModelGraphList(object):
    def __init__(self,node_dim,edge_dim,nb_classes,num_layers=1,learning_rate=0.1,mu=0.1,node_indim=-1):
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.n_classes=nb_classes
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.activation=tf.nn.relu
        self.mu=mu
        self.learn_edge=True
        self.optalg = tf.train.AdamOptimizer(self.learning_rate)
        self.stack_instead_add=False

        if node_indim==-1:
            self.node_indim=self.node_dim
        else:
            self.node_indim=node_indim



    def create_model(self):
        self.nb_node    = tf.placeholder(tf.int32,(), name='nb_node')
        self.node_input = tf.placeholder(tf.float32, [None, self.node_dim], name='X_')
        self.y_input    = tf.placeholder(tf.float32, [None, self.n_classes], name='Y')
        self.EA_input   =  tf.placeholder(tf.float32, name='EA_input')
        self.NA_input   =  tf.placeholder(tf.float32, name='NA_input')


        std_dev_in=float(1.0/ float(self.node_dim))
        self.Wnode  = tf.Variable(np.identity(self.node_dim, dtype=np.float32), name='Wnode')
        #self.Wnode= tf.Variable(tf.random_normal([self.node_dim,self.node_dim],mean=0.0,stddev=std_dev_in, dtype=np.float32), name='Wnode')
        self.Bnode = tf.Variable(tf.zeros([self.node_dim]), name='Bnode',dtype=np.float32)

        #self.Wedge  = tf.Variable(tf.random_normal([self.edge_dim],mean=0.0,stddev=edge_dim, dtype=np.float32, name='Wedge'))
        self.Wedge  = tf.Variable(tf.ones([1,self.edge_dim], dtype=np.float32, name='Wedge'))

        self.Bedge = tf.Variable(tf.zeros([self.edge_dim]), name='Bedge',dtype=np.float32)


        self.Wnode_layers=[]
        self.Bnode_layers=[]
        self.Wed_layers=[]
        self.Bed_layers=[]

        #Should Project edge as well ...
        if self.node_indim!=self.node_dim:
            Wnl0 = tf.Variable(tf.random_uniform((self.node_dim, self.node_indim),
                                                                   -1.0 / math.sqrt(self.node_dim),
                                                                   1.0 / math.sqrt(self.node_dim)),name='Wnl0',dtype=tf.float32)
        else:
            Wnl0 = tf.Variable(tf.eye(self.node_dim),name='Wnl0',dtype=tf.float32)

        Bnl0 = tf.Variable(tf.zeros([self.node_indim]), name='Bnl0',dtype=tf.float32)
        Wel0 =tf.Variable(tf.ones([1,self.edge_dim], dtype=np.float32, name='Wel0'))

        #self.Wed_layers.append(Wel0)
        for i in range(self.num_layers):
            if self.stack_instead_add:
                Wnli =tf.Variable(tf.random_uniform( (2*self.node_indim, self.node_indim),
                                                               -1.0 / math.sqrt(self.node_indim),
                                                               1.0 / math.sqrt(self.node_indim)),name='Wnl',dtype=tf.float32)
            else:
                Wnli =tf.Variable(tf.random_uniform( (self.node_indim, self.node_indim),
                                                                   -1.0 / math.sqrt(self.node_indim),
                                                                   1.0 / math.sqrt(self.node_indim)),name='Wnl',dtype=tf.float32)

            Bnli = tf.Variable(tf.zeros([self.node_indim]), name='Bnl'+str(i),dtype=tf.float32)

            Weli = tf.Variable(tf.ones([1, self.edge_dim],dtype=tf.float32))

            Beli = tf.Variable(tf.zeros([self.edge_dim]), name='Bel'+str(i),dtype=tf.float32)

            self.Wnode_layers.append(Wnli)
            self.Bnode_layers.append(Bnli)
            self.Wed_layers.append  (Weli)
            self.Bed_layers.append(Beli)




        #TODO Do we project the firt layer or not ?
        # Initialize the weights and biases for a simple one full connected network
        if self.stack_instead_add:
            self.W_classif = tf.Variable(tf.random_uniform((2*self.node_indim, self.n_classes),
                                                           -1.0 / math.sqrt(self.node_dim),
                                                           1.0 / math.sqrt(self.node_dim)),
                                        name="W_classif",dtype=np.float32)
        else:
            self.W_classif = tf.Variable(tf.random_uniform((self.node_indim, self.n_classes),
                                                           -1.0 / math.sqrt(self.node_dim),
                                                           1.0 / math.sqrt(self.node_dim)),
                                        name="W_classif",dtype=np.float32)
        self.B_classif = tf.Variable(tf.zeros([self.n_classes]), name='B_classif',dtype=np.float32)


        self.H = self.activation(tf.add(tf.matmul(self.node_input,self.Wnode),self.Bnode))
        self.hidden_layers=[self.H]

        I = tf.eye(self.nb_node)

        if self.num_layers==1:
            #TODO Add Multiple Layers
            Hi_=tf.matmul(self.hidden_layers[-1],self.Wnode)
            #Em =(tf.matmul(self.Wedge,self.EA_input,b_is_sparse=True))
            Em =(tf.matmul(self.Wedge,self.EA_input))
            Z=tf.reshape(Em,tf.stack([self.nb_node,self.nb_node]))
            #Zn=tf.matmul(self.NA_input,Z)

            if self.stack_instead_add:
                P= tf.matmul(Z,Hi_)
                Hp= tf.concat([Hi_,P],1)
            else:
                Hp= tf.matmul(Z+I,Hi_)
            Hi=self.activation(Hp)
            self.hidden_layers.append(Hi)

        elif self.num_layers>1:
            #TODO Fix activation before convolve or after ..
            H0 = self.activation(tf.add(tf.matmul(self.node_input,Wnl0),Bnl0))

            if self.stack_instead_add:
                Em0 =(tf.matmul(Wel0,self.EA_input))
                Z0=tf.reshape(Em0,tf.stack([self.nb_node,self.nb_node]))
                P= tf.matmul(Z0,H0)
                Hp= tf.concat([H0,P],1)
                self.hidden_layers=[Hp]
            else:
                self.hidden_layers=[H0]

            for i in range(self.num_layers):
                Hi_ = tf.matmul(self.hidden_layers[-1],self.Wnode_layers[i])
                Emi = (tf.matmul(self.Wed_layers[i],self.EA_input))
                Z=tf.reshape(Emi,tf.stack([self.nb_node,self.nb_node]))
                if self.stack_instead_add:
                    P= tf.matmul(Z,Hi_)
                    Hp= tf.concat([Hi_,P],1)
                    Hi=self.activation(Hp)
                    self.hidden_layers.append([Hi])
                else:
                    Hp= tf.matmul(Z+I,Hi_)
                    Hi=self.activation(Hp)
                self.hidden_layers.append(Hi)

        self.logits =tf.add(tf.matmul(self.hidden_layers[-1],self.W_classif),self.B_classif)

        cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
        self.loss = tf.reduce_mean(cross_entropy_source)+self.mu*tf.nn.l2_loss(self.W_classif) +self.mu*tf.nn.l2_loss(self.Wedge)


        self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.grads_and_vars = self.optalg.compute_gradients(self.loss)
        self.train_step = self.optalg.apply_gradients(self.grads_and_vars)


        # Add ops to save and restore all the variables.
        self.init = tf.global_variables_initializer()

    def train(self,session,n_node,X,EA,Y,NA,n_iter=1,verbose=False):
        #TrainEvalSet Here
        for i in range(n_iter):
            feed_batch={
                        self.nb_node:n_node,
                        self.node_input:X,
                        self.EA_input:EA,
                        self.y_input:Y,
                        self.NA_input:NA,
            }
            Ops =session.run([self.train_step,self.loss], feed_dict=feed_batch)
            if verbose:
                print('Training Loss',Ops[1])



    def test(self,session,n_node,X,EA,Y,NA):
        #TrainEvalSet Here
        feed_batch={
                        self.nb_node:n_node,
                        self.node_input:X,
                        self.EA_input:EA,
                        self.y_input:Y,
                        self.NA_input:NA,
        }
        Ops =session.run([self.loss,self.accuracy], feed_dict=feed_batch)
        print('Test Loss',Ops[0],' Test Accuracy:',Ops[1])
        return Ops[1]

