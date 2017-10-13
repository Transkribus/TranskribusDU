# -*- coding: utf-8 -*-
"""
@author: Stéphane Clinchant

Code for Original GCN is in /opt/MLS_db/usr/sclincha/GraphConvolutionNets/gcn

"""

import tensorflow as tf
import math
import numpy as np
import scipy.sparse as sp
import random


def init_glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    #Taken from the GCN_code
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def init_normal(shape,stddev,name=None):
    initial=tf.random_normal(shape, mean=0.0, stddev=stddev, dtype=np.float32)
    return tf.Variable(initial, name=name)


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
                Hi = self.activation(tf.matmul(N,Hi_)+Hi_)
                self.hidden_layers.append(Hi)

        #Wrong Here ....
        #self.logits = self.activation(tf.add(tf.matmul(self.hidden_layers[-1],self.W_classif),self.B_classif))
        self.logits =tf.add(tf.matmul(self.hidden_layers[-1],self.W_classif),self.B_classif)
        #Le code rajoute du Dropout aussi

        #cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y_input)
        cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
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





class GCNModelGraphList(object):
    def __init__(self,node_dim,edge_dim,nb_classes,num_layers=1,learning_rate=0.1,mu=0.1,node_indim=-1,nconv_edge=1):
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
        self.nconv_edge=nconv_edge

        if node_indim==-1:
            self.node_indim=self.node_dim
        else:
            self.node_indim=node_indim


    def convolve(self,Wedge,EA,H,nb_node,nconv):
        Em =(tf.matmul(Wedge,EA))
        print('EM',Em.get_shape())
        print (nb_node,nconv)
        #Use activation here or not ?
        Z=tf.reshape(Em,(nconv,nb_node,nb_node))

        Cops=[]
        for i in range(nconv):
            Hi=tf.matmul(Z[i],H)
            Cops.append(Hi)

        P=tf.concat(Cops,1)
        return P


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

        #Try changing init_normal by init_glorot ...
        self.Wedge  = tf.Variable(tf.random_normal( [int(self.nconv_edge),int(self.edge_dim)],mean=0.0,stddev=1.0), dtype=tf.float32, name='Wedge')
        #self.Wedge  = tf.Variable(tf.ones([self.nconv_edge,self.edge_dim], dtype=np.float32, name='Wedge'))

        #self.Wedge_logit = init_glorot([int(self.n_classes), int(self.edge_dim)], name='Wedge_logit')
        #self.Wedge_logit = init_glorot([1, int(self.edge_dim)], name='Wedge_logit')
        #self.Yt          = init_glorot([int(self.n_classes), int(self.n_classes)], name='Yt')

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
        self.Wel0 =tf.Variable(tf.random_normal([int(self.nconv_edge),int(self.edge_dim)],mean=0.0,stddev=1.0), dtype=np.float32, name='Wel0')

        #self.Wed_layers.append(Wel0)
        for i in range(self.num_layers-1):
            if self.stack_instead_add:
                Wnli =tf.Variable(tf.random_uniform( (self.node_indim*self.nconv_edge+self.node_indim, self.node_indim),
                                                               -1.0 / math.sqrt(self.node_indim),
                                                               1.0 / math.sqrt(self.node_indim)),name='Wnl',dtype=tf.float32)
                print('Wnli shape',Wnli.get_shape())
            else:
                Wnli =tf.Variable(tf.random_uniform( (self.node_indim, self.node_indim),
                                                                   -1.0 / math.sqrt(self.node_indim),
                                                                   1.0 / math.sqrt(self.node_indim)),name='Wnl',dtype=tf.float32)

            Bnli = tf.Variable(tf.zeros([self.node_indim]), name='Bnl'+str(i),dtype=tf.float32)

            #Weli = tf.Variable(tf.ones([int(self.nconv_edge),int(self.edge_dim)],dtype=tf.float32))
            Weli = tf.Variable(tf.random_normal([int(self.nconv_edge), int(self.edge_dim)], mean=0.0, stddev=1.0),
                               dtype=np.float32, name='Wel_')

            Beli = tf.Variable(tf.zeros([self.edge_dim]), name='Bel'+str(i),dtype=tf.float32)

            self.Wnode_layers.append(Wnli)
            self.Bnode_layers.append(Bnli)
            self.Wed_layers.append  (Weli)
            self.Bed_layers.append(Beli)




        #TODO Do we project the firt layer or not ?
        # Initialize the weights and biases for a simple one full connected network
        if self.stack_instead_add:
            self.W_classif = tf.Variable(tf.random_uniform((self.node_indim*self.nconv_edge+self.node_indim, self.n_classes),
                                                           -1.0 / math.sqrt(self.node_dim),
                                                           1.0 / math.sqrt(self.node_dim)),
                                        name="W_classif",dtype=np.float32)
        else:
            self.W_classif = tf.Variable(tf.random_uniform((self.node_indim*self.nconv_edge, self.n_classes),
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
            #Em =(tf.matmul(self.Wedge,self.EA_input))
            #Z=tf.reshape(Em,tf.stack([self.nb_node,self.nb_node]))
            P =self.convolve(self.Wedge,self.EA_input,Hi_,self.nb_node,self.nconv_edge)
            if self.stack_instead_add:
                #P= tf.matmul(Z,Hi_)
                Hp= tf.concat([Hi_,P],1)
            else:
                #Hp= tf.matmul(Z+I,Hi_) #If multiple edge, can not add as it does not have the same dimensionality
                Hp= P+Hi_
            Hi=self.activation(Hp)
            Hi_shape = Hi.get_shape()
            print(Hi_shape)
            self.hidden_layers.append(Hi)

        elif self.num_layers>1:
            #TODO Fix activation before convolve or after ..
            #TODO Fix probelem of id betwwen layers layer 0 adn 1 are mixed
            H0 = self.activation(tf.add(tf.matmul(self.node_input,Wnl0),Bnl0))

            if self.stack_instead_add:
                #Em0 =(tf.matmul(Wel0,self.EA_input))
                #Z0=tf.reshape(Em0,tf.stack([self.nb_node,self.nb_node]))
                #P= tf.matmul(Z0,H0)
                P=self.convolve(self.Wel0,self.EA_input,H0,self.nb_node,self.nconv_edge)
                Hp= tf.concat([H0,P],1)
                self.hidden_layers=[Hp]
            else:
                self.hidden_layers=[H0]

            print('H0_shape',self.hidden_layers[0].get_shape())
            for i in range(self.num_layers-1):
                Hi_ = tf.matmul(self.hidden_layers[-1],self.Wnode_layers[i])
                print('Hi_shape',Hi_.get_shape())
                #Emi = (tf.matmul(self.Wed_layers[i],self.EA_input))
                P=self.convolve(self.Wed_layers[i],self.EA_input,Hi_,self.nb_node,self.nconv_edge)
                #Z=tf.reshape(Emi,tf.stack([self.nb_node,self.nb_node]))
                print('P_shape',P.get_shape())
                if self.stack_instead_add:
                    #P= tf.matmul(Z,Hi_)
                    Hp= tf.concat([Hi_,P],1)
                    Hi=self.activation(Hp)
                    Hi_shape= Hi.get_shape()
                    print(Hi_shape)
                    self.hidden_layers.append([Hi])
                else:
                    #Hp= tf.matmul(Z+I,Hi_)
                    Hp= P+Hi_ #Looks like it is going to break with multiple convolution here
                    Hi=self.activation(Hp)
                self.hidden_layers.append(Hi)

        self.logits =tf.add(tf.matmul(self.hidden_layers[-1],self.W_classif),self.B_classif)
        #print('Logits ...')
        #print(self.logits_.get_shape())
        #Convolve the Logits as well
        #works only with 1 convolve
        #self.logits_convolve =self.convolve(self.Wedge_logit,self.EA_input,self.logits_,self.nb_node,1)
        #print(self.logits_convolve.get_shape())
        #Reduce_sum, reduce_max ?here not clear
        #self.A=tf.matmul(self.logits_convolve,self.Yt) #I should not take the sum over neighbor but the max here
        #print(self.A.get_shape())

        #self.logits= tf.add(self.logits_,tf.reduce_sum(tf.matmul(self.logits_convolve,self.Yt),axis=1))
        #self.logits = tf.add(tf.nn.dropout(self.logits_,0.5), self.A)
        #self.logits = self.logits_
        cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
        #cross_entropy_neighbor = tf.nn.softmax_cross_entropy_with_logits(logits=self.A, labels=self.y_input)
        #cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.A, labels=self.y_input)

        # Global L2 Regulization
        if self.num_layers > 1:
            in_layers_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.Wnode_layers]) * self.mu
            self.loss = tf.reduce_mean(cross_entropy_source)  + self.mu * tf.nn.l2_loss(
                self.W_classif) + self.mu * tf.nn.l2_loss(self.Wnode) + in_layers_l2
        else:
            self.loss = tf.reduce_mean(cross_entropy_source) + self.mu * tf.nn.l2_loss(
                self.W_classif) + self.mu * tf.nn.l2_loss(self.Wnode)

        #self.loss = tf.reduce_mean(cross_entropy_source)+self.mu*tf.nn.l2_loss(self.W_classif) +self.mu*tf.nn.l2_loss(self.Wedge)


        self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.grads_and_vars = self.optalg.compute_gradients(self.loss)
        self.train_step = self.optalg.apply_gradients(self.grads_and_vars)


        # Add ops to save and restore all the variables.
        self.init = tf.global_variables_initializer()

    def get_Wedge(self,session):
        if self.num_layers>1:
            L0=session.run([self.Wel0,self.Wed_layers])
            We0=L0[0]
            list_we=[We0]
            for we in L0[1]:
                list_we.append(we)
            return list_we
        else:
            raise NotImplementedError

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



    def test(self,session,n_node,X,EA,Y,NA,verbose=True):
        #TrainEvalSet Here
        feed_batch={
                        self.nb_node:n_node,
                        self.node_input:X,
                        self.EA_input:EA,
                        self.y_input:Y,
                        self.NA_input:NA,
        }
        Ops =session.run([self.loss,self.accuracy], feed_dict=feed_batch)
        if verbose:
            print('Test Loss',Ops[0],' Test Accuracy:',Ops[1])
        return Ops[1]


    def train_lG(self,session,gcn_graph_train):
        for g in gcn_graph_train:
            self.train(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA, n_iter=1)


    def test_lG(self,session,gcn_graph_test,verbose=True):
        '''
        Test on a list of Graph
        :param session:
        :param gcn_graph_test:
        :return:
        '''
        acc_tp = 0.0
        nb_node_total = 0.0
        mean_acc_test = []

        for g in gcn_graph_test:
            acc = self.test(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA, verbose=False)
            mean_acc_test.append(acc)
            nb_node_total += g.X.shape[0]
            acc_tp += acc * g.X.shape[0]

        g_acc =np.mean(mean_acc_test)
        node_acc =acc_tp / nb_node_total

        if verbose:
            print('Mean Graph Accuracy', '%.4f' %g_acc)
            print('Mean Node  Accuracy', '%.4f' %node_acc)

        return g_acc,node_acc


    def train_with_validation_set(self,session,graph_train,graph_val,max_iter,eval_iter=10,patience=7,graph_test=None):
        best_val_acc=0.0
        wait=0
        stop_training=False
        stopped_iter=max_iter
        train_accuracies=[]
        validation_accuracies=[]
        test_accuracies=[]

        start_monitoring_val_acc=False

        for i in range(max_iter):
            if stop_training:
                break

            if i % eval_iter == 0:
                print('\nEpoch', i)
                _, tr_acc = self.test_lG(session, graph_train,verbose=False)
                print(' Train Acc', '%.4f' % tr_acc)
                train_accuracies.append(tr_acc)

                _,node_acc=self.test_lG(session,graph_val,verbose=False)
                print(' Valid Acc', '%.4f' % node_acc)
                validation_accuracies.append(node_acc)

                if graph_test:
                    _,test_acc=self.test_lG(session,graph_test,verbose=False)
                    print('  Test Acc', '%.4f' % test_acc)
                    test_accuracies.append(test_acc)

                #TODO min_delta
                #if tr_acc>0.99:
                #    start_monitoring_val_acc=True

                if node_acc > best_val_acc:
                    best_val_acc=node_acc
                    wait = 0
                else:
                    if wait >= patience:
                        stopped_iter = i
                        stop_training = True
                    wait += 1
            else:
                random.shuffle(graph_train)
                for g in graph_train:
                    self.train(session, g.X.shape[0], g.X, g.EA, g.Y, g.NA, n_iter=1)

        mean_acc = []
        print('Stopped Model Training after',stopped_iter)
        print('Val Accuracies',validation_accuracies)

        print('Final Training Accuracy')
        _,node_train_acc=self.test_lG(session,graph_train)
        print('Train Mean Accuracy','%.4f' % node_train_acc)

        print('Final Valid Acc')
        self.test_lG(session,graph_val)

        R = {}
        R['train_acc'] = train_accuracies
        R['val_acc'] = validation_accuracies
        R['test_acc'] = test_accuracies
        R['stopped_iter'] = stopped_iter

        R['W_edge'] =self.get_Wedge(session)
        if graph_test:

            _, final_test_acc = self.test_lG(session, graph_test)
            print('Final Test Acc','%.4f' % final_test_acc)
            R['final_test_acc'] = final_test_acc

        return R




class GCNBaselineGraphList(object):
    '''
    Class for A Deep Standard GCN model for a graph list
    '''
    def __init__(self,node_dim,nb_classes,num_layers=1,learning_rate=0.1,mu=0.1,node_indim=-1,
                 dropout_rate=0.0,dropout_mode=0):
        self.node_dim=node_dim
        self.n_classes=nb_classes
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.activation=tf.nn.relu
        self.mu=mu
        self.optalg = tf.train.AdamOptimizer(self.learning_rate)
        self.convolve_last=False
        self.dropout_rate=dropout_rate
        #0 No dropout 1, Node Dropout at input 2 Standard dropout for layer
        # check logit layer
        self.dropout_mode=dropout_mode

        if node_indim==-1:
            self.node_indim=self.node_dim
        else:
            self.node_indim=node_indim

    def create_model(self):
        self.nb_node    = tf.placeholder(tf.int32,(), name='nb_node')
        self.node_input = tf.placeholder(tf.float32, [None, self.node_dim], name='X_')
        self.y_input    = tf.placeholder(tf.float32, [None, self.n_classes], name='Y')
        self.NA_input   = tf.placeholder(tf.float32, name='NA_input') #Normalized Adjacency Matrix Here
        self.dropout_p  = tf.placeholder(tf.float32,(), name='dropout_prob')


        #std_dev_in=float(1.0/ float(self.node_dim))
        self.Wnode_layers=[]
        self.Bnode_layers=[]


        std_dev_input=float(1.0/ float(self.node_dim))
        std_dev_indim=float(1.0/ float(self.node_indim))


        if self.node_indim!=self.node_dim:
            self.Wnode = init_glorot((self.node_dim,self.node_indim),name='Wn0')
            #self.Wnode = init_normal((self.node_dim, self.node_indim),std_dev_input,name='Wn0')
        else:
            self.Wnode = tf.Variable(tf.eye(self.node_dim),name='Wn0',dtype=tf.float32)

        self.Bnode = tf.Variable(tf.zeros([self.node_indim]), name='Bnode',dtype=tf.float32)


        for i in range(self.num_layers-1):
            Wnli =init_glorot((self.node_indim, self.node_indim),name='Wnl'+str(i))
            #Wnli = init_normal((self.node_indim, self.node_indim),std_dev_indim, name='Wnl' + str(i))
            self.Wnode_layers.append(Wnli)
            #The GCN do not seem to use a bias term
            #Bnli = tf.Variable(tf.zeros([self.node_indim]), name='Bnl'+str(i),dtype=tf.float32)
            #self.Bnode_layers.append(Bnli)

        self.W_classif = init_glorot((self.node_indim, self.n_classes),name="W_classif")
        self.B_classif = tf.Variable(tf.zeros([self.n_classes]), name='B_classif',dtype=np.float32)


        #Input Layer
        #Check the self-loop . Is included in the normalized adjacency matrix
        #Check Residual Connections as weel for deeper models

        #add dropout_placeholder ... to differentiate train and test
        #x = tf.nn.dropout(x, 1 - self.dropout)


        #Dropout some nodes at the input of the graph ?
        #Should I dropout in upper layers as well ?
        #This way this forces the net to infer the node labels from its neighbors only
        #Here I dropout the features, but node the edges ..

        self.node_dropout_ind =  tf.nn.dropout(tf.ones([self.nb_node],dtype=tf.float32),1-self.dropout_p)
        self.ND = tf.diag(self.node_dropout_ind)



        if self.dropout_mode==1:
            self.H = self.activation(tf.matmul(self.ND,tf.add(tf.matmul(self.node_input, self.Wnode), self.Bnode)))
            self.hidden_layers = [self.H]
        else:
            self.H = self.activation(tf.add(tf.matmul(self.node_input, self.Wnode), self.Bnode))
            self.hidden_layers = [self.H]

        for i in range(self.num_layers-1):
            if self.dropout_mode==2:
                Hp = tf.nn.dropout(self.hidden_layers[-1],1-self.dropout_p)
                Hi_ = tf.matmul(Hp, self.Wnode_layers[i])
            else:
                Hi_ = tf.matmul(self.hidden_layers[-1], self.Wnode_layers[i])
            Hi = self.activation(tf.matmul(self.NA_input, Hi_))
            self.hidden_layers.append(Hi)


        #This dropout the logits as in GCN
        if self.dropout_mode==2:
            Hp = tf.nn.dropout(self.hidden_layers[-1], 1 - self.dropout_p)
            self.hidden_layers.append(Hp)

        if self.convolve_last is True:
            logit_0     = tf.add(tf.matmul(self.hidden_layers[-1], self.W_classif), self.B_classif)
            self.logits =  tf.matmul(self.NA_input,logit_0) #No activation function here
        else:
            self.logits =tf.add(tf.matmul(self.hidden_layers[-1],self.W_classif),self.B_classif)

        cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)

        #Global L2 Regulization
        #TODO Add L2 on the Edges as well ?
        if self.num_layers>1:
            in_layers_l2=tf.add_n([ tf.nn.l2_loss(v) for v in self.Wnode_layers ]) * self.mu
            self.loss = tf.reduce_mean(cross_entropy_source)+self.mu*tf.nn.l2_loss(self.W_classif) +self.mu*tf.nn.l2_loss(self.Wnode)+in_layers_l2
        else:
            self.loss = tf.reduce_mean(cross_entropy_source) + self.mu * tf.nn.l2_loss(self.W_classif) + self.mu * tf.nn.l2_loss(self.Wnode)


        self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.grads_and_vars = self.optalg.compute_gradients(self.loss)
        self.train_step = self.optalg.apply_gradients(self.grads_and_vars)


        # Add ops to save and restore all the variables.
        self.init = tf.global_variables_initializer()

    def train(self,session,n_node,X,Y,NA,n_iter=1,verbose=False):
        #TrainEvalSet Here
        for i in range(n_iter):
            feed_batch={
                        self.nb_node:n_node,
                        self.node_input:X,
                        self.y_input:Y,
                        self.NA_input:NA,
                        self.dropout_p:self.dropout_rate
            }
            Ops =session.run([self.train_step,self.loss], feed_dict=feed_batch)
            if verbose:
                print('Training Loss',Ops[1])



    def test(self,session,n_node,X,Y,NA,verbose=True):
        #TrainEvalSet Here
        feed_batch={
                        self.nb_node:n_node,
                        self.node_input:X,
                        self.y_input:Y,
                        self.NA_input:NA,
                        self.dropout_p: 0.0
        }
        Ops =session.run([self.loss,self.accuracy], feed_dict=feed_batch)
        if verbose:
            print('Test Loss',Ops[0],' Test Accuracy:',Ops[1])
        return Ops[1]

