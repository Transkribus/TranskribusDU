# -*- coding: utf-8 -*-
"""
@author: Stéphane Clinchant

"""

import tensorflow as tf
import math
import numpy as np
import scipy.sparse as sp
import random
import sklearn
import sklearn.metrics
import time

def init_glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def init_normal(shape,stddev,name=None):
    initial=tf.random_normal(shape, mean=0.0, stddev=stddev, dtype=np.float32)
    return tf.Variable(initial, name=name)


class DummyGCNModel(object):
    '''
    First Implementation of GCN. Working only on a single graph.
    This class was developped to perform the unit test on the iris dataset
    '''

    def __init__(self,gcn_dataset,num_layers=1,learning_rate=0.1,mu=0.1):

        self.dataset =gcn_dataset
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.activation=tf.nn.relu
        self.mu=mu
        self.learn_edge=False #if false standard GCN, else it learning 1 convolutions for edges

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
            #Build the Edge-Adjacency Matrix. Very naive and stupid implemenation ...
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


        # Initialize the weights and biases for a simple one full connected network
        self.W_classif = tf.Variable(tf.random_uniform((self.node_dim, self.n_classes),
                                                       -1.0 / math.sqrt(self.node_dim),
                                                       1.0 / math.sqrt(self.node_dim)),
                                     name="W_classif",dtype=np.float32)
        self.B_classif = tf.Variable(tf.zeros([self.n_classes]), name='B_classif',dtype=np.float32)


        self.H = self.activation(tf.add(tf.matmul(self.node_input,self.Wnode),self.Bnode))
        self.hidden_layers=[self.H]


        #Computing here some constant matrix for the convolution of GCN Model
        Dinv_ = np.diag(np.power(self.dataset.A.sum(axis=1),-0.5))
        N=tf.constant(np.dot(Dinv_,self.dataset.A+np.identity(self.dataset.A.shape[0]).dot(Dinv_)),dtype=np.float32)



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


        self.logits =tf.add(tf.matmul(self.hidden_layers[-1],self.W_classif),self.B_classif)


        cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)

        self.loss = tf.reduce_mean(cross_entropy_source)+self.mu*tf.nn.l2_loss(self.W_classif) +self.mu*tf.nn.l2_loss(self.Wedge)


        self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


        #self.optalg = tf.train.AdagradOptimizer(self.learning_rate)
        self.optalg = tf.train.AdamOptimizer(self.learning_rate)

        self.grads_and_vars = self.optalg.compute_gradients(self.loss)
        self.train_step = self.optalg.apply_gradients(self.grads_and_vars)


        # Add ops to save and restore all the variables.
        self.init = tf.global_variables_initializer()


    def train(self,session,n_iter=10):
        #TrainEvalSet Here
        for i in range(n_iter):
            feed_batch={self.node_input:self.dataset.X,
                        self.y_input:self.dataset.Y
            }
            Ops =session.run([self.train_step,self.loss,self.accuracy], feed_dict=feed_batch)
            print('Training Loss',Ops[1],'Accuracy:',Ops[2])



class MultiGraphNN(object):
    '''
    Abstract Class for a Neural Net learned on a graph list

    '''
    def train_lG(self,session,gcn_graph_train):
        '''
        Train an a list of graph
        :param session:
        :param gcn_graph_train:
        :return:
        '''
        for g in gcn_graph_train:
            self.train(session, g, n_iter=1)


    def test_lG(self,session,gcn_graph_test,verbose=True):
        '''
        Test on a list of Graph
        :param session:
        :param gcn_graph_test:
        :return:
        '''
        acc_tp = np.float64(0.0)
        nb_node_total = np.float64(0.0)
        mean_acc_test = []

        for g in gcn_graph_test:
            acc = self.test(session, g, verbose=False)
            mean_acc_test.append(acc)
            nb_node_total += g.X.shape[0]
            acc_tp += acc * g.X.shape[0]

        g_acc =np.mean(mean_acc_test)
        node_acc =acc_tp / nb_node_total

        if verbose:
            print('Mean Graph Accuracy', '%.4f' %g_acc)
            print('Mean Node  Accuracy', '%.4f' %node_acc)

        return g_acc,node_acc

    def predict_lG(self,session,gcn_graph_predict,verbose=True):
        '''
        Predict for a list of graph
        :param session:
        :param gcn_graph_test:
        :return:
        '''
        lY_pred=[]

        for g in gcn_graph_predict:
            gY_pred = self.predict(session, g, verbose=verbose)
            lY_pred.append(gY_pred)


        return lY_pred


    def get_nb_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            #print(shape)
            #print(len(shape))
            variable_parameters = 1
            for dim in shape:
                #print(dim)
                variable_parameters *= dim.value
            #print(variable_parameters)
            total_parameters += variable_parameters
        print(total_parameters)

    def train_with_validation_set(self,session,graph_train,graph_val,max_iter,eval_iter=10,patience=7,graph_test=None,save_model_path=None):
        '''
        Implements training with a validation set
        The model is trained and accuracy is measure on a validation sets
        In addition, the model can be save and one can perform early stopping thanks to the patience argument

        :param session:
        :param graph_train: the list of graph to train on
        :param graph_val:   the list of graph used for validation
        :param max_iter:  maximum number of epochs
        :param eval_iter: evaluate every eval_iter
        :param patience: stopped training if accuracy is not improved on the validation set after patience_value
        :param graph_test: Optional. If a test set is provided, then accuracy on the test set is reported
        :param save_model_path: checkpoints filename to save the model.
        :return: A Dictionary with training accuracies, validations accuracies and test accuracies if any, and the Wedge parameters
        '''
        best_val_acc=0.0
        wait=0
        stop_training=False
        stopped_iter=max_iter
        train_accuracies=[]
        validation_accuracies=[]
        test_accuracies=[]
        conf_mat=[]

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

                if save_model_path:
                    save_path = self.saver.save(session, save_model_path,global_step=i)

                if graph_test:
                    _,test_acc=self.test_lG(session,graph_test,verbose=False)
                    print('  Test Acc', '%.4f' % test_acc)
                    test_accuracies.append(test_acc)


                    #Ypred = self.predict_lG(session, graph_test,verbose=False)
                    #Y_true_flat = []
                    #Ypred_flat = []
                    #for graph, ypred in zip(graph_test, Ypred):
                    #    ytrue = np.argmax(graph.Y, axis=1)
                    #    Y_true_flat.extend(ytrue)
                    #    Ypred_flat.extend(ypred)
                    #cm = sklearn.metrics.confusion_matrix(Y_true_flat, Ypred_flat)
                    #conf_mat.append(cm)


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
                    self.train(session, g, n_iter=1)
        #Final Save
        #if save_model_path:
        #save_path = self.saver.save(session, save_model_path, global_step=i)
        #TODO Add the final step
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
        R['confusion_matrix'] = conf_mat
        #R['W_edge'] =self.get_Wedge(session)
        if graph_test:

            _, final_test_acc = self.test_lG(session, graph_test)
            print('Final Test Acc','%.4f' % final_test_acc)
            R['final_test_acc'] = final_test_acc

        return R


class Logit(MultiGraphNN):
    '''
    Logistic Regression for MultiGraph
    '''
    def __init__(self,node_dim,nb_classes,learning_rate=0.1,mu=0.1,node_indim=-1):
        self.node_dim=node_dim
        self.n_classes=nb_classes
        self.learning_rate=learning_rate
        self.activation=tf.nn.relu
        self.mu=mu
        self.optalg = tf.train.AdamOptimizer(self.learning_rate)
        self.stack_instead_add=False
        self.train_Wn0=True

        if node_indim==-1:
            self.node_indim=self.node_dim
        else:
            self.node_indim=node_indim

    def create_model(self):
        '''
        Create the tensorflow graph for the model
        :return:
        '''
        self.nb_node    = tf.placeholder(tf.int32,(), name='nb_node')
        self.node_input = tf.placeholder(tf.float32, [None, self.node_dim], name='X_')
        self.y_input    = tf.placeholder(tf.float32, [None, self.n_classes], name='Y')


        self.Wnode_layers=[]
        self.Bnode_layers=[]


        self.W_classif = tf.Variable(tf.random_uniform((self.node_indim, self.n_classes),
                                                           -1.0 / math.sqrt(self.node_dim),
                                                           1.0 / math.sqrt(self.node_dim)),
                                        name="W_classif",dtype=np.float32)
        self.B_classif = tf.Variable(tf.zeros([self.n_classes]), name='B_classif',dtype=np.float32)


        self.logits =tf.add(tf.matmul(self.node_input,self.W_classif),self.B_classif)
        cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)

        # Global L2 Regulization
        self.loss = tf.reduce_mean(cross_entropy_source)  + self.mu * tf.nn.l2_loss(self.W_classif)


        self.pred = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.correct_prediction = tf.equal(self.pred, tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


        self.grads_and_vars = self.optalg.compute_gradients(self.loss)

        self.train_step = self.optalg.apply_gradients(self.grads_and_vars)


        # Add ops to save and restore all the variables.
        self.init = tf.global_variables_initializer()
        self.saver= tf.train.Saver(max_to_keep=5)

        print('Number of Params:')
        self.get_nb_params()


    def save_model(self, session, model_filename):
        print("Saving Model")
        save_path = self.saver.save(session, model_filename)

    def restore_model(self, session, model_filename):
        self.saver.restore(session, model_filename)
        print("Model restored.")


    def train(self,session,graph,verbose=False,n_iter=1):
        '''
        Apply a train operation, ie sgd step for a single graph
        :param session:
        :param graph: a graph from GCN_Dataset
        :param verbose:
        :param n_iter: (default 1) number of steps to perform sgd for this graph
        :return:
        '''
        #TrainEvalSet Here
        for i in range(n_iter):
            #print('Train',X.shape,EA.shape)

            feed_batch = {

                self.nb_node: graph.X.shape[0],
                self.node_input: graph.X,
                self.y_input: graph.Y,
            }

            Ops =session.run([self.train_step,self.loss], feed_dict=feed_batch)
            if verbose:
                print('Training Loss',Ops[1])



    def test(self,session,graph,verbose=True):
        '''
        Test return the loss and accuracy for the graph passed as argument
        :param session:
        :param graph:
        :param verbose:
        :return:
        '''

        feed_batch = {

            self.nb_node: graph.X.shape[0],
            self.node_input: graph.X,
            self.y_input: graph.Y,
        }

        Ops =session.run([self.loss,self.accuracy], feed_dict=feed_batch)
        if verbose:
            print('Test Loss',Ops[0],' Test Accuracy:',Ops[1])
        return Ops[1]


    def predict(self,session,graph,verbose=True):
        '''
        Does the prediction
        :param session:
        :param graph:
        :param verbose:
        :return:
        '''
        feed_batch = {
            self.nb_node: graph.X.shape[0],
            self.node_input: graph.X,
        }
        Ops = session.run([self.pred], feed_dict=feed_batch)
        if verbose:
            print('Got Prediction for:',Ops[0].shape)
        return Ops[0]






class EdgeConvNet(MultiGraphNN):
    '''
    Edge-GCN Model for a graph list
    '''

    def __init__(self,node_dim,edge_dim,nb_classes,num_layers=1,learning_rate=0.1,mu=0.1,node_indim=-1,nconv_edge=1,
                 residual_connection=False,shared_We=False):
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
        self.residual_connection=residual_connection
        self.shared_We = shared_We
        self.optim_mode=0 #deprecated
        self.fast_convolve=False
        self.init_fixed=False
        self.logit_convolve=False
        self.train_Wn0=True

        self.dropout_rate_edge_feat= 0.0
        self.dropout_rate_edge = 0.0
        self.dropout_rate_node = 0.0
        self.dropout_rate_H    = 0.0


        if node_indim==-1:
            self.node_indim=self.node_dim
        else:
            self.node_indim=node_indim


    def fastconvolve(self,Wedge,F,S,T,H,nconv,Sshape,nb_edge,dropout_p_edge,dropout_p_edge_feat,stack=True, use_dropout=False):
        '''

        :param Wedge: Parameter matrix for edge convolution, with hape (n_conv_edge,edge_dim)
        :param F: The Edge Feature Matrix
        :param S: the Source (Node,Edge) matrix in sparse format
        :param T: the Target (Node,Edge) matrix in sparse format
        :param H: The current node layer
        :param nconv: The numbder of edge convolutions.
        :param Sshape: The shapeof matrix S, and T
        :param nb_edge: The number of edges
        :param stack: whether to concat all the convolutions or add them
        :return: a tensor P of shape ( nconv, node_dim) if stack else P is [node_dim]
        '''

        #F is n_edge time nconv

        #TODO if stack is False we could simply sum,the convolutions and do S diag(sum)T
        #It woudl be faster
        #UnitTest for that
        FW= tf.matmul(F,Wedge,transpose_b=True)
        #
        #SHould I use dropout below too to dropout edges ?
        if use_dropout:
            FW= tf.nn.dropout(FW, 1.0 -dropout_p_edge_feat)

        self.conv =tf.unstack(FW,axis=1)
        Cops=[]

        #Here we transospose the target tensor direclty
        Tr = tf.SparseTensor(indices=T, values=tf.ones([nb_edge], dtype=tf.float32), dense_shape=[Sshape[1],Sshape[0]])
        Tr = tf.sparse_reorder(Tr)
        #if use_dropout:
        #    Tr =tf.nn.dropout(Tr, 1.0 -dropout_p_edge)
        TP = tf.sparse_tensor_dense_matmul(Tr,H)



        for i, cw in enumerate(self.conv):
            #SD= tf.SparseTensor(indices=S,values=cw,dense_shape=[nb_node,nb_edge])
            #Warning, pay attention to the ordering of edges
            if use_dropout:
                cwd = tf.nn.dropout(cw, 1.0 -dropout_p_edge)
                SD = tf.SparseTensor(indices=S, values=cwd, dense_shape=Sshape)
            else:
                SD = tf.SparseTensor(indices=S, values=cw, dense_shape=Sshape)
            SD =tf.sparse_reorder(SD)
            #Does this dropout depends on the convolution ?
            #if use_dropout:
            #    SD = tf.nn.dropout(SD, 1.0 - dropout_p_edge)

            Hi =tf.sparse_tensor_dense_matmul(SD,TP)
            Cops.append(Hi)

        if stack is True:
            #If stack we concatenate all the different convolutions
            P=tf.concat(Cops,1)
        else:
            #Else we take the mean
            P=1.0/(tf.cast(nconv,tf.float32))*tf.add_n(Cops)
            #print('p_add_n',P.get_shape())
        return P


    def create_model(self):
        '''
        Create the tensorflow graph for the model
        :return:
        '''
        self.nb_node    = tf.placeholder(tf.int32,(), name='nb_node')
        self.nb_edge = tf.placeholder(tf.int32, (), name='nb_edge')
        self.node_input = tf.placeholder(tf.float32, [None, self.node_dim], name='X_')
        self.y_input    = tf.placeholder(tf.float32, [None, self.n_classes], name='Y')
        self.EA_input   = tf.placeholder(tf.float32, name='EA_input')
        self.NA_input   = tf.placeholder(tf.float32, name='NA_input')
        self.dropout_p_H    = tf.placeholder(tf.float32,(), name='dropout_prob_H')
        self.dropout_p_node = tf.placeholder(tf.float32, (), name='dropout_prob_N')
        self.dropout_p_edge = tf.placeholder(tf.float32, (), name='dropout_prob_edges')
        self.dropout_p_edge_feat = tf.placeholder(tf.float32, (), name='dropout_prob_edgefeat')
        self.S          = tf.placeholder(tf.float32, name='S')
        self.Ssparse    = tf.placeholder(tf.int64, name='Ssparse') #indices
        self.Sshape     = tf.placeholder(tf.int64, name='Sshape') #indices

        self.T          = tf.placeholder(tf.float32,[None,None], name='T')
        self.Tsparse    = tf.placeholder(tf.int64, name='Tsparse')

        #self.S_indice = tf.placeholder(tf.in, [None, None], name='S')

        self.F          = tf.placeholder(tf.float32,[None,None], name='F')

        #TODO Add Bias
        #TODO Dropout Options, node, layer node, convolution,mixture of this --> each variant -> options
        # how to dropout the convolutions FW

        std_dev_in=float(1.0/ float(self.node_dim))

        self.Wnode_layers=[]
        self.Bnode_layers=[]
        self.Wed_layers=[]
        self.Bed_layers=[]

        #Should Project edge as well ...
        train_var=[]

        if self.node_indim!=self.node_dim:
            Wnl0 = tf.Variable(tf.random_uniform((self.node_dim, self.node_indim),
                                                                   -1.0 / math.sqrt(self.node_dim),
                                                                   1.0 / math.sqrt(self.node_dim)),name='Wnl0',dtype=tf.float32)
        else:
            Wnl0 = tf.Variable(tf.eye(self.node_dim),name='Wnl0',dtype=tf.float32,trainable=self.train_Wn0)

        Bnl0 = tf.Variable(tf.zeros([self.node_indim]), name='Bnl0',dtype=tf.float32)
        #self.Wel0 =tf.Variable(tf.random_normal([int(self.nconv_edge),int(self.edge_dim)],mean=0.0,stddev=1.0), dtype=np.float32, name='Wel0')
        if self.init_fixed:
            self.Wel0 = tf.Variable(100*tf.ones([int(self.nconv_edge),int(self.edge_dim)]), name='Wel0',dtype=tf.float32)
        else:
            self.Wel0 = init_glorot([int(self.nconv_edge),int(self.edge_dim)],name='Wel0')

        print('Wel0',self.Wel0.get_shape())
        train_var.extend([Wnl0,Bnl0])
        train_var.append(self.Wel0)

        #Parameter for convolving the logits
        if self.logit_convolve:
            self.Wel_logits = init_glorot([int(self.nconv_edge),int(self.edge_dim)],name='Wel_logit')
            self.logits_Transition = tf.Variable(tf.ones([int(self.n_classes), int(self.n_classes)]), name='logit_Transition')

        #self.Wed_layers.append(Wel0)
        for i in range(self.num_layers-1):
            if self.stack_instead_add:
                Wnli =tf.Variable(tf.random_uniform( (self.node_indim*self.nconv_edge+self.node_indim, self.node_indim),
                                                               -1.0 / math.sqrt(self.node_indim),
                                                               1.0 / math.sqrt(self.node_indim)),name='Wnl',dtype=tf.float32)
                print('Wnli shape',Wnli.get_shape())
            else:
                Wnli =tf.Variable(tf.random_uniform( (2*self.node_indim, self.node_indim),
                                                                   -1.0 / math.sqrt(self.node_indim),
                                                                   1.0 / math.sqrt(self.node_indim)),name='Wnl',dtype=tf.float32)

            #Bnli = tf.Variable(tf.zeros([self.node_indim]), name='Bnl'+str(i),dtype=tf.float32)

            #Weli = tf.Variable(tf.ones([int(self.nconv_edge),int(self.edge_dim)],dtype=tf.float32))
            Weli= init_glorot([int(self.nconv_edge), int(self.edge_dim)], name='Wel_')
            #Weli = tf.Variable(tf.random_normal([int(self.nconv_edge), int(self.edge_dim)], mean=0.0, stddev=1.0),
            #                   dtype=np.float32, name='Wel_')
            #Beli = tf.Variable(tf.zeros([self.edge_dim]), name='Bel'+str(i),dtype=tf.float32)

            self.Wnode_layers.append(Wnli)
            #self.Bnode_layers.append(Bnli)
            self.Wed_layers.append  (Weli)
            #self.Bed_layers.append(Beli)

        train_var.extend((self.Wnode_layers))
        train_var.extend((self.Wed_layers))

        self.Hnode_layers=[]



        #TODO Do we project the firt layer or not ?
        # Initialize the weights and biases for a simple one full connected network
        if self.stack_instead_add:
            self.W_classif = tf.Variable(tf.random_uniform((self.node_indim*self.nconv_edge+self.node_indim, self.n_classes),
                                                           -1.0 / math.sqrt(self.node_dim),
                                                           1.0 / math.sqrt(self.node_dim)),
                                        name="W_classif",dtype=np.float32)
        else:
            self.W_classif = tf.Variable(tf.random_uniform((2*self.node_indim, self.n_classes),
                                                           -1.0 / math.sqrt(self.node_dim),
                                                           1.0 / math.sqrt(self.node_dim)),
                                        name="W_classif",dtype=np.float32)
        self.B_classif = tf.Variable(tf.zeros([self.n_classes]), name='B_classif',dtype=np.float32)

        train_var.append((self.W_classif))
        train_var.append((self.B_classif))


        #Use for true add
        #I = tf.eye(self.nb_node)

        self.node_dropout_ind = tf.nn.dropout(tf.ones([self.nb_node], dtype=tf.float32), 1 - self.dropout_p_node)
        self.ND = tf.diag(self.node_dropout_ind)

        edge_dropout = self.dropout_rate_edge> 0.0 or self.dropout_rate_edge_feat > 0.0
        print('Edge Dropout',edge_dropout, self.dropout_rate_edge,self.dropout_rate_edge_feat)
        if self.num_layers==1:
            self.H = self.activation(tf.add(tf.matmul(self.node_input, Wnl0), Bnl0))
            self.hidden_layers = [self.H]
            print("H shape",self.H.get_shape())


            P = self.fastconvolve(self.Wel0,self.F,self.Ssparse,self.Tsparse,self.H,self.nconv_edge,self.Sshape,self.nb_edge,
                                  self.dropout_p_edge,self.dropout_p_edge_feat,stack=self.stack_instead_add,use_dropout=edge_dropout)

            Hp = tf.concat([self.H, P], 1)
            #Hp= P+self.H

            Hi=self.activation(Hp)
            #Hi_shape = Hi.get_shape()
            #print(Hi_shape)
            self.hidden_layers.append(Hi)

        elif self.num_layers>1:

            if self.dropout_rate_node>0.0:
                H0 = self.activation(tf.matmul(self.ND,tf.add(tf.matmul(self.node_input, Wnl0), Bnl0)))
            else:
                H0 = self.activation(tf.add(tf.matmul(self.node_input,Wnl0),Bnl0))

            self.Hnode_layers.append(H0)

            #TODO Default to fast convolve but we change update configs, train and test flags
            P = self.fastconvolve(self.Wel0, self.F, self.Ssparse, self.Tsparse, H0, self.nconv_edge, self.Sshape,self.nb_edge,
                                  self.dropout_p_edge,self.dropout_p_edge_feat, stack=self.stack_instead_add, use_dropout=edge_dropout)

            Hp = tf.concat([H0, P], 1)
            self.hidden_layers = [Hp]


            for i in range(self.num_layers-1):

                if self.dropout_rate_H > 0.0:
                    Hi_ = tf.nn.dropout(tf.matmul(self.hidden_layers[-1], self.Wnode_layers[i]), 1-self.dropout_p_H)
                else:
                    Hi_ = tf.matmul(self.hidden_layers[-1], self.Wnode_layers[i])

                if self.residual_connection:
                    Hi_= tf.add(Hi_,self.Hnode_layers[-1])

                self.Hnode_layers.append(Hi_)

                print('Hi_shape',Hi_.get_shape())
                print('Hi prevous shape',self.hidden_layers[-1].get_shape())

                P = self.fastconvolve(self.Wed_layers[i], self.F, self.Ssparse, self.Tsparse, Hi_, self.nconv_edge,self.Sshape, self.nb_edge,
                                      self.dropout_p_edge,self.dropout_p_edge_feat, stack=self.stack_instead_add, use_dropout=edge_dropout)


                Hp = tf.concat([Hi_, P], 1)
                Hi = self.activation(Hp)

                self.hidden_layers.append(Hi)


        if self.logit_convolve is False:
            self.logits =tf.add(tf.matmul(self.hidden_layers[-1],self.W_classif),self.B_classif)
            cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
        else:
        # TODO Convolve Logits and learn a transition matrix   #Reduce_sum, reduce_max ? use Dropout ?
            #Ideally I should convolve the predition and not the logits
            #I would need to have my own cross entropy function , smooth it
            #add entropy regulaztion
            #this is the current logit multiply by  Transition matrix
            #check that we are not summing indeed in a correct direction
            self.logits = tf.add(tf.matmul(self.hidden_layers[-1], self.W_classif), self.B_classif)
            self.logits_T = tf.matmul(self.logits,self.logits_Transition)
            self.logits_convolve= tf.nn.dropout(self.fastconvolve(self.Wel_logits,self.F,self.Ssparse,self.Tsparse,self.logits_T,10,self.Sshape, self.nb_edge, stack=False),keep_prob=0.5)
            #does not work. investigate ...
            cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits+self.logits_convolve, labels=self.y_input)

        # Global L2 Regulization
        self.loss = tf.reduce_mean(cross_entropy_source)  + self.mu * tf.nn.l2_loss(self.W_classif)


        self.pred = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.correct_prediction = tf.equal(self.pred, tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


        self.grads_and_vars = self.optalg.compute_gradients(self.loss)

        self.gv_Gn=[]
        #TODO Experiment with gradient noise
        #if self.stack_instead_add:
        #    for grad, var in self.grads_and_vars:
        #        print(grad,var)
        #        if grad is not None:
        #            self.gv_Gn.append( ( tf.add(grad, tf.random_normal(tf.shape(grad), stddev=0.00001)),var)    )

        #self.gv_Gn = [(tf.add(grad, tf.random_normal(tf.shape(grad), stddev=0.00001)), val) for grad, val in self.grads_and_vars if  is not None]

        self.train_step = self.optalg.apply_gradients(self.grads_and_vars)


        # Add ops to save and restore all the variables.
        self.init = tf.global_variables_initializer()
        self.saver= tf.train.Saver(max_to_keep=0)

        print('Number of Params:')
        self.get_nb_params()


    def save_model(self, session, model_filename):
        print("Saving Model")
        save_path = self.saver.save(session, model_filename)

    def restore_model(self, session, model_filename):
        self.saver.restore(session, model_filename)
        print("Model restored.")


    def get_Wedge(self,session):
        '''
        Return the parameters for the Edge Convolutions
        :param session:
        :return:
        '''
        if self.num_layers>1:
            L0=session.run([self.Wel0,self.Wed_layers])
            We0=L0[0]
            list_we=[We0]
            for we in L0[1]:
                list_we.append(we)
            return list_we
        else:
            L0=session.run([self.Wel0])
            We0=L0[0]
            list_we=[We0]
            return list_we

    def train(self,session,graph,verbose=False,n_iter=1):
        '''
        Apply a train operation, ie sgd step for a single graph
        :param session:
        :param graph: a graph from GCN_Dataset
        :param verbose:
        :param n_iter: (default 1) number of steps to perform sgd for this graph
        :return:
        '''
        #TrainEvalSet Here
        for i in range(n_iter):
            #print('Train',X.shape,EA.shape)
            #print('DropoutEdges',self.dropout_rate_edge)
            feed_batch = {

                self.nb_node: graph.X.shape[0],
                self.nb_edge: graph.F.shape[0],
                self.node_input: graph.X,
                self.Ssparse: np.array(graph.Sind, dtype='int64'),
                self.Sshape: np.array([graph.X.shape[0], graph.F.shape[0]], dtype='int64'),
                self.Tsparse: np.array(graph.Tind, dtype='int64'),
                self.F: graph.F,
                self.y_input: graph.Y,
                self.dropout_p_H: self.dropout_rate_H,
                self.dropout_p_node: self.dropout_rate_node,
                self.dropout_p_edge: self.dropout_rate_edge,
                self.dropout_p_edge_feat: self.dropout_rate_edge_feat
            }

            Ops =session.run([self.train_step,self.loss], feed_dict=feed_batch)
            if verbose:
                print('Training Loss',Ops[1])



    def test(self,session,graph,verbose=True):
        '''
        Test return the loss and accuracy for the graph passed as argument
        :param session:
        :param graph:
        :param verbose:
        :return:
        '''

        feed_batch = {

            self.nb_node: graph.X.shape[0],
            self.nb_edge: graph.F.shape[0],
            self.node_input: graph.X,

            self.Ssparse: np.array(graph.Sind, dtype='int64'),
            self.Sshape: np.array([graph.X.shape[0], graph.F.shape[0]], dtype='int64'),
            self.Tsparse: np.array(graph.Tind, dtype='int64'),

            self.F: graph.F,
            self.y_input: graph.Y,
            self.dropout_p_H: 0.0,
            self.dropout_p_node: 0.0,
            self.dropout_p_edge: 0.0,
            self.dropout_p_edge_feat: 0.0
        }

        Ops =session.run([self.loss,self.accuracy], feed_dict=feed_batch)
        if verbose:
            print('Test Loss',Ops[0],' Test Accuracy:',Ops[1])
        return Ops[1]


    def predict(self,session,graph,verbose=True):
        '''
        Does the prediction
        :param session:
        :param graph:
        :param verbose:
        :return:
        '''
        feed_batch = {
            self.nb_node: graph.X.shape[0],
            self.nb_edge: graph.F.shape[0],
            self.node_input: graph.X,
            # fast_gcn.S: np.asarray(graph.S.todense()).squeeze(),
            # fast_gcn.Ssparse: np.vstack([graph.S.row,graph.S.col]),
            self.Ssparse: np.array(graph.Sind, dtype='int64'),
            self.Sshape: np.array([graph.X.shape[0], graph.F.shape[0]], dtype='int64'),
            self.Tsparse: np.array(graph.Tind, dtype='int64'),
            # fast_gcn.T: np.asarray(graph.T.todense()).squeeze(),
            self.F: graph.F,
            self.dropout_p_H: 0.0,
            self.dropout_p_node: 0.0,
            self.dropout_p_edge: 0.0,
            self.dropout_p_edge_feat: 0.0
        }
        Ops = session.run([self.pred], feed_dict=feed_batch)
        if verbose:
            print('Got Prediction for:',Ops[0].shape)
        return Ops[0]


    def train_All_lG(self,session,gcn_graph_train):
        '''
        Train an a list of graph
        :param session:
        :param gcn_graph_train:
        :return:
        '''

        raise NotImplementedError
        Xg=[g.X for g in gcn_graph_train]
        Yg=[g.Y for g in gcn_graph_train]
        Eg=[g.E for g in gcn_graph_train]

        #Node Ids are not the same for edges

        for g in gcn_graph_train:
            self.train(session, g, n_iter=1)



class GraphConvNet(MultiGraphNN):
    '''
    A Deep Standard GCN model for a graph list
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
            Wnli =init_glorot((2*self.node_indim, self.node_indim),name='Wnl'+str(i))
            #Wnli = init_normal((self.node_indim, self.node_indim),std_dev_indim, name='Wnl' + str(i))
            self.Wnode_layers.append(Wnli)
            #The GCN do not seem to use a bias term
            #Bnli = tf.Variable(tf.zeros([self.node_indim]), name='Bnl'+str(i),dtype=tf.float32)
            #self.Bnode_layers.append(Bnli)

        self.W_classif = init_glorot((2*self.node_indim, self.n_classes),name="W_classif")
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
            #self.H = self.activation(tf.matmul(self.ND,tf.add(tf.matmul(self.node_input, self.Wnode), self.Bnode)))
            P0 = self.activation(tf.matmul(self.ND, tf.add(tf.matmul(self.node_input, self.Wnode), self.Bnode)))
            self.hidden_layers = [self.H]
        else:
            H0 =tf.add(tf.matmul(self.node_input, self.Wnode), self.Bnode)
            P0 =tf.matmul(self.NA_input, H0)  # we should forget the self loop
            H0_ = self.activation(tf.concat([H0, P0], 1))
            self.hidden_layers=[H0_]
            #self.H = self.activation(tf.add(tf.matmul(self.node_input, self.Wnode), self.Bnode))
            #self.hidden_layers = [self.H]



        for i in range(self.num_layers-1):
            if self.dropout_mode==2:
                Hp = tf.nn.dropout(self.hidden_layers[-1],1-self.dropout_p)
                Hi_ = tf.matmul(Hp, self.Wnode_layers[i])
            else:
                Hi_ = tf.matmul(self.hidden_layers[-1], self.Wnode_layers[i])
            P =tf.matmul(self.NA_input, Hi_) #we should forget the self loop
            #Hp = tf.concat([H0, P], 1)
            Hi = self.activation(tf.concat([Hi_,P],1))
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
        self.loss = tf.reduce_mean(cross_entropy_source) + self.mu * tf.nn.l2_loss(self.W_classif)

        self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.grads_and_vars = self.optalg.compute_gradients(self.loss)
        self.train_step = self.optalg.apply_gradients(self.grads_and_vars)


        print('Number of Parameters')
        self.get_nb_params()
        # Add ops to save and restore all the variables.
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

    def train(self,session,g,n_iter=1,verbose=False):
        #TrainEvalSet Here
        for i in range(n_iter):
            feed_batch={
                        self.nb_node:g.X.shape[0],
                        self.node_input:g.X,
                        self.y_input:g.Y,
                        self.NA_input:g.NA,
                        self.dropout_p:self.dropout_rate
            }
            Ops =session.run([self.train_step,self.loss], feed_dict=feed_batch)
            if verbose:
                print('Training Loss',Ops[1])



    def test(self,session,g,verbose=True):
        #TrainEvalSet Here
        feed_batch={
                        self.nb_node:g.X.shape[0],
                        self.node_input:g.X,
                        self.y_input:g.Y,
                        self.NA_input:g.NA,
                        self.dropout_p: 0.0
        }
        Ops =session.run([self.loss,self.accuracy], feed_dict=feed_batch)
        if verbose:
            print('Test Loss',Ops[0],' Test Accuracy:',Ops[1])
        return Ops[1]
