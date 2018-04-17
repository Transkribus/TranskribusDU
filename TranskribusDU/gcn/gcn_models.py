# -*- coding: utf-8 -*-
"""
@author: Stéphane Clinchant

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
import scipy.sparse as sp
import random


import sklearn
import sklearn.metrics
import time
#TODO Clean this
from gcn.gcn_datasets import GCNDataset
import warnings
try:
    from . import gcn_datasets
except:
    import gcn_datasets

def init_glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def init_normal(shape,stddev,name=None):
    initial=tf.random_normal(shape, mean=0.0, stddev=stddev, dtype=np.float32)
    return tf.Variable(initial, name=name)


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
                    test_graph_acc,test_acc=self.test_lG(session,graph_test,verbose=False)
                    print('  Test Acc', '%.4f' % test_acc,'%.4f' % test_graph_acc)
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
                    #Yt = session.run(self.logits_Transition)
                    #print('Yt',Yt)

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
        print('Validation Accuracies',['%03.2f'% sx for sx in validation_accuracies])

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

            val = R['val_acc']
            print('Validation scores', val)

            epoch_index = np.argmax(val)
            print('Best performance on val set: Epoch', epoch_index,val[epoch_index])
            print('Test Performance from val', test_accuracies[epoch_index])

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

    #Variable ignored by the set_learning_options
    _setter_variables={
        "node_dim":True,"edge_dim":True,"nb_class":True,
        "num_layers":True,"lr":True,"mu":True,
        "node_indim":True,"nconv_edge":True,
        "nb_iter":True,"ratio_train_val":True}


    def __init__(self,node_dim,edge_dim,nb_classes,num_layers=1,learning_rate=0.1,mu=0.1,node_indim=-1,nconv_edge=1,
                 ):
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.n_classes=nb_classes
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.activation=tf.nn.tanh
        #self.activation=tf.nn.relu
        self.mu=mu
        self.optalg = tf.train.AdamOptimizer(self.learning_rate)
        self.stack_instead_add=False
        self.nconv_edge=nconv_edge
        self.residual_connection=False#deprecated
        self.shared_We = False#deprecated
        self.optim_mode=0 #deprecated
        self.init_fixed=False #ignore --for test purpose
        self.logit_convolve=False#ignore --for test purpose
        self.train_Wn0=True #ignore --for test purpose

        self.dropout_rate_edge_feat= 0.0
        self.dropout_rate_edge = 0.0
        self.dropout_rate_node = 0.0
        self.dropout_rate_H    = 0.0

        self.use_conv_weighted_avg=False
        self.use_edge_mlp=False
        self.edge_mlp_dim = 5

        if node_indim==-1:
            self.node_indim=self.node_dim
        else:
            self.node_indim=node_indim

    def set_learning_options(self,dict_model_config):
        """
        Set all learning options that not directly accessible from the constructor

        :param kwargs:
        :return:
        """
        print(dict_model_config)
        for attrname,val in dict_model_config.items():
            #We treat the activation function differently as we can not pickle/serialiaze python function
            if attrname=='activation_name':
                if val=='relu':
                    self.activation=tf.nn.relu
                elif val=='tanh':
                    self.activation=tf.nn.tanh
                else:
                    raise Exception('Invalid Activation Function')
            if attrname=='stack_instead_add' or attrname=='stack_convolutions':
                self.stack_instead_add=val
            if attrname not in self._setter_variables:
                try:
                    print('set',attrname,val)
                    setattr(self,attrname,val)
                except AttributeError:
                    warnings.warn("Ignored options for ECN"+attrname+':'+val)



    def fastconvolve(self,Wedge,Bedge,F,S,T,H,nconv,Sshape,nb_edge,dropout_p_edge,dropout_p_edge_feat,
                     stack=True, use_dropout=False,zwe=None,use_weighted_average=False,
                     use_edge_mlp=False,Wedge_mlp=None,Bedge_mlp=None):
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
        #It would be faster

        #Drop convolution individually t
        if use_dropout:
        #if False:
            conv_dropout_ind = tf.nn.dropout(tf.ones([nconv], dtype=tf.float32), 1 - dropout_p_edge_feat)
            ND_conv = tf.diag(conv_dropout_ind)

            FW_ = tf.matmul(F, Wedge, transpose_b=True) + Bedge
            FW  = tf.matmul(FW_,ND_conv)

        elif use_edge_mlp:
            #Wedge mlp is a shared variable across layer which project edge in a lower dim
            FW0 = tf.nn.tanh( tf.matmul(F,Wedge_mlp) +Bedge_mlp )
            print('FW0', FW0.get_shape())
            FW = tf.matmul(FW0, Wedge, transpose_b=True) + Bedge
            print('FW', FW.get_shape())
        else:
            FW = tf.matmul(F, Wedge, transpose_b=True) + Bedge
            print('FW', FW.get_shape())


        self.conv =tf.unstack(FW,axis=1)
        Cops=[]


        Tr = tf.SparseTensor(indices=T, values=tf.ones([nb_edge], dtype=tf.float32), dense_shape=[Sshape[1],Sshape[0]])
        Tr = tf.sparse_reorder(Tr)
        TP = tf.sparse_tensor_dense_matmul(Tr,H)



        for i, cw in enumerate(self.conv):
            #SD= tf.SparseTensor(indices=S,values=cw,dense_shape=[nb_node,nb_edge])
            #Warning, pay attention to the ordering of edges
            if use_weighted_average:
                cw = zwe[i]*cw
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

    @staticmethod
    def logitconvolve_fixed(pY,Yt,A_indegree):
        '''
        Tentative Implement of a fixed logit convolve without taking into account edge features
        '''

        #warning we should test that Yt is column normalized
        pY_Yt = tf.matmul(pY,Yt,transpose_b=True)

        #TODO A is dense but shoudl be sparse ....
        P =tf.matmul(A_indegree,pY_Yt)
        return P


    def create_model_stack_convolutions(self):
        #Create All the Variables
        for i in range(self.num_layers - 1):
            if i == 0:
                Wnli = tf.Variable(
                    tf.random_uniform((self.node_dim * self.nconv_edge + self.node_dim, self.node_indim),
                                      -1.0 / math.sqrt(self.node_indim),
                                      1.0 / math.sqrt(self.node_indim)), name='Wnl', dtype=tf.float32)
            else:

                Wnli = tf.Variable(
                    tf.random_uniform((self.node_indim * self.nconv_edge + self.node_indim, self.node_indim),
                                      -1.0 / math.sqrt(self.node_indim),
                                      1.0 / math.sqrt(self.node_indim)), name='Wnl', dtype=tf.float32)
            print('Wnli shape', Wnli.get_shape())
            Bnli = tf.Variable(tf.zeros([self.node_indim]), name='Bnl' + str(i), dtype=tf.float32)

            Weli = init_glorot([int(self.nconv_edge), int(self.edge_dim)], name='Wel_')
            # Weli = tf.Variable(tf.random_normal([int(self.nconv_edge), int(self.edge_dim)], mean=0.0, stddev=1.0),
            #                   dtype=np.float32, name='Wel_')
            Beli = tf.Variable(0.01 * tf.ones([self.nconv_edge]), name='Bel' + str(i), dtype=tf.float32)

            self.Wnode_layers.append(Wnli)
            self.Bnode_layers.append(Bnli)
            self.Wed_layers.append(Weli)
            self.Bed_layers.append(Beli)

        self.train_var.extend((self.Wnode_layers))
        self.train_var.extend((self.Wed_layers))

        self.Hnode_layers = []

        self.W_classif = tf.Variable(
                tf.random_uniform((self.node_indim * self.nconv_edge + self.node_indim, self.n_classes),
                                  -1.0 / math.sqrt(self.node_dim),
                                  1.0 / math.sqrt(self.node_dim)),
                name="W_classif", dtype=np.float32)

        self.B_classif = tf.Variable(tf.zeros([self.n_classes]), name='B_classif', dtype=np.float32)

        self.train_var.append((self.W_classif))
        self.train_var.append((self.B_classif))


        self.node_dropout_ind = tf.nn.dropout(tf.ones([self.nb_node], dtype=tf.float32), 1 - self.dropout_p_node)
        self.ND = tf.diag(self.node_dropout_ind)

        edge_dropout = self.dropout_rate_edge > 0.0 or self.dropout_rate_edge_feat > 0.0
        print('Edge Dropout', edge_dropout, self.dropout_rate_edge, self.dropout_rate_edge_feat)
        if self.num_layers == 1:
            self.H = self.activation(tf.add(tf.matmul(self.node_input, self.Wnl0), self.Bnl0))
            self.hidden_layers = [self.H]
            print("H shape", self.H.get_shape())

            P = self.fastconvolve(self.Wel0, self.Bel0, self.F, self.Ssparse, self.Tsparse, self.H, self.nconv_edge,
                                  self.Sshape, self.nb_edge,
                                  self.dropout_p_edge, self.dropout_p_edge_feat, stack=self.stack_instead_add,
                                  use_dropout=edge_dropout)

            Hp = tf.concat([self.H, P], 1)
            # Hp= P+self.H

            Hi = self.activation(Hp)
            # Hi_shape = Hi.get_shape()
            # print(Hi_shape)
            self.hidden_layers.append(Hi)

        elif self.num_layers > 1:

            if self.dropout_rate_node > 0.0:
                H0 = self.activation(tf.matmul(self.ND, tf.add(tf.matmul(self.node_input, self.Wnl0), self.Bnl0)))
            else:
                H0 = self.activation(tf.add(tf.matmul(self.node_input, self.Wnl0), self.Bnl0))

            self.Hnode_layers.append(H0)

            # TODO Default to fast convolve but we change update configs, train and test flags
            P = self.fastconvolve(self.Wel0, self.Bel0, self.F, self.Ssparse, self.Tsparse, H0, self.nconv_edge,
                                  self.Sshape, self.nb_edge,
                                  self.dropout_p_edge, self.dropout_p_edge_feat, stack=self.stack_instead_add,
                                  use_dropout=edge_dropout,
                                  )
            Hp = tf.concat([H0, P], 1)

            # TODO add activation Here.
            # self.hidden_layers = [self.activation(Hp)]
            self.hidden_layers = [Hp]

            for i in range(self.num_layers - 1):

                if self.dropout_rate_H > 0.0:
                    Hi_ = tf.nn.dropout(tf.matmul(self.hidden_layers[-1], self.Wnode_layers[i]) + self.Bnode_layers[i],
                                        1 - self.dropout_p_H)
                else:
                    Hi_ = tf.matmul(self.hidden_layers[-1], self.Wnode_layers[i]) + self.Bnode_layers[i]

                if self.residual_connection:
                    Hi_ = tf.add(Hi_, self.Hnode_layers[-1])

                self.Hnode_layers.append(Hi_)

                print('Hi_shape', Hi_.get_shape())
                print('Hi prevous shape', self.hidden_layers[-1].get_shape())

                P = self.fastconvolve(self.Wed_layers[i], self.Bed_layers[i], self.F, self.Ssparse, self.Tsparse, Hi_,
                                      self.nconv_edge, self.Sshape, self.nb_edge,
                                      self.dropout_p_edge, self.dropout_p_edge_feat, stack=self.stack_instead_add,
                                      use_dropout=edge_dropout,
                                      )
                Hp = tf.concat([Hi_, P], 1)
                Hi = self.activation(Hp)
                self.hidden_layers.append(Hi)

    def create_model_sum_convolutions(self):

        #self.Wed_layers.append(Wel0)
        for i in range(self.num_layers-1):

            if i==0:
                Wnli = tf.Variable(tf.random_uniform((2 * self.node_indim, self.node_indim),
                                                     -1.0 / math.sqrt(self.node_indim),
                                                     1.0 / math.sqrt(self.node_indim)), name='Wnl',
                                   dtype=tf.float32)
            else:
                Wnli =tf.Variable(tf.random_uniform( (2*self.node_indim, self.node_indim),
                                                               -1.0 / math.sqrt(self.node_indim),
                                                               1.0 / math.sqrt(self.node_indim)),name='Wnl',dtype=tf.float32)

            Bnli = tf.Variable(tf.zeros([self.node_indim]), name='Bnl'+str(i),dtype=tf.float32)

            Weli= init_glorot([int(self.nconv_edge), int(self.edge_dim)], name='Wel_')
            #Weli = tf.Variable(tf.random_normal([int(self.nconv_edge), int(self.edge_dim)], mean=0.0, stddev=1.0),
            #                   dtype=np.float32, name='Wel_')
            Beli = tf.Variable(0.01*tf.ones([self.nconv_edge]), name='Bel'+str(i),dtype=tf.float32)

            self.Wnode_layers.append(Wnli)
            self.Bnode_layers.append(Bnli)
            self.Wed_layers.append  (Weli)
            self.Bed_layers.append(Beli)

        self.train_var.extend((self.Wnode_layers))
        self.train_var.extend((self.Wed_layers))

        self.Hnode_layers=[]




        self.W_classif = tf.Variable(tf.random_uniform((2*self.node_indim, self.n_classes),
                                                           -1.0 / math.sqrt(self.node_dim),
                                                           1.0 / math.sqrt(self.node_dim)),
                                        name="W_classif",dtype=np.float32)
        self.B_classif = tf.Variable(tf.zeros([self.n_classes]), name='B_classif',dtype=np.float32)

        self.train_var.append((self.W_classif))
        self.train_var.append((self.B_classif))



        self.node_dropout_ind = tf.nn.dropout(tf.ones([self.nb_node], dtype=tf.float32), 1 - self.dropout_p_node)
        self.ND = tf.diag(self.node_dropout_ind)

        edge_dropout = self.dropout_rate_edge> 0.0 or self.dropout_rate_edge_feat > 0.0
        print('Edge Dropout',edge_dropout, self.dropout_rate_edge,self.dropout_rate_edge_feat)
        if self.num_layers==1:
            self.H = self.activation(tf.add(tf.matmul(self.node_input, Wnl0), Bnl0))
            self.hidden_layers = [self.H]
            print("H shape",self.H.get_shape())


            P = self.fastconvolve(self.Wel0,self.Bel0,self.F,self.Ssparse,self.Tsparse,self.H,self.nconv_edge,self.Sshape,self.nb_edge,
                                  self.dropout_p_edge,self.dropout_p_edge_feat,stack=self.stack_instead_add,use_dropout=edge_dropout)

            Hp = tf.concat([self.H, P], 1)
            Hi=self.activation(Hp)
            self.hidden_layers.append(Hi)

        elif self.num_layers>1:

            if self.dropout_rate_node>0.0:
                H0 = self.activation(tf.matmul(self.ND,tf.add(tf.matmul(self.node_input,self.Wnl0), self.Bnl0)))
            else:
                H0 = self.activation(tf.add(tf.matmul(self.node_input,self.Wnl0),self.Bnl0))

            self.Hnode_layers.append(H0)

            #TODO Default to fast convolve but we change update configs, train and test flags
            P = self.fastconvolve(self.Wel0,self.Bel0, self.F, self.Ssparse, self.Tsparse, H0, self.nconv_edge, self.Sshape,self.nb_edge,
                                  self.dropout_p_edge,self.dropout_p_edge_feat, stack=self.stack_instead_add, use_dropout=edge_dropout,
                                  )

            if self.use_conv_weighted_avg:
                Hp = self.zH[0] * H0 + P
            else:
                Hp = tf.concat([H0, P], 1)

            #TODO add activation Here.
            #self.hidden_layers = [self.activation(Hp)]
            self.hidden_layers = [Hp]

            for i in range(self.num_layers-1):

                if self.dropout_rate_H > 0.0:
                    Hi_ = tf.nn.dropout(tf.matmul(self.hidden_layers[-1], self.Wnode_layers[i]) + self.Bnode_layers[i], 1-self.dropout_p_H)
                else:
                    Hi_ = tf.matmul(self.hidden_layers[-1], self.Wnode_layers[i]) + self.Bnode_layers[i]

                if self.residual_connection:
                    Hi_= tf.add(Hi_,self.Hnode_layers[-1])

                self.Hnode_layers.append(Hi_)

                print('Hi_shape',Hi_.get_shape())
                print('Hi prevous shape',self.hidden_layers[-1].get_shape())

                P = self.fastconvolve(self.Wed_layers[i],self.Bed_layers[i], self.F, self.Ssparse, self.Tsparse, Hi_, self.nconv_edge,self.Sshape, self.nb_edge,
                                      self.dropout_p_edge,self.dropout_p_edge_feat, stack=self.stack_instead_add, use_dropout=edge_dropout
                                      )

                Hp = tf.concat([Hi_, P], 1)
                Hi = self.activation(Hp)
                self.hidden_layers.append(Hi)

    def create_model(self):
        '''
        Create the tensorflow graph for the model
        :return:
        '''
        self.nb_node = tf.placeholder(tf.int32, (), name='nb_node')
        self.nb_edge = tf.placeholder(tf.int32, (), name='nb_edge')
        self.node_input = tf.placeholder(tf.float32, [None, self.node_dim], name='X_')
        self.y_input = tf.placeholder(tf.float32, [None, self.n_classes], name='Y')
        self.dropout_p_H = tf.placeholder(tf.float32, (), name='dropout_prob_H')
        self.dropout_p_node = tf.placeholder(tf.float32, (), name='dropout_prob_N')
        self.dropout_p_edge = tf.placeholder(tf.float32, (), name='dropout_prob_edges')
        self.dropout_p_edge_feat = tf.placeholder(tf.float32, (), name='dropout_prob_edgefeat')
        self.S = tf.placeholder(tf.float32, name='S')
        self.Ssparse = tf.placeholder(tf.int64, name='Ssparse')  # indices
        self.Sshape = tf.placeholder(tf.int64, name='Sshape')  # indices

        self.T = tf.placeholder(tf.float32, [None, None], name='T')
        self.Tsparse = tf.placeholder(tf.int64, name='Tsparse')

        self.F = tf.placeholder(tf.float32, [None, None], name='F')


        std_dev_in = float(1.0 / float(self.node_dim))

        self.Wnode_layers = []
        self.Bnode_layers = []
        self.Wed_layers = []
        self.Bed_layers = []
        self.zed_layers = []

        self.Wedge_mlp_layers = []
        # Should Project edge as well ...
        self.train_var = []

        # if self.node_indim!=self.node_dim:
        #    Wnl0 = tf.Variable(tf.random_uniform((self.node_dim, self.node_indim),
        #                                                           -1.0 / math.sqrt(self.node_dim),
        #                                                           1.0 / math.sqrt(self.node_dim)),name='Wnl0',dtype=tf.float32)
        #
        self.Wnl0 = tf.Variable(tf.eye(self.node_dim), name='Wnl0', dtype=tf.float32, trainable=self.train_Wn0)

        self.Bnl0 = tf.Variable(tf.zeros([self.node_dim]), name='Bnl0', dtype=tf.float32)

        if self.init_fixed: #For testing Purposes
            self.Wel0 = tf.Variable(100 * tf.ones([int(self.nconv_edge), int(self.edge_dim)]), name='Wel0',
                                    dtype=tf.float32)
        # self.Wel0 =tf.Variable(tf.random_normal([int(self.nconv_edge),int(self.edge_dim)],mean=0.0,stddev=1.0), dtype=np.float32, name='Wel0')
        self.Wel0 = init_glorot([int(self.nconv_edge), int(self.edge_dim)], name='Wel0')
        self.Bel0 = tf.Variable(0.01 * tf.ones([self.nconv_edge]), name='Bel0', dtype=tf.float32)


        print('Wel0', self.Wel0.get_shape())
        self.train_var.extend([self.Wnl0, self.Bnl0])
        self.train_var.append(self.Wel0)

        if self.stack_instead_add:
            self.create_model_stack_convolutions()

        else:
            self.create_model_sum_convolutions()

        self.logits = tf.add(tf.matmul(self.hidden_layers[-1], self.W_classif), self.B_classif)
        cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
        # Global L2 Regulization
        self.loss = tf.reduce_mean(cross_entropy_source) + self.mu * tf.nn.l2_loss(self.W_classif)

        self.pred = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.correct_prediction = tf.equal(self.pred, tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.grads_and_vars = self.optalg.compute_gradients(self.loss)

        self.gv_Gn = []
        self.train_step = self.optalg.apply_gradients(self.grads_and_vars)

        # Add ops to save and restore all the variables.
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=0)

        print('Number of Params:')
        self.get_nb_params()

    def create_model_old(self):
        '''
        Create the tensorflow graph for the model
        :return:
        '''
        self.nb_node    = tf.placeholder(tf.int32,(), name='nb_node')
        self.nb_edge = tf.placeholder(tf.int32, (), name='nb_edge')
        self.node_input = tf.placeholder(tf.float32, [None, self.node_dim], name='X_')
        self.y_input    = tf.placeholder(tf.float32, [None, self.n_classes], name='Y')
        #self.EA_input   = tf.placeholder(tf.float32, name='EA_input')
        #self.NA_input   = tf.placeholder(tf.float32, name='NA_input')
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


        #self.NA_indegree = tf.placeholder(tf.float32, name='NA_indegree')


        std_dev_in=float(1.0/ float(self.node_dim))

        self.Wnode_layers=[]
        self.Bnode_layers=[]
        self.Wed_layers=[]
        self.Bed_layers=[]
        #REFACT1 self.zed_layers = []

        #REFACT1 self.Wedge_mlp_layers=[]
        #Should Project edge as well ...
        self.train_var=[]

        #if self.node_indim!=self.node_dim:
        #    Wnl0 = tf.Variable(tf.random_uniform((self.node_dim, self.node_indim),
        #                                                           -1.0 / math.sqrt(self.node_dim),
        #                                                           1.0 / math.sqrt(self.node_dim)),name='Wnl0',dtype=tf.float32)
        #else:
        self.Wnl0 = tf.Variable(tf.eye(self.node_dim),name='Wnl0',dtype=tf.float32,trainable=self.train_Wn0)

        self.Bnl0 = tf.Variable(tf.zeros([self.node_dim]), name='Bnl0',dtype=tf.float32)


        #self.Wel0 =tf.Variable(tf.random_normal([int(self.nconv_edge),int(self.edge_dim)],mean=0.0,stddev=1.0), dtype=np.float32, name='Wel0')
        if self.init_fixed:
            self.Wel0 = tf.Variable(100*tf.ones([int(self.nconv_edge),int(self.edge_dim)]), name='Wel0',dtype=tf.float32)

        elif self.use_edge_mlp:
            self.Wel0 = init_glorot([int(self.nconv_edge), int(self.edge_mlp_dim)], name='Wel0')
        else:
            self.Wel0 = init_glorot([int(self.nconv_edge),int(self.edge_dim)],name='Wel0')

        self.Bel0 = tf.Variable(0.01*tf.ones([self.nconv_edge]), name='Bel0' , dtype=tf.float32)

        #RF self.zel0 = tf.Variable(tf.ones([self.nconv_edge]), name='zel0' , dtype=tf.float32)
        #RF self.zH = tf.Variable(tf.ones([self.num_layers]),name='zH',dtype=tf.float32)

        print('Wel0',self.Wel0.get_shape())
        self.train_var.extend([self.Wnl0,self.Bnl0])
        self.train_var.append(self.Wel0)

        #Parameter for convolving the logits
        ''' REFACT1
        if self.logit_convolve:
            #self.Wel_logits = init_glorot([int(self.nconv_edge),int(self.edge_dim)],name='Wel_logit')
            #self.Belg = tf.Variable(tf.zeros( [int(self.nconv_edge)]), name='Belogit' , dtype=tf.float32)
            self.Wel_logits = tf.Variable(tf.zeros([int(1),int(self.edge_dim)]), name='Wlogit0',dtype=tf.float32,trainable=False)
            self.Belg = tf.Variable(tf.ones( [int(1)]), name='Belogit' , dtype=tf.float32)
            #self.logits_Transition = 1.0*tf.Variable(tf.ones([int(self.n_classes) , int(self.n_classes)]), name='logit_Transition')
            self.logits_Transition=init_glorot([int(self.n_classes), int(self.n_classes)], name='Wel_')

        self.Wmlp_edge_0= init_glorot([int(self.edge_dim), int(self.edge_mlp_dim)], name='Wedge_mlp')
        self.Bmlp_edge_0= tf.Variable(tf.ones([self.edge_mlp_dim]),name='Wedge_mlp',dtype=tf.float32)
        '''

        #self.Wed_layers.append(Wel0)
        for i in range(self.num_layers-1):
            if self.stack_instead_add:
                if i==0:
                    Wnli = tf.Variable(
                        tf.random_uniform((self.node_dim * self.nconv_edge + self.node_dim, self.node_indim),
                                          -1.0 / math.sqrt(self.node_indim),
                                          1.0 / math.sqrt(self.node_indim)), name='Wnl', dtype=tf.float32)
                else:

                    Wnli =tf.Variable(tf.random_uniform( (self.node_indim*self.nconv_edge+self.node_indim, self.node_indim),
                                                               -1.0 / math.sqrt(self.node_indim),
                                                               1.0 / math.sqrt(self.node_indim)),name='Wnl',dtype=tf.float32)
                print('Wnli shape',Wnli.get_shape())

            elif self.use_conv_weighted_avg:
                Wnli = tf.Variable(
                    tf.random_uniform((self.node_indim, self.node_indim),
                                      -1.0 / math.sqrt(self.node_indim),
                                      1.0 / math.sqrt(self.node_indim)), name='Wnl', dtype=tf.float32)

                #Wnli = tf.eye(self.node_dim,dtype=tf.float32)

                print('Wnli shape', Wnli.get_shape())


            else:
                if i==0:
                    Wnli = tf.Variable(tf.random_uniform((2 * self.node_indim, self.node_indim),
                                                         -1.0 / math.sqrt(self.node_indim),
                                                         1.0 / math.sqrt(self.node_indim)), name='Wnl',
                                       dtype=tf.float32)
                else:
                    Wnli =tf.Variable(tf.random_uniform( (2*self.node_indim, self.node_indim),
                                                                   -1.0 / math.sqrt(self.node_indim),
                                                                   1.0 / math.sqrt(self.node_indim)),name='Wnl',dtype=tf.float32)

            Bnli = tf.Variable(tf.zeros([self.node_indim]), name='Bnl'+str(i),dtype=tf.float32)

            #Weli = tf.Variable(tf.ones([int(self.nconv_edge),int(self.edge_dim)],dtype=tf.float32))

            if self.use_edge_mlp:
                #self.Wel0 = init_glorot([int(self.nconv_edge), int(self.edge_mlp_dim)], name='Wel0')
                Weli = init_glorot([int(self.nconv_edge), int(self.edge_mlp_dim)], name='Wel_')
                Beli = tf.Variable(0.01 * tf.ones([self.nconv_edge]), name='Bel' + str(i), dtype=tf.float32)

                #RF Wmlp_edge_i = init_glorot([int(self.edge_dim), int(self.edge_mlp_dim)], name='Wedge_mlp'+str(i))
                #RF Bmlp_edge_i = tf.Variable(tf.ones([self.edge_mlp_dim]), name='Bedge_mlp'+str(i), dtype=tf.float32)
                #RF self.Wedge_mlp_layers.append(Wmlp_edge_i)

            else:
                Weli= init_glorot([int(self.nconv_edge), int(self.edge_dim)], name='Wel_')
                #Weli = tf.Variable(tf.random_normal([int(self.nconv_edge), int(self.edge_dim)], mean=0.0, stddev=1.0),
                #                   dtype=np.float32, name='Wel_')
                Beli = tf.Variable(0.01*tf.ones([self.nconv_edge]), name='Bel'+str(i),dtype=tf.float32)

                #RF Wmlp_edge_i = init_glorot([int(self.edge_dim), int(self.edge_mlp_dim)], name='Wedge_mlp' + str(i))
                #RF Bmlp_edge_i = tf.Variable(tf.ones([self.edge_mlp_dim]), name='Bedge_mlp' + str(i), dtype=tf.float32)
                #RF self.Wedge_mlp_layers.append(Wmlp_edge_i)

            #zeli = tf.Variable(tf.ones([self.nconv_edge]),name='zel'+str(i),dtype=tf.float32)


            self.Wnode_layers.append(Wnli)
            self.Bnode_layers.append(Bnli)
            self.Wed_layers.append  (Weli)
            self.Bed_layers.append(Beli)
            #self.zed_layers.append(zeli)

        self.train_var.extend((self.Wnode_layers))
        self.train_var.extend((self.Wed_layers))

        self.Hnode_layers=[]



        #TODO Do we project the firt layer or not ?
        # Initialize the weights and biases for a simple one full connected network
        if self.stack_instead_add:
            self.W_classif = tf.Variable(tf.random_uniform((self.node_indim*self.nconv_edge+self.node_indim, self.n_classes),
                                                           -1.0 / math.sqrt(self.node_dim),
                                                           1.0 / math.sqrt(self.node_dim)),
                                        name="W_classif",dtype=np.float32)

        elif self.use_conv_weighted_avg:
            self.W_classif = tf.Variable(tf.random_uniform((self.node_indim, self.n_classes),
                                                           -1.0 / math.sqrt(self.node_dim),
                                                           1.0 / math.sqrt(self.node_dim)),
                                         name="W_classif", dtype=np.float32)
        else:
            self.W_classif = tf.Variable(tf.random_uniform((2*self.node_indim, self.n_classes),
                                                           -1.0 / math.sqrt(self.node_dim),
                                                           1.0 / math.sqrt(self.node_dim)),
                                        name="W_classif",dtype=np.float32)
        self.B_classif = tf.Variable(tf.zeros([self.n_classes]), name='B_classif',dtype=np.float32)

        self.train_var.append((self.W_classif))
        self.train_var.append((self.B_classif))


        #Use for true add
        #I = tf.eye(self.nb_node)

        self.node_dropout_ind = tf.nn.dropout(tf.ones([self.nb_node], dtype=tf.float32), 1 - self.dropout_p_node)
        self.ND = tf.diag(self.node_dropout_ind)

        edge_dropout = self.dropout_rate_edge> 0.0 or self.dropout_rate_edge_feat > 0.0
        print('Edge Dropout',edge_dropout, self.dropout_rate_edge,self.dropout_rate_edge_feat)
        if self.num_layers==1:
            self.H = self.activation(tf.add(tf.matmul(self.node_input, self.Wnl0), self.Bnl0))
            self.hidden_layers = [self.H]
            print("H shape",self.H.get_shape())


            P = self.fastconvolve(self.Wel0,self.Bel0,self.F,self.Ssparse,self.Tsparse,self.H,self.nconv_edge,self.Sshape,self.nb_edge,
                                  self.dropout_p_edge,self.dropout_p_edge_feat,stack=self.stack_instead_add,use_dropout=edge_dropout)

            Hp = tf.concat([self.H, P], 1)
            #Hp= P+self.H

            Hi=self.activation(Hp)
            #Hi_shape = Hi.get_shape()
            #print(Hi_shape)
            self.hidden_layers.append(Hi)

        elif self.num_layers>1:

            if self.dropout_rate_node>0.0:
                H0 = self.activation(tf.matmul(self.ND,tf.add(tf.matmul(self.node_input, self.Wnl0), self.Bnl0)))
            else:
                H0 = self.activation(tf.add(tf.matmul(self.node_input,self.Wnl0),self.Bnl0))

            self.Hnode_layers.append(H0)

            #TODO Default to fast convolve but we change update configs, train and test flags
            P = self.fastconvolve(self.Wel0,self.Bel0, self.F, self.Ssparse, self.Tsparse, H0, self.nconv_edge, self.Sshape,self.nb_edge,
                                  self.dropout_p_edge,self.dropout_p_edge_feat, stack=self.stack_instead_add, use_dropout=edge_dropout,
                                  )
                                  #RF zwe=self.zel0,
                                  #RF use_weighted_average=self.use_conv_weighted_avg,
                                  #RF use_edge_mlp=self.use_edge_mlp,
                                  #RFWedge_mlp=self.Wmlp_edge_0,
                                  #RF Bedge_mlp=self.Bmlp_edge_0)

            if self.use_conv_weighted_avg:
                Hp = self.zH[0] * H0 + P
            else:
                Hp = tf.concat([H0, P], 1)

            #TODO add activation Here.
            #self.hidden_layers = [self.activation(Hp)]
            self.hidden_layers = [Hp]


            for i in range(self.num_layers-1):

                if self.dropout_rate_H > 0.0:
                    Hi_ = tf.nn.dropout(tf.matmul(self.hidden_layers[-1], self.Wnode_layers[i]) + self.Bnode_layers[i], 1-self.dropout_p_H)
                else:
                    Hi_ = tf.matmul(self.hidden_layers[-1], self.Wnode_layers[i]) + self.Bnode_layers[i]

                if self.residual_connection:
                    Hi_= tf.add(Hi_,self.Hnode_layers[-1])

                self.Hnode_layers.append(Hi_)

                print('Hi_shape',Hi_.get_shape())
                print('Hi prevous shape',self.hidden_layers[-1].get_shape())

                P = self.fastconvolve(self.Wed_layers[i],self.Bed_layers[i], self.F, self.Ssparse, self.Tsparse, Hi_, self.nconv_edge,self.Sshape, self.nb_edge,
                                      self.dropout_p_edge,self.dropout_p_edge_feat, stack=self.stack_instead_add, use_dropout=edge_dropout,
                                      )
                                      # zwe=self.zed_layers[i],
                                      # use_weighted_average=self.use_conv_weighted_avg,
                                      # use_edge_mlp=self.use_edge_mlp,
                                      # Wedge_mlp=self.Wedge_mlp_layers[i],
                                      #RF Bedge_mlp=self.Bmlp_edge_0)


                if self.use_conv_weighted_avg:
                    Hp = self.zH[i+1]* Hi_ + P

                else:
                    Hp = tf.concat([Hi_, P], 1)

                Hi = self.activation(Hp)

                self.hidden_layers.append(Hi)



        self.logits =tf.add(tf.matmul(self.hidden_layers[-1],self.W_classif),self.B_classif)
        cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
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
                self.dropout_p_edge_feat: self.dropout_rate_edge_feat,
                #self.NA_indegree:graph.NA_indegree
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
            self.dropout_p_edge_feat: 0.0,
            #self.NA_indegree: graph.NA_indegree
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
            self.dropout_p_edge_feat: 0.0,
            #self.NA_indegree: graph.NA_indegree
        }
        Ops = session.run([self.pred], feed_dict=feed_batch)
        if verbose:
            print('Got Prediction for:',Ops[0].shape)
        return Ops[0]


    def train_All_lG(self,session,graph_train,graph_val, max_iter, eval_iter = 10, patience = 7, graph_test = None, save_model_path = None):
        '''

        Merge all the graph and train on them

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
        best_val_acc = 0.0
        wait = 0
        stop_training = False
        stopped_iter = max_iter
        train_accuracies = []
        validation_accuracies = []
        test_accuracies = []
        conf_mat = []

        start_monitoring_val_acc = False

        # Not Efficient to compute this for

        merged_graph = gcn_datasets.GCNDataset.merge_allgraph(graph_train)

        self.train(session, merged_graph, n_iter=1)

        for i in range(max_iter):
            if stop_training:
                break

            if i % eval_iter == 0:
                print('\nEpoch', i)
                _, tr_acc = self.test_lG(session, graph_train, verbose=False)
                print(' Train Acc', '%.4f' % tr_acc)
                train_accuracies.append(tr_acc)

                _, node_acc = self.test_lG(session, graph_val, verbose=False)
                print(' Valid Acc', '%.4f' % node_acc)
                validation_accuracies.append(node_acc)

                if save_model_path:
                    save_path = self.saver.save(session, save_model_path, global_step=i)

                if graph_test:
                    _, test_acc = self.test_lG(session, graph_test, verbose=False)
                    print('  Test Acc', '%.4f' % test_acc)
                    test_accuracies.append(test_acc)


                    # Ypred = self.predict_lG(session, graph_test,verbose=False)
                    # Y_true_flat = []
                    # Ypred_flat = []
                    # for graph, ypred in zip(graph_test, Ypred):
                    #    ytrue = np.argmax(graph.Y, axis=1)
                    #    Y_true_flat.extend(ytrue)
                    #    Ypred_flat.extend(ypred)
                    # cm = sklearn.metrics.confusion_matrix(Y_true_flat, Ypred_flat)
                    # conf_mat.append(cm)

                # TODO min_delta
                # if tr_acc>0.99:
                #    start_monitoring_val_acc=True

                if node_acc > best_val_acc:
                    best_val_acc = node_acc
                    wait = 0
                else:
                    if wait >= patience:
                        stopped_iter = i
                        stop_training = True
                    wait += 1
            else:

                self.train(session, merged_graph, n_iter=1)
        # Final Save
        # if save_model_path:
        # save_path = self.saver.save(session, save_model_path, global_step=i)
        # TODO Add the final step
        mean_acc = []
        print('Stopped Model Training after', stopped_iter)
        print('Val Accuracies', validation_accuracies)

        print('Final Training Accuracy')
        _, node_train_acc = self.test_lG(session, graph_train)
        print('Train Mean Accuracy', '%.4f' % node_train_acc)

        print('Final Valid Acc')
        self.test_lG(session, graph_val)

        R = {}
        R['train_acc'] = train_accuracies
        R['val_acc'] = validation_accuracies
        R['test_acc'] = test_accuracies
        R['stopped_iter'] = stopped_iter
        R['confusion_matrix'] = conf_mat
        # R['W_edge'] =self.get_Wedge(session)
        if graph_test:
            _, final_test_acc = self.test_lG(session, graph_test)
            print('Final Test Acc', '%.4f' % final_test_acc)
            R['final_test_acc'] = final_test_acc

        return R




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


class EdgeLogit(Logit):
    '''
    Logistic Regression for MultiGraph
    '''

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

            nb_edge =graph.E.shape[0]
            half_edge  =nb_edge/2
            feed_batch = {

                self.nb_node: graph.EC.shape[0], #here we pass the number of edges
                self.node_input: graph.EC,
                self.y_input: graph.Yedge,

                #self.nb_node: half_edge, #here we pass the number of edges
                #self.node_input: graph.F[:half_edge],
                #self.y_input: graph.Yedge[:half_edge],
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

        nb_edge = graph.E.shape[0]
        half_edge = nb_edge / 2

        feed_batch = {

            self.nb_node: graph.EC.shape[0],
            self.node_input: graph.EC,
            self.y_input: graph.Yedge,

            #self.nb_node: half_edge,  # here we pass the number of edges
            #self.node_input: graph.F[:half_edge],
            #self.y_input: graph.Yedge[:half_edge],
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

        nb_edge = graph.E.shape[0]
        half_edge = nb_edge / 2

        feed_batch = {
            self.nb_node: graph.EC.shape[0],
            self.node_input: graph.EC,

            #self.nb_node: half_edge,  # here we pass the number of edges
            #self.node_input: graph.F[:half_edge],
            #self.y_input: graph.Yedge[:, half_edge],
        }
        Ops = session.run([self.pred], feed_dict=feed_batch)
        if verbose:
            print('Got Prediction for:',Ops[0].shape)
        return Ops[0]



#TODO Benchmark on Snake, GCN, ECN, graphAttNet vs Cora
#TODO Refactorize Code
#TODO Add L2 Regularization
#TODO Stack or Add Convolution -> Reduce the size
# Force the attention to preserve the node information i.e alpha'= 0.8 I +0.2 alpha
# Force doing attention only for the logit ?
# with a two layer
# Branching factor --> Attention
# Logit Layer and Attention
# There is one diff with ECN the feature for the edges are dynamically calculated
# whereas for GAT they are conditionned on the currect Node features
# 0.88
# Do a dot product attention or something different ...
# Change Initialization of the attention vector
# with Attn vector equal  [x;0] the attention keeps the source features and do not propagate ...
# Should reduce Nb of parameters
# This notion of edges is completely arbritrary
# We could at all nodes in the graph to see whether there are some depencies, no ?, interesting exp

class GraphAttNet(MultiGraphNN):
    '''
    Graph Attention Network
    '''

    # Variable ignored by the set_learning_options
    _setter_variables = {
        "node_dim": True, "edge_dim": True, "nb_class": True,
        "num_layers": True, "lr": True,
        "node_indim": True, "nb_attention": True,
        "nb_iter": True, "ratio_train_val": True}

    def __init__(self,node_dim,nb_classes,num_layers=1,learning_rate=0.1,node_indim=-1,nb_attention=3
                 ):
        self.node_dim=node_dim

        self.n_classes=nb_classes
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.activation=tf.nn.elu
        #self.activation=tf.nn.relu

        self.optalg = tf.train.AdamOptimizer(self.learning_rate)
        self.stack_instead_add=False
        self.residual_connection=False#deprecated
        self.mu=0.0
        self.dropout_rate_node = 0.0
        self.dropout_rate_attention = 0.0
        self.nb_attention=nb_attention
        self.distinguish_node_from_neighbor=False
        self.original_model=False
        self.attn_type=0
        self.dense_model=False

        if node_indim==-1:
            self.node_indim=self.node_dim
        else:
            self.node_indim=node_indim

    #TODO GENERIC Could be move in MultigraphNN
    def set_learning_options(self,dict_model_config):
        """
        Set all learning options that not directly accessible from the constructor

        :param kwargs:
        :return:
        """
        print(dict_model_config)
        for attrname,val in dict_model_config.items():
            #We treat the activation function differently as we can not pickle/serialiaze python function
            if attrname=='activation_name':
                if val=='relu':
                    self.activation=tf.nn.relu
                elif val=='tanh':
                    self.activation=tf.nn.tanh
                else:
                    raise Exception('Invalid Activation Function')
            if attrname=='stack_instead_add' or attrname=='stack_convolutions':
                self.stack_instead_add=val
            if attrname not in self._setter_variables:
                try:
                    print('set',attrname,val)
                    setattr(self,attrname,val)
                except AttributeError:
                    warnings.warn("Ignored options for ECN"+attrname+':'+val)

    def dense_graph_attention_layer(self,H,W,A,nb_node,dropout_attention,dropout_node,use_dropout=False):
        '''
        Implement a dense attention layer where every node is connected to everybody
        :param A:
        :param H:
        :param W:
        :param dropout_attention:
        :param dropout_node:
        :param use_dropout:
        :return:
        '''

        '''
        for all i,j aHi + bHj
        repmat all first column contains H1 second columns H2  etc
        diag may be a special case
        
        
        '''
        with tf.name_scope('graph_att_dense_attn'):

            P = tf.matmul(H, W)
            Aij_forward = tf.expand_dims(A[0], 0)  # attention vector for forward edge and backward edge
            Aij_backward = tf.expand_dims(A[1], 0)  # Here we assume it is the same on contrary to the paper
            # Compute the attention weight for target node, ie a . Whj if j is the target node
            att_target_node = tf.matmul(P, Aij_backward,transpose_b=True)
            # Compute the attention weight for the source node, ie a . Whi if j is the target node
            att_source_node = tf.matmul(P, Aij_forward, transpose_b=True)

            Asrc_vect = tf.tile(att_source_node,[nb_node,1])
            Asrc = tf.reshape(Asrc_vect,[nb_node,nb_node])

            Atgt_vect = tf.tile(att_target_node, [nb_node,1])
            Atgt = tf.reshape(Atgt_vect, [nb_node, nb_node])


            Att = tf.nn.leaky_relu(Asrc+Atgt)
            #Att = tf.nn.leaky_relu(Asrc)
            alphas = tf.nn.softmax(Att)

            # dropout is done after the softmax
            if use_dropout:
                print('... using dropout for attention layer')
                alphasD = tf.nn.dropout(alphas, 1.0 - dropout_attention)
                P_D = tf.nn.dropout(P, 1.0 - dropout_node)
                alphasP = tf.matmul(alphasD, P_D)
                return alphasD, alphasP
            else:
                # We compute the features given by the attentive neighborhood
                alphasP = tf.matmul(alphas, P)
                return alphas, alphasP

    #TODO Change the transpose of the A parameter
    def simple_graph_attention_layer(self,H,W,A,S,T,Adjind,Sshape,nb_edge,
                                     dropout_attention,dropout_node,
                                     use_dropout=False,add_self_loop=False,attn_type=0):
        '''
        :param H: The current node feature
        :param W: The node projection for this layer
        :param A: The attention weight vector: a
        :param S: The source edge matrix indices
        :param T: The target edge matrix indices
        :param Adjind: The adjcency matrix indices
        :param Sshape: Shape of S
        :param nb_edge: Number of edge
        :param dropout_attention: dropout_rate for the attention
        :param use_dropout: wether to use dropout
        :param add_self_loop: wether to add edge (i,i)
        :return: alphas,nH
            where alphas is the attention-based adjancency matrix alpha[i,j] correspond to alpha_ij
            nH correspond to the new features for this layer ie alphas(H,W)
        '''

        with tf.name_scope('graph_att_net_attn'):
            # This has shape (nb_node,in_dim) and correspond to the project W.h in the paper
            P=tf.matmul(H,W)
            #print(P.get_shape())

            #This has shape #shape,(nb_edge,nb_node)
            #This sparse tensor contains target nodes for edges.
            #The indices are [edge_idx,node_target_index]
            Tr = tf.SparseTensor(indices=T, values=tf.ones([nb_edge], dtype=tf.float32),
                                 dense_shape=[Sshape[1], Sshape[0]])
            Tr = tf.sparse_reorder(Tr) # reorder so that sparse operations work correctly
            # This tensor has shape(nb_edge,in_dim) and contains the node target projection, ie Wh
            TP = tf.sparse_tensor_dense_matmul(Tr, P,name='TP')

            # This has shape #shape,(nb_node,nb_edge)
            # This sparse tensor contains source nodes for edges.
            # The indices are [node_source_index,edge_idx]
            SD = tf.SparseTensor(indices=S, values=tf.ones([nb_edge],dtype=tf.float32), dense_shape=Sshape)
            SD = tf.sparse_reorder(SD) #shape,(nb_edge,nb_node)
            # This tensor has shape(nb_edge,in_dim) and contains the node source projection, ie Wh
            SP = tf.sparse_tensor_dense_matmul(tf.sparse_transpose(SD), P,name='SP') #shape(nb_edge,in_dim)
            #print('SP', SP.get_shape())


            #Deprecated
            if attn_type==1:
                #Mutlitplication Attn Module
                Aij_forward = A  # attention vector for forward edge and backward edge
                Aij_backward = A  # Here we assume it is the same on contrary to the paper
                # Compute the attention weight for target node, ie a . Whj if j is the target node
                att_target_node = tf.multiply(TP,  Aij_forward[0])
                # Compute the attention weight for the source node, ie a . Whi if j is the target node
                att_source_node = tf.multiply(SP, Aij_backward[0])

                # The attention values for the edge ij is the sum of attention of node i and j
                # Attn( node_i, node_j) = Sum_k (a_k)^2 Hik Hjk Is this what we want ?
                att_source_target_node = tf.reduce_sum( tf.multiply(att_source_node,att_target_node),axis=1)
                attn_values = tf.nn.leaky_relu( att_source_target_node)
            #
            elif attn_type==2:
                #Inspired by learning to rank approach on w(x+-x-)
                # Attn( node_i, node_j) = Sum_k (a_k)  (Hik- Hjk) Is this what we want ?
                att_source_target_node = tf.reduce_sum( tf.multiply(SP-TP,A[0]),axis=1)
                attn_values = tf.nn.leaky_relu( att_source_target_node)

            else:
                Aij_forward=tf.expand_dims(A[0],0)  # attention vector for forward edge and backward edge
                Aij_backward=tf.expand_dims(A[1],0) # Here we assume it is the same on contrary to the paper
                # Compute the attention weight for target node, ie a . Whj if j is the target node
                att_target_node  =tf.matmul(TP,Aij_backward,transpose_b=True)
                # Compute the attention weight for the source node, ie a . Whi if j is the target node
                att_source_node = tf.matmul(SP,Aij_forward,transpose_b=True)

                # The attention values for the edge ij is the sum of attention of node i and j
                attn_values = tf.nn.leaky_relu(tf.squeeze(att_target_node) + tf.squeeze(att_source_node))

            # From that we build a sparse adjacency matrix containing the correct values
            # which we then feed to a sparse softmax
            AttAdj = tf.SparseTensor(indices=Adjind, values=attn_values, dense_shape=[Sshape[0], Sshape[0]])
            AttAdj = tf.sparse_reorder(AttAdj)

            #Note very efficient to do this, we should add the loop in the preprocessing
            if add_self_loop:
                node_indices=tf.range(Sshape[0])
                #Sparse Idendity
                Aij_forward = tf.expand_dims(A[0], 0)
                id_indices = tf.stack([node_indices, node_indices], axis=1)
                val =tf.squeeze(tf.matmul(P,Aij_forward,transpose_b=True))
                spI = tf.SparseTensor(indices=id_indices,values=2.0*val,dense_shape=[Sshape[0], Sshape[0]])

                AttAdj_I = tf.sparse_add(AttAdj,spI)
                alphas = tf.sparse_softmax(AttAdj_I)

            else:
                alphas = tf.sparse_softmax(AttAdj)

            #dropout is done after the softmax
            if use_dropout:
                print('... using dropout for attention layer')
                alphasD = tf.SparseTensor(indices=alphas.indices,values=tf.nn.dropout(alphas.values, 1.0 - dropout_attention),dense_shape=alphas.dense_shape)
                P_D =tf.nn.dropout(P,1.0-dropout_node)
                alphasP = tf.sparse_tensor_dense_matmul(alphasD, P_D)
                return alphasD, alphasP
            else:
                #We compute the features given by the attentive neighborhood
                alphasP = tf.sparse_tensor_dense_matmul(alphas,P)
                return alphas,alphasP



    def _create_original_model(self):
        std_dev_in = float(1.0 / float(self.node_dim))
        self.use_dropout = self.dropout_rate_attention > 0 or self.dropout_rate_node > 0
        self.hidden_layer = []
        attns0 = []

        # Define the First Layer from the Node Input
        for a in range(self.nb_attention):
            # H0 = Maybe do a common H0 and have different attention parameters
            # Change the attention, maybe ?
            # Do multiplicative
            # How to add edges here
            # Just softmax makes a differences
            # I could stack [current_node,representation; edge_features;] and do a dot product on that
            Wa = init_glorot([int(self.node_dim), int(self.node_indim)], name='Wa0' + str(a))
            va = init_glorot([2, int(self.node_indim)], name='va0' + str(a))

            if self.distinguish_node_from_neighbor:
                H0 = tf.matmul(self.node_input, Wa)
                attns.append(H0)

            _, nH = self.simple_graph_attention_layer(self.node_input, Wa, va, self.Ssparse, self.Tsparse, self.Aind,
                                                      self.Sshape, self.nb_edge, self.dropout_p_attn,
                                                      self.dropout_p_node,
                                                      use_dropout=self.use_dropout, add_self_loop=True)
            attns0.append(nH)
        self.hidden_layer.append(
            self.activation(tf.concat(attns0, axis=-1)))  # Now dims should be indim*self.nb_attention

        # Define Intermediate Layers
        for i in range(1, self.num_layers):
            attns = []
            for a in range(self.nb_attention):
                if self.distinguish_node_from_neighbor:
                    Wia = init_glorot(
                        [int(self.node_indim * self.nb_attention + self.node_indim), int(self.node_indim)],
                        name='Wa' + str(i) + '_' + str(a))
                else:
                    Wia = init_glorot([int(self.node_indim * self.nb_attention), int(self.node_indim)],
                                      name='Wa' + str(i) + '_' + str(a))

                via = init_glorot([2, int(self.node_indim)], name='va' + str(i) + '_' + str(a))
                _, nH = self.simple_graph_attention_layer(self.hidden_layer[-1], Wia, via, self.Ssparse, self.Tsparse,
                                                          self.Aind,
                                                          self.Sshape, self.nb_edge, self.dropout_p_attn,
                                                          self.dropout_p_node,
                                                          use_dropout=self.use_dropout, add_self_loop=True)
                attns.append(nH)

            self.hidden_layer.append(self.activation(tf.concat(attns, axis=-1)))

        # Define Logit Layer
        out = []
        for i in range(self.nb_attention):
        #for i in range(1):
            logits_a = init_glorot([int(self.node_indim * self.nb_attention), int(self.n_classes)],
                                   name='Logita' + '_' + str(a))
            via = init_glorot([2, int(self.n_classes)], name='LogitA' + '_' + str(a))
            _, nL = self.simple_graph_attention_layer(self.hidden_layer[-1], logits_a, via, self.Ssparse, self.Tsparse,
                                                      self.Aind,
                                                      self.Sshape, self.nb_edge, self.dropout_p_attn,
                                                      self.dropout_p_node,
                                                      use_dropout=self.use_dropout, add_self_loop=True)
            out.append(nL)

        self.logits = tf.add_n(out) / self.nb_attention
        #self.logits = out[0]


    def _create_nodedistint_model(self):
        '''
        Create a model the separe node distinct models
        :return:
        '''

        std_dev_in = float(1.0 / float(self.node_dim))
        self.use_dropout = self.dropout_rate_attention > 0 or self.dropout_rate_node > 0
        self.hidden_layer = []
        attns0 = []

        # Define the First Layer from the Node Input
        Wa = tf.eye(int(self.node_dim), name='I0')
        H0 = tf.matmul(self.node_input, Wa)
        attns0.append(H0)

        I = tf.Variable(tf.eye(self.node_dim), trainable=False)
        for a in range(self.nb_attention):
            # H0 = Maybe do a common H0 and have different attention parameters
            # Change the attention, maybe ?
            # Do multiplicative
            # How to add edges here
            # Just softmax makes a differences
            # I could stack [current_node,representation; edge_features;] and do a dot product on that
            va = init_glorot([2, int(self.node_dim)], name='va0' + str(a))


            _, nH = self.simple_graph_attention_layer(H0, I, va, self.Ssparse, self.Tsparse, self.Aind,
                                                      self.Sshape, self.nb_edge, self.dropout_p_attn,
                                                      self.dropout_p_node,
                                                      use_dropout=self.use_dropout, add_self_loop=False,attn_type=self.attn_type)
            attns0.append(nH)
        self.hidden_layer.append(
            self.activation(tf.concat(attns0, axis=-1)))  # Now dims should be indim*self.nb_attention

        # Define Intermediate Layers
        for i in range(1, self.num_layers):
            attns = []
            if i == 1:
                previous_layer_dim =int(self.node_dim * self.nb_attention + self.node_dim)
                Wia = init_glorot([previous_layer_dim, int(self.node_indim)],
                    name='Wa' + str(i) + '_' + str(a))
            else:
                previous_layer_dim = int(self.node_indim * self.nb_attention + self.node_indim)
                Wia = init_glorot( [previous_layer_dim, int(self.node_indim)], name='Wa' + str(i) + '_' + str(a))

            Hi = tf.matmul(self.hidden_layer[-1], Wia)
            attns.append(Hi)
            Ia = tf.Variable(tf.eye(self.node_indim), trainable=False)

            for a in range(self.nb_attention):
                via = init_glorot([2, int(self.node_indim)], name='va' + str(i) + '_' + str(a))
                _, nH = self.simple_graph_attention_layer(Hi, Ia, via, self.Ssparse, self.Tsparse,
                                                          self.Aind,
                                                          self.Sshape, self.nb_edge, self.dropout_p_attn,
                                                          self.dropout_p_node,
                                                          use_dropout=self.use_dropout, add_self_loop=False,attn_type=self.attn_type)
                attns.append(nH)

            self.hidden_layer.append(self.activation(tf.concat(attns, axis=-1)))

        # Define Logit Layer
        #TODO Add Attention on Logit Layer
        #It would not cost too much to add an attn mecha once I get the logits
        #If x,y are indicated in the node feature then we can implicitly find the type of edges that we are using ...


        if self.num_layers>1:
            logits_a = init_glorot([int(self.node_indim * self.nb_attention+self.node_indim), int(self.n_classes)],
                                   name='Logita' + '_' + str(a))
        else:
            logits_a = init_glorot([int(self.node_dim * self.nb_attention + self.node_dim), int(self.n_classes)],
                                   name='Logita' + '_' + str(a))
        Bc = tf.ones([int(self.n_classes)], name='LogitA' + '_' + str(a))
        # self.logits = tf.add_n(out) / self.nb_attention
        self.logits = tf.matmul(self.hidden_layer[-1],logits_a) +Bc

    def _create_densegraph_model(self):
        '''
        Create a model the separe node distinct models
        :return:
        '''

        std_dev_in = float(1.0 / float(self.node_dim))
        self.use_dropout = self.dropout_rate_attention > 0 or self.dropout_rate_node > 0
        self.hidden_layer = []
        attns0 = []

        # Define the First Layer from the Node Input
        Wa = tf.eye(int(self.node_dim), name='I0')
        H0 = tf.matmul(self.node_input, Wa)
        attns0.append(H0)

        I = tf.Variable(tf.eye(self.node_dim), trainable=False)
        for a in range(self.nb_attention):
            # H0 = Maybe do a common H0 and have different attention parameters
            # Change the attention, maybe ?
            # Do multiplicative
            # How to add edges here
            # Just softmax makes a differences
            # I could stack [current_node,representation; edge_features;] and do a dot product on that
            va = init_glorot([2, int(self.node_dim)], name='va0' + str(a))


            _, nH = self.dense_graph_attention_layer(H0, I, va, self.nb_node, self.dropout_p_attn,
                                                      self.dropout_p_node,
                                                      use_dropout=self.use_dropout)
            attns0.append(nH)
        self.hidden_layer.append(
            self.activation(tf.concat(attns0, axis=-1)))  # Now dims should be indim*self.nb_attention

        # Define Intermediate Layers
        for i in range(1, self.num_layers):
            attns = []
            if i == 1:
                previous_layer_dim =int(self.node_dim * self.nb_attention + self.node_dim)
                Wia = init_glorot([previous_layer_dim, int(self.node_indim)],
                    name='Wa' + str(i) + '_' + str(a))
            else:
                previous_layer_dim = int(self.node_indim * self.nb_attention + self.node_indim)
                Wia = init_glorot( [previous_layer_dim, int(self.node_indim)], name='Wa' + str(i) + '_' + str(a))

            Hi = tf.matmul(self.hidden_layer[-1], Wia)
            attns.append(Hi)
            Ia = tf.Variable(tf.eye(self.node_indim), trainable=False)

            for a in range(self.nb_attention):
                via = init_glorot([2, int(self.node_indim)], name='va' + str(i) + '_' + str(a))
                _, nH = self.dense_graph_attention_layer(Hi, Ia, via, self.nb_node,
                                                          self.dropout_p_attn,
                                                          self.dropout_p_node,
                                                          use_dropout=self.use_dropout)
                attns.append(nH)

            self.hidden_layer.append(self.activation(tf.concat(attns, axis=-1)))

        # Define Logit Layer
        #TODO Add Attention on Logit Layer
        #It would not cost too much to add an attn mecha once I get the logits
        #If x,y are indicated in the node feature then we can implicitly find the type of edges that we are using ...


        if self.num_layers>1:
            logits_a = init_glorot([int(self.node_indim * self.nb_attention+self.node_indim), int(self.n_classes)],
                                   name='Logita' + '_' + str(a))
        else:
            logits_a = init_glorot([int(self.node_dim * self.nb_attention + self.node_dim), int(self.n_classes)],
                                   name='Logita' + '_' + str(a))
        Bc = tf.ones([int(self.n_classes)], name='LogitA' + '_' + str(a))
        # self.logits = tf.add_n(out) / self.nb_attention
        self.logits = tf.matmul(self.hidden_layer[-1],logits_a) +Bc

    def create_model(self):
        '''
        Create the tensorflow graph for the model
        :return:
        '''
        self.nb_node    = tf.placeholder(tf.int32,(), name='nb_node')
        self.nb_edge = tf.placeholder(tf.int32, (), name='nb_edge')
        self.node_input = tf.placeholder(tf.float32, [None, self.node_dim], name='X_')
        self.y_input    = tf.placeholder(tf.float32, [None, self.n_classes], name='Y')
        #self.dropout_p_H    = tf.placeholder(tf.float32,(), name='dropout_prob_H')
        self.dropout_p_node = tf.placeholder(tf.float32, (), name='dropout_prob_N')
        self.dropout_p_attn = tf.placeholder(tf.float32, (), name='dropout_prob_edges')
        self.S          = tf.placeholder(tf.float32, name='S')
        self.Ssparse    = tf.placeholder(tf.int64, name='Ssparse') #indices
        self.Sshape     = tf.placeholder(tf.int64, name='Sshape') #indices
        self.Aind       =tf.placeholder(tf.int64, name='Sshape') #Adjacency indices
        self.T          = tf.placeholder(tf.float32,[None,None], name='T')
        self.Tsparse    = tf.placeholder(tf.int64, name='Tsparse')


        #self.S_indice = tf.placeholder(tf.in, [None, None], name='S')
        #self.F          = tf.placeholder(tf.float32,[None,None], name='F')

        if self.original_model:
            self._create_original_model()

        elif self.dense_model:
            self._create_densegraph_model()
        else:
            self._create_nodedistint_model()

        cross_entropy_source = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_input)
        self.loss = tf.reduce_mean(cross_entropy_source)

        self.pred = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.correct_prediction = tf.equal(self.pred, tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.grads_and_vars = self.optalg.compute_gradients(self.loss)
        self.train_step = self.optalg.apply_gradients(self.grads_and_vars)
        

        # Add ops to save and restore all the variables.
        self.init = tf.global_variables_initializer()
        self.saver= tf.train.Saver(max_to_keep=0)

        print('Number of Params:')
        self.get_nb_params()



    #TODO Move in MultigraphNN
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
            #print('DropoutEdges',self.dropout_rate_edge)
            Aind = np.array(np.stack([graph.Sind[:, 0], graph.Tind[:, 1]], axis=-1), dtype='int64')
            feed_batch = {

                self.nb_node: graph.X.shape[0],
                self.nb_edge: graph.F.shape[0],
                self.node_input: graph.X,
                self.Ssparse: np.array(graph.Sind, dtype='int64'),
                self.Sshape: np.array([graph.X.shape[0], graph.F.shape[0]], dtype='int64'),
                self.Tsparse: np.array(graph.Tind, dtype='int64'),
                #self.F: graph.F,
                self.Aind: Aind,
                self.y_input: graph.Y,
                #self.dropout_p_H: self.dropout_rate_H,
                self.dropout_p_node: self.dropout_rate_node,
                self.dropout_p_attn: self.dropout_rate_attention,
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
        Aind = np.array(np.stack([graph.Sind[:, 0], graph.Tind[:, 1]], axis=-1), dtype='int64')
        feed_batch = {

            self.nb_node: graph.X.shape[0],
            self.nb_edge: graph.F.shape[0],
            self.node_input: graph.X,

            self.Ssparse: np.array(graph.Sind, dtype='int64'),
            self.Sshape: np.array([graph.X.shape[0], graph.F.shape[0]], dtype='int64'),
            self.Tsparse: np.array(graph.Tind, dtype='int64'),
            self.Aind: Aind,
            #self.F: graph.F,
            self.y_input: graph.Y,
            #self.dropout_p_H: 0.0,
            self.dropout_p_node: 0.0,
            self.dropout_p_attn: 0.0,
            #self.dropout_p_edge_feat: 0.0,
            #self.NA_indegree: graph.NA_indegree
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
        Aind = np.array(np.stack([graph.Sind[:, 0], graph.Tind[:, 1]], axis=-1), dtype='int64')
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
            #self.F: graph.F,
            self.Aind: Aind,
            self.dropout_p_node: 0.0,
            self.dropout_p_attn: 0.0,
        }
        Ops = session.run([self.pred], feed_dict=feed_batch)
        if verbose:
            print('Got Prediction for:',Ops[0].shape)
        return Ops[0]


    #TODO Move that MultiGraphNN
    def train_All_lG(self,session,graph_train,graph_val, max_iter, eval_iter = 10, patience = 7, graph_test = None, save_model_path = None):
        '''

        Merge all the graph and train on them

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
        best_val_acc = 0.0
        wait = 0
        stop_training = False
        stopped_iter = max_iter
        train_accuracies = []
        validation_accuracies = []
        test_accuracies = []
        conf_mat = []

        start_monitoring_val_acc = False

        # Not Efficient to compute this for

        merged_graph = gcn_datasets.GCNDataset.merge_allgraph(graph_train)

        self.train(session, merged_graph, n_iter=1)

        for i in range(max_iter):
            if stop_training:
                break

            if i % eval_iter == 0:
                print('\nEpoch', i)
                _, tr_acc = self.test_lG(session, graph_train, verbose=False)
                print(' Train Acc', '%.4f' % tr_acc)
                train_accuracies.append(tr_acc)

                _, node_acc = self.test_lG(session, graph_val, verbose=False)
                print(' Valid Acc', '%.4f' % node_acc)
                validation_accuracies.append(node_acc)

                if save_model_path:
                    save_path = self.saver.save(session, save_model_path, global_step=i)

                if graph_test:
                    _, test_acc = self.test_lG(session, graph_test, verbose=False)
                    print('  Test Acc', '%.4f' % test_acc)
                    test_accuracies.append(test_acc)

                # TODO min_delta
                # if tr_acc>0.99:
                #    start_monitoring_val_acc=True

                if node_acc > best_val_acc:
                    best_val_acc = node_acc
                    wait = 0
                else:
                    if wait >= patience:
                        stopped_iter = i
                        stop_training = True
                    wait += 1
            else:

                self.train(session, merged_graph, n_iter=1)
        # Final Save
        # if save_model_path:
        # save_path = self.saver.save(session, save_model_path, global_step=i)
        # TODO Add the final step
        mean_acc = []
        print('Stopped Model Training after', stopped_iter)
        print('Val Accuracies', validation_accuracies)

        print('Final Training Accuracy')
        _, node_train_acc = self.test_lG(session, graph_train)
        print('Train Mean Accuracy', '%.4f' % node_train_acc)

        print('Final Valid Acc')
        self.test_lG(session, graph_val)

        R = {}
        R['train_acc'] = train_accuracies
        R['val_acc'] = validation_accuracies
        R['test_acc'] = test_accuracies
        R['stopped_iter'] = stopped_iter
        R['confusion_matrix'] = conf_mat
        # R['W_edge'] =self.get_Wedge(session)
        if graph_test:
            _, final_test_acc = self.test_lG(session, graph_test)
            print('Final Test Acc', '%.4f' % final_test_acc)
            R['final_test_acc'] = final_test_acc

        return R