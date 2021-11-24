# -*- coding: utf-8 -*-

"""
    ECN Model
    
    Copyright NAVER(C) 2018, 2019  Stéphane Clinchant, Animesh Prasad
         Hervé Déjean, Jean-Luc Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os
#import tensorflow as tf
import pickle
import glob
import random
import gc
import fnmatch
import shutil
import traceback
import json
from optparse import OptionParser

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ModuleNotFoundError:
    import tensorflow as tf

import numpy as np
import scipy.sparse as sp

try: #to ease the use without proper Python installation
    from common.trace import traceln
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    from common.trace import traceln
from common.chrono import chronoOn, chronoOff, pretty_time_delta
from common.TestReport import TestReport
from common.LabelBinarizer2 import LabelBinarizer2

from util.gzip_pkl_list import PklList
import util.Tracking as Tracking

from graph.GraphModel import GraphModel, GraphModelException, GraphModelNoEdgeException
from graph.Graph import Graph, GraphException
import gcn.gcn_models as gcn_models
from gcn.gcn_datasets import GCNDataset

from tasks.DU_Task          import DU_Task
import tasks.DU_ECN_Task    as DU_ECN_Task 


traceln("VERSION: tensorflow.__version__=", tf.__version__)

class DU_Model_ECN(GraphModel):
    sSurname = "ecn"
    pkl_queue_size = 10  # in --loader mode, how many pkl pre-loaded in RAM?

    def __init__(self, sName, sModelDir):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        """
        super().__init__(sName, sModelDir)

        self.gcn_model  = None
        self.tf_graph   = None
        self.tf_session = None

        # A binarizer that uses 2 columns for binary classes (instead of only 1)
        self.labelBinarizer = LabelBinarizer2()

    def getMetadataComment(self):
        s = super().getMetadataComment()
        s += "\n" + repr(self.model_config)
        return s

    @staticmethod
    def getBaselineConfig():
        '''
        Return A Baseline Edge Conv Net Configuration with 3 layers and 10 convolutions per layer
        :return:
        '''
        assert False, "code to be checked"
        config={}
        config['name'] = '3Layers-10conv-stack'
        config['nb_iter'] = 2000
        config['lr'] = 0.001
        config['stack_convolutions'] = True
        config['mu'] = 0.0
        config['num_layers'] = 3
        config['node_indim'] = -1  # INDIM =2 not working here
        config['nconv_edge'] = 10
        config['fast_convolve'] = True
        config['dropout_rate_edge'] = 0.0
        config['dropout_rate_edge_feat'] = 0.0
        config['dropout_rate_node'] = 0.0
        config['ratio_train_val']=0.15

        return config


    def configureLearner(self,**kwargs):

        """
        To configure the learner: pass a dictionary using the ** argument-passing method
        """
        #Pass arguments
        self.model_config=kwargs

    def getModelFilename(self):
        return os.path.join(self.sDir, self.sName+"._."+self.sSurname+".bestmodel.ckpt")

    def getTmpModelFilename(self):
        return os.path.join(self.sDir, self.sName+"._."+self.sSurname+".tmpmodel.ckpt")

    def getValScoreFilename(self):
        return os.path.join(self.sDir, self.sName+"._."+self.sSurname+'.validation_scores.pkl')

    def getlabelBinarizerFilename(self):
        return os.path.join(self.sDir, self.sName+"._."+self.sSurname+'.label_binarizer.pkl')

    def getModelConfigFilename(self):
        return os.path.join(self.sDir, self.sName+"._."+self.sSurname+'.model_config.pkl')


    @classmethod
    def addConjugateRevertedEdge(cls, lG, lX):
        """
        our Graph class relies on undirected edges.
        Here we need directed edges, so we create them.
        """
        lX_new = []
        for _g, X  in zip(lG, lX):
            (nf_dual, edge_dual, ef_dual) = X
            try:
                reverted_edge_dual = edge_dual[:,[1,0]]
                double_edge_dual = np.vstack((edge_dual, reverted_edge_dual))
                double_ef_dual   = np.vstack((ef_dual  , ef_dual))
            except IndexError as e:
                if edge_dual.size == 0:
                    assert ef_dual.size == 0
                    # ok, no edge, that's it!
                    double_edge_dual, double_ef_dual = edge_dual, ef_dual
                else:
                    raise e
            
            lX_new.append((nf_dual, double_edge_dual, double_ef_dual))
        return lX_new

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def get_lX(self, lGraph):
        """
        Compute node and edge features and return one X matrix for each graph as a list
        return a list of X, a list of Y matrix
        """
        lX=[]
        nb_nodes=0
        nb_edges=0
        for g in lGraph:
            #TODO This could move in the Graph Code ...
            X = g.getX(self._node_transformer, self._edge_transformer)
            
            # ECN need directed edges, but for the conjugate graph, we proceed differently
            # because the features remain the same in both directions.
            if g.bConjugate:
                [(node_features, new_edges, new_edges_feat)] = self.addConjugateRevertedEdge([g], [X])
            else:
                # in conjugate graph mode, we "duplicate" the edges in the conjugate graph
                (node_features, edges, edge_features) = X
                g.revertEdges()
                (node_features, edges_reverse, edge_features_reverse) = g.getX(self._node_transformer, self._edge_transformer)
                new_edges=np.vstack([edges, edges_reverse])
                new_edges_feat =np.vstack([edge_features,edge_features_reverse])
            nb_edges += new_edges.shape[0]
            nb_nodes += node_features.shape[0]
            lX.append((node_features, new_edges, new_edges_feat) )

        traceln("\t\t ECN #nodes=%d  #edges=%d " % (nb_nodes, nb_edges))
        return lX

    def convert_lX_lY_to_GCNDataset(self,lX, lY,training=False,test=False,predict=False):
        gcn_list = []
        graph_id = 0

        # This has state information here --> move that to DU_Model_ECN ...
        lys = []
        for _, ly in zip(lX, lY):
            lys.extend(list(ly))
        #print (lys)            

        if training:
            self.labelBinarizer.fit(lys) # note: can call it multiple times, it is fitted once, at first call

        for lx, ly  in zip(lX, lY):
            nf = lx[0]
            edge = lx[1]
            ef = lx[2]
            nb_node = nf.shape[0]

            graph = GCNDataset(str(graph_id))
            graph.X = nf
            if training or test:
                graph.Y = self.labelBinarizer.transform(ly)
                                
            elif predict:
                graph.Y = -np.ones((nb_node, len(self.labelBinarizer.classes_)), dtype='i')
            else:
                raise Exception('Invalid Usage: one of train,test,predict should be true')
            # We are making the adacency matrix here

            # print(edger)
            A1 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])), shape=(nb_node, nb_node))
            # A2 = sp.coo_matrix((np.ones(edger.shape[0]), (edger[:, 0], edger[:, 1])), shape=(nb_node, nb_node))
            graph.A = A1  # + A2

            # JL: unued??   edge_normalizer = Normalizer()
            # Normalize EA

            E0 = np.hstack([edge, ef])  # check order
            # E1 = np.hstack([edger, efr])  # check order

            graph.E = E0
            #graph.compute_NA()
            graph.compute_NodeEdgeMat()

            gcn_list.append(graph)
            graph_id += 1

        return gcn_list

    def convert_X_to_GCNDataset(self, X):
        """
        Same code as above, dedicated to the predict mode (no need  for Y)
        """
        graph_id = 0

        nf      = X[0]
        edge    = X[1]
        ef      = X[2]
        nb_node = nf.shape[0]

        graph = GCNDataset(str(graph_id))
        graph.X = nf
        graph.Y = -np.ones((nb_node, len(self.labelBinarizer.classes_)), dtype='i')

        A1 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])), shape=(nb_node, nb_node))
        # A2 = sp.coo_matrix((np.ones(edger.shape[0]), (edger[:, 0], edger[:, 1])), shape=(nb_node, nb_node))
        graph.A = A1  # + A2

        # JL: unued??   edge_normalizer = Normalizer()
        # Normalize EA

        E0 = np.hstack([edge, ef])  # check order
        # E1 = np.hstack([edger, efr])  # check order

        graph.E = E0
        #graph.compute_NA()
        graph.compute_NodeEdgeMat()

        return graph

    def convert_X_Y_to_GCNDataset(self, X, Y, graph_id = 0):
        """
        for train X,Y that are loaded on the fly from the pkl
        """
        (nf, edge, ef) = X
        nb_node = nf.shape[0]

        graph = GCNDataset(str(graph_id))
        graph.X = nf
        graph.Y = self.labelBinarizer.transform(Y)
                            
        # print(edger)
        A1 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1]))
                           , shape=(nb_node, nb_node))
        graph.A = A1  # + A2
        graph.E = np.hstack([edge, ef])
        graph.compute_NodeEdgeMat()

        return graph

    def _init_TF_graph(self):
        tf_graph = tf.Graph()
        with tf_graph.as_default():
            self._init_model()
            #This create a session containing the correct model
            self.restore()
        self.tf_graph = tf_graph
        
    def _init_model(self):
        '''
        Create the tensorflow graph.
        This function assume that self.model_config contains all the appropriate variables
        to set the model
        This function is called in the train operation and in the load function for testing  on new documents
        :return:
        '''
        self.gcn_model = gcn_models.EdgeConvNet(self.model_config['node_dim'], self.model_config['edge_dim'], self.model_config['nb_class'],
                                                num_layers=self.model_config['num_layers'],
                                                learning_rate=self.model_config['lr'],
                                                mu=self.model_config['mu'],
                                                node_indim=self.model_config['node_indim'],
                                                nconv_edge=self.model_config['nconv_edge'],
                                                )
        self.gcn_model.set_learning_options(self.model_config)
        self.gcn_model.create_model()

#         self.gcn_model.stack_instead_add = self.model_config['stack_convolutions']
#         #TODO Clean Constructor
#         if 'activation' in self.model_config:
#             self.gcn_model.activation = self.model_config['activation']
# 
#         if 'fast_convolve' in self.model_config:
#             self.gcn_model.fast_convolve = self.model_config['fast_convolve']
# 
#         if 'dropout_rate_edge' in self.model_config:
#             self.gcn_model.dropout_rate_edge = self.model_config['dropout_rate_edge']
#             print('Dropout Edge', self.gcn_model.dropout_rate_edge)
# 
#         if 'dropout_rate_edge_feat' in self.model_config:
#             self.gcn_model.dropout_rate_edge_feat = self.model_config['dropout_rate_edge_feat']
#             print('Dropout Edge', self.gcn_model.dropout_rate_edge_feat)
# 
#         if 'dropout_rate_node' in self.model_config:
#             self.gcn_model.dropout_rate_node = self.model_config['dropout_rate_node']
#             print('Dropout Node', self.gcn_model.dropout_rate_node)


    def _cleanTmpCheckpointFiles(self):
        '''
        When a model is trained, tensorflow checkpoint files are created every 10 epochs
        This functions cleans unecessary checkpoint files
        :return: the number of files deleted
        '''
        dir_path = os.path.dirname(self.getTmpModelFilename())
        modelsFiles = os.listdir(dir_path)
        found_files = fnmatch.filter(modelsFiles, os.path.basename(self.getTmpModelFilename()) + '*')

        for tmp_f in found_files:
            os.remove(os.path.join(dir_path, tmp_f))

        nb_clean=len(found_files)
        traceln("\t- cleaning previous model : ", nb_clean, ' files deleted')
        return nb_clean

    def _getlGlXlY(self, lG):
        """
        get lX, lY from the list of graph
        since the conjugate computation may lead to errors, we skip those graph
        return lG, lX, lY
        """
        lGraphNew, lX, lY = [], [], []
        for _ig, _g in enumerate(lG):
            try:
                [X], [Y] = self.get_lX_lY([_g])
                lX.append(X)
                lY.append(Y)
                lGraphNew.append(_g)
            except GraphException as e:
                traceln("Skipping graph of index %d because %s" % (_ig, str(e)))
        return lGraphNew, lX, lY
    
    def _prepare_for_train(self, lGraph, lGraph_vld):
        """
        Prepare for training eCN or EnsembleECN
        """
        traceln('ECN Training ', self.sName)
        traceln("\t- computing features on training set")
        traceln("\t\t #nodes=%d  #edges=%d " % Graph.getNodeEdgeTotalNumber(lGraph))
        chronoOn()
        
        # here we might get trouble when computing the conjugate graph...
        # lX, lY = self.get_lX_lY(lGraph)
        lGraph, lX, lY = self._getlGlXlY(lGraph)
        self._computeModelCaracteristics(lX)  # we discover here dynamically the number of features of nodes and edges
        # self._tNF_EF contains the number of node features and edge features
        traceln("\t\t %s" % self._getNbFeatureAsText())
        traceln("\t [%.1fs] done\n" % chronoOff())
        nb_class = len(lGraph[0].getLabelNameList())  #Is it better to do Y.shape ?
        traceln("\t- %d classes (%s)" % (nb_class, lGraph[0].getLabelNameList()))
        traceln("\tRAM size of train X,Y features %.1f %s" % self.format_size(self.get_lX_size(lX) + self.get_lY_size(lY)))
                        
        traceln("\t- retrieving or creating model...")

        self.model_config['node_dim'] = self._tNF_EF[0]
        self.model_config['edge_dim'] = self._tNF_EF[1]
        self.model_config['nb_class'] = nb_class

        if False: 
            with open ('linear_reg', 'wb') as save_file:
                pickle.dump((lX,lY), save_file, pickle.HIGHEST_PROTOCOL)
            
        #This converts the lX,lY in the format necessary for GCN Models
        gcn_graph = self.convert_lX_lY_to_GCNDataset(lX,lY,training=True)
        del lX, lY
        gc.collect()

        #Save the label Binarizer for prediction usage
        fd_lb =open(self.getlabelBinarizerFilename(),'wb')
        pickle.dump(self.labelBinarizer,fd_lb)
        fd_lb.close()

        #TODO Save the validation set too to reproduce experiments
        random.shuffle(gcn_graph)
        
        if lGraph_vld:
            gcn_graph_train = gcn_graph
            # lX_vld, lY_vld = self.get_lX_lY(lGraph_vld)
            lGraph_vld, lX_vld, lY_vld = self._getlGlXlY(lGraph_vld)
            traceln("\tRAM size of valid X,Y features %.1f %s" % self.format_size(self.get_lX_size(lX_vld) + self.get_lY_size(lY_vld)))
            gcn_graph_val = self.convert_lX_lY_to_GCNDataset(lX_vld, lY_vld, test=True)
            del lX_vld, lY_vld
        else:
            #Get a validation set from the training set
            split_idx = max(1, int(self.model_config['ratio_train_val'] * len(gcn_graph)))
            traceln(" - using %d train graphs as validation graphs" % split_idx)
            gcn_graph_train = []
            gcn_graph_val = []
            gcn_graph_val.extend(gcn_graph[:split_idx])
            gcn_graph_train.extend(gcn_graph[split_idx:])
        traceln("%d training graphs --  %d validation graphs"%(len(gcn_graph_train), len(gcn_graph_val)))
        self._cleanTmpCheckpointFiles()
        gc.collect()

        return gcn_graph_train, gcn_graph_val
        
    def train(self, lGraph, lGraph_vld, bWarmStart=True, expiration_timestamp=None, verbose=0):
        """
        Return a model trained using the given labelled graphs.
        The train method is expected to save the model into self.getModelFilename(), at least at end of training
        If bWarmStart==True, The model is loaded from the disk, if any, and if fresher than given timestamp, and training restarts

        if some baseline model(s) were set, they are also trained, using the node features

        """

        gcn_graph_train, gcn_graph_val = self._prepare_for_train(lGraph, lGraph_vld)

        self._train(gcn_graph_train, gcn_graph_val)
        
    def _train(self, gcn_graph_train, gcn_graph_val):
        """
        In ECN Ensemble mode, many steps have been alread performed by the ECNEnsemble model
        """
        patience = self.model_config['patience'] if 'patience' in self.model_config else self.model_config['nb_iter']
        
        #Save the model config in order to restore the model later
        fd_mc = open(self.getModelConfigFilename(), 'wb')
        pickle.dump(self.model_config, fd_mc)
        fd_mc.close()
        
        #This call the ECN internal constructor and defines the tensorflow graph
        tf_graph=tf.Graph()
        with tf_graph.as_default():
            self._init_model()
        self.tf_graph = tf_graph
        
        # SC  with tf.Session() as session:
        # Animesh
        with tf.Session(graph=self.tf_graph) as session:
            session.run([self.gcn_model.init])

            R = self.gcn_model.train_with_validation_set(session, gcn_graph_train, gcn_graph_val, self.model_config['nb_iter'],
                                                    eval_iter=10, patience=patience,
                                                    save_model_path=self.getTmpModelFilename())
            f = open(self.getValScoreFilename(), 'wb')
            pickle.dump(R, f)
            f.close()

        #This save the model
        self._getBestModelVal()
        self._cleanTmpCheckpointFiles()
        
        #We reopen a session here and load the selected model if we need one
        self.restore()

    # ------------   Using pickled X, Y ------------
    def _prepare_for_train_XY(self, lsTrnDir, lsVldDir):
        """
        Prepare for training eCN or EnsembleECN
        - loading the pickles  (trn and vld)
        - converting to TF data (trn and vld)
        - fitting a label binarizer and storing it  (trn)
        - shuffling data (trn)
        """
        traceln('ECN Training ', self.sName)
        # traceln("\t- computing features on training set")
        # traceln("\t\t #nodes=%d  #edges=%d " % Graph.getNodeEdgeTotalNumber(lGraph))
        chronoOn()
        
        # temporary code
        lX, lY = list(), list()
        for sTrnDir in lsTrnDir:
            _lXY = list(DU_Task.getPickledXY(sTrnDir, blPickleXY=True))
            _lX, _lY = zip(*_lXY)
            lX.extend(_lX)
            lY.extend(_lY)
        del _lXY, _lX, _lY
        traceln("\tRAM size of train X,Y features %.1f %s" % self.format_size(self.get_lX_size(lX) + self.get_lY_size(lY)))

        traceln("\t- creating model...")
        
        #This converts the lX,lY in the format necessary for GCN Models
        gcn_graph = self.convert_lX_lY_to_GCNDataset(lX,lY,training=True)
        del lX, lY
        gc.collect()

        #Save the label Binarizer for prediction usage
        fd_lb =open(self.getlabelBinarizerFilename(),'wb')
        pickle.dump(self.labelBinarizer,fd_lb)
        fd_lb.close()

        random.shuffle(gcn_graph)
        gcn_graph_train = gcn_graph
        
        # lX_vld, lY_vld = self.get_lX_lY(lGraph_vld)
        lX_vld, lY_vld = list(), list()
        for sVldDir in lsVldDir:
            _lXY = list(DU_Task.getPickledXY(sVldDir, blPickleXY=True))
            _lX, _lY = zip(*_lXY)
            lX_vld.extend(_lX)
            lY_vld.extend(_lY)
        del _lXY, _lX, _lY
        traceln("\tRAM size of valid X,Y features %.1f %s" % self.format_size(self.get_lX_size(lX_vld) + self.get_lY_size(lY_vld)))
        
        gcn_graph_val = self.convert_lX_lY_to_GCNDataset(lX_vld, lY_vld, test=True)
        del lX_vld, lY_vld

        traceln("%d training graphs --  %d validation graphs"%(len(gcn_graph_train), len(gcn_graph_val)))
        self._cleanTmpCheckpointFiles()

        gc.collect()
        traceln("\t [%.1fs] done\n" % chronoOff())

        return gcn_graph_train, gcn_graph_val

    def train_XY(self, lsTrnDir, lsVldDir, bWarmStart=True, expiration_timestamp=None, verbose=0):
        """
        Return a model trained using the given pickled X,Y.
        The train method is expected to save the model into self.getModelFilename(), at least at end of training
        If bWarmStart==True, The model is loaded from the disk, if any, and if fresher than given timestamp, and training restarts

        if some baseline model(s) were set, they are also trained, using the node features

        """

        gcn_graph_train, gcn_graph_val = self._prepare_for_train_XY(lsTrnDir, lsVldDir)

        self._train_XY(gcn_graph_train, gcn_graph_val)
        
    def _train_XY(self, gcn_graph_train, gcn_graph_val, bTrnAcc=True):
        """
        In ECN Ensemble mode, many steps have been alread performed by the ECNEnsemble model
        """
        patience = self.model_config['patience'] if 'patience' in self.model_config else self.model_config['nb_iter']
        
        #Save the model config in order to restore the model later
        fd_mc = open(self.getModelConfigFilename(), 'wb')
        pickle.dump(self.model_config, fd_mc)
        fd_mc.close()
        
        #This call the ECN internal constructor and defines the tensorflow graph
        tf_graph=tf.Graph()
        with tf_graph.as_default():
            self._init_model()
        self.tf_graph = tf_graph
        
        # SC  with tf.Session() as session:
        # Animesh
        with tf.Session(graph=self.tf_graph) as session:
            session.run([self.gcn_model.init])

            R = self.gcn_model.train_with_validation_set(session, gcn_graph_train, gcn_graph_val, self.model_config['nb_iter'],
                                                    eval_iter=10, patience=patience,
                                                    save_model_path=self.getTmpModelFilename()
                                                    , bTrnAcc=False  # too costly
                                                    )
            f = open(self.getValScoreFilename(), 'wb')
            pickle.dump(R, f)
            f.close()

        #This save the model
        self._getBestModelVal()
        self._cleanTmpCheckpointFiles()
        
        #We reopen a session here and load the selected model if we need one
        self.restore()
    # ----------------------------------------------
    def _getBestModelVal(self):
        val_pickle = self.getValScoreFilename()
        traceln("\t- reading training info from...",val_pickle)
        f = open(val_pickle, 'rb')
        R = pickle.load(f)
        f.close()

        val = R['val_acc']
        #traceln('Validation scores', ['%03.2f'% sx for sx in val])
        epoch_index = np.argmax(val)
        traceln('BEST PERFORMANCE (acc= %.2f %%) on valid set at Epoch %d'
                % (100*max(val), 10 * epoch_index))

        model_path = self.getTmpModelFilename()+"-"+ str(10 * epoch_index)

        dir_path=os.path.dirname(self.getTmpModelFilename())
        fnames =os.listdir(dir_path)

        #Find all the checkpoints files for that epoch
        found_files=fnmatch.filter(fnames, os.path.basename(model_path)+'*')
        #traceln(found_files)
        #Now copy the files with the final model name
        for m in found_files:
            f_src=os.path.join(dir_path,m)
            f_suffix = m[len(os.path.basename(model_path)):]
            f_dst=self.getModelFilename()+f_suffix
            shutil.copy(f_src,f_dst)
            traceln('Copying  Final Model files ', f_src,' ',f_dst)

 
    # ------------   Using pickled X, Y with a delayed loader ------------
    def _prepare_for_train_XY_loader(self, lsTrnDir, lsVldDir):
        """
        Prepare for training eCN or EnsembleECN
        - loading the pickles  (trn and vld)
        - converting to TF data (trn and vld)
        - fitting a label binarizer and storing it  (trn)
        - shuffling data (trn)
        """
        traceln('ECN Training using a data loader', self.sName)
        # traceln("\t- computing features on training set")
        # traceln("\t\t #nodes=%d  #edges=%d " % Graph.getNodeEdgeTotalNumber(lGraph))
        chronoOn()
        
        # expand the list of pickle file into a single list
        lsTrnPkl = []
        for sTrnDir in lsTrnDir:
            lsTrnPkl.extend( list(glob.iglob(os.path.join(sTrnDir, "[0-9]*[0-9].pkl"))) )
        
        # fit and save the label binarizer
        # make it an instance of our very special PklList class, without processing
        LtXY = PklList(self.pkl_queue_size, None,
                        lsTrnPkl)        
        lAllY = []
        for _X, Y in  LtXY: lAllY.extend(Y)
        traceln("\tRAM size of train Y features %.1f %s" % self.format_size(self.get_lY_size(lAllY)))
        self.labelBinarizer.fit(lAllY) # note: can call it multiple times, it is fitted once, at first call
        del lAllY
        #Save the label Binarizer for prediction usage
        fd_lb =open(self.getlabelBinarizerFilename(),'wb')
        pickle.dump(self.labelBinarizer,fd_lb)
        fd_lb.close()
        gc.collect()
        
        # make it an instance of our very special PklList class
        gcn_train_graph = PklList(self.pkl_queue_size, self.convert_X_Y_to_GCNDataset
                                  , lsTrnPkl)
        random.shuffle(gcn_train_graph)

        traceln("\t- creating model...")
        
        # lX_vld, lY_vld = self.get_lX_lY(lGraph_vld)
        lX_vld, lY_vld = list(), list()
        for sVldDir in lsVldDir:
            _lXY = list(DU_Task.getPickledXY(sVldDir, blPickleXY=True))
            _lX, _lY = zip(*_lXY)
            lX_vld.extend(_lX)
            lY_vld.extend(_lY)
        del _lXY, _lX, _lY
        traceln("\tRAM size of valid X,Y features %.1f %s" % self.format_size(self.get_lX_size(lX_vld) + self.get_lY_size(lY_vld)))
        
        gcn_graph_val = self.convert_lX_lY_to_GCNDataset(lX_vld, lY_vld, test=True)
        del lX_vld, lY_vld

        traceln("%d training graphs --  %d validation graphs"%(len(lsTrnPkl), len(gcn_graph_val)))
        self._cleanTmpCheckpointFiles()

        gc.collect()
        traceln("\t [%.1fs] done\n" % chronoOff())
        return gcn_train_graph, gcn_graph_val

    def train_XY_loader(self, lsTrnDir, lsVldDir, bWarmStart=True, expiration_timestamp=None, verbose=0):
        """
        Return a model trained using the given pickled X,Y.
        The train method is expected to save the model into self.getModelFilename(), at least at end of training
        If bWarmStart==True, The model is loaded from the disk, if any, and if fresher than given timestamp, and training restarts

        if some baseline model(s) were set, they are also trained, using the node features

        """
        gcn_train_graph, gcn_graph_val = self._prepare_for_train_XY_loader(lsTrnDir, lsVldDir)

        self._train_XY(gcn_train_graph, gcn_graph_val, bTrnAcc=False)
        
        #traceln("Average read operations:  loading time=%.3fs   waiting time=%.3fs" % gcn_train_graph.averageLoadWaitTime())
        traceln(gcn_train_graph.averageTimeReport())
        
    # ----------------------------------------------
    def _getNbFeatureAsText(self):
        """
        return the number of node features and the number of edge features as a textual message
        """
        return "#features nodes=%d  edges=%d "%self._tNF_EF

    def _computeModelCaracteristics(self, lX):
        """
        We discover dynamically the number of features. Pretty convenient for developer.
        Drawback: if the feature extractor code changes, predicting with a stored model will crash without beforehand catch
        """
        self._tNF_EF = (lX[0][0].shape[1], lX[0][2].shape[1]) #number of node features,  number of edge features
        return self._tNF_EF

    def gridsearch(self, lGraph):
        """
        Return a model trained using the given labelled graphs, by grid search (see http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
        The train method is expected to save the model into self.getModelFilename(), at least at end of training

        if some baseline model(s) were set, they are also trained, using the node features

        """
        raise Exception("Method must be overridden")

    def save(self):
        """
        Save a trained model
        """
        # by default, save the baseline models
        sBaselineFile = self.getBaselineFilename()
        self.gzip_cPickle_dump(sBaselineFile, self.getBaselineModelList())
        return sBaselineFile

        # --- Model loading/writing -------------------------------------------------------------
    def load(self, expiration_timestamp=None):
        """
        Load myself from disk
        If an expiration timestamp is given, the model stored on disk must be fresher than timestamp
        return self or raise a ModelException
        """
        # by default, load the baseline models
        sBaselineFile = self.getBaselineFilename()
        try:
            self._lMdlBaseline = self._loadIfFresh(sBaselineFile, expiration_timestamp, self.gzip_cPickle_load)
        except GraphModelException:
            traceln('no baseline model found : %s' % (sBaselineFile))
        self.loadTransformers(expiration_timestamp)

        fd_mc = open(self.getModelConfigFilename(), 'rb')
        self.model_config=pickle.load(fd_mc)
        fd_mc.close()

        fd_lb = open(self.getlabelBinarizerFilename(), 'rb')
                
        self.labelBinarizer = pickle.load(fd_lb)
        fd_lb.close()

        self._init_TF_graph()
        
        return self

        
    def train_baselines(self,mdl_name = 'lr', class_weight={0:1, 1:4.5}):
        # For: Testing on basic models
        def check (i, l1, l2):
            feat_size = l2[0].shape[0]
            a = None
            b = None
            c  = np.zeros(feat_size)
            for item1, item2 in zip(l1, l2):
                if item1[0] ==  i:
                    a = item2
                if item1[1] == i:
                    b = item2
            if a is None:
                a =c
            if b is None:
                b = c
            return np.hstack([a,b])
            
            import pickle as p
            from sklearn import svm
            from sklearn import linear_model
            with open ('linear_reg', 'rb') as read_file:
                lX, lY = p.load(read_file)
            
            l = []
            
            for g in lX:
                _l = []
                for index, items in enumerate(g[0]):
                    _l.append(np.hstack([items,  check(index, g[1], g[2])]))
                l.append(_l)
            
            
            l  = np.array([item for sublist in l for item in sublist])
            lY = np.array([item for sublist in lY for item in sublist])
            
            if mdl_name != 'lr':
                mdl = svm.SVC(kernel='rbf', C=0.001, class_weight=class_weight, probability=True)
            else:
                mdl  =  linear_model.LogisticRegression(C=0.001, class_weight=class_weight)
            mdl.fit(l, lY)
            return mdl

        def predict_baselines(self,lX,mdl, prob=True): 
            # For: Testing on basic models
            def check (i, l1, l2):
                feat_size = l2[0].shape[0]
                a = None
                b = None
                c  = np.zeros(feat_size)
                for item1, item2 in zip(l1, l2):
                    if item1[0] ==  i:
                        a = item2
                    if item1[1] == i:
                        b = item2
                if a is None:
                    a = c
                if b is None:
                    b = c
                return np.hstack([a,b])    
              
              
              
            l = []
            for g in lX:
                _l = []
                for index, items in enumerate(g[0]):
                    _l.append(np.hstack([items,  check(index, g[1], g[2])]))
                l.append(_l)
            
            lY_pred_proba = []
            if prob:
                for g in l:
                    lY_pred_proba.append(mdl.predict_proba(g))
            else:
                for g in l:
                    lY_pred_proba.append(mdl.predict(g))
            
            return lY_pred_proba




    def _prepare_for_test(self, lGraph):

        traceln("\t- computing features on test set")
        traceln("\t\t #nodes=%d  #edges=%d " % Graph.getNodeEdgeTotalNumber(lGraph))
        chronoOn()

        # lX, lY = self.get_lX_lY(lGraph)
        lGraph, lX, lY = self._getlGlXlY(lGraph)
        traceln("\t [%.1fs] done\n" % chronoOff())

        gcn_graph_test = self.convert_lX_lY_to_GCNDataset(lX, lY, training=False,test=True)
        
        return gcn_graph_test, lX, lY
     
    def test(self, lGraph
             , lsDocName=None
             , predict_proba=False
             ):
        """
        Test the model using those graphs and report results on stderr

        if some baseline model(s) were set, they are also tested

        Return a Report object
        """
        #Assume the model was created or loaded
        assert lGraph

        gcn_graph_test, lX, lY = self._prepare_for_test(lGraph)
        
        lLabelName = lGraph[0].getLabelNameList()

        chronoOn("test2")
        session=self.tf_session
        if predict_proba:
            #TODO Should split that function diryt
            lY_pred_proba = self.gcn_model.predict_prob_lG(session, gcn_graph_test, verbose=False)
            traceln(" [%.1fs] done\n" % chronoOff("test2"))

            ret = lY_pred_proba
        else:
            #pdb.set_trace()
            lY_pred = self.gcn_model.predict_lG(session, gcn_graph_test, verbose=False)
                        
#             # Convert to list as Python pickle does not  seem like the array while the list can be pickled
#             lY_list = []
#             for x in lY_pred:
#                 lY_list.append(list(x))

            traceln(" [%.1fs] done\n" % chronoOff("test2"))
#             tstRpt = TestReport(self.sName, lY_list, lY, lLabelName, lsDocName=lsDocName)
            tstRpt = TestReport(self.sName, lY_pred, lY, lLabelName, lsDocName=lsDocName)
            lBaselineTestReport = self._testBaselines(lX, lY, lLabelName, lsDocName=lsDocName)
            tstRpt.attach(lBaselineTestReport)
            
            ret = tstRpt

        # do some garbage collection
        del lX, lY
        gc.collect()
        
        return ret


    #TODO Test This ;#Is this deprecated ?
    def testFiles(self, lsFilename, loadFun,bBaseLine=False):
        """
        Test the model using those files. The corresponding graphs are loaded using the loadFun function (which must return a singleton list).
        It reports results on stderr

        if some baseline model(s) were set, they are also tested

        Return a Report object
        """
        lY, lY_pred = [], []
        lLabelName = None
        traceln("- predicting on test set")
        chronoOn("testFiles")

        for sFilename in lsFilename:
            
            lg = loadFun(sFilename)  # returns a singleton list
            for g in lg:
                if g.bConjugate: g.computeEdgeLabels()
                [X], [Y] = self.get_lX_lY([g])
                if len(Y) <= 1:
                    traceln("\tY.shape = %s --> skipping this file" % Y.shape)
                else:
                    gcn_graph_test = self.convert_lX_lY_to_GCNDataset([X], [Y], training=False, test=True)
                    if lLabelName == None:
                        lLabelName = g.getEdgeLabelNameList() if g.bConjugate else g.getLabelNameList()
                        traceln("\t #nodes=%d  #edges=%d " % Graph.getNodeEdgeTotalNumber([g]))
                        tNF_EF = (X[0].shape[1], X[2].shape[1])
                        traceln("node-dim,edge-dim", tNF_EF)
    
                    [Y_pred] = self.gcn_model.predict_lG(self.tf_session, gcn_graph_test, verbose=False)
                    lY_pred.append(Y_pred)
                    
                    lY.append(Y)
                del g  # this can be very large
                gc.collect()

        traceln("[%.1fs] done\n" % chronoOff("testFiles"))

        tstRpt = TestReport(self.sName, lY_pred, lY, lLabelName, lsDocName=lsFilename)


        # ??? why commented out?
        #TODO
        # if bBaseLine:
            # lBaselineTestReport = self._testBaselinesEco(lX, lY, lLabelName, lsDocName=lsFilename)
            # tstRpt.attach(lBaselineTestReport)

        del lY, lY_pred
        gc.collect()
        
        return tstRpt

    def restore(self):
        traceln(" start tf session; loading checkpoint")
        # SC                session=tf.Session()

        session=tf.Session(graph=self.tf_graph)
        session.run(self.gcn_model.init)
        self.gcn_model.restore_model(session, self.getModelFilename())
        traceln(" ... done loaded ",self.sName)
        self.tf_session = session
        return session


    def predict(self, g, bProba=False):
        """
        predict the class of each node of the graph
        return a numpy array, which is a 1-dim array of size the number of nodes of the graph.
        """
        [X] = self.get_lX([g])
        if X[1].shape[0] == 0: raise GraphModelNoEdgeException  # no edge in this graph!

        gcn_graph_test = self.convert_X_to_GCNDataset(X)
        #lY_pred = self.gcn_model.predict_lG(self.tf_session, gcn_graph_test, verbose=False)
        [Y_pred] = self.gcn_model.predict_prob_lG(self.tf_session, [gcn_graph_test], verbose=True)     
        
        # SC        lY_pred = self.gcn_model.predict_lG(session, gcn_graph_test, verbose=False)
        # return lY_pred[0]
        if bProba:
            return Y_pred
        else:
            return np.argmax(Y_pred, 1)


    def getModelInfo(self):
        """
        Get some basic model info
        Return a textual report
        """
        return "ECN_Model"




class DU_Model_GAT(DU_Model_ECN):
    sSurname = "gat"

    def __init__(self, sName, sModelDir):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        """
        #super(ChildB, self).__init__()
        super(DU_Model_GAT,self).__init__(sName,sModelDir)


    def _init_model(self):
        '''
        Create the tensorflow graph.
        This function assume that self.model_config contains all the appropriate variables
        to set the model
        This function is called in the train operation and in the load function for testing  on new documents
        :return:
        '''
        self.gcn_model = gcn_models.GraphAttNet(self.model_config['node_dim'], self.model_config['nb_class'],
                                                num_layers=self.model_config['num_layers'],
                                                learning_rate=self.model_config['lr'],
                                                node_indim=self.model_config['node_indim'],
                                                nb_attention=self.model_config['nb_attention'],
                                                )
        self.gcn_model.set_learning_options(self.model_config)
        self.gcn_model.create_model()

    def getModelInfo(self):
        """
        Get some basic model info
        Return a textual report
        """
        return "GAT_Model"


class DU_Ensemble_ECN(DU_Model_ECN):
    sSurname = "ecn_ensemble"
    
    def __init__(self, sName, sModelDir):
        super(DU_Ensemble_ECN,self).__init__(sName, sModelDir)
        self.tf_graphs=[]
        self.models=[]

    def _init_model(self):
        '''
        Create the tensorflow graph.
        This function assume that self.model_config contains all the appropriate variables
        to set the model
        This function is called in the train operation and in the load function for testing  on new documents
        :return:
        '''
        traceln(' ---- Ensemble config')
        assert self.model_config['ratio_train_val'], "The ensemble as a whole must have one ratio 'ratio_train_val'"       
        for iMdl, model_config in enumerate(self.model_config['ecn_ensemble']):
            if model_config['type']=='ecn':
                sName = self.getSubModelName(iMdl, model_config['name'])
                
                traceln(' ---- Configuration of ', sName)
                du_model = DU_Model_ECN(sName, self.sDir)
                
                du_model.configureLearner(**model_config)
                
                du_model.model_config['name']            = sName
                try:
                    traceln(" -- Max iteration forced to %d"%du_model.model_config['nb_iter'])
                    du_model.model_config['nb_iter'] = self.model_config['nb_iter']
                except KeyError: pass
                du_model.model_config['ratio_train_val'] = self.model_config['ratio_train_val']
                du_model.model_config['node_dim']        = self.model_config['node_dim']
                du_model.model_config['edge_dim']        = self.model_config['edge_dim']
                du_model.model_config['nb_class']        = self.model_config['nb_class']
                
                du_model._init_model()

                self.models.append(du_model)
            else:
                raise Exception('Invalid ECN Model')
        traceln('Ensemble config done: %d models initialized' % len(self.models))
            
    def getSubModelName(self, iMdl, sSubModelName):
        """
        Make a name that is globally unique and art of the family of names for this ECN Ensemble
        """
        return self.sName + "._.%d_" % iMdl + sSubModelName
    
    def train(self, lGraph, lGraph_vld, bWarmStart=True, expiration_timestamp=None, verbose=0):
        
        gcn_graph_train, gcn_graph_val = self._prepare_for_train(lGraph, lGraph_vld)
        
        self._init_model()

        #Save the model config in order to restore the model later
        with open(self.getModelConfigFilename(), 'wb') as fd_mc:
            pickle.dump(self.model_config, fd_mc)
        
        traceln(' ---- Ensemble ECN Training')
        for du_model in self.models:
            traceln(" ---- training %s" % du_model.model_config['name'])
            du_model._tNF_EF = self._tNF_EF  # not sure it is required
            du_model._train(gcn_graph_train, gcn_graph_val)
#         #TODO assert label binarizer are the same

    def load(self, expiration_timestamp=None):
        """
        Load myself from disk
        If an expiration timestamp is given, the model stored on disk must be fresher than timestamp
        return self or raise a ModelException
        """
        # by default, load the baseline models
        sBaselineFile = self.getBaselineFilename()
        try:
            self._lMdlBaseline = self._loadIfFresh(sBaselineFile, expiration_timestamp, self.gzip_cPickle_load)
        except GraphModelException:
            traceln('no baseline model found : %s' % (sBaselineFile))
        self.loadTransformers(expiration_timestamp)

        traceln('Loading Ensemble of Models')
        fd_mc = open(self.getModelConfigFilename(), 'rb')
        self.model_config=pickle.load(fd_mc)
        fd_mc.close()
        
        fd_lb = open(self.getlabelBinarizerFilename(), 'rb')
        self.labelBinarizer = pickle.load(fd_lb)
        fd_lb.close()

        #Should load all the submodels with their config and picklers, load transformers
        #This recreate all the DU_model
        #Still unclear if the load should load all the submodels
        #In principle yes
        self._init_model()
        
        for iMdl, du_model in enumerate(self.models):
            #Load each model, init and restore the checkpoint
            # Create also a corresponding tf.Session
            # du_model.load()  the load method does too many inapropriate things
            du_model.setTranformers(self.getTransformers())
            du_model.labelBinarizer = self.labelBinarizer

            fd_mc = open(du_model.getModelConfigFilename(), 'rb')
            du_model.model_config=pickle.load(fd_mc)
            fd_mc.close()
            
            du_model._init_TF_graph()
            
        return self


    def getModelInfo(self):
        """
        Get some basic model info
        Return a textual report
        """
        return "Ensemble_ECN_Model"



    def test(self, lGraph,lsDocName=None,predict_proba=False):
        """
        Test the model using those graphs and report results on stderr

        if some baseline model(s) were set, they are also tested

        Return a Report object
        """
        #Assume the model was created or loaded

        assert lGraph
        
        gcn_graph_test, lX, lY = self._prepare_for_test(lGraph)

        lLabelName = lGraph[0].getLabelNameList()

        lY_pred_proba=[]
        for du_model in self.models:
            model_pred = du_model.gcn_model.predict_prob_lG(du_model.tf_session
                                                            , gcn_graph_test
                                                            , verbose=False)
            lY_pred_proba.append(model_pred)

        traceln(' -- Number of Models : ',len(lY_pred_proba))
        
        lY_pred,_ = DU_Ensemble_ECN.average_prediction(lY_pred_proba)
        tstRpt = TestReport(self.sName, lY_pred, lY, lLabelName, lsDocName=lsDocName)

        # do some garbage collection
        del lX, lY
        gc.collect()

        return tstRpt

    @staticmethod
    def average_prediction(lY_pred_proba):
        '''
        Average the predictions
        :param lY_pred_proba:
        :return:
        '''
        nb_models = float(len(lY_pred_proba))
        nb_graphs =len(lY_pred_proba[0])


        lY_pred = []
        avg_P=[]
        model_idx = range(len(lY_pred_proba))
        for gi in range(nb_graphs):
            gi_l = [lY_pred_proba[m][gi] for m in model_idx]
            avg_proba = np.sum(gi_l, axis=0) / nb_models
            #traceln(avg_proba)
            avg_P.append(avg_proba)
            lY_pred.append(np.argmax(avg_proba, axis=1))
        return lY_pred, avg_P

    # TODO Test This ;#Is this deprecated ?
    def testFiles(self, lsFilename, loadFun, bBaseLine=False):
        """
        Test the model using those files. The corresponding graphs are loaded using the loadFun function (which must return a singleton list).
        It reports results on stderr

        if some baseline model(s) were set, they are also tested

        Return a Report object
        """
        traceln("- predicting on test set")
        chronoOn("testFiles")
        lX, lY, lY_pred = [], [], []
        lLabelName = None

        for sFilename in lsFilename:
            
            lg = loadFun(sFilename)  # returns a singleton list
            for g in lg:
                if g.bConjugate: g.computeEdgeLabels()
                [X], [Y] = self.get_lX_lY([g])
                
                [gcn_graph_test] = self.convert_lX_lY_to_GCNDataset([X], [Y], training=False, test=True)

                if lLabelName == None:
                    lLabelName = g.getLabelNameList()
                    traceln("\t #nodes=%d  #edges=%d " % Graph.getNodeEdgeTotalNumber([g]))
                    tNF_EF = (X[0].shape[1], X[2].shape[1])
                    traceln("node-dim,edge-dim", tNF_EF)

                m_pred=[]
                for du_model in self.models:
                    [Y_pred] = du_model.gcn_model.predict_prob_lG(du_model.tf_session, [gcn_graph_test], verbose=False)
                    m_pred.append([Y_pred])
                    
                [Y_pred], [_Y_pred_proba] = DU_Ensemble_ECN.average_prediction(m_pred)
                
                #lX.append(X)
                lY.append(Y)
                lY_pred.append(Y_pred)
                del _Y_pred_proba
                g.detachFromDoc()
                del g  # this can be very large
            gc.collect()

        traceln("[%.1fs] done\n" % chronoOff("testFiles"))

        tstRpt = TestReport(self.sName, lY_pred, lY, lLabelName, lsDocName=lsFilename)

        del lX, lY, lY_pred
        gc.collect()

        return tstRpt


    def predict(self, g, bProba=False):
        """
        predict the class of each node of the graph
        return a numpy array, which is a 1-dim array of size the number of nodes of the graph.
        """
        
        [X] = self.get_lX([g])
        if X[1].shape[0] == 0: raise GraphModelNoEdgeException  # no edge in this graph!

        gcn_graph_test = self.convert_X_to_GCNDataset(X)
        
        m_pred=[]
        for du_model in self.models:
            assert not du_model.tf_session is None
            [Y_pred] = du_model.gcn_model.predict_prob_lG(du_model.tf_session, [gcn_graph_test], verbose=False)
            m_pred.append([Y_pred])
            
        [y_pred], [y_pred_proba] = DU_Ensemble_ECN.average_prediction(m_pred)
        
        if bProba:
            return y_pred_proba
        else:
            return y_pred
            

if __name__ == "__main__":
    
    usage = """
Train and test a ECN or EnsembleECN model using data pickled (generated with --lpkl option)
"""
    
    parser = OptionParser(usage=usage, version=0.1)
    
    parser.add_option("--trn"       , dest='lTrn'       ,  action="append"  , type="string"
                      , help="Train or continue previous training session using the given folder of pickled X,Y.")    
    parser.add_option("--vld"       , dest='lVld'       ,  action="append"  , type="string"
                      , help="Use this validation pickle folder while training.")    
    parser.add_option("--tst"       , dest='lTst'       ,  action="append"  , type="string"
                      , help="Test a model using the given pickle folder.")    
    parser.add_option("--max_iter"  , dest='max_iter'   ,  action="store"   , type="int"        # "append" would allow doing a gridsearch on max_iter...
                      , help="Maximum number of iterations allowed")    
    parser.add_option("--seed"      , dest='seed'       ,  action="store"   , type="int"        # "append" would allow doing a gridsearch on max_iter...
                      , help="Randomizer seed")  
    parser.add_option("--loader"    , dest='bPklLoader'        ,  action="store_true"
                      , help="Use a data loader of .pkl files to reduce RAM usage to a minimum")  
    
    parser.add_option("--mlflow", dest='bMLFlow',  action="store_true"
                      , help="Record the output of the experiment in MLFlow")
    parser.add_option("--mlflow_uri", dest='sMLFlowURI',  action="store", type="string"
                      , help="Record the output of the experiment in MLFlow at the given URI")
    parser.add_option("--mlflow_exp", "--mlflow_expe", dest='sMLFlowExp',  action="store", type="string"
                      , help="Record the output of the experiment in MLFlow under the given experience")
    parser.add_option("--mlflow_run", dest='sMLFlowRun',  action="store", type="string"
                      , help="Record the output of the experiment in MLFlow under the given run")

    DU_ECN_Task.DU_ECN_Task         .updateStandardOptionsParser(parser)
    DU_ECN_Task.DU_Ensemble_ECN_Task.updateStandardOptionsParser(parser)
 
    (options, args) = parser.parse_args()
    
    if options.bECN:
        clsModel        = DU_Model_ECN
        dLearnerConfig  = DU_ECN_Task.DU_ECN_Task         .getStandardLearnerConfig(options)
    elif options.bECNEnsemble:
        clsModel        = DU_Ensemble_ECN
        dLearnerConfig  = DU_ECN_Task.DU_Ensemble_ECN_Task.getStandardLearnerConfig(options)
    else:
        traceln("ERROR: Specify the type of model to create")
        exit(2)
    
    
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("ERROR: specify a model folder and a model name!")
        exit(1)

    mdl = clsModel(sModelName, sModelDir)
    
    mdl.configureLearner(**dLearnerConfig)
    
    # DU_Task 172 
    # self.config_extractor_kwargs = dFeatureConfig
    # mdl.saveConfiguration( (self.config_extractor_kwargs, self.config_learner_kwargs) )
 
    
    lsTrnDir = options.lTrn
    lsVldDir = options.lVld
    
    assert lsTrnDir, "ERROR: expected at least one training data folder"
    
    # model characteristics
    bOk = True
    for i, sTrnDir in enumerate(lsTrnDir):
        nbClass, nbNodeFeat, nbEdgeFeat, nbGraph, config_extractor_kwargs = DU_Task.getPickledDef(sTrnDir)
        traceln("TRN %d classes  |  %d node features  |  %d edge features  |  %s" % (nbClass, nbNodeFeat, nbEdgeFeat, sTrnDir))
        if i == 0:
            tDimMdl = (nbClass, nbNodeFeat, nbEdgeFeat)
            mdl.model_config['nb_class'], mdl.model_config['node_dim'], mdl.model_config['edge_dim'] = tDimMdl
        else:
            if (nbClass, nbNodeFeat, nbEdgeFeat) != tDimMdl: 
                traceln("ERROR: Inconsistent dimensions! %s | %s" % ((nbClass, nbNodeFeat, nbEdgeFeat), sTrnDir))
                bOk = False
                
    for i, sVldDir in enumerate(lsVldDir):
        nbClass, nbNodeFeat, nbEdgeFeat, nbGraph, _onfig_extractor_kwargs = DU_Task.getPickledDef(sVldDir)
        traceln("VLD %d classes  |  %d node features  |  %d edge features  |  %s" % (nbClass, nbNodeFeat, nbEdgeFeat, sVldDir))
        if (nbClass, nbNodeFeat, nbEdgeFeat) != tDimMdl: 
            traceln("ERROR: Inconsistent dimensions! %s | %s" % ((nbClass, nbNodeFeat, nbEdgeFeat), sVldDir))
            bOk = False

    
    # ----------        Tracking stuff
    if options.sMLFlowURI:
        if options.sMLFlowURI == "-" or options.sMLFlowURI.startswith("file"):
            # tracking in local files
            Tracking.set_tracking()
        else:
            Tracking.set_tracking_uri(options.sMLFlowURI)
    elif options.bMLFlow or options.sMLFlowExp:
        Tracking.set_tracking_uri()
    else:
        Tracking.set_no_tracking()
        
    # MLFLow Experiment name
    _s = options.sMLFlowExp if options.sMLFlowExp else "DU_Model_ECN"
    Tracking.set_experiment(_s)
    traceln("Tracking experiment = ", _s)
    # MLFLow Run name
    _s = options.sMLFlowRun if options.sMLFlowRun else sModelName
    Tracking.start_run(_s)
    traceln("Tracking run = ", _s)
    if os.environ.get("SLURM_JOB_ID"): Tracking.log_param("SLURM_JOB_ID", os.environ.get("SLURM_JOB_ID"))
    
    Tracking.log_artifact_string("General", json.dumps({
          "main"              : str(os.path.abspath(sys.argv[0]))
        , "main.args"         : str(sys.argv[1:])
#         , "main.graph_class"  : self.getGraphClass().__name__
#         , "main.graph_mode"   : self.getGraphClass().getGraphMode()
        , "main.ModelDir"     : os.path.abspath(sModelDir) 
        , "main.ModelName"    : sModelName
        , "main.model_class"  : clsModel.__name__
        , "main.seed"         : options.seed
        }, indent=True))
    Tracking.log_artifact_string("Options", str(options))
    Tracking.log_artifact_string("Options.True", str({k:v for k,v in options.__dict__.items() if bool(v)}))
    Tracking.log_artifact_string("LearningParam" 
                                 , json.dumps(dLearnerConfig, indent=True))
    Tracking.log_artifact_string("Data", json.dumps({'lTrn':options.lTrn, 'lVld':options.lVld}
                                                    , indent=True))

    # Goooo!
    mdl.saveConfiguration( (config_extractor_kwargs, dLearnerConfig) )
    traceln("\t - configuration: ", dLearnerConfig )
   
    chronoOn("DU_Model_ECN_train")
    
    if options.bPklLoader:
        mdl.train_XY_loader(lsTrnDir, lsVldDir)
    else:
        mdl.train_XY(lsTrnDir, lsVldDir)
        
    tTrn = chronoOff("DU_Model_ECN_train")
    traceln(" training done [%.1f s]  (%s)" % (tTrn, pretty_time_delta(tTrn)))
    
        
        
        
        













