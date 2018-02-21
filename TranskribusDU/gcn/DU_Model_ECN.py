from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

import tensorflow as tf

import pickle
import os.path
import random

import gcn.gcn_models as gcn_models

from gcn.gcn_datasets import GCNDataset
from crf.Model import Model
from crf.Graph import Graph
from crf.TestReport import TestReport

from common.chrono import chronoOn, chronoOff
try: #to ease the use without proper Python installation
    from common.trace import traceln
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    from common.trace import traceln


from sklearn.preprocessing import LabelBinarizer,Normalizer,LabelEncoder
import numpy as np
import scipy.sparse as sp

class DU_Model_ECN(Model):
    def __init__(self, sName, sModelDir):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        """
        self.sName = sName

        if os.path.exists(sModelDir):
            assert os.path.isdir(sModelDir), "%s exists and is not a directory" % sModelDir
        else:
            os.mkdir(sModelDir)
        self.sDir = sModelDir

        self._node_transformer = None
        self._edge_transformer = None

        self._lMdlBaseline = []  # contains possibly empty list of models
        self.bTrainEdgeBaseline = False

        self._nbClass = None

        self.ecn_model=None

        #We should pickle that
        self.labelBinarizer = LabelBinarizer()


    @staticmethod
    def getBaselineConfig():
        '''
        Return A Baseline Edge Conv Net Configuration with 3 layers and 10 convolutions per layer
        :return:
        '''
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

        '''
               lr=0.001,
               stack_convolutions=True,
               mu=0.0,
               num_layers=3,
               node_indim=-1,
               nconv_edge=10,
               fast_convolve = True,
               dropout_rate_edge = 0.0,
               dropout_rate_edge_feat = 0.0,
               dropout_rate_node = 0.0,
               nb_iter=2000,
               ratio_train_val=0.15,
               activation=tf.nn.tanh,
               ):
               '''

    def getModelFilename(self):
        return os.path.join(self.sDir, self.sName+"_tfmodel.ckpt")

    def getValScoreFilename(self):
        return os.path.join(self.sDir, self.sName + '.validation_scores.pkl')

    def getlabelBinarizerFilename(self):
        return os.path.join(self.sDir, self.sName + 'label_binarizer.pkl')

    def get_lX_lY(self, lGraph):
        """
        Compute node and edge features and return one X matrix for each graph as a list
        return a list of X, a list of Y matrix
        """
        # unfortunately, zip returns tuples and pystruct requires lists... :-/

        lX =self.get_lX(lGraph)
        lY =[g.getY() for g in lGraph]
        assert(len(lX)==len(lY))
        return lX,lY

    def get_lX(self, lGraph):
        """
        Compute node and edge features and return one X matrix for each graph as a list
        return a list of X, a list of Y matrix
        """
        #TODO Load reverse arcs ...
        lX=[]
        for g in lGraph:
            #TODO This could move in the Graph Code ...
            (node_features, edges, edge_features) = g.getX(self._node_transformer, self._edge_transformer)
            g.revertEdges()
            (node_features, edges_reverse, edge_features_reverse) = g.getX(self._node_transformer, self._edge_transformer)
            new_edges=np.vstack([edges, edges_reverse])
            new_edges_feat =np.vstack([edge_features,edge_features_reverse])
            lX.append((node_features,new_edges,new_edges_feat) )
        return lX

    def convert_lX_lY_to_GCNDataset(self,lX, lY,training=False):
        gcn_list = []
        graph_id = 0

        # This has state information here --> move that to DU_Model_ECN ...
        lys = []
        for _, ly in zip(lX, lY):
            lys.extend(list(ly))

        if training:
            self.labelBinarizer.fit(lys)

        for lx, ly  in zip(lX, lY):
            nf = lx[0]
            edge = lx[1]
            ef = lx[2]

            graph = GCNDataset(str(graph_id))
            graph.X = nf
            if training:
                graph.Y = self.labelBinarizer.transform(ly)
            else:
                graph.Y = -np.ones((nb_node, len(self.labelBinarizer.classes_)), dtype='i')
            # We are making the adacency matrix here
            nb_node = nf.shape[0]
            # print(edger)
            A1 = sp.coo_matrix((np.ones(edge.shape[0]), (edge[:, 0], edge[:, 1])), shape=(nb_node, nb_node))
            # A2 = sp.coo_matrix((np.ones(edger.shape[0]), (edger[:, 0], edger[:, 1])), shape=(nb_node, nb_node))
            graph.A = A1  # + A2

            edge_normalizer = Normalizer()
            # Normalize EA

            E0 = np.hstack([edge, ef])  # check order
            # E1 = np.hstack([edger, efr])  # check order

            graph.E = E0
            #graph.compute_NA()
            graph.compute_NodeEdgeMat()

            gcn_list.append(graph)
            graph_id += 1

        return gcn_list

    def train(self, lGraph, bWarmStart=True, expiration_timestamp=None,verbose=0):
        """
        Return a model trained using the given labelled graphs.
        The train method is expected to save the model into self.getModelFilename(), at least at end of training
        If bWarmStart==True, The model is loaded from the disk, if any, and if fresher than given timestamp, and training restarts

        if some baseline model(s) were set, they are also trained, using the node features

        """
        print('ECN Training')
        traceln("\t- computing features on training set")
        traceln("\t\t #nodes=%d  #edges=%d " % Graph.getNodeEdgeTotalNumber(lGraph))
        chronoOn()
        lX, lY = self.get_lX_lY(lGraph)

        self._computeModelCaracteristics(lX)  # we discover here dynamically the number of features of nodes and edges
        # self._tNF_EF contains the number of node features and edge features
        traceln("\t\t %s" % self._getNbFeatureAsText())
        traceln("\t [%.1fs] done\n" % chronoOff())

        traceln("\t- retrieving or creating model...")

        nb_class =self.getNbClass() #Is it better to do Y.shape ?


        self.gcn_model = gcn_models.EdgeConvNet(self._tNF_EF[0], self._tNF_EF[1], nb_class,
                                                num_layers=self.model_config['num_layers'],
                                                learning_rate=self.model_config['lr'],
                                                mu=self.model_config['mu'],
                                                node_indim=self.model_config['node_indim'],
                                                nconv_edge=self.model_config['nconv_edge'],
                                                )

        self.gcn_model.stack_instead_add = self.model_config['stack_convolutions']

        if 'activation' in self.model_config:
            self.gcn_model.activation = self.model_config['activation']

        if 'fast_convolve' in self.model_config:
            self.gcn_model.fast_convolve = self.model_config['fast_convolve']

        if 'dropout_rate_edge' in self.model_config:
            self.gcn_model.dropout_rate_edge = self.model_config['dropout_rate_edge']
            print('Dropout Edge', self.gcn_model.dropout_rate_edge)

        if 'dropout_rate_edge_feat' in self.model_config:
            self.gcn_model.dropout_rate_edge_feat = self.model_config['dropout_rate_edge_feat']
            print('Dropout Edge', self.gcn_model.dropout_rate_edge_feat)

        if 'dropout_rate_node' in self.model_config:
            self.gcn_model.dropout_rate_node = self.model_config['dropout_rate_node']
            print('Dropout Node', self.gcn_model.dropout_rate_node)

        self.gcn_model.create_model()

        # Split Training to get some validation
        # ratio_train_val=0.2

        gcn_graph = self.convert_lX_lY_to_GCNDataset(lX,lY,training=True)

        #Save the label Binarizer for prediction usage
        fd_lb =open(self.getlabelBinarizerFilename(),'wb')
        pickle.dump(self.labelBinarizer,fd_lb)
        fd_lb.close()

        split_idx = int(self.model_config['ratio_train_val'] * len(gcn_graph))
        random.shuffle(gcn_graph)
        gcn_graph_train = []
        gcn_graph_val = []

        gcn_graph_val.extend(gcn_graph[:split_idx])
        gcn_graph_train.extend(gcn_graph[split_idx:])


        with tf.Session() as session:
            session.run([self.gcn_model.init])

            R = self.gcn_model.train_with_validation_set(session, gcn_graph_train, gcn_graph_val, self.model_config['nb_iter'],
                                                    eval_iter=10, patience=1000,
                                                    save_model_path=self.getModelFilename())
            f = open(self.getValScoreFilename(), 'wb')
            pickle.dump(R, f)
            f.close()


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

    def test(self, lGraph):
        """
        Test the model using those graphs and report results on stderr

        if some baseline model(s) were set, they are also tested

        Return a Report object
        """
        raise Exception("Method must be overridden")

    def testFiles(self, lsFilename, loadFun):
        """
        Test the model using those files. The corresponding graphs are loaded using the loadFun function (which must return a singleton list).
        It reports results on stderr

        if some baseline model(s) were set, they are also tested

        Return a Report object
        """
        raise Exception("Method must be overridden")

    def predict(self, graph):
        """
        predict the class of each node of the graph
        return a numpy array, which is a 1-dim array of size the number of nodes of the graph.
        """
        raise Exception("Method must be overridden")

    def getModelInfo(self):
        """
        Get some basic model info
        Return a textual report
        """
        return ""
