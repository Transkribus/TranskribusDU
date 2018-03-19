from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os,sys

import tensorflow as tf

import pickle
import os.path
import random

import gcn.gcn_models as gcn_models

from gcn.gcn_datasets import GCNDataset
from crf.Model import Model, ModelException
from crf.Graph import Graph
from crf.TestReport import TestReport

from common.chrono import chronoOn, chronoOff
try: #to ease the use without proper Python installation
    from common.trace import traceln
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    from common.trace import traceln

import gc
from sklearn.preprocessing import LabelBinarizer,Normalizer,LabelEncoder
import numpy as np
import scipy.sparse as sp
import fnmatch
import shutil

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

        self.gcn_model=None

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
        return os.path.join(self.sDir, self.sName+"_bestmodel.ckpt")

    def getTmpModelFilename(self):
        return os.path.join(self.sDir, self.sName+"_tmpmodel.ckpt")

    def getValScoreFilename(self):
        return os.path.join(self.sDir, self.sName + '.validation_scores.pkl')

    def getlabelBinarizerFilename(self):
        return os.path.join(self.sDir, self.sName + 'label_binarizer.pkl')

    def getModelConfigFilename(self):
        return os.path.join(self.sDir, self.sName + 'model_config.pkl')

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
        lX=[]
        nb_nodes=0
        nb_edges=0
        for g in lGraph:
            #TODO This could move in the Graph Code ...
            (node_features, edges, edge_features) = g.getX(self._node_transformer, self._edge_transformer)
            nb_nodes+=node_features.shape[0]
            g.revertEdges()
            (node_features, edges_reverse, edge_features_reverse) = g.getX(self._node_transformer, self._edge_transformer)
            new_edges=np.vstack([edges, edges_reverse])
            new_edges_feat =np.vstack([edge_features,edge_features_reverse])
            nb_edges += new_edges.shape[0]
            lX.append((node_features,new_edges,new_edges_feat) )


        traceln("\t\t ECN #nodes=%d  #edges=%d " % (nb_nodes,nb_edges))
        return lX

    def convert_lX_lY_to_GCNDataset(self,lX, lY,training=False,test=False,predict=False):
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
        '''
        self.gcn_model.stack_instead_add = self.model_config['stack_convolutions']
        #TODO Clean Constructor
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
        '''
        self.gcn_model.set_learning_options(self.model_config)
        self.gcn_model.create_model()

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

        self.model_config['node_dim']=self._tNF_EF[0]
        self.model_config['edge_dim']=self._tNF_EF[1]
        self.model_config['nb_class']=nb_class

        #This call the ECN internal constructor and defines the tensorflow graph
        self._init_model()

        #This converts the lX,lY in the format necessary for GCN Models
        gcn_graph = self.convert_lX_lY_to_GCNDataset(lX,lY,training=True)

        #Save the label Binarizer for prediction usage
        fd_lb =open(self.getlabelBinarizerFilename(),'wb')
        pickle.dump(self.labelBinarizer,fd_lb)
        fd_lb.close()
        #Save the model config in order to restore the model later
        fd_mc = open(self.getModelConfigFilename(), 'wb')
        pickle.dump(self.model_config, fd_mc)
        fd_mc.close()

        #Get a validation set from the training set
        split_idx = int(self.model_config['ratio_train_val'] * len(gcn_graph))
        random.shuffle(gcn_graph)
        gcn_graph_train = []
        gcn_graph_val = []

        gcn_graph_val.extend(gcn_graph[:split_idx])
        gcn_graph_train.extend(gcn_graph[split_idx:])

        self._cleanTmpCheckpointFiles()

        patience = self.model_config['patience'] if 'patience' in self.model_config else self.model_config['nb_iter']
        with tf.Session() as session:
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


    def _getBestModelVal(self):
        val_pickle = self.getValScoreFilename()
        traceln("\t- reading training info from...",val_pickle)
        f = open(val_pickle, 'rb')
        R = pickle.load(f)
        val = R['val_acc']
        #print('Validation scores', ['%03.2f'% sx for sx in val])
        epoch_index = np.argmax(val)
        traceln('\t Best performance on val set: Epoch:', epoch_index)

        model_path = self.getTmpModelFilename()+"-"+ str(10 * epoch_index)

        dir_path=os.path.dirname(self.getTmpModelFilename())
        fnames =os.listdir(dir_path)

        #Find all the checkpoints files for that epoch
        found_files=fnmatch.filter(fnames, os.path.basename(model_path)+'*')
        #print(found_files)
        #Now copy the files with the final model name
        for m in found_files:
            f_src=os.path.join(dir_path,m)
            f_suffix = m[len(os.path.basename(model_path)):]
            f_dst=self.getModelFilename()+f_suffix
            shutil.copy(f_src,f_dst)
            traceln('Copying  Final Model files ', f_src,' ',f_dst)

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
        except ModelException:
            traceln('no baseline model found : %s' % (sBaselineFile))
        self.loadTransformers(expiration_timestamp)

        fd_mc = open(self.getModelConfigFilename(), 'rb')
        self.model_config=pickle.load(fd_mc)
        fd_mc.close()

        fd_lb = open(self.getlabelBinarizerFilename(), 'rb')
        self.labelBinarizer = pickle.load(fd_lb)
        fd_lb.close()

        self._init_model()
        return self

    def test(self, lGraph,lsDocName=None):
        """
        Test the model using those graphs and report results on stderr

        if some baseline model(s) were set, they are also tested

        Return a Report object
        """
        #Assume the model was created or loaded

        assert lGraph
        lLabelName = lGraph[0].getLabelNameList()
        traceln("\t- computing features on test set")
        traceln("\t\t #nodes=%d  #edges=%d " % Graph.getNodeEdgeTotalNumber(lGraph))
        chronoOn()
        lX, lY = self.get_lX_lY(lGraph)

        traceln("\t [%.1fs] done\n" % chronoOff())

        gcn_graph_test = self.convert_lX_lY_to_GCNDataset(lX, lY, training=False,test=True)

        chronoOn("test2")
        with tf.Session() as session:
            session.run(self.gcn_model.init)
            self.gcn_model.restore_model(session, self.getModelFilename())
            lY_pred = self.gcn_model.predict_lG(session, gcn_graph_test, verbose=False)
            #end_time = time.time()
            #print("--- %s seconds ---" % (end_time - start_time))
            #print('Number of graphs:', len(lY_pred))

            # Convert to list as Python pickle does not  seem like the array while the list can be pickled
            lY_list = []
            for x in lY_pred:
                lY_list.append(list(x))

        traceln(" [%.1fs] done\n" % chronoOff("test2"))
        tstRpt = TestReport(self.sName, lY_list, lY, lLabelName, lsDocName=lsDocName)

        #lBaselineTestReport = self._testBaselines(lX, lY, lLabelName, lsDocName=lsDocName)
        #tstRpt.attach(lBaselineTestReport)


        # do some garbage collection
        del lX, lY
        gc.collect()

        return tstRpt


    #TODO Test This ;#Is this deprecated ?
    def testFiles(self, lsFilename, loadFun,bBaseLine=False):
        """
        Test the model using those files. The corresponding graphs are loaded using the loadFun function (which must return a singleton list).
        It reports results on stderr

        if some baseline model(s) were set, they are also tested

        Return a Report object
        """

        lX, lY, lY_pred = [], [], []
        lLabelName = None
        traceln("- predicting on test set")
        chronoOn("testFiles")


        with tf.Session() as session:
            session.run(self.gcn_model.init)
            self.gcn_model.restore_model(session, self.getModelFilename())


            for sFilename in lsFilename:
                [g] = loadFun(sFilename)  # returns a singleton list
                [X], [Y] = self.get_lX_lY([g])

                gcn_graph_test = self.convert_lX_lY_to_GCNDataset([X], [Y], training=False, test=True)
                if lLabelName == None:
                    lLabelName = g.getLabelNameList()
                    traceln("\t #nodes=%d  #edges=%d " % Graph.getNodeEdgeTotalNumber([g]))
                    tNF_EF = (X[0].shape[1], X[2].shape[1])
                    traceln("node-dim,edge-dim", tNF_EF)
                else:
                    assert lLabelName == g.getLabelNameList(), "Inconsistency among label spaces"
                lY_pred_ = self.gcn_model.predict_lG(session, gcn_graph_test, verbose=False)

                lY_pred.append(lY_pred_[0])
                lX.append(X)
                lY.append(Y)
                g.detachFromDOM()
                del g  # this can be very large
                gc.collect()

        traceln("[%.1fs] done\n" % chronoOff("testFiles"))

        tstRpt = TestReport(self.sName, lY_pred, lY, lLabelName, lsDocName=lsFilename)

        if bBaseLine:
            lBaselineTestReport = self._testBaselinesEco(lX, lY, lLabelName, lsDocName=lsFilename)
            tstRpt.attach(lBaselineTestReport)

        del lX, lY
        gc.collect()
        
        return tstRpt

    def restore(self):
        traceln(" start tf session; loading checkpoint")
        session=tf.Session()
        session.run(self.gcn_model.init)
        self.gcn_model.restore_model(session, self.getModelFilename())
        traceln(" ... done")
        return session

    def predict(self, g,session):
        """
        predict the class of each node of the graph
        return a numpy array, which is a 1-dim array of size the number of nodes of the graph.
        """
        lLabelName = None

        [X], [Y] = self.get_lX_lY([g])
        gcn_graph_test = self.convert_lX_lY_to_GCNDataset([X], [Y], training=False, predict=True)
        if lLabelName is None:
            lLabelName = g.getLabelNameList()
            traceln("\t #nodes=%d  #edges=%d " % Graph.getNodeEdgeTotalNumber([g]))
            tNF_EF = (X[0].shape[1], X[2].shape[1])
            traceln("node-dim,edge-dim:", tNF_EF)
        else:
            assert lLabelName == g.getLabelNameList(), "Inconsistency among label spaces"
        lY_pred = self.gcn_model.predict_lG(session, gcn_graph_test, verbose=False)
        return lY_pred[0]

    def getModelInfo(self):
        """
        Get some basic model info
        Return a textual report
        """
        return "ECN_Model"




class DU_Model_GAT(DU_Model_ECN):
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
