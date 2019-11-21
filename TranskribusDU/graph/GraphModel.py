# -*- coding: utf-8 -*-

"""
    Train, test, predict steps for a graph-based model

    Structured machine learning, currently using graph-CRF or Edge Convolution Network

    Copyright NAVER(C) 2016-2019 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import os
import sys
import gzip, json
from io import open
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import scipy.sparse as sp

from common.trace import  traceln
from common.chrono import chronoOn, chronoOff
from common.TestReport import TestReport, TestReportConfusion


class GraphModelException(Exception):
    """
    Exception specific to this class
    """
    pass

class GraphModelNoEdgeException(Exception):
    """
    Exception specific to this class: absence of edge in the graph
    """
    pass


class GraphModel:

    _balancedWeights = False   #  Uniform does same or better, in general

    sSurname = ""  #surname is added to each generated filename, e.g. crf, ecn, ...
    
    def __init__(self, sName, sModelDir):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        """
        self.sName = sName
        
        if os.path.exists(sModelDir):
            assert os.path.isdir(sModelDir), "%s exists and is not a directory"%sModelDir
        else:
            os.mkdir(sModelDir)
        self.sDir = sModelDir

        self._node_transformer   = None
        self._edge_transformer   = None
        self._lMdlBaseline       = []  #contains possibly empty list of models
        self.bTrainEdgeBaseline  = False
        
        self._nbClass = None
        
    def configureLearner(self, **kwargs):
        """
        To configure the learner: pass a dictionary using the ** argument-passing method
        """
        raise Exception("Method must be overridden")

    def setName(self, sName):
        self.sName = sName
        
    # --- Utilities -------------------------------------------------------------
    def getMetadataComment(self):
        """
        Return an informative short string for storing a metadata comment in output XML
        """
        return "%s: %s (%s)" % (self.__class__.__name__, self.sName, os.path.abspath(self.sDir))

        
    def getModelFilename(self):
        return os.path.join(self.sDir, self.sName+'._.'+self.sSurname+".model.pkl")
    def getTransformerFilename(self):
        return os.path.join(self.sDir, self.sName+'._.'+              ".transf.pkl")
    def getConfigurationFilename(self):
        return os.path.join(self.sDir, self.sName+'._.'+self.sSurname+".config.json")
    def getBaselineFilename(self):
        return os.path.join(self.sDir, self.sName+'._.'+self.sSurname+".baselines.pkl")
    def getTrainDataFilename(self, name):
        return os.path.join(self.sDir, self.sName+'._.'+self.sSurname+".tlXlY_%s.pkl"%name)
        
    @classmethod
    def _getParamsFilename(cls, sDir, sName):
        return os.path.join(sDir, sName+"._."+"_params.json")

    def setNbClass(self, lNbClass):
        """
        in multitype case we get a list of class number (one per type)
        """
        self._nbClass = lNbClass

    def _getNbFeatureAsText(self):
        """
        return the number of node features and the number of edge features as a textual message
        """
        return "#features nodes=%d  edges=%d "%self._tNF_EF
        
    # --- Model loading/writing -------------------------------------------------------------
    def load(self, expiration_timestamp=None):
        """
        Load myself from disk
        If an expiration timestamp is given, the model stored on disk must be fresher than timestamp
        return self or raise a GraphModelException
        """
        #by default, load the baseline models
        sBaselineFile = self.getBaselineFilename()
        try:
            self._lMdlBaseline =  self._loadIfFresh(sBaselineFile, expiration_timestamp, self.gzip_cPickle_load)
        except GraphModelException:
            traceln('no baseline model found : %s' %(sBaselineFile)) 
        self.loadTransformers(expiration_timestamp)
            
        return self
            
    def storeBestParams(self, dBestModelParameters):
        """
        Store those best parameters (generally a dictionary) under that name if given otherwise under the model's name
        """
        sFN = self._getParamsFilename(self.sDir, self.sName)
        traceln("-+- Storing best parameters in ", sFN)
        with open(sFN, "w") as fd:
            fd.write(json.dumps(dBestModelParameters, sort_keys=True))
    
    @classmethod
    def loadBestParams(cls, sDir, sName):
        """
        Load from disk the previously stored best parameters under that name or model's name
        """
        sFN = cls._getParamsFilename(sDir, sName)
        traceln("-+- Reading best parameters from ", sFN)
        with open(sFN, "r") as fd:
            dBestModelParameters = json.loads(fd.read())
        return dBestModelParameters
        
    def _loadIfFresh(self, sFilename, expiration_timestamp, loadFun):
        """
        Look for the given file
        If it is fresher than given timestamp, attempt to load it using the loading function, and return the data
        Raise a GraphModelException otherwise
        """
        traceln("\t- loading pre-computed data from: %s"%sFilename)
        dat = None
        if os.path.exists(sFilename):
            traceln("\t\t file found on disk: %s"%sFilename)
            if expiration_timestamp is None or os.path.getmtime(sFilename) > expiration_timestamp:
                #OK, it is fresh
                traceln("\t\t file is fresh")
                dat = loadFun(sFilename)
            else:
                traceln("\t\t file is rotten, ignoring it.")
                raise GraphModelException("File %s found but too old."%sFilename)
        else:
            traceln("\t\t no such file : %s"%sFilename)
            raise GraphModelException("File %s not found."%sFilename)
        return dat
    
    def gzip_cPickle_dump(cls, sFilename, dat):
        with gzip.open(sFilename, "wb") as zfd:
                pickle.dump( dat, zfd, protocol=2)
    gzip_cPickle_dump = classmethod(gzip_cPickle_dump)

    def gzip_cPickle_load(cls, sFilename):
        with gzip.open(sFilename, "rb") as zfd:
                return pickle.load(zfd)        
    gzip_cPickle_load = classmethod(gzip_cPickle_load)
    
    # --- TRANSFORMERS ---------------------------------------------------
    def setTranformers(self, t_node_transformer_edge_transformer):
        """
        Set the type of transformers 
        takes as input a tuple: (node_transformer, edge_transformer)
        return True 
        """
        self._node_transformer, self._edge_transformer = t_node_transformer_edge_transformer        
        return True

    def getTransformers(self):
        """
        return the node and edge transformers.
        This method is useful to clean them before saving them on disk
        """
        return self._node_transformer, self._edge_transformer

    def saveTransformers(self):
        """
        Save the transformer on disk
        return the filename
        """
        sTransfFile = self.getTransformerFilename()
        self.gzip_cPickle_dump(sTransfFile, (self._node_transformer, self._edge_transformer))
        return sTransfFile
        
    def loadTransformers(self, expiration_timestamp=0):
        """
        Look on disk for some already fitted transformers, and load them 
        If a timestamp is given, ignore any disk data older than it and raises an exception
        Return True
        Raise an GraphModelException if nothing good can be found on disk
        """
        sTransfFile = self.getTransformerFilename()

        dat =  self._loadIfFresh(sTransfFile, expiration_timestamp, self.gzip_cPickle_load)
        self._node_transformer, self._edge_transformer = dat        
        return True
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def get_lX_lY(self, lGraph):
        """
        Compute node and edge features and return one X matrix for each graph as a list
        return a list of X, a list of Y matrix
        """
        # unfortunately, zip returns tuples and pystruct requires lists... :-/

        lX = self.get_lX(lGraph)
        lY = self.get_lY(lGraph)
        assert(len(lX)==len(lY))
        return lX,lY

    def get_lX(self, lGraph):
        """
        Compute node and edge features and return one X matrix for each graph as a list
        return a list of X, a list of Y matrix
        """
        return [g.getX(self._node_transformer, self._edge_transformer) for g in lGraph]

    def get_lY(self, lGraph):
        """
        Compute node and edge features and return one X matrix for each graph as a list
        return a list of X, a list of Y matrix
        """
        return [g.getY() for g in lGraph]


    def saveConfiguration(self, config_data):
        """
        Save the configuration on disk
        return the filename
        """
        sConfigFile = self.getConfigurationFilename()
        traceln("\tConfiguration file: ", sConfigFile)
        if sys.version_info >= (3,0): #boring problem...
            with open(sConfigFile, "w") as fd:
                json.dump(config_data, fd, indent=4, sort_keys=True)
        else: 
            with open(sConfigFile, "wb") as fd:
                json.dump(config_data, fd, indent=4, sort_keys=True)
        return sConfigFile
        
    # --- TRAIN / TEST / PREDICT BASELINE MODELS ------------------------------------------------
    def _computeModelCaracteristics(self, lX):
        """
        We discover dynamically the number of features. Pretty convenient for developer.
        Drawback: if the feature extractor code changes, predicting with a stored model will crash without beforehand catch
        """
        self._tNF_EF = (lX[0][0].shape[1], lX[0][2].shape[1]) #number of node features,  number of edge features
        return self._tNF_EF
    
    def setBaselineModelList(self, mdlBaselines):
        """
        set one or a list of sklearn model(s):
        - they MUST be initialized, so that the fit method can be called at train time
        - they MUST accept the sklearn usual predict method
        - they SHOULD support a concise __str__ method
        They will be trained with the node features, from all nodes of all training graphs
        """
        #the baseline model(s) if any
        if type(mdlBaselines) in [list, tuple]:
            self._lMdlBaseline = mdlBaselines
        else:
            self._lMdlBaseline = list(mdlBaselines) if mdlBaselines else []  #singleton or None
        return
    
    def getBaselineModelList(self):
        """
        return the list of baseline models
        """
        return self._lMdlBaseline

    def _get_X_flat(self,lX):
        '''
        Return a matrix view X from a list of graph
        Handle sparse matrix as well
        :param lX: 
        :return: 
        '''

        is_sparse=False
        node_feat_mat_list=[]
        for (node_feature,_,_) in lX:
            if sp.issparse(node_feature):
                is_sparse=True
            node_feat_mat_list.append(node_feature)

        if is_sparse:
            X_flat = sp.vstack(node_feat_mat_list)
        else:
            X_flat = np.vstack(node_feat_mat_list)

        return X_flat


    def _trainBaselines(self, lX, lY):
        """
        Train the baseline models, if any
        """
        if self._lMdlBaseline:
            X_flat =self._get_X_flat(lX)
            Y_flat = np.hstack(lY)
            if False:
                with open("XY_flat.pkl", "wb") as fd: 
                    pickle.dump((X_flat, Y_flat), fd)
            for mdlBaseline in self._lMdlBaseline:
                chronoOn()
                traceln("\t - training baseline model: %s"%str(mdlBaseline))
                mdlBaseline.fit(X_flat, Y_flat)
                traceln("\t [%.1fs] done\n"%chronoOff())
            del X_flat, Y_flat
        
        if self.bTrainEdgeBaseline:
            traceln(' - training edge baseline')
            self._trainEdgeBaseline(lX, lY) #we always train a predefined model on edges
        
        return True

    def _testBaselines(self, lX, lY, lLabelName=None, lsDocName=None):
        """
        test the baseline models, 
        return a test report list, one per baseline method
        """
        if lsDocName: assert len(lX) == len(lsDocName), "Internal error"
        
        lTstRpt = []
        if self._lMdlBaseline:
            X_flat =self._get_X_flat(lX)
            Y_flat = np.hstack(lY)
            for mdl in self._lMdlBaseline:   #code in extenso, to call del on the Y_pred_flat array...
                chronoOn("_testBaselines")
                Y_pred_flat = mdl.predict(X_flat)
                traceln("\t\t [%.1fs] done\n"%chronoOff("_testBaselines"))
                lTstRpt.append( TestReport(str(mdl), Y_pred_flat, Y_flat, lLabelName, lsDocName=lsDocName) )
                
            del X_flat, Y_flat, Y_pred_flat
        return lTstRpt                                                                              
    
    def _testBaselinesEco(self, lX, lY, lLabelName=None, lsDocName=None):
        """
        test the baseline models, WITHOUT MAKING A HUGE X IN MEMORY
        return a test report list, one per baseline method
        """
        if lsDocName: assert len(lX) == len(lsDocName), "Internal error"
        lTstRpt = []
        for mdl in self._lMdlBaseline:   #code in extenso, to call del on the Y_pred_flat array...
            chronoOn()
            #using a COnfusionMatrix-based test report object, we can accumulate results
            oTestReportConfu = TestReportConfusion(str(mdl), list(), lLabelName, lsDocName=lsDocName)
            for X,Y in zip(lX, lY):
                Y_pred = mdl.predict(X) #I suspect a bug here. (JLM June 2017) Because X_flat is probably required.
                oTestReportConfu.accumulate( TestReport(str(mdl), Y_pred, Y, lLabelName, lsDocName=lsDocName) )
            traceln("\t\t [%.1fs] done\n"%chronoOff())
            lTstRpt.append( oTestReportConfu )
        return lTstRpt                                                                              
    
#     def predictBaselines(self, X):
#         """
#         predict with the baseline models, 
#         return a list of 1-dim numpy arrays
#         """
#         return [mdl.predict(X) for mdl in self._lMdlBaseline]

    # --- TRAIN / TEST / PREDICT ------------------------------------------------
    def train(self, lGraph, bWarmStart=True, expiration_timestamp=None,verbose=0):
        """
        Return a model trained using the given labelled graphs.
        The train method is expected to save the model into self.getModelFilename(), at least at end of training
        If bWarmStart==True, The model is loaded from the disk, if any, and if fresher than given timestamp, and training restarts
        
        if some baseline model(s) were set, they are also trained, using the node features
        
        """
        raise Exception("Method must be overridden")

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
        #by default, save the baseline models
        sBaselineFile = self.getBaselineFilename()
        self.gzip_cPickle_dump(sBaselineFile, self.getBaselineModelList())
        return sBaselineFile
    
    def test(self, lGraph,lsDocName=None):
        """
        Test the model using those graphs and report results on stderr
        
        if some baseline model(s) were set, they are also tested
        
        Return a Report object
        """
        raise Exception("Method must be overridden")

    def testFiles(self, lsFilename, loadFun,bBaseLine=False):
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
    
    # ----
    @classmethod
    def setBalancedWeights(cls, bBalanced=True):
        """
        By default, we use a uniform weighting scheme.
        A balanced by class is possible.
        """
        cls._balancedWeights = bBalanced
        return bBalanced
    
    @classmethod
    def computeClassWeight_balanced(cls, lY):
        """
        Returns
        -------
        class_weight_vect : ndarray, shape (n_classes,)
            Array with class_weight_vect[i] the weight for i-th class        
        """
        Y = np.hstack(lY)
        Y_unique = np.unique(Y)
        class_weights = compute_class_weight("balanced", Y_unique, Y)
        
        # make sure it sums to n_classes  (like the 'balanced' one in pystruct)
        class_weights = len(Y_unique) * class_weights / np.sum(class_weights)
        del Y, Y_unique
        
        return class_weights

    @classmethod
    def computeClassWeight_uniform(cls, _):
        #Pystruct does uniform by default
        return None

    def computeClassWeight(self, lY):
        """
        This is tricky. Uniform weight for now.
        In our experience, uniform class weight is same or better.
        In addition, in multi-type, the weighting scheme is hard to design.
        """
        if self._balancedWeights:
            return self.computeClassWeight_balanced(lY)
        else:
            return self.computeClassWeight_uniform(lY) # by default

# --- AUTO-TESTS ------------------------------------------------------------------
def test_computeClassWeight():
    a = np.array([1,1,2], dtype=np.int32)
    b = np.array([2,1,3], dtype=np.int32)        
    mdl = GraphModel("test", ".")   
    mdl.setBalancedWeights()
    cw = mdl.computeClassWeight([a,b])
    ref_cw = 6.0/3.0*np.array([1/3.0, 1/2.0, 1/1.0])
    assert ((cw - ref_cw) <0.001).all()    

