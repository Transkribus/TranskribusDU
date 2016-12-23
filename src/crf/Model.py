# -*- coding: utf-8 -*-

"""
    Train, test, predict steps for a CRF model

    Copyright Xerox(C) 2016 JL. Meunier

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import os
import cPickle, gzip, json
import types
import gc

import numpy as np

from pystruct.utils import SaveLogger

from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

from pystruct.learners import OneSlackSSVM
from pystruct.models import EdgeFeatureGraphCRF

from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM

from common.trace import  traceln
from common.chrono import chronoOn, chronoOff

from TestReport import TestReport

class ModelException(Exception):
    pass

class Model:
    
    def __init__(self, sName, sModelDir):
        """
        a CRF model, with a name and a folder where it will be stored or retrieved from
        """
        self.sName = sName
        
        if os.path.exists(sModelDir):
            assert os.path.isdir(sModelDir)
        else:
            os.mkdir(sModelDir)
        self.sDir = sModelDir

        self._node_transformer   = None
        self._edge_transformer   = None
        
        self._lMdlBaseline       = None  #contains possibly empty list of models
            
    def configureLearner(self, **kwargs):
        """
        To configure the learner: pass a dictionary using the ** argument-passing method
        """
        raise Exception("Method must be overridden")
        
    # --- Utilities -------------------------------------------------------------
    def getModelFilename(self):
        return os.path.join(self.sDir, self.sName+"_model.pkl")
    def getTransformerFilename(self):
        return os.path.join(self.sDir, self.sName+"_transf.pkl")
    def getConfigurationFilename(self):
        return os.path.join(self.sDir, self.sName+"_config.json")
    
    def load(self):
        """
        Load myself from disk
        return self or raise a ModelException
        """
        raise Exception("Method must be overridden")
        
    def _loadIfFresh(self, sFilename, expiration_timestamp, loadFun):
        """
        Look for the given file
        If it is fresher than given timestamp, attempt to load it using the loading function, and return the data
        Raise a ModelException otherwise
        """
        traceln("\t- loading pre-computed data from: %s"%sFilename)
        dat = None
        if os.path.exists(sFilename):
            traceln("\t\t file found on disk: %s"%sFilename)
            if os.path.getmtime(sFilename) > expiration_timestamp:
                #OK, it is fresh
                traceln("\t\t file is fresh")
                dat = loadFun(sFilename)
            else:
                traceln("\t\t file is rotten, ignoring it.")
                raise ModelException("File %s found but too old."%sFilename)
        else:
            traceln("\t\t no such file : %s"%sFilename)
            raise ModelException("File %s not found."%sFilename)
        return dat
    
    def gzip_cPickle_dump(cls, sFilename, dat):
        with gzip.open(sFilename, "wb") as zfd:
                cPickle.dump( dat, zfd, protocol=2)
    gzip_cPickle_dump = classmethod(gzip_cPickle_dump)

    def gzip_cPickle_load(cls, sFilename):
        with gzip.open(sFilename, "rb") as zfd:
                return cPickle.load(zfd)        
    gzip_cPickle_load = classmethod(gzip_cPickle_load)
    
    # --- TRANSFORMER FITTING ---------------------------------------------------
    def setTranformers(self, (node_transformer, edge_transformer)):
        """
        Set the type of transformers 
        return True 
        """
        self._node_transformer, self._edge_transformer = node_transformer, edge_transformer        
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
        
    def saveConfiguration(self, config_data):
        """
        Save the configuration on disk
        return the filename
        """
        sConfigFile = self.getConfigurationFilename()
        with open(sConfigFile, "wb") as fd:
            json.dump(config_data, fd, indent=4, sort_keys=True)
        return sConfigFile
        
    def loadTransformers(self, expiration_timestamp=0):
        """
        Look on disk for some already fitted transformers, and load them 
        If a timestamp is given, ignore any disk data older than it and raises an exception
        Return True
        Raise an ModelException if nothing good can be found on disk
        """
        sTransfFile = self.getTransformerFilename()

        dat =  self._loadIfFresh(sTransfFile, expiration_timestamp, self.gzip_cPickle_load)
        self._node_transformer, self._edge_transformer = dat        
        return True
        
    def transformGraphs(self, lGraph, bLabelled=False):
        """
        Compute node and edge features and return one X matrix for each graph as a list
        If bLabelled==True, return the Y matrix for each as a list
        return either:
         - a list of X and a list of Y
         - a list of X
        """
        lX = [g.buildNodeEdgeMatrices(self._node_transformer, self._edge_transformer) for g in lGraph]
        if bLabelled:
            lY = [g.buildLabelMatrix() for g in lGraph]
            return lX, lY
        else:
            return lX

    # --- TRAIN / TEST / PREDICT BASELINE MODELS ------------------------------------------------
    
    def setBaselineModels(self, mdlBaselines):
        """
        set one or a list of sklearn model(s):
        - they MUST be initialized, so that the fit method can be called at train time
        - they MUST accept the sklearn usual predict method
        They will be trained with the node features, from all nodes of all training graphs
        """
        #the baseline model(s) if any
        if type(mdlBaselines) in [types.ListType, types.TupleType]:
            self._lMdlBaseline = mdlBaselines
        else:
            self._lMdlBaseline = list(mdlBaselines) if mdlBaselines else []  #singleton or None
        return
    
    def getBaselineModelList(self):
        """
        return the list of baseline models
        """
        return self._lMdlBaseline
    
    def _trainBaselines(self, lX, lY):
        """
        Train the baseline models, if any
        """
        if self._lMdlBaseline:
            X_flat = np.vstack( [node_features for (node_features, _, _) in lX] )
            Y_flat = np.hstack(lY)
            for mdlBaseline in self._lMdlBaseline:
                chronoOn()
                traceln("\t - training baseline model: %s"%str(mdlBaseline))
                mdlBaseline.fit(X_flat, Y_flat)
                traceln("\t [%.1fs] done\n"%chronoOff())
            del X_flat, Y_flat
        return 
                  
    def _testBaselines(self, lX, lY):
        """
        test the baseline models, return a test report list
        """
        lTstRpt = []
        if self._lMdlBaseline:
            X_flat = np.vstack( [node_features for (node_features, _, _) in lX] )
            Y_flat = np.hstack(lY)
            lTstRpt = list()
            for mdl in self._lMdlBaseline:   #code in extenso, to call del on the Y_pred_flat array...
                Y_pred_flat = mdl.predict(X_flat)
                lTstRpt.append( TestReport(str(mdl), Y_pred_flat, Y_flat) )
                del Y_pred_flat
            del X_flat, Y_flat
        return lTstRpt                                                                              
    
    def predictBaselines(self, X):
        """
        predict with the baseline models, return a list of 1-dim numpy arrays
        """
        return [mdl.predict(X) for mdl in self._lMdlBaseline]

    # --- TRAIN / TEST / PREDICT ------------------------------------------------
    def train(self, lGraph, bWarmStart=True, expiration_timestamp=None):
        """
        Return a model trained using the given labelled graphs.
        The train method is expected to save the model into self.getModelFilename(), at least at end of training
        If bWarmStart==True, The model is loaded from the disk, if any, and if fresher than given timestamp, and training restarts
        
        if some baseline model(s) were set, they are also trained, using the node features
        
        """
        raise Exception("Method must be overridden")

    def test(self, lGraph, lsClassName=None, lConstraints=[]):
        """
        Test the model using those graphs and report results on stderr
        lConstraints may contain a list of logical constraints per graph.
        This list has the form: [(logical-operator, indices, state, negated), ...]
        
        if some baseline model(s) were set, they are also tested
        
        Return a Report object
        """
        raise Exception("Method must be overridden")

    def predict(self, graph, constraints=None):
        """
        predict the class of each node of the graph
        constraints may contain a list of logical constraints of the form: [(logical-operator, indices, state, negated), ...]
        return a numpy array, which is a 1-dim array of size the number of nodes of the graph. 
        """
        raise Exception("Method must be overridden")
            
    def computeClassWeight(cls, lY):
        Y = np.hstack(lY)
        Y_unique = np.unique(Y)
        class_weights = compute_class_weight("balanced", Y_unique, Y)
        del Y, Y_unique
        return class_weights
    computeClassWeight = classmethod(computeClassWeight)

    def test_report(self, Y, Y_pred, lsClassName=None):
        """
        compute the confusion matrix and classification report.
        Print them on stderr and return the accuracy global score and the report
        """
        
        #we need to include all clasname that appear in the dataset or in the predicted labels (well... I guess so!)
        if lsClassName:
            setSeenCls = set()
            for _Y in [Y, Y_pred]:
                setSeenCls = setSeenCls.union( np.unique(_Y).tolist() )
            lsSeenClassName = [ cls for (i, cls) in enumerate(lsClassName) if i in setSeenCls]
            
        traceln("Line=True class, column=Prediction")
        a = confusion_matrix(Y, Y_pred)
#not nice because of formatting of numbers possibly different per line
#         if lsClassName:
#             #Let's show the class names
#             s1 = ""
#             for sLabel, aLine in zip(lsSeenClassName, a):
#                 s1 += "%20s %s\n"%(sLabel, aLine)
#         else:
#             s1 = str(a)
        s1 = str(a)
        if lsClassName:
            lsLine = s1.split("\n")    
            assert len(lsLine)==len(lsSeenClassName)
            s1 = "\n".join( ["%20s  %s"%(sLabel, sLine) for (sLabel, sLine) in zip(lsSeenClassName, lsLine)])    
        traceln(s1)

        if lsClassName:
            s2 = classification_report(Y, Y_pred, target_names=lsSeenClassName)
        else:
            s2 = classification_report(Y, Y_pred)
        traceln(s2)
        
        fScore = accuracy_score(Y, Y_pred)
        s3 = "(unweighted) Accuracy score = %.2f"% fScore
        traceln(s3)
        
        return fScore, "\n\n".join([s1,s2,s3])
    test_report = classmethod(test_report)    

# --- AUTO-TESTS ------------------------------------------------------------------
def test_computeClassWeight():
    a = np.array([1,1,2], dtype=np.int32)
    b = np.array([2,1,3], dtype=np.int32)        
    cw = Model.computeClassWeight([a,b])
    ref_cw = 11.0/3.0*np.array([2/11.0, 3/11.0, 6/11.0])
    assert ((cw - ref_cw) <0.1).all()
    
def test_test_report():
    lsClassName = ['OTHER', 'catch-word', 'header', 'heading', 'marginalia', 'page-number']
    Y = np.array([0,  2, 3, 2, 5], dtype=np.int32)
    f, _ = Model.test_report(Y, np.array([0,  2, 3, 2, 5], dtype=np.int32), None)
    assert f == 1.0
    f, _ = Model.test_report(Y, np.array([0,  2, 3, 2, 5], dtype=np.int32), lsClassName)
    assert f == 1.0
    f, _ = Model.test_report(Y, np.array([0,  2, 3, 2, 2], dtype=np.int32), lsClassName)
    assert f == 0.8
    f, _ = Model.test_report(Y, np.array([0,  2, 3, 2, 4], dtype=np.int32), lsClassName)
    assert f == 0.8    