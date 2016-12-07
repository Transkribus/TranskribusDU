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
import cPickle, gzip

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

from common.trace import trace, traceln
from common.chrono import chronoOn, chronoOff



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

        self.node_transformer   = None
        self.edge_transformer   = None
        
        self.lsClassName        = None  #for test reports
                
    # --- Utilities -------------------------------------------------------------
    def getModelFilename(self):
        return os.path.join(self.sDir, self.sName+"_model.pkl")
    def getTransformerFilename(self):
        return os.path.join(self.sDir, self.sName+"_transf.pkl")
    
    def _loadIfFresh(self, sFilename, expiration_timestamp, loadFun):
        """
        Look for the given file
        If it is fresher than given timestamp, attempt to load it using the loading function, and return the data
        Raise an exception otherwise
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
                raise Exception("File %s found but too old."%sFilename)
        else:
            traceln("\t\t no such file : %s"%sFilename)
            raise Exception("File %s not found."%sFilename)
        return dat
    
    def gzip_cPickle_dump(cls, sFilename, dat):
        with gzip.open(sFilename, "wb") as zfd:
                cPickle.dump( dat, zfd, protocol=2)
    gzip_cPickle_dump = classmethod(gzip_cPickle_dump)

    def gzip_cPickle_load(cls, sFilename, dat):
        with gzip.open(sFilename, "rb") as zfd:
                return cPickle.load(zfd)        
    gzip_cPickle_load = classmethod(gzip_cPickle_load)
    
    # --- TRANSFORMER FITTING ---------------------------------------------------
    def loadFittedTransformers(self, expiration_timestamp=0):
        """
        Look on disk for some already fitted transformers, and load them 
        If a timestamp is given, ignore any disk data older than it and raises an exception
        Return True
        Raise an exception if nothing good can be found on disk
        """
        sTransfFile = self.getTransformerFilename()

        dat =  self._loadIfFresh(sTransfFile, expiration_timestamp, self.gzip_cPickle_load)
        self.node_transformer, self.edge_transformer = dat        
        return True
        
    def setTranformers(self, (node_transformer, edge_transformer)):
        """
        Set the type of transformers 
        return True 
        """
        self.node_transformer, self.edge_transformer = node_transformer, edge_transformer        
        return True

    def getTransformers(self):
        """
        return the node and edge transformers.
        This method is useful to clean them before saving them on disk
        """
        return self.node_transformer, self.edge_transformer

    def fitTranformers(self, lGraph):
        """
        Fit the transformers using the graphs
        return True 
        """
        lAllNode = [nd for g in lGraph for nd in g.lNode]
        self.node_transformer.fit(lAllNode)
        del lAllNode #trying to free the memory!
        
        lAllEdge = [edge for g in lGraph for edge in g.lEdge]
        self.edge_transformer.fit(lAllEdge)
        del lAllEdge
        
        return True

    def saveTransformers(self):
        """
        Save the transformer on disk
        return the filename
        """
        sTransfFile = self.getTransformerFilename()
        self.gzip_cPickle_dump(sTransfFile, (self.node_transformer, self.edge_transformer))
        return sTransfFile
        
    def transformGraphs(self, lGraph, bLabelled=False):
        """
        Compute node and edge features and return one X matrix for each graph as a list
        If bLabelled==True, return the Y matrix for each as a list
        return either:
         - a list of X and a list of Y
         - a list of X
        """
        lX = [g.buildNodeEdgeMatrices(self.node_transformer, self.edge_transformer) for g in lGraph]
        if bLabelled:
            lY = [g.buildLabelMatrix() for g in lGraph]
            return lX, lY
        else:
            return lX
                
    # --- TRAIN / TEST / PREDICT ------------------------------------------------
    def train(self, lGraph, bWarmStart=True, expiration_timestamp=None):
        """
        Return a model trained using the given labelled graphs.
        If bWarmStart==True, The model is loaded from the disk, if any, and if fresher than given timestamp, and training restarts
        """
        raise Exception("Method must be overridden")

    def computeClassWeight(cls, lY):
        Y = np.hstack(lY)
        Y_unique = np.unique(Y)
        class_weights = compute_class_weight("balanced", Y_unique, Y)
        del Y, Y_unique
        return class_weights
    computeClassWeight = classmethod(computeClassWeight)

    def setClassNames(self, lsClassName):
        """
        Test reports are easier to read with good class names instead of integers
        """
        self.lsClassName = lsClassName
        return lsClassName
    
    def test_report(self, Y, Y_pred):
        """
        compute the confusion matrix and classification report.
        Print them on stderr and return them
        """
        s1 = confusion_matrix(Y, Y_pred)
        traceln(s1)
        
        if self.lsClassName:
            s2 = classification_report(Y, Y_pred, target_names=self.lsClassName)
        else:
            s2 = classification_report(Y, Y_pred)
        traceln(s2)
        
        s3 = "(unweighted) Accuracy score = %.2f"%accuracy_score(Y, Y_pred) 
        traceln(s3)
        
        return "\n\n".join([s1,s2,s3])
        

# --- AUTO-TESTS ------------------------------------------------------------------
def test_computeClassWeight():
    a = np.array([1,1,2], dtype=np.int32)
    b = np.array([2,1,3], dtype=np.int32)        
    cw = Model.computeClassWeight([a,b])
    ref_cw = 11.0/3.0*np.array([2/11.0, 3/11.0, 6/11.0])
    assert ((cw - ref_cw) <0.1).all()
    
