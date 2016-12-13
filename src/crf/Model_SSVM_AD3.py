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

import numpy as np

# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.metrics import classification_report

from pystruct.utils import SaveLogger
from pystruct.learners import OneSlackSSVM
from pystruct.models import EdgeFeatureGraphCRF

from common.trace import traceln
from common.chrono import chronoOn, chronoOff
from Model import Model

class Model_SSVM_AD3(Model):
    #default values for the solver
    solver_C                = .1 
    solver_njobs            = 4
    solver_inference_cache  = 50
    solver_tol              = .1
    solver_save_every       = 50     #save every 50 iterations,for warm start
    solver_max_iter         = 1000
    
    def __init__(self, sName, sModelDir):
        """
        a CRF model, that uses SSVM and AD3, with a name and a folder where it will be stored or retrieved from
        """
        Model.__init__(self, sName, sModelDir)
        self.ssvm = None
        
    def configureLearner(self, inference_cache=None, C=None, tol=None, njobs=None, save_every=None, max_iter=None):
        if None != inference_cache  : self.solver_inference_cache   = inference_cache
        if None != C                : self.solver_C                 = C
        if None != tol              : self.solver_tol               = tol
        if None != njobs            : self.solver_njobs             = njobs
        if None != save_every       : self.solver_save_every        = save_every
        if None != max_iter         : self.solver_max_iter          = max_iter

    def load(self, expiration_timestamp=None):
        """
        Load myself from disk
        If an expiration timestamp is given, the mdeol stored on disk must be fresher than timestamp
        return self or raise a ModelException
        """
        self.ssvm = self._loadIfFresh(self.getModelFilename(), expiration_timestamp, lambda x: SaveLogger(x).load())
        self.loadTransformers(expiration_timestamp)
        return self
    
    # --- TRAIN / TEST / PREDICT ------------------------------------------------
    def train(self, lGraph, bWarmStart=True, expiration_timestamp=None):
        """
        Train a CRF model using the list of labelled graph as training
        if bWarmStart if True, try to continue from previous training, IF the stored model is older than expiration_timestamp!!
            , otherwise, starts from scratch
        return nothing
        """
        traceln("\t- computing features on training set")
        lX, lY = self.transformGraphs(lGraph, True)
        traceln("\t done")

        traceln("\t- retrieving or creating feature extractors...")
        self.ssvm = None
        sModelFN = self.getModelFilename()
        if bWarmStart:
            try:
                self.ssvm = self._loadIfFresh(sModelFN, expiration_timestamp, lambda x: SaveLogger(x).load())
                traceln("\t- warmstarting!")
            except Exception as e:
                self.ssvm = None
                traceln("\t- Cannot warmstart: %s"%e.message)
            #self.ssvm is either None or containing a nice ssvm model!!
        
        if not self.ssvm:
            traceln("\t- creating a new SSVM-trained CRF model")
            
            traceln("\t\t- computing class weight:")
            clsWeights = self.computeClassWeight(lY)
            traceln("\t\t\t%s"%clsWeights)
            
            crf = EdgeFeatureGraphCRF(inference_method='ad3', class_weight=clsWeights)
    
            self.ssvm = OneSlackSSVM(crf
                                , inference_cache=self.solver_inference_cache, C=self.solver_C, tol=self.solver_tol, n_jobs=self.solver_njobs
                                , logger=SaveLogger(sModelFN, save_every=self.solver_save_every)
                                , max_iter=self.solver_max_iter                                        
                                , show_loss_every=10, verbose=1)
            bWarmStart = False
        chronoOn()
        traceln("\t - training graph-based model")
        traceln("\t\t solver parameters:"
                    , " inference_cache=",self.solver_inference_cache
                    , " C=",self.solver_C, " tol=",self.solver_tol, " n_jobs=",self.solver_njobs)
        traceln("\t  #features nodes=%d  edges=%d "%(lX[0][0].shape[1], lX[0][2].shape[1]))
        self.ssvm.fit(lX, lY, warm_start=bWarmStart)
        traceln("\t [%.1fs] done (graph-based model is trained) \n"%chronoOff())

    def test(self, lGraph, lsClassName=None):
        """
        Test the model using those graphs and report results on stderr
        Return the textual report
        """
        traceln("\t- computing features on test set")
        lX, lY = self.transformGraphs(lGraph, True)
        traceln("\t  #features nodes=%d  edges=%d "%(lX[0][0].shape[1], lX[0][2].shape[1]))
        traceln("\t done")

        traceln("\t- predicting on test set")
        lY_pred = self.ssvm.predict(lX) 
        traceln("\t done")
        Y_flat = np.hstack(lY)
        Y_pred_flat = np.hstack(lY_pred)
        del lX, lY, lY_pred
        return self.test_report(Y_flat, Y_pred_flat, lsClassName)

    def predict(self, graph, constraints=None):
        """
        predict the class of each node of the graph
        return a numpy array, which is a 1-dim array of size the number of nodes of the graph. 
        """
        [X] = self.transformGraphs([graph])
        traceln("\t  #features nodes=%d  edges=%d "%(X[0].shape[1], X[2].shape[1]))
        if constraints:
            [Y] = self.ssvm.predict([X], constraints=constraints)
        else:
            [Y] = self.ssvm.predict([X])
            
        return Y
        
# --- AUTO-TESTS ------------------------------------------------------------------
