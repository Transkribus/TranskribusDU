# -*- coding: utf-8 -*-

"""
    Train, test, predict steps for a CRF model
    - CRF model is EdgeFeatureGraphCRF  (unary and pairwise potentials)
    - Train using SSM
    - Predict using AD3

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
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import sys, os
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

from sklearn.utils.class_weight import compute_class_weight
    
from pystruct.utils import SaveLogger
from pystruct.models import NodeTypeEdgeFeatureGraphCRF

try: #to ease the use without proper Python installation
    from common.trace import traceln
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    from common.trace import traceln

from common.chrono import chronoOn, chronoOff
from crf.Model_SSVM_AD3 import Model_SSVM_AD3

from .TestReport import TestReport, TestReportConfusion

class Model_SSVM_AD3_Multitype(Model_SSVM_AD3):
    #default values for the solver
    C                = .1 
    njobs            = 4
    inference_cache  = 50
    tol              = .1
    save_every       = 50     #save every 50 iterations,for warm start
    max_iter         = 1000
    
    def __init__(self, sName, sModelDir):
        """
        a CRF model, that uses SSVM and AD3, with a name and a folder where it will be stored or retrieved from
        """
        Model_SSVM_AD3.__init__(self, sName, sModelDir)
    
    # ------------------------------------------- TYPED CRF MEMO -------------------------------------------
    #     Parameters
    #     ----------
    #     n_types : number of node types
    #     
    #     l_n_states : list of int, default=None
    #         Number of states per type of variables. 
    # 
    #     l_n_features : list of int, default=None
    #         Number of features per type of node. 
    # 
    #     a_n_edge_features: an array of shape (n_types, n_types) giving the number of features per pair of types
    #     
    #     NOTE: there should always be at least 1 feature for any pairs of types which has some edge in the graph.
    #           To mimic GraphCRF, pass 1 and make a constant feature of 1.0 for all those edges.  
    #     
    #     class_weight : None, or list of array-like (ndim=1)
    #         Class weights. If a list of array-like is passed, the Ith one must have length equal to l_n_states[i]
    #         None means equal class weights (across node types)
    # 
    #     X and Y
    #     -------
    #     Node features are given as a list of n_types arrays of shape (n_type_nodes, n_type_features):
    #         - n_type_nodes is the number of nodes of that type
    #         - n_type_features is the number of features for this type of node
    #     
    #     Edges are given as a list of n_types x n_types arrays of shape (n_type_edges, 2). 
    #         Columns are resp.: node index (in corresponding node type), node index (in corresponding node type)
    #     
    #     Edge features are given as a list of n_types x n_types arrays of shape (n_type_type_edge, n_type_type_edge_features)
    #         - n_type_type_edge is the number of edges of type type_type
    #         - n_type_type_edge_features is the number of features for edge of type type_type
    #         
    #     An instance ``X`` is represented as a tuple ``([node_features, ..], [edges, ..], [edge_features, ..])`` 
    # 
    #     Labels ``Y`` are given as one array of shape (n_nodes)   
    #         Labels are numbered from 0 so that each label across types is encoded by a unique integer.
    #         
    #         Look at flattenY and unflattentY if you want to pass/obtain list of labels per type, with first label of each type being encoded by 0
    # ------------------------------------------------------------------------------------------------------

    def get_lX_lY(self, lGraph):
        """
        Compute node and edge features and return one X matrix for each graph as a list
        return a list of X, a list of Y matrix
        """
        lX, lY = Model_SSVM_AD3.get_lX_lY(self, lGraph)
        
        nbEdge = sum( e.shape[0] for (_lNF, lE, _lEF) in lX for e in lE)
        traceln(" CRF multi-type model: %d edges" % nbEdge)
        
        return lX, lY

    def get_lX(self, lGraph):
        """
        Compute node and edge features and return one X matrix for each graph as a list
        return a list of X, a list of Y matrix
        """
        lX = Model_SSVM_AD3.get_lX(self, lGraph)

        nbEdge = sum( e.shape[0] for (_lNF, lE, _lEF) in lX for e in lE)
        traceln(" CRF multi-type model: %d edges" % nbEdge)
        
        return lX


    # --- UTILITIES -------------------------------------------------------------
    def _computeModelCaracteristics(self, lX):
        """
        We discover dynamically the number of features. Pretty convenient for developer.
        Drawback: if the feature extractor code changes, predicting with a stored model will crash without beforehand catch
        
        return a triplet:
        0 - nbClass
        1 - list of node feature number per type 
        2 - list of edge feature number per type x type
        """
        lNF, lE, lEF = lX[0]    #we assume the lX is properly constructed (and they have all correct shape! even if dim0=0
        self.nbType = len(lNF)
        assert len(lE)  == self.nbType*self.nbType, \
            "SW Error: Badly constructed X: "       \
            "expected %d Edge matrices, got %d" % (self.nbType*self.nbType,
                                                   len(lE))
            
        assert len(lEF) == self.nbType*self.nbType, \
            "SW Error: Badly constructed X: "       \
            "expected %d Edge Feature matrices"     \
            ", got %d" % (self.nbType*self.nbType, len(lEF))
            
        self.lNodeFeatNb = [NF.shape[1] for NF in lNF]
        self.lEdgeFeatNb = [ [lEF[i*self.nbType+j].shape[1] for i in range(self.nbType)] for j in range(self.nbType)]
        return self.nbType, self.lNodeFeatNb, self.lEdgeFeatNb
    
    def _getNbFeatureAsText(self):
        """
        return the number of node features and the number of edge features as a textual message
        """
        return " %d types - #features: (nodes) %s   (edges) %s"%(self.nbType, self.lNodeFeatNb, self.lEdgeFeatNb)

    # --- TRAIN / TEST / PREDICT BASELINE MODELS ------------------------------------------------
    
    def _getXY_forType(self, lX, lY, type_index):
        """
        The node features are grouped by type of nodes.
        Given a type, we need to stack the feature of nodes of that type and extract their labels
        """
        X_flat = np.vstack( node_features[type_index] for (node_features, _, _) in lX )
        
        lY_type = []
        for X, Y in zip(lX, lY):
            node_features = X[0]  #list of node feature matrices, per type
            n_node_before_the_type = sum( node_features[i].shape[0] for i in range(type_index) )      #how many node in previous types?
            n_node_of_type = node_features[type_index].shape[0] 
            Y_type = Y[n_node_before_the_type:n_node_before_the_type+n_node_of_type]
            lY_type.append( Y_type )
        Y_flat = np.hstack(lY_type)
        
        del lY_type
        return X_flat, Y_flat
    
    def _trainBaselines(self, lX, lY):
        """
        Train the baseline models, if any
        """
        if self._lMdlBaseline:
            for itype in range(self.nbType):
                X_flat, Y_flat = self._getXY_forType(lX, lY, itype)
                if False:
                    with open("XY_flat_Type%d.pkl"%(itype), "wb") as fd: 
                        pickle.dump((X_flat, Y_flat), fd)
                for mdlBaseline in self._lMdlBaseline:
                    chronoOn()
                    traceln("\t - training baseline model: %s"%str(mdlBaseline))
                    mdlBaseline[itype].fit(X_flat, Y_flat)
                    traceln("\t [%.1fs] done\n"%chronoOff())
                del X_flat, Y_flat
        return True
                  
    def _testBaselines(self, lX, lY, lLabelName=None, lsDocName=None):
        """
        test the baseline models, 
        return a test report list, one per baseline method
        """
        if lsDocName: assert len(lX) == len(lsDocName), "Internal error"
        
        lTstRpt = []
        if self._lMdlBaseline:
            for itype in range(self.nbType):
                X_flat, Y_flat = self._getXY_forType(lX, lY, itype)
                traceln("\t\t type %d   #nodes=%d  #features=%d"%(itype, X_flat.shape[0], X_flat.shape[1]))
                for mdl in self._lMdlBaseline:   #code in extenso, to call del on the Y_pred_flat array...
                    chronoOn("_testBaselines_T")
                    Y_pred_flat = mdl[itype].predict(X_flat)
                    traceln("\t\t [%.1fs] done\n"%chronoOff("_testBaselines_T"))
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
            chronoOn("_testBaselinesEco_T")
            #using a COnfusionMatrix-based test report object, we can accumulate results
            oTestReportConfu = TestReportConfusion(str(mdl), list(), lLabelName, lsDocName=lsDocName)
            for X,Y in zip(lX, lY):
                for itype in range(self.nbType):
                    X_flat, Y_flat = self._getXY_forType([X], [Y], itype)
                    Y_flat_pred = mdl[itype].predict(X_flat)
                    oTestReportConfu.accumulate( TestReport(str(mdl), Y_flat_pred, Y_flat, lLabelName, lsDocName=lsDocName) )
            traceln("\t\t [%.1fs] done\n"%chronoOff("_testBaselinesEco_T"))
            lTstRpt.append( oTestReportConfu )
        return lTstRpt                                                                              
    
#     def predictBaselines(self, X):
#         """
#         predict with the baseline models, 
#         return a list of 1-dim numpy arrays
#         """
#         return [mdl.predict(X) for mdl in self._lMdlBaseline]

    # --- EDGE BASELINE -------------------------------------------------------------
    #no time to write this code
    def getEdgeModel(self):
        """
        Logisitic regression model for edges
        """
        return None
        
    def _getEdgeXEdgeY(self, lX, lY):
        """
        return X,Y for each edge
        The edge label is in [0, ntype^2-1]
        """
        return None
    
    def _trainEdgeBaseline(self, lX, lY):
        """
        Here we train a logistic regression model to predict the pair of labels of each edge.
        This code assume single type
        """

        return True

    def _testEdgeBaselines(self, lX, lY, lLabelName=None, lsDocName=None):
        """
        test the edge baseline model, 
        return a test report list (a singleton for now)
        """
        return []
       
    # --- TRAIN / TEST / PREDICT ------------------------------------------------
    def _getCRFModel(self, clsWeights=None):

        crf = NodeTypeEdgeFeatureGraphCRF(self.nbType,          # How many node types?
                                          self._nbClass,        # How many states per type?
                                          self.lNodeFeatNb,     # How many node features per type?
                                          self.lEdgeFeatNb,     # How many edge features per type x type?
                                          inference_method="ad3",
                                          l_class_weight = clsWeights 
                                          )
        return crf 


    def computeClassWeight_balanced(self, lY):
        """
        We compute a normalized balanced set of weights per type
        
        UNUSED as of March 2018 (showed worse results on ABP table)
        """
        l_class_weights = []
        
        iTypPrev = 0
        Y = np.hstack(lY)
        for ityp in range(self.nbType):
            iTypNext = iTypPrev + self._nbClass[ityp]
            Y_typ = np.extract(np.logical_and(iTypPrev <= Y, Y < iTypNext), Y)
            Y_typ_unique = np.unique(Y_typ)
            class_weights = compute_class_weight("balanced", Y_typ_unique, Y_typ)
            
            l_class_weights.append( class_weights / np.linalg.norm(class_weights) )
            del Y_typ, Y_typ_unique
            iTypPrev = iTypNext
        del Y
        
        return l_class_weights


    
