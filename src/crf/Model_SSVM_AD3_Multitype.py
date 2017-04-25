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
import sys, os, types
import gc

from sklearn.model_selection import GridSearchCV  #0.18.1 REQUIRES NUMPY 1.12.1 or more recent
    
from pystruct.utils import SaveLogger
from pystruct.learners import OneSlackSSVM
from pystruct.models import NodeTypeEdgeFeatureGraphCRF

try: #to ease the use without proper Python installation
    from common.trace import traceln
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    from common.trace import traceln

from common.chrono import chronoOn, chronoOff
from crf.Model_SSVM_AD3 import Model_SSVM_AD3

from Graph import Graph
from TestReport import TestReport

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
        assert len(lE)  == self.nbType*self.nbType, "SW Error: Badly constructed X: expected %d Edge matrices"         % self.nbType*self.nbType
        assert len(lEF) == self.nbType*self.nbType, "SW Error: Badly constructed X: expected %d Edge Feature matrices" % self.nbType*self.nbType
        self.lNodeFeatNb = [NF.shape[1] for NF in lNF]
        self.lEdgeFeatNb = [EF.shape[1] for EF in lEF]
        return self.nbType, self.lNodeFeatNb, self.lEdgeFeatNb
    
    def _getNbFeatureAsText(self):
        """
        return the number of node features and the number of edge features as a textual message
        """
        return " %d types - #features: (nodes) %s   (edges) %s"%(self.nbType, self.lNodeFeatNb, self.lEdgeFeatNb)
    
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

    @classmethod
    def computeClassWeight(cls, lY):
        """
        This is tricky. Uniform weight for now.
        """
        return None
    
# --- MAIN: DISPLAY STORED MODEL INFO ------------------------------------------------------------------

if __name__ == "__main__":
    try:
        sModelDir, sModelName = sys.argv[1:3]
    except:
        print "Usage: %s <model-dir> <model-name>"%sys.argv[0]
        print "Display some info regarding the stored model"
        exit(1)
        
    mdl = Model_SSVM_AD3_Multitype(sModelName, sModelDir)
    print "Loading %s"%mdl.getModelFilename()
    if False:
        mdl.load()  #loads all sub-models!!
    else:
        mdl.ssvm = mdl._loadIfFresh(mdl.getModelFilename(), None, lambda x: SaveLogger(x).load())

    print mdl.getModelInfo()
    
    import matplotlib.pyplot as plt
    plt.plot(mdl.ssvm.loss_curve_)
    plt.ylabel("Loss")
    plt.show()

    
    
