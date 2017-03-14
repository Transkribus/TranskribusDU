# -*- coding: utf-8 -*-

"""
    Node and edge feature transformers to extract features for PageXml based on Logit classifiers
    
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
import sys

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier  #for multilabel classif
#sklearn has changed and sklearn.grid_search.GridSearchCV will disappear in next release or so
#so it is recommended to use instead sklearn.model_selection
#BUT on Linux, unplickling of the model fails
#=> change only on Windows
#JLM 2017-03-10
if sys.platform == "win32":
    from sklearn.model_selection import GridSearchCV
else:
    from sklearn.grid_search import GridSearchCV
    
from Transformer import Transformer
from Transformer_PageXml import  NodeTransformerTextEnclosed

from Edge import HorizontalEdge, VerticalEdge, SamePageEdge, CrossPageEdge

dGridSearch_CONF = {'C':[0.1, 0.5, 1.0, 2.0] }  #Grid search parameters for Logit training
dGridSearch_CONF = {'C':[0.1] }  #Grid search parameters for Logit training

DEBUG=0

#------------------------------------------------------------------------------------------------------
class NodeTransformerLogit(Transformer):
    """
    we will get a list of blocks belonging to N classes.
    we train a logit classifier for those classes, as well as a multilabel classifier for the neighor of those classes

    the built feature vector is 2*N long
    """
    dGridSearch_LR_conf = dGridSearch_CONF
    
    def __init__(self, nbClass=None, n_feat_node=1000, t_ngrams_node=(2,4), b_node_lc=False, n_jobs=1):
        """
        input: 
        - number of classes
        - number of ngram
        - ngram min/max size
        - lowercase or not
        - njobs when fitting the logit using grid search
        if n_feat_node is negative, or 0, or None, we use all possible ngrams
        """
        Transformer.__init__(self)
        
        self.nbClass = nbClass
        self.n_feat_node, self.t_ngrams_node, self.b_node_lc = n_feat_node, t_ngrams_node, b_node_lc
        self.n_jobs = n_jobs
        
        self.text_pipeline = None    # feature extractor
        self.mdl_main      = None    # the main model predicting among the nbClass classes
        self.mdl_neighbor  = None    # the neighborhood model predicting zero to many of the classes
        
    
    def fit(self, X, y=None):
        """
        This tranformer needs the graphs to be fitted properly - see fitByGraph
        """
        return self
    
    def fitByGraph(self, lGraph, lAllNode=None):
        """
        we need to train 2 Logit: one to predict the node class, another to predict the class of the neighborhhod
        """
        self.text_pipeline = Pipeline([  ('selector'       , NodeTransformerTextEnclosed())
                                       , ('tf'             , TfidfVectorizer(lowercase=self.b_node_lc
                                                                        #, max_features=10000
                                                                        , analyzer = 'char'
                                                                        , ngram_range=self.t_ngrams_node)) #(2,6)), #we can use it separately from the pipleline once fitted
                                       , ('word_selector'  , SelectKBest(chi2, k=self.n_feat_node))
                                       ])
        # the y
        if lAllNode==None: lAllNode = [nd for g in lGraph for nd in g.lNode]
        y = np.array([nd.cls for nd in lAllNode], dtype=np.int)
        assert self.nbClass == len(np.unique(y)), "ERROR: some class is not represented in the training set"
        
        #fitting the textual feature extractor
        self.text_pipeline.fit(lAllNode, y)
        
        #extracting textual features
        x = self.text_pipeline.transform(lAllNode)
        
        #creating and training the main logit model
        lr = LogisticRegression(class_weight='balanced')
        self.mdl_main = GridSearchCV(lr , self.dGridSearch_LR_conf, refit=True, n_jobs=self.n_jobs)        
        self.mdl_main.fit(x, y)
        del y
        if DEBUG: print self.mdl_main
        
        #now fit a multiclass multilabel logit to predict if a node is neighbor with at least one node of a certain class, for each class
        #Shape = (nb_tot_nodes x nb_tot_labels)
        y = np.vstack([g.getNeighborClassMask() for g in lGraph])  #we get this from the graph object. 
        assert y.shape[0] == len(lAllNode)
        
        lr = LogisticRegression(class_weight='balanced')
        gslr = GridSearchCV(lr , self.dGridSearch_LR_conf, refit=True, n_jobs=self.n_jobs)        
        self.mdl_neighbor = OneVsRestClassifier(gslr, n_jobs=self.n_jobs)
        self.mdl_neighbor.fit(x, y)
#         y_pred = self.mdl_neighbor.predict(x)
#         import TestReport
#         for i in range(y_pred.shape[1]):
#             o = TestReport.TestReport(i, y_pred[i], y[i])
#             print o
        del x, y
        if DEBUG: print self.mdl_neighbor

        return self
        
    def transform(self, lNode):
        """
        return the 2 logit scores
        """
        a = np.zeros( ( len(lNode), 2*self.nbClass ), dtype=np.float64)     #for each class: is_of_class? is_neighbor_of_class?
        
        x = self.text_pipeline.transform(lNode)

        a[...,0:self.nbClass]                   = self.mdl_main     .predict_proba(x)
        a[...,  self.nbClass:2*self.nbClass]    = self.mdl_neighbor .predict_proba(x)
#         for i, nd in enumerate(lNode):
#             print i, nd, a[i]
        if DEBUG: print a
        return a


#------------------------------------------------------------------------------------------------------
class EdgeTransformerLogit(Transformer):
    """
    we will get a list of edges belonging to N classes.
    we train a logit classifier for those classes, as well as a multilabel classifier for the neighor of those classes

    the built feature vector is 2*N long
    """
    dGridSearch_LR_conf = dGridSearch_CONF
    
    def __init__(self, nbClass, ndTrnsfLogit):
        """
        input: 
        - number of classes
        - number of ngram
        - ngram min/max size
        - lowercase or not
        - njobs when fitting the logit using grid search
        if n_feat_edge is negative, or 0, or None, we use all possible ngrams
        """
        Transformer.__init__(self)
        
        self.nbClass = nbClass
        self.transfNodeLogit = ndTrnsfLogit #fitted node transformer
        
    def transform(self, lEdge):
        """
        return the 2 logit scores
        """
        lEdgeClass = [HorizontalEdge, VerticalEdge, CrossPageEdge]
        nbEdgeClass = len(lEdgeClass)
        
        nbFeatPerNode = 2*self.nbClass              #for each class: is_of_class? is_neighbor_of_class?
        nbFeatPerEdgeClass = 2 * nbFeatPerNode      #2 nodes
        d_iEdgeClass = { cls:i*nbFeatPerEdgeClass for i,cls in enumerate(lEdgeClass) }  #shift by edge class
        
        a = np.zeros( ( len(lEdge), nbEdgeClass * nbFeatPerEdgeClass ), dtype=np.float64) 
        
        #slow but safer to code
        for i, edge in enumerate(lEdge):
            iEdgeClass = d_iEdgeClass[edge.__class__] 
            for nd, iNode in [ (edge.A, 0), (edge.B, nbFeatPerNode)]:
                a[i, iEdgeClass+iNode:iEdgeClass+iNode+nbFeatPerNode] = self.transfNodeLogit.transform([nd])
            
#         for i, edg in enumerate(lEdge):
#             print i, edg, a[i]
        
        return a


