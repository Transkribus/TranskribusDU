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

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.linear_model import LogisticRegression
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
        
    
    def fit(self, lAllNode, y=None):
        """
        Here we need to train 2 Logit: one to predcit the node class, another to predict the class of the neighborhhod
        """
        assert not y, "Did not expect a 'y' parameter. We get the labels from the node themselves."
        
        self.text_pipeline = Pipeline([  ('selector'       , NodeTransformerTextEnclosed())
                                       , ('tf'             , TfidfVectorizer(lowercase=self.b_node_lc
                                                                        #, max_features=10000
                                                                        , analyzer = 'char'
                                                                        , ngram_range=self.t_ngrams_node)) #(2,6)), #we can use it separately from the pipleline once fitted
                                       , ('word_selector'  , SelectKBest(chi2, k=self.n_feat_node))
                                       ])
        # the y
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
                        
        return self
    
    def transform(self, lNode):
        """
        return the 2 logit scores
        """
        a = np.zeros( ( len(lNode), 2*self.nbClass ), dtype=np.float64) 
        
        x = self.text_pipeline.transform(lNode)

        a[...,0:self.nbClass] = self.mdl_main.predict_proba(x)
        
#         for i, nd in enumerate(lNode):
#             print i, nd, a[i]
        
        return a


#------------------------------------------------------------------------------------------------------
class EdgeTransformerLogit(Transformer):
    """
    we will get a list of edges belonging to N classes.
    we train a logit classifier for those classes, as well as a multilabel classifier for the neighor of those classes

    the built feature vector is 2*N long
    """
    dGridSearch_LR_conf = dGridSearch_CONF
    
    def __init__(self, nbClass=None, n_feat_edge=1000, t_ngrams_edge=(2,4), b_edge_lc=False, n_jobs=1):
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
        self.n_feat_edge, self.t_ngrams_edge, self.b_edge_lc = n_feat_edge, t_ngrams_edge, b_edge_lc
        self.n_jobs = n_jobs
        
        self.text_pipeline = None    # feature extractor
        self.mdl_main      = None    # the main model predicting among the nbClass classes
        self.mdl_neighbor  = None    # the neighborhood model predicting zero to many of the classes
        
    
    def fit(self, lAlledge, y=None):
        """
        Here we need to train 2 Logit: one to predcit the edge class, another to predict the class of the neighborhhod
        """
        assert not y, "Did not expect a 'y' parameter. We get the labels from the edge themselves."
        
        self.text_pipeline = Pipeline([  ('selector'       , EdgeTransformerTextEnclosed())
                                       , ('tf'             , TfidfVectorizer(lowercase=self.b_edge_lc
                                                                        #, max_features=10000
                                                                        , analyzer = 'char'
                                                                        , ngram_range=self.t_ngrams_edge)) #(2,6)), #we can use it separately from the pipleline once fitted
                                       , ('word_selector'  , SelectKBest(chi2, k=self.n_feat_edge))
                                       ])
        # the y
        y = np.array([nd.cls for nd in lAlledge], dtype=np.int)
        assert self.nbClass == len(np.unique(y)), "ERROR: some class is not represented in the training set"
        
        #fitting the textual feature extractor
        self.text_pipeline.fit(lAlledge, y)
        
        #extracting textual features
        x = self.text_pipeline.transform(lAlledge)
        
        #creating and training the main logit model
        lr = LogisticRegression(class_weight='balanced')
        self.mdl_main = GridSearchCV(lr , self.dGridSearch_LR_conf, refit=True, n_jobs=self.n_jobs)        
        self.mdl_main.fit(x, y)
                        
        return self
    
    def transform(self, ledge):
        """
        return the 2 logit scores
        """
        a = np.zeros( ( len(ledge), 2*self.nbClass ), dtype=np.float64) 
        
        x = self.text_pipeline.transform(ledge)

        a[...,0:self.nbClass] = self.mdl_main.predict_proba(x)
        
#         for i, nd in enumerate(ledge):
#             print i, nd, a[i]
        
        return a


