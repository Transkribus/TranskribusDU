# -*- coding: utf-8 -*-

"""
    Logit-based PageXml features
    
    After discussion with Stéphane Clinchant and Hervé Déjean, we will use the score of several logit multiclass classifiers 
    instead of selecting ngrams.
    The rationale is that logit deals well with large numbers of unigrams.
    
    In more details:
    - given N classes,
    - we extract a large number of ngrams
    - we train a N-multiclass logit classifier
    - we train a N-multiclass, multilabel logit classifier that predicts given a node and for all classes, if this node is neighbor of a node of that class
    - The extractor results i 2 * N scores
    
    Copyright Xerox(C) 2017 JL. Meunier

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

from sklearn.pipeline import Pipeline, FeatureUnion
#not robust to empty arrays, so use our robust intermediary class instead
#from sklearn.preprocessing import StandardScaler
from crf.Transformer import RobustStandardScaler as StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from crf.Transformer import SparseToDense
from crf.Transformer_PageXml import NodeTransformerTextEnclosed, NodeTransformerTextLen, NodeTransformerXYWH, NodeTransformerNeighbors, Node1HotFeatures
from crf.Transformer_PageXml import Edge1HotFeatures, EdgeBooleanFeatures, EdgeNumericalSelector, EdgeTransformerSourceText, EdgeTransformerTargetText
from crf.PageNumberSimpleSequenciality import PageNumberSimpleSequenciality

from FeatureDefinition import FeatureDefinition

from Transformer_Logit import NodeTransformerLogit, EdgeTransformerLogit

class FeatureDefinition_PageXml_LogitExtractor(FeatureDefinition):

    """
    We will fit a logistic classifier
    """    
    def __init__(self, nbClass=None
                     , n_feat_node=None, t_ngrams_node=None, b_node_lc=None
                     , n_feat_edge=None, t_ngrams_edge=None, b_edge_lc=None
                     , n_jobs=1): 
        FeatureDefinition.__init__(self, nbClass)
        assert nbClass, "Error: indicate the numbe of classes"
        self.nbClass = nbClass
        self.n_feat_node, self.t_ngrams_node, self.b_node_lc = n_feat_node, t_ngrams_node, b_node_lc
        self.n_feat_edge, self.t_ngrams_edge, self.b_edge_lc = n_feat_edge, t_ngrams_edge, b_edge_lc
        
#         tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_node_lc, max_features=self.n_feat_node
#                                                                                   , analyzer = 'char', ngram_range=self.t_ngrams_node #(2,6)
#                                                                                   , dtype=np.float64)
        """
        I tried to parallelize this code but I'm getting an error on Windows:
        
  File "c:\Local\meunier\git\TranskribusDU\src\crf\FeatureDefinition_PageXml_logit.py", line 144, in fitTranformers
    self._node_transformer.fit(lAllNode)
  File "C:\Anaconda2\lib\site-packages\sklearn\pipeline.py", line 709, in fit
    for _, trans, _ in self._iter())
  File "C:\Anaconda2\lib\site-packages\sklearn\externals\joblib\parallel.py", line 768, in __call__
    self.retrieve()
  File "C:\Anaconda2\lib\site-packages\sklearn\externals\joblib\parallel.py", line 719, in retrieve
    raise exception
TypeError: can't pickle PyCapsule objects        

(virtual_python_pystruct) (C:\Anaconda2) c:\tmp_READ\tuto>python -c "import sklearn; print sklearn.__version__"
0.18.1
        => I force n_jobs to 1
        
        """
        n_jobs = 1
        
        
        n_jobs_NodeTransformerLogit = max(1, n_jobs/2)  #half of the jobs for the NodeTransformerLogit, the rets for the others
        
        #we keep a ref onto it because its fitting needs not only all the nodes, but also additional info, available on the graph objects
        self._node_transf_logit = NodeTransformerLogit(nbClass, self.n_feat_node, self.t_ngrams_node, self.b_node_lc, n_jobs=n_jobs_NodeTransformerLogit)

        node_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                    ("text", self._node_transf_logit)
                                    , 
                                    ("textlen", Pipeline([
                                                         ('selector', NodeTransformerTextLen()),
                                                         ('textlen', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                       )
                                    , ("xywh", Pipeline([
                                                         ('selector', NodeTransformerXYWH()),
                                                         ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                       )
                                    , ("neighbors", Pipeline([
                                                         ('selector', NodeTransformerNeighbors()),
                                                         ('neighbors', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                       )
                                    , ("1hot", Pipeline([
                                                         ('1hot', Node1HotFeatures())  #does the 1-hot encoding directly
                                                         ])
                                       )
#                                     , ('ocr' , Pipeline([
#                                                          ('ocr', NodeOCRFeatures())
#                                                          ])
#                                        )
#                                     , ('pnumre' , Pipeline([
#                                                          ('pnumre', NodePNumFeatures())
#                                                          ])
#                                        )                                          
#                                     , ("doc", Pipeline([
#                                                          ('zero', Zero2Features()) 
#                                                          #THIS ONE MUST BE LAST, because it include a placeholder column for the doculent-level tfidf
#                                                          ])
#                                        )                                          
                                      ], n_jobs=max(1, n_jobs - n_jobs_NodeTransformerLogit))

        lEdgeFeature = [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("1hot", Pipeline([
                                                         ('1hot', Edge1HotFeatures(PageNumberSimpleSequenciality()))
                                                         ])
                                        )
                                    , ("boolean", Pipeline([
                                                         ('boolean', EdgeBooleanFeatures())
                                                         ])
                                        )
                                    , ("numerical", Pipeline([
                                                         ('selector', EdgeNumericalSelector()),
                                                         ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                        )
                                    , ("nodetext", EdgeTransformerLogit(nbClass, self._node_transf_logit))
                        ]
                        
        edge_transformer = FeatureUnion( lEdgeFeature, n_jobs=n_jobs )
          
        #return _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        self._node_transformer = node_transformer
        self._edge_transformer = edge_transformer
        
#         #dirty trick to enable testing the logit models
#         self._node_transformer._testable_extractor_ = self._node_transf_logit
        
    def fitTranformers(self, lGraph):
        """
        Fit the transformers using the graphs
        return True 
        """
        lAllNode = [nd for g in lGraph for nd in g.lNode]
        self._node_transformer.fit(lAllNode)
        self._node_transf_logit.fitByGraph(lGraph, lAllNode)
        del lAllNode #trying to free the memory!
        
        lAllEdge = [edge for g in lGraph for edge in g.lEdge]
        self._edge_transformer.fit(lAllEdge)
        del lAllEdge
        
        return True
        
    def getTransformers(self):
        """
        return (node transformer, edge transformer)
        """
        return self._node_transformer, self._edge_transformer
    
    def cleanTransformers(self):
        """
        the TFIDF transformers are keeping the stop words => huge pickled file!!!
        
        Here the fix is a bit rough. There are better ways....
        JL
        """
        #TODO
        print "TODO: cleanTransformers"
#         self._node_transformer.transformer_list[0][1].steps[1][1].stop_words_ = None   #is 1st in the union...
#         for i in [2, 3, 4, 5, 6, 7]:
#             self._edge_transformer.transformer_list[i][1].steps[1][1].stop_words_ = None   #are 3rd and 4th in the union....
        return self._node_transformer, self._edge_transformer        

