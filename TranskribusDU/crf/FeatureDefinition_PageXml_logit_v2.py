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
import sys

from sklearn.pipeline import Pipeline, FeatureUnion
#not robust to empty arrays, so use our robust intermediary class instead
#from sklearn.preprocessing import StandardScaler
from crf.Transformer import EmptySafe_StandardScaler as StandardScaler
from crf.Transformer_PageXml import NodeTransformerTextLen, NodeTransformerXYWH_v2, NodeTransformerNeighbors, Node1HotFeatures
from crf.Transformer_PageXml import Edge1HotFeatures, EdgeBooleanFeatures_v2, EdgeNumericalSelector
from crf.PageNumberSimpleSequenciality import PageNumberSimpleSequenciality

from FeatureDefinition import FeatureDefinition

from Transformer_Logit import NodeTransformerLogit, EdgeTransformerLogit

class FeatureDefinition_PageXml_LogitExtractorV2(FeatureDefinition):

    """
    We will fit a logistic classifier
    """    
    def __init__(self, nbClass=None
                     , n_feat_node=None, t_ngrams_node=None, b_node_lc=None
                     , n_feat_edge=None, t_ngrams_edge=None, b_edge_lc=None
                     , n_jobs=1): 
        FeatureDefinition.__init__(self)
        assert nbClass, "Error: indicate the number of classes"
        self.nbClass = nbClass
        self.n_feat_node, self.t_ngrams_node, self.b_node_lc = n_feat_node, t_ngrams_node, b_node_lc
        self.n_feat_edge, self.t_ngrams_edge, self.b_edge_lc = n_feat_edge, t_ngrams_edge, b_edge_lc
        
#         tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_node_lc, max_features=self.n_feat_node
#                                                                                   , analyzer = 'char', ngram_range=self.t_ngrams_node #(2,6)
#                                                                                   , dtype=np.float64)
        """
        - loading pre-computed data from: CV_5/model_A_fold_1_transf.pkl
                 no such file : CV_5/model_A_fold_1_transf.pkl
Traceback (most recent call last):
  File "/opt/project/read/jl_git/TranskribusDU/src/tasks/DU_GTBooks_5labels.py", line 216, in <module>
    oReport = doer._nfold_RunFoldFromDisk(options.iFoldRunNum, options.warm)
  File "/opt/project/read/jl_git/TranskribusDU/src/tasks/DU_CRF_Task.py", line 481, in _nfold_RunFoldFromDisk
    oReport = self._nfold_RunFold(iFold, ts_trn, lFilename_trn, train_index, test_index, bWarm=bWarm)
  File "/opt/project/read/jl_git/TranskribusDU/src/tasks/DU_CRF_Task.py", line 565, in _nfold_RunFold
    fe.fitTranformers(lGraph_trn)
  File "/opt/project/read/jl_git/TranskribusDU/src/crf/FeatureDefinition_PageXml_logit_v2.py", line 141, in fitTranformers
    self._node_transformer.fit(lAllNode)
  File "/opt/project/read/VIRTUALENV_PYTHON_FULL_type/lib/python2.7/site-packages/sklearn/pipeline.py", line 712, in fit
    for _, trans, _ in self._iter())
  File "/opt/project/read/VIRTUALENV_PYTHON_FULL_type/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.py", line 768, in __call__
    self.retrieve()
  File "/opt/project/read/VIRTUALENV_PYTHON_FULL_type/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.py", line 719, in retrieve
    raise exception
RuntimeError: maximum recursion depth exceeded
"""
        """
        I guess this is due to the cyclic links to node's neighbours.
        But joblib.Parallel uses cPickle, so we cannot specialize the serialization of the Block objects.
        
        JLM April 2017
        """
        n_jobs_from_graph = 1   #we cannot pickl the list of graph, so n_jobs = 1 for this part!
#         n_jobs_NodeTransformerLogit = max(1, n_jobs/2)  #half of the jobs for the NodeTransformerLogit, the rets for the others
        n_jobs_NodeTransformerLogit = max(1, n_jobs - 1)
        
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
                                                         ('selector', NodeTransformerXYWH_v2()),
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
                                      ], n_jobs=n_jobs_from_graph)

        lEdgeFeature = [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("1hot", Pipeline([
                                                         ('1hot', Edge1HotFeatures(PageNumberSimpleSequenciality()))
                                                         ])
                                        )
                                    , ("boolean", Pipeline([
                                                         ('boolean', EdgeBooleanFeatures_v2())
                                                         ])
                                        )
                                    , ("numerical", Pipeline([
                                                         ('selector', EdgeNumericalSelector()),
                                                         ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                        )
                                    , ("nodetext", EdgeTransformerLogit(nbClass, self._node_transf_logit))
                        ]
                        
        edge_transformer = FeatureUnion( lEdgeFeature, n_jobs=n_jobs_from_graph )
          
        #return _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        self._node_transformer = node_transformer
        self._edge_transformer = edge_transformer

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

    
