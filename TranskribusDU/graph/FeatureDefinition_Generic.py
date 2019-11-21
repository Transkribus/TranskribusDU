# -*- coding: utf-8 -*-

"""
    Standard PageXml features:
    - not using the page information
    - using a QuantileTransformer for numerical features instead of a StandardScaler

    No link with DOm or JSON => named GENERIC

    Copyright Xerox(C) 2016, 2019 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

#not robust to empty arrays, so use our robust intermediary class instead
#from sklearn.preprocessing import StandardScaler
from .Transformer import SparseToDense
from .FeatureDefinition import FeatureDefinition
from .Transformer import EmptySafe_QuantileTransformer as QuantileTransformer

from .Transformer_Generic import NodeTransformerTextEnclosed
from .Transformer_Generic import NodeTransformerTextLen
from .Transformer_Generic import NodeTransformerXYWH
from .Transformer_Generic import NodeTransformerNeighbors
from .Transformer_Generic import EdgeBooleanAlignmentFeatures
from .Transformer_Generic import EdgeNumericalSelector_noText


class FeatureDefinition_Generic(FeatureDefinition):
    
    n_QUANTILES = 64
    
    def __init__(self
                 , n_tfidf_node=None, t_ngrams_node=None, b_tfidf_node_lc=None
                 #, n_tfidf_edge=None, t_ngrams_edge=None, b_tfidf_edge_lc=None
                 ):
        FeatureDefinition.__init__(self)
        
        self.n_tfidf_node, self.t_ngrams_node, self.b_tfidf_node_lc = n_tfidf_node, t_ngrams_node, b_tfidf_node_lc
        # self.n_tfidf_edge, self.t_ngrams_edge, self.b_tfidf_edge_lc = n_tfidf_edge, t_ngrams_edge, b_tfidf_edge_lc

        tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc
                                                 , max_features=self.n_tfidf_node
                                                 , analyzer = 'char'
                                                 , ngram_range=self.t_ngrams_node #(2,6)
                                                 , dtype=np.float64)
                
        node_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("text", Pipeline([
                                                       ('selector', NodeTransformerTextEnclosed()),
#                                                         ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_node_lc, max_features=self.n_tfidf_node
#                                                                                   , analyzer = 'char', ngram_range=self.tNODE_NGRAMS #(2,6)
#                                                                                   , dtype=np.float64)),
                                                       ('tfidf', tdifNodeTextVectorizer), #we can use it separately from the pipleline once fitted
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                        ) 
                                    , ("textlen", Pipeline([
                                                         ('selector', NodeTransformerTextLen()),
                                                         ('textlen', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                    , ("xywh", Pipeline([
                                                         ('selector', NodeTransformerXYWH()),
                                                         #v1 ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('xywh', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                    , ("neighbors", Pipeline([
                                                         ('selector', NodeTransformerNeighbors()),
                                                         #v1 ('neighbors', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('neighbors', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                      ])
    
        lEdgeFeature = [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("boolean", Pipeline([
                                                         ('boolean', EdgeBooleanAlignmentFeatures())
                                                         ])
                                        )
                                    , ("numerical", Pipeline([
                                                         ('selector', EdgeNumericalSelector_noText()),
                                                         #v1 ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('numerical', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                        )
                        ]
                        
        edge_transformer = FeatureUnion( lEdgeFeature )
          
        #return _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        self._node_transformer = node_transformer
        self._edge_transformer = edge_transformer
        self.tfidfNodeTextVectorizer = None #tdifNodeTextVectorizer
        
    def cleanTransformers(self):
        """
        the TFIDF transformers are keeping the stop words => huge pickled file!!!
        
        Here the fix is a bit rough. There are better ways....
        JL
        """
        self._node_transformer.transformer_list[0][1].steps[1][1].stop_words_ = None   #is 1st in the union...
        
#         if self.bMirrorPage:
#             imax = 9
#         else:
#             imax = 7
#         for i in range(3, imax):
#             self._edge_transformer.transformer_list[i][1].steps[1][1].stop_words_ = None   #are 3rd and 4th in the union....
        return self._node_transformer, self._edge_transformer        


