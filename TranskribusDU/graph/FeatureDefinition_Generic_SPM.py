# -*- coding: utf-8 -*-

"""
    Standard PageXml features:
    - not using the page information
    - using a QuantileTransformer for numerical features instead of a StandardScaler
    - using Google's SentencePiece for representing textual information
    No link with DOM or JSON => named GENERIC

    Copyright Naver Labs Europe(C) 2019 JL. Meunier
"""


from sklearn.pipeline import Pipeline, FeatureUnion

#not robust to empty arrays, so use our robust intermediary class instead
#from sklearn.preprocessing import StandardScaler
from .Transformer import SparseToDense
from .FeatureDefinition import FeatureDefinition
from .Transformer import EmptySafe_QuantileTransformer as QuantileTransformer

from .Transformer_Generic import NodeTransformerTextSentencePiece
from .Transformer_Generic import NodeTransformerTextLen
from .Transformer_Generic import NodeTransformerXYWH
from .Transformer_Generic import NodeTransformerNeighbors
from .Transformer_Generic import EdgeBooleanAlignmentFeatures
from .Transformer_Generic import EdgeNumericalSelector_noText


class FeatureDefinition_Generic_SPM(FeatureDefinition):
    
    n_QUANTILES = 64
    
    def __init__(self
                 , sSPModel
                 ):
        FeatureDefinition.__init__(self)
        
        node_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("text_spm", NodeTransformerTextSentencePiece(sSPModel)
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
        
