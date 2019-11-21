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

from sklearn.pipeline import Pipeline, FeatureUnion

from .FeatureDefinition import FeatureDefinition
from .Transformer import EmptySafe_QuantileTransformer as QuantileTransformer

from .Transformer_Generic import NodeTransformerXYWH
from .Transformer_Generic import NodeTransformerNeighbors
from .Transformer_Generic import EdgeBooleanAlignmentFeatures
from .Transformer_Generic import EdgeNumericalSelector_noText


class FeatureDefinition_Generic_noText(FeatureDefinition):
    
    n_QUANTILES = 16
    
    def __init__(self): 
        FeatureDefinition.__init__(self)
        
        node_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("xywh", Pipeline([
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
        
    
#     def cleanTransformers(self):
#         """
#         the TFIDF transformers are keeping the stop words => huge pickled file!!!
#         
#         Here the fix is a bit rough. There are better ways....
#         JL
#         """
#         self._node_transformer.transformer_list[0][1].steps[1][1].stop_words_ = None   #is 1st in the union...
#         for i in [2, 3, 4, 5, 6, 7]:
#             self._edge_transformer.transformer_list[i][1].steps[1][1].stop_words_ = None   #are 3rd and 4th in the union....
#         return self._node_transformer, self._edge_transformer        


