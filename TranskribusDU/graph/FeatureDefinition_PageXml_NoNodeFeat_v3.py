# -*- coding: utf-8 -*-

"""
    Node feature = 1.0  (no node features!)
    Standard PageXml features for edges, but (v3) using a QuantileTransformer for numerical features instead of a StandardScaler

    Copyright Xerox(C) 2017 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




from sklearn.pipeline import Pipeline, FeatureUnion

#not robust to empty arrays, so use our robust intermediary class instead
#from sklearn.preprocessing import StandardScaler
from .Transformer import EmptySafe_QuantileTransformer as QuantileTransformer
from .Transformer_PageXml import Node1ConstantFeature
from .Transformer_PageXml import NodeTransformerXYWH_v2, NodeTransformerNeighbors, Node1HotFeatures
from .Transformer_PageXml import Edge1HotFeatures, EdgeBooleanFeatures_v2, EdgeNumericalSelector, EdgeTypeFeature_HV
from .PageNumberSimpleSequenciality import PageNumberSimpleSequenciality
from .FeatureDefinition import FeatureDefinition


class FeatureDefinition_PageXml_NoNodeFeat_v3(FeatureDefinition):
    
    n_QUANTILES = 16
    
    def __init__(self): 
        FeatureDefinition.__init__(self)
                
        node_transformer = Node1ConstantFeature()  # feature vector per node = [1.0]
    
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
        

class FeatureDefinition_PageXml_StandardOnes_noText_noEdgeFeat_v3(FeatureDefinition):
    
    n_QUANTILES = 16
    
    def __init__(self): 
        FeatureDefinition.__init__(self)
        
        node_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                    ("xywh", Pipeline([
                                                         ('selector', NodeTransformerXYWH_v2()),
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
                                    , ("1hot", Pipeline([
                                                         ('1hot', Node1HotFeatures())  #does the 1-hot encoding directly
                                                         ])
                                       )
                                      ])
    
        edge_transformer = Node1ConstantFeature()  # feature vector per node = [1.0]
          
        #return _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        self._node_transformer = node_transformer
        self._edge_transformer = edge_transformer
        self.tfidfNodeTextVectorizer = None #tdifNodeTextVectorizer


class FeatureDefinition_PageXml_StandardOnes_noText_noEdge2Feat_v3(FeatureDefinition):
    
    n_QUANTILES = 16
    
    def __init__(self): 
        FeatureDefinition.__init__(self)
                
        node_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                    ("xywh", Pipeline([
                                                         ('selector', NodeTransformerXYWH_v2()),
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
                                    , ("1hot", Pipeline([
                                                         ('1hot', Node1HotFeatures())  #does the 1-hot encoding directly
                                                         ])
                                       )
                                      ])
    
        edge_transformer = EdgeTypeFeature_HV()  # feature vector per node = [1.0]
          
        #return _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        self._node_transformer = node_transformer
        self._edge_transformer = edge_transformer
        self.tfidfNodeTextVectorizer = None #tdifNodeTextVectorizer
