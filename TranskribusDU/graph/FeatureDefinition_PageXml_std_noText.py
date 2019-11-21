# -*- coding: utf-8 -*-

"""
    Standard PageXml features, but using a QuantileTransformer for numerical features instead of a StandardScaler

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




from sklearn.pipeline import Pipeline, FeatureUnion

#not robust to empty arrays, so use our robust intermediary class instead
#from sklearn.preprocessing import StandardScaler
from .Transformer import EmptySafe_QuantileTransformer as QuantileTransformer
from .Transformer import TransformerListByType
from .Transformer_PageXml import NodeTransformerXYWH, NodeTransformerNeighbors, Node1HotFeatures_noText
from .Transformer_PageXml import EdgeBooleanFeatures, EdgeNumericalSelector_noText
from .FeatureDefinition import FeatureDefinition


class FeatureDefinition_PageXml_StandardOnes_noText(FeatureDefinition):
    
    n_QUANTILES = 16
    
    def __init__(self): 
        FeatureDefinition.__init__(self)
        
#         self.n_tfidf_node, self.t_ngrams_node, self.b_tfidf_node_lc = n_tfidf_node, t_ngrams_node, b_tfidf_node_lc
#         self.n_tfidf_edge, self.t_ngrams_edge, self.b_tfidf_edge_lc = n_tfidf_edge, t_ngrams_edge, b_tfidf_edge_lc

#         tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc, max_features=self.n_tfidf_node
#                                                                                   , analyzer = 'char', ngram_range=self.t_ngrams_node #(2,6)
#                                                                                   , dtype=np.float64)
        
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
                                    , ("1hot", Pipeline([
                                                         ('1hot', Node1HotFeatures_noText())  #does the 1-hot encoding directly
                                                         ])
                                       )
                                      ])
    
        lEdgeFeature = [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("boolean", Pipeline([
                                                         ('boolean', EdgeBooleanFeatures())
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


class FeatureDefinition_T_PageXml_StandardOnes_noText_v4(FeatureDefinition):
    """
    Multitype version:
    so the node_transformer actually is a list of node_transformer of length n_class
       the edge_transformer actually is a list of node_transformer of length n_class^2
       
    We also inherit from FeatureDefinition_T !!!
    """ 
    n_QUANTILES = 16
       
    def __init__(self, **kwargs):
        FeatureDefinition.__init__(self, **kwargs)

        nbTypes = self._getTypeNumber(kwargs)
        
        node_transformer = TransformerListByType([ FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
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
                                    , ("1hot", Pipeline([
                                                         ('1hot', Node1HotFeatures_noText())  #does the 1-hot encoding directly
                                                         ])
                                       )
                                      ]) for _i in range(nbTypes) ])
    
        edge_transformer = TransformerListByType([ FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                      ("boolean", Pipeline([
                                                         ('boolean', EdgeBooleanFeatures())
                                                         ])
                                        )
                                    , ("numerical", Pipeline([
                                                         ('selector', EdgeNumericalSelector_noText()),
                                                         #v1 ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('numerical', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                        )
                                          ] ) for _i in range(nbTypes*nbTypes) ])
          
        #return _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        self._node_transformer = node_transformer
        self._edge_transformer = edge_transformer
        
