# -*- coding: utf-8 -*-

"""
    Standard PageXml features

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

#not robust to empty arrays, so use our robust intermediary class instead
#from sklearn.preprocessing import StandardScaler
from graph.Transformer import EmptySafe_QuantileTransformer as QuantileTransformer
from graph.Transformer import SparseToDense
from graph.Transformer_PageXml import NodeTransformerXYWH, NodeTransformerNeighbors, Node1HotFeatures
from graph.Transformer_PageXml import Edge1HotFeatures, EdgeBooleanFeatures, EdgeNumericalSelector
from graph.Transformer_PageXml import NodeTransformerTextEnclosed, NodeTransformerTextLen
from graph.Transformer_PageXml import EdgeTransformerSourceText, EdgeTransformerTargetText
from graph.PageNumberSimpleSequenciality import PageNumberSimpleSequenciality
from graph.FeatureDefinition import FeatureDefinition

from PageXmlSeparatorRegion import Separator_boolean, Separator_num


class FeatureDefinition_PageXml_StandardOnes(FeatureDefinition):

    n_QUANTILES = 16
    bSeparator = False
    def __init__(self, n_tfidf_node=None, t_ngrams_node=None, b_tfidf_node_lc=None
                     , n_tfidf_edge=None, t_ngrams_edge=None, b_tfidf_edge_lc=None
                     , bMirrorPage=True, bMultiPage=True): 
        FeatureDefinition.__init__(self)
        
        self.n_tfidf_node, self.t_ngrams_node, self.b_tfidf_node_lc = n_tfidf_node, t_ngrams_node, b_tfidf_node_lc
        self.n_tfidf_edge, self.t_ngrams_edge, self.b_tfidf_edge_lc = n_tfidf_edge, t_ngrams_edge, b_tfidf_edge_lc
        self.bMirrorPage = bMirrorPage
        self.bMultiPage  = bMultiPage
        tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc, max_features=self.n_tfidf_node
                                                                                  , analyzer = 'char', ngram_range=self.t_ngrams_node #(2,6)
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
                                    , 
                                    ("textlen", Pipeline([
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
                                    , ("1hot", Pipeline([
                                                         ('1hot', Node1HotFeatures()) #does the 1-hot encoding directly
                                                         ])
                                       )
                                    #, ("sem", Pipeline([
                                    #                     ('sem', NodeSemanticLabels())  #add semantic labels
                                    #                     ])
                                    #  )  # Added  by Animesh
#                                     , ('ocr' , Pipeline([
#                                                          ('ocr', NodeOCRFeatures())
#                                                          ])
#                                        )
#                                     , ('pnumre' , Pipeline([
#                                                          ('pnumre', NodePNumFeatures())
#                                                          ])
#                                        )                                          
#                                     , ("doc_tfidf", Pipeline([
#                                                          ('zero', Zero2Features()) 
#                                                          #THIS ONE MUST BE LAST, because it include a placeholder column for the doculent-level tfidf
#                                                          ])
#                                        )                                          
                                      ])
    
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
                                                         #v1 ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ('numerical', QuantileTransformer(n_quantiles=self.n_QUANTILES, copy=False))  #use in-place scaling
                                                         ])
                                        )
#                                     , ("sourcetext0", Pipeline([
#                                                        ('selector', EdgeTransformerSourceText(0, bMirrorPage=bMirrorPage, bMultiPage=bMultiPage)),
#                                                        ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
#                                                                                  , analyzer = 'char', ngram_range=self.t_ngrams_edge  #(2,6)
#                                                                                  , dtype=np.float64)),
#                                                        ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
#                                                        ])
#                                        )
#                                     , ("targettext0", Pipeline([
#                                                        ('selector', EdgeTransformerTargetText(0, bMirrorPage=bMirrorPage, bMultiPage=bMultiPage)),
#                                                        ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
#                                                                                  , analyzer = 'char', ngram_range=self.t_ngrams_edge
#                                                                                  #, analyzer = 'word', ngram_range=self.tEDGE_NGRAMS
#                                                                                  , dtype=np.float64)),
#                                                        ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
#                                                        ])
#                                        )
#                                     , ("sourcetext1", Pipeline([
#                                                        ('selector', EdgeTransformerSourceText(1, bMirrorPage=bMirrorPage, bMultiPage=bMultiPage)),
#                                                        ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
#                                                                                  , analyzer = 'char', ngram_range=self.t_ngrams_edge  #(2,6)
#                                                                                  , dtype=np.float64)),
#                                                        ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
#                                                        ])
#                                        )
#                                     , ("targettext1", Pipeline([
#                                                        ('selector', EdgeTransformerTargetText(1, bMirrorPage=bMirrorPage, bMultiPage=bMultiPage)),
#                                                        ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
#                                                                                  , analyzer = 'char', ngram_range=self.t_ngrams_edge
#                                                                                  #, analyzer = 'word', ngram_range=self.tEDGE_NGRAMS
#                                                                                  , dtype=np.float64)),
#                                                        ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
#                                                        ])
#                                        )
                                    ]
        
        
        if self.bSeparator:
            lEdgeFeature = lEdgeFeature + [ 
                  ('sprtr_bool', Separator_boolean())
                , ('sprtr_num' , Separator_num())
                ]
                
        if bMultiPage:
            lEdgeFeature.extend([("sourcetext2", Pipeline([
                                                       ('selector', EdgeTransformerSourceText(2, bMirrorPage=bMirrorPage, bMultiPage=bMultiPage)),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.t_ngrams_edge  #(2,6)
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                                , ("targettext2", Pipeline([
                                                       ('selector', EdgeTransformerTargetText(2, bMirrorPage=bMirrorPage, bMultiPage=bMultiPage)),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.t_ngrams_edge
                                                                                 #, analyzer = 'word', ngram_range=self.tEDGE_NGRAMS
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )                        
                        ])
                        
        edge_transformer = FeatureUnion( lEdgeFeature )
          
        #return _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        self._node_transformer = node_transformer
        self._edge_transformer = edge_transformer
        self.tfidfNodeTextVectorizer = tdifNodeTextVectorizer
        
    def cleanTransformers(self):
        """
        the TFIDF transformers are keeping the stop words => huge pickled file!!!
        
        Here the fix is a bit rough. There are better ways....
        JL
        """
        self._node_transformer.transformer_list[0][1].steps[1][1].stop_words_ = None   #is 1st in the union...
        
        if self.bMirrorPage:
            imax = 9
        else:
            imax = 7
#         for i in range(3, imax):
#             self._edge_transformer.transformer_list[i][1].steps[1][1].stop_words_ = None   #are 3rd and 4th in the union....
        return self._node_transformer, self._edge_transformer        

    
class FeatureDefinition_PageXml_StandardOnes_SEP(FeatureDefinition_PageXml_StandardOnes):
    bSeparator = True

