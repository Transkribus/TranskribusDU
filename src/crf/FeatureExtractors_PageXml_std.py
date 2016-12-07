# -*- coding: utf-8 -*-

"""
    Standard PageXml feature extractors
    

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

import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from crf.Transformer import SparseToDense
from crf.Transformer_PageXml import NodeTransformerTextEnclosed, NodeTransformerTextLen, NodeTransformerXYWH, Node1HotFeatures
from crf.Transformer_PageXml import Edge1HotFeatures, EdgeNumericalSelector, EdgeTransformerSourceText, EdgeTransformerTargetText
from crf.PageNumberSimpleSequenciality import PageNumberSimpleSequenciality

from FeatureExtractors import FeatureExtractors

class FeatureExtractors_PageXml_StandardOnes(FeatureExtractors):
    def __init__(self, n_tfidf_node, t_ngrams_node, b_tfidf_node_lc
                     , n_tfidf_edge, t_ngrams_edge, b_tfidf_edge_lc): 
        FeatureExtractors.__init__(self)
        
        self.n_tfidf_node, self.t_ngrams_node, self.b_tfidf_node_lc = n_tfidf_node, t_ngrams_node, b_tfidf_node_lc
        self.n_tfidf_edge, self.t_ngrams_edge, self.b_tfidf_edge_lc = n_tfidf_edge, t_ngrams_edge, b_tfidf_edge_lc

    def getTransformers(self):
        """
        return the node and edge feature extractors, as well as the tfidf extractor
        """
        tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc, max_features=self.n_tfidf_node
                                                                                  , analyzer = 'char', ngram_range=self.t_ngrams_node #(2,6)
                                                                                  , dtype=np.float64)
        
        node_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see clean_transformers method!!!!
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
                                                         ('textlen', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                       )
                                    , ("xywh", Pipeline([
                                                         ('selector', NodeTransformerXYWH()),
                                                         ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
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
#                                     , ("doc_tfidf", Pipeline([
#                                                          ('zero', Zero2Features()) 
#                                                          #THIS ONE MUST BE LAST, because it include a placeholder column for the doculent-level tfidf
#                                                          ])
#                                        )                                          
                                      ])
    
        lEdgeFeature = [  #CAREFUL IF YOU CHANGE THIS - see clean_transformers method!!!!
                                      ("boolean", Pipeline([
#                                                          ('boolean', Edge1HotFeatures(Dodge.DodgePlan.plan_GraphML_Sequence.PageNumberSequenciality()))
                                                         ('boolean', Edge1HotFeatures(PageNumberSimpleSequenciality()))
                                                         ])
                                        )
                                    , ("numerical", Pipeline([
                                                         ('selector', EdgeNumericalSelector()),
                                                         ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                        )
                                    , ("sourcetext", Pipeline([
                                                       ('selector', EdgeTransformerSourceText()),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.t_ngrams_edge  #(2,6)
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                                    , ("targettext", Pipeline([
                                                       ('selector', EdgeTransformerTargetText()),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.t_ngrams_edge
                                                                                 #, analyzer = 'word', ngram_range=self.tEDGE_NGRAMS
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                        ]
                        
        edge_transformer = FeatureUnion( lEdgeFeature )
          
        #return node_transformer, edge_transformer, tdifNodeTextVectorizer
        return node_transformer, edge_transformer

    def clean_transformers(self, (node_transformer, edge_transformer)):
        """
        the TFIDF transformers are keeping the stop words => huge pickled file!!!
        
        Here the fix is a bit rough. There are better ways....
        JL
        """
        node_transformer.transformer_list[0][1].steps[1][1].stop_words_ = None   #is 1st in the union...
        edge_transformer.transformer_list[2][1].steps[1][1].stop_words_ = None   #are 3rd and 4th in the union....
        edge_transformer.transformer_list[3][1].steps[1][1].stop_words_ = None        
        return node_transformer, edge_transformer
        

    
