# -*- coding: utf-8 -*-

"""
    First DU task for StAZH
    
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
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

from pystruct.learners import OneSlackSSVM
from pystruct.models import EdgeFeatureGraphCRF

from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM


from DU_CRF_Task import DU_CRF_Task
from crf.Graph_MultiPageXml_TextRegion import Graph_MultiPageXml_TextRegion
from crf.Label_PageXml import Label_PageXml

from crf.Transformer import SparseToDense
from crf.Transformer_PageXml import *
from crf.PageNumberSimpleSequenciality import PageNumberSimpleSequenciality
 
class My_Label(Label_PageXml):
    #            1            2            3        4                5
    _lsLabel = ['catch-word', 'header', 'heading', 'marginalia', 'page-number']
    
class DU_StAZH_a(DU_CRF_Task, Graph_MultiPageXml_TextRegion, My_Label):

    n_tfidf_node    = 300
    n_tfidf_edge    = 300
    tNODE_NGRAMS    = (2,4)
    tEDGE_NGRAMS    = (2,4)
    
    #NO __init__
    
    def getTransformers(self):
        """
        return the node and edge feature extractors, as well as the tfidf extractor
        """
        tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc, max_features=self.n_tfidf_node
                                                                                  , analyzer = 'char', ngram_range=self.tNODE_NGRAMS #(2,6)
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
                                                                                 , analyzer = 'char', ngram_range=self.tEDGE_NGRAMS  #(2,6)
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                                    , ("targettext", Pipeline([
                                                       ('selector', EdgeTransformerTargetText()),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.tEDGE_NGRAMS
                                                                                 #, analyzer = 'word', ngram_range=self.tEDGE_NGRAMS
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                        ]
                        
        edge_transformer = FeatureUnion( lEdgeFeature )
          
        return node_transformer, edge_transformer, tdifNodeTextVectorizer

