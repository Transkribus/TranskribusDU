# -*- coding: utf-8 -*-

"""
    Standard PageXml features

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
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from crf.Transformer import SparseToDense
from crf.Transformer_PageXml import NodeTransformerTextEnclosed, NodeTransformerTextLen, NodeTransformerXYWH, NodeTransformerNeighbors, Node1HotFeatures
from crf.Transformer_PageXml import Edge1HotFeatures, EdgeBooleanFeatures, EdgeNumericalSelector, EdgeTransformerSourceText, EdgeTransformerTargetText
from crf.PageNumberSimpleSequenciality import PageNumberSimpleSequenciality

from FeatureDefinition import FeatureDefinition

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#Should be able to discriminate between chi2 and mutual info, right ?
#(X, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)[source]¶



class FeatureDefinition_PageXml_FeatSelect(FeatureDefinition):

    def __init__(self, n_tfidf_node=None, t_ngrams_node=None, b_tfidf_node_lc=None
                     , n_tfidf_edge=None, t_ngrams_edge=None, b_tfidf_edge_lc=None,feat_select=None):
        FeatureDefinition.__init__(self)

        self.n_tfidf_node, self.t_ngrams_node, self.b_tfidf_node_lc = n_tfidf_node, t_ngrams_node, b_tfidf_node_lc
        self.n_tfidf_edge, self.t_ngrams_edge, self.b_tfidf_edge_lc = n_tfidf_edge, t_ngrams_edge, b_tfidf_edge_lc

        tdifNodeTextVectorizer = CountVectorizer(lowercase=self.b_tfidf_node_lc, max_features=10000
                                                                                  , analyzer = 'char', ngram_range=self.t_ngrams_node) #(2,6)

        #tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc, max_features=10000
         #                                                                         , analyzer = 'char', ngram_range=self.t_ngrams_node) #(2,6)

        self.feature_selection=False
        feat_selector=None

        if feat_select=='chi2':
            self.feature_selection=True
            feat_selector=SelectKBest(chi2, k=self.n_tfidf_node)


        if feat_selector:
            node_transformer_debug = Pipeline([('selector', NodeTransformerTextEnclosed()),
                                               ('tf', tdifNodeTextVectorizer), #we can use it separately from the pipleline once fitted
                                                ('word_selector',feat_selector),
                                               ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                               ])
        else:
            node_transformer_debug= Pipeline([('selector', NodeTransformerTextEnclosed()),
                                               ('tf', tdifNodeTextVectorizer), #we can use it separately from the pipleline once fitted
                                               ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                               ])





        node_transformer = FeatureUnion( [  #CAREFUL IF YOU CHANGE THIS - see cleanTransformers method!!!!
                                    ("text", Pipeline([
                                                       ('selector', NodeTransformerTextEnclosed()),
                                                       ('tf', tdifNodeTextVectorizer), #we can use it separately from the pipleline once fitted
                                                        #('chi2',feat_selector),
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
                                    , ("neighbors", Pipeline([
                                                         ('selector', NodeTransformerNeighbors()),
                                                         ('neighbors', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                       )
                                    , ("1hot", Pipeline([
                                                         ('1hot', Node1HotFeatures())  #does the 1-hot encoding directly
                                                         ])
                                       )
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
                                                         ('numerical', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                                                         ])
                                        )
                                    , ("sourcetext0", Pipeline([
                                                       ('selector', EdgeTransformerSourceText(0)),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.t_ngrams_edge  #(2,6)
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                                    , ("targettext0", Pipeline([
                                                       ('selector', EdgeTransformerTargetText(0)),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.t_ngrams_edge
                                                                                 #, analyzer = 'word', ngram_range=self.tEDGE_NGRAMS
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                                    , ("sourcetext1", Pipeline([
                                                       ('selector', EdgeTransformerSourceText(1)),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.t_ngrams_edge  #(2,6)
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                                    , ("targettext1", Pipeline([
                                                       ('selector', EdgeTransformerTargetText(1)),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.t_ngrams_edge
                                                                                 #, analyzer = 'word', ngram_range=self.tEDGE_NGRAMS
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                                    , ("sourcetext2", Pipeline([
                                                       ('selector', EdgeTransformerSourceText(2)),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.t_ngrams_edge  #(2,6)
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                                    , ("targettext2", Pipeline([
                                                       ('selector', EdgeTransformerTargetText(2)),
                                                       ('tfidf', TfidfVectorizer(lowercase=self.b_tfidf_edge_lc, max_features=self.n_tfidf_edge
                                                                                 , analyzer = 'char', ngram_range=self.t_ngrams_edge
                                                                                 #, analyzer = 'word', ngram_range=self.tEDGE_NGRAMS
                                                                                 , dtype=np.float64)),
                                                       ('todense', SparseToDense())  #pystruct needs an array, not a sparse matrix
                                                       ])
                                       )
                        ]

        edge_transformer = FeatureUnion( lEdgeFeature )
        #return _node_transformer, _edge_transformer, tdifNodeTextVectorizer
        self._node_transformer = node_transformer_debug
        self._edge_transformer = edge_transformer
        self.tfidfNodeTextVectorizer = tdifNodeTextVectorizer

    def getTransformers(self):
        """
        return (node transformer, edge transformer)
        """
        #return self._node_transformer, self._edge_transformer
        return self._node_transformer, self._edge_transformer

    def cleanTransformers(self):
        """
        the TFIDF transformers are keeping the stop words => huge pickled file!!!

        Here the fix is a bit rough. There are better ways....
        JL
        """
        #Do not Clean
        print('TODO Cleaning Method not implemented yet .....')

        #self._node_transformer.transformer_list[0][1].steps[1][1].stop_words_ = None   #is 1st in the union...
        #for i in [2, 3, 4, 5, 6, 7]:
        #    self._edge_transformer.transformer_list[i][1].steps[1][1].stop_words_ = None   #are 3rd and 4th in the union....
        return self._node_transformer


    @staticmethod
    def getNodeTextSelectedFeatures(node_transformer):
        #I have the impression this implem is not efficient
        #as we still keep the 10,000 words from the vectorizer ...
        #TODO Combine objects CountVectorizer with features selection that update and clean the vocabulary
        cvect=node_transformer.named_steps['tf']
        #Index to Word String array
        I2S_array =np.array(cvect.get_feature_names())
        if hasattr(node_transformer,'feature_selection') and node_transformer.feature_selection is True:
            fs=node_transformer.named_steps['word_select']
            selected_indices=fs.get_support(indices=True)
            return I2S_array[selected_indices]
        else:
            return I2S_array


