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
from crf.Transformer_PageXml import NodeTransformerTextEnclosed, NodeTransformerTextLen, NodeTransformerXYWH, NodeTransformerXYWH_v2, NodeTransformerNeighbors, Node1HotFeatures,NodeTransformerNeighborsAllText
from crf.Transformer_PageXml import Edge1HotFeatures, EdgeBooleanFeatures,EdgeBooleanFeatures_v2, EdgeNumericalSelector, EdgeTransformerSourceText, EdgeTransformerTargetText
from crf.PageNumberSimpleSequenciality import PageNumberSimpleSequenciality
from crf.Transformer_PageXml import NodeEdgeTransformer

from FeatureDefinition import FeatureDefinition

#from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#Should be able to discriminate between chi2 and mutual info, right ?
#(X, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)[source]

from crf.FeatureSelection import pointwise_mutual_information_score,mutual_information,SelectRobinBest

#feat_selector=SelectRobinBest(mutual_information



def chi2_scores(X,y):
    #This return  only the scores of the chi2 function ...
    # Define as a function as chi_score = lambda x,y : chi2(x,y)[0] #this can not be pickled ...
    return chi2(X,y)[0]



class FeatureDefinition_PageXml_FeatSelect(FeatureDefinition):

    def __init__(self, n_tfidf_node=None, t_ngrams_node=None, b_tfidf_node_lc=None
                     , n_tfidf_edge=None, t_ngrams_edge=None, b_tfidf_edge_lc=None
                     ,feat_select=None,text_neighbors=False, n_tfidf_node_neighbors=500
                     ,XYWH_v2=False, edge_features=False):

        FeatureDefinition.__init__(self)

        self.n_tfidf_node, self.t_ngrams_node, self.b_tfidf_node_lc = n_tfidf_node, t_ngrams_node, b_tfidf_node_lc
        self.n_tfidf_edge, self.t_ngrams_edge, self.b_tfidf_edge_lc = n_tfidf_edge, t_ngrams_edge, b_tfidf_edge_lc

        self.text_neighbors=text_neighbors
        self.n_tfidf_node_neighbors=n_tfidf_node_neighbors
        self.XYWH_v2=XYWH_v2
        self.edge_features=edge_features
        #TODO n_jobs=4


        if feat_select=='chi2':
            feat_selector=SelectKBest(chi2, k=self.n_tfidf_node)

        elif feat_select == 'mi_rr':
            print('Using Mutual Information Round Robin as Feature Selection')
            feat_selector=SelectRobinBest(mutual_information,k=self.n_tfidf_node)
            feat_selector_neigh=SelectRobinBest(mutual_information,k=self.n_tfidf_node)

        elif feat_select =='chi2_rr':
            #chi_score = lambda x,y : chi2(x,y)[0] #this can not be pickled ...
            feat_selector=SelectRobinBest(chi2_scores, k=self.n_tfidf_node)
            feat_selector_neigh=SelectRobinBest(chi2_scores, k=self.n_tfidf_node)

        elif feat_select=='tf' or feat_select is None:
            feat_selector=None

        else:
            raise ValueError('Invalid Feature Selection method',feat_select)


        if feat_selector:
            tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc,max_features=10000, analyzer = 'char', ngram_range=self.t_ngrams_node) #(2,6)

            text_pipeline = Pipeline([('selector', NodeTransformerTextEnclosed()),
                                               ('tf', tdifNodeTextVectorizer),
                                                ('word_selector',feat_selector),
                                               #('todense', SparseToDense()) #Here we don't need to convert to Dense anymore
                                               ])
        else:
            tdifNodeTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc, max_features=self.n_tfidf_node
                                                                                  , analyzer = 'char', ngram_range=self.t_ngrams_node #(2,6)
                                                                                  , dtype=np.float64)
            text_pipeline= Pipeline([('selector', NodeTransformerTextEnclosed()),
                                               ('tf', tdifNodeTextVectorizer),
                                               #('todense', SparseToDense()) #Here we don't need to convert to Dense anymore
                                               ])


        node_transformer_ops =[("text", text_pipeline) ,
                                    ("textlen", Pipeline([
                                                         ('selector', NodeTransformerTextLen()),
                                                         ('textlen', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
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
                                      )]

        if self.XYWH_v2 is True:
            feat_xy=("xywh", Pipeline([('selector', NodeTransformerXYWH_v2()),
                               ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                              ]) )

        else:
            feat_xy=("xywh", Pipeline([('selector', NodeTransformerXYWH()),
                               ('xywh', StandardScaler(copy=False, with_mean=True, with_std=True))  #use in-place scaling
                              ]) )

        node_transformer_ops.append(feat_xy)



        if text_neighbors:
            #BY DEFAULT we use chi2
            if self.n_tfidf_node_neighbors>0:
                feat_selector_neigh=SelectKBest(chi2, k=self.n_tfidf_node_neighbors)
                neighborsTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc,analyzer = 'char', ngram_range=self.t_ngrams_node) #(2,6)
                neighbors_text_pipeline = Pipeline([('selector', NodeTransformerNeighborsAllText()),
                                                   ('tf_neighbors', neighborsTextVectorizer),
                                                    ('feat_selector',feat_selector_neigh),
                                                   ])

            else:
                neighborsTextVectorizer = TfidfVectorizer(lowercase=self.b_tfidf_node_lc,analyzer = 'char', ngram_range=self.t_ngrams_node) #(2,6)
                neighbors_text_pipeline = Pipeline([('selector', NodeTransformerNeighborsAllText()),
                                                   ('tf_neighbors', neighborsTextVectorizer)
                                                    ])
            node_transformer_ops.append(('text_neighbors',neighbors_text_pipeline))


        print(node_transformer_ops)


        node_aggregated_edge_features=[('1hot_edge',NodeEdgeTransformer(Edge1HotFeatures(PageNumberSimpleSequenciality()),agg_func='sum')) ]
        node_aggregated_edge_features.append(('boolean_edge',NodeEdgeTransformer(EdgeBooleanFeatures_v2(),agg_func='sum')) )
        #Aggregated Numerical Features do not make a lot of sense here ....

        if edge_features:
            node_transformer_ops.extend(node_aggregated_edge_features)

        print(node_transformer_ops)
        node_transformer = FeatureUnion(node_transformer_ops)


        #Minimal EdgeFeature Here
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
                        ]


        edge_transformer = FeatureUnion(lEdgeFeature)
        self._node_transformer = node_transformer
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
        #TODO Better Cleaning for feature selection
        self._node_transformer.transformer_list[0][1].steps[1][1].stop_words_ = None   #is 1st in the union...
        return self._node_transformer


    @staticmethod
    def getNodeTextSelectedFeatures(node_transformer):
        #I have the impression this implem is not efficient
        #as we still keep the 10,000 words from the vectorizer ...
        #TODO Combine objects CountVectorizer with features selection that update and clean the vocabulary

        text_pipeline =node_transformer.transformer_list[0][1]
        cvect=node_transformer.transformer_list[0][1].named_steps['tf']
        #Index to Word String array
        I2S_array =np.array(cvect.get_feature_names())
        #if hasattr(node_transformer,'feature_selection') and node_transformer.feature_selection is True:
        if 'word_select' in text_pipeline.named_steps:
            fs=node_transformer.named_steps['word_select']
            selected_indices=fs.get_support(indices=True)
            return I2S_array[selected_indices]
        else:
            return I2S_array


