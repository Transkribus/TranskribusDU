# -*- coding: utf-8 -*-

"""
    Root class of node and edge trasnformers (to extract features)
    

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import sklearn.pipeline

from common.trace import traceln


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        
    def fit(self, l, y=None):
        return self

    def transform(self, l):
        assert False, "Specialize this method!"

    def cleanTransformers(self):
        """
        some transformers benefit from being cleaned before saved on disk...
        """
        pass   


class Pipeline(sklearn.pipeline.Pipeline):
    def cleanTransformers(self):
        """
        some transformers benefit from being cleaned before saved on disk...
        """
        for o in self.steps:
            try:
                o[1].cleanTransformers()
            except Exception as e:
                traceln("Cleaning warning:graph.Pipeline: ", e) 
                   
    def __str__(self):
#         return "- Pipeline %s: [\n%s\n\t]" % (self.__class__, "\n   ".join(map(lambda o: str(o), self.steps)))
        sep = "\n   "
        return "- Pipeline %s: [%s%s\n]" % (self.__class__, sep, sep.join(map(lambda o: str(o[1]), self.steps)))


class FeatureUnion(sklearn.pipeline.FeatureUnion):
    def cleanTransformers(self):
        """
        some transformers benefit from being cleaned before saved on disk...
        """
        for o in self.transformer_list:
            try:
                o[1].cleanTransformers()
            except Exception as e:
                traceln("Cleaning warning:graph.FeatureUnion: ", e) 

    def __str__(self):
        sep = "\n   "
        return "- FeatureUnion %s: [%s%s\n]" % (self.__class__, sep, sep.join(map(lambda o: str(o[1]), self.transformer_list)))

    
class SparseToDense(Transformer):
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, o):
        return o.toarray()


    
class TransformerListByType(list, Transformer):
    """
    This is a list of transformer by type (node type or edge type)
    """
        
    def fit(self, l, _y=None):
        """
        self is a list of transformers, one per type
        l is a list of objects 
        """
        return [ t.fit(l) for t in self]
    
    def transform(self, l):
        """
        self is a list of transformers, one per type
        l is a list of feature matrix, one per type
        """
        assert len(self) == len(l), "Internal error"
        return [t.transform(lo) for t, lo in zip(self, l)]


#before 12/9/2017: class RobustStandardScaler(StandardScaler):
class EmptySafe_StandardScaler(StandardScaler):
    """
    Same as its super class apart that it  does not crash when th einput array is empty
    """
    def transform(self, X, y=None, copy=None):
        assert y == None
        assert copy == None
        try:
            #return StandardScaler.transform(self, X, y, copy)
            return StandardScaler.transform(self, X)
        except ValueError as e:
            if X.shape[0] == 0:
                return X #just do not crash
            else:
                #StandardScaler failed for some other reason
                raise e

    def cleanTransformers(self):
        """
        some transformers benefit from being cleaned before saved on disk...
        """
        pass   

#we keep this name, but it is misleading with sklearn.preprocessing.RobustScaler
RobustStandardScaler = EmptySafe_StandardScaler


class EmptySafe_QuantileTransformer(QuantileTransformer):
#bad name class RobustStandardScaler(StandardScaler):
    """
    Same as its super class apart that it  does not crash when th einput array is empty
    """
    def transform(self, X, y=None, copy=None):
        try:
            return QuantileTransformer.transform(self, X)
        except ValueError as e:
            if X.shape[0] == 0:
                return X #just do not crash
            else:
                raise e

    def cleanTransformers(self):
        """
        some transformers benefit from being cleaned before saved on disk...
        """
        pass   

# ----------------------------------------------------------------------------
class CorrelatedQuantileTransformer(EmptySafe_QuantileTransformer):
    """
    preserves the correlation between features by computing the quantiles 
    globally accross features.
    
    Limitation: does not support sparse data yet
    
    JL Meunier 9/9/2020
    """
    
    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.
        
        Same as parent apart that the quantiles are computed globally

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        if self.ignore_implicit_zeros:
            warnings.warn("'ignore_implicit_zeros' takes effect only with"
                          " sparse matrix. This parameter has no effect.")

        n_samples, n_features = X.shape
        references = self.references_ * 100

        self.quantiles_ = []
# numpy original code (1.18.5)
#         for col in X.T:
#             if self.subsample < n_samples:
#                 subsample_idx = random_state.choice(n_samples,
#                                                     size=self.subsample,
#                                                     replace=False)
#                 col = col.take(subsample_idx, mode='clip')
#             self.quantiles_.append(np.nanpercentile(col, references))
        if self.subsample < n_samples:
            # TODO
            subsample_idx = random_state.choice(n_samples,
                                                    size=self.subsample,
                                                    replace=False)
            _X = X.take(subsample_idx, axis=0, mode='clip')
            quantiles_ = np.nanpercentile(_X, references)
        else:
            quantiles_ = np.nanpercentile( X, references)
            # e.g. array([ 1., 15., 77.])
            
        self.quantiles_ = np.transpose([quantiles_] * n_features)
        # Due to floating-point precision error in `np.nanpercentile`,
        # make sure that quantiles are monotonically increasing.
        # Upstream issue in numpy:
        # https://github.com/numpy/numpy/issues/14685
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def _sparse_fit(self, X, random_state):
        raise Exception("Not yet implemented for sparse data.")    


if __name__ == "__main__":
    print(np.__version__)