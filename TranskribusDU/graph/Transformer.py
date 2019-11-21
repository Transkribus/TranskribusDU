# -*- coding: utf-8 -*-

"""
    Root class of node and edge trasnformers (to extract features)
    

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




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
