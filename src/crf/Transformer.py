# -*- coding: utf-8 -*-

"""
    Root class of node and edge trasnformers (to extract features)
    

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
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        
    def fit(self, l, y=None):
        return self

    def transform(self, l):
        assert False, "Specialize this method!"
        
class SparseToDense(Transformer):
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, o):
        return o.toarray()


    
class TransformerListByType(list, Transformer):
    """
    This is a list of transformer by type (node type or edge type)
    """
        
    def fit(self, l, y=None):
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
        return [ t.transform(lo) for t, lo in zip(self, l)]

class RobustStandardScaler(StandardScaler):
    """
    Same as its super class apart that it  does not crash when th einput array is empty
    """
    def transform(self, X, y=None, copy=None):
        try:
            return StandardScaler.transform(self, X, y, copy)
        except ValueError as e:
            if X.shape[0] == 0:
                return X #just do not crash
            else:
                #StandardScaler failed for some other reason
                raise e
