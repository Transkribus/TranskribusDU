# -*- coding: utf-8 -*-

"""
    Specialized label binarizer, based on sklearn.preprocessing.LabelBinarizer
    
    For 2 classes, it creates a 2-column array
    
    (sklearn version shrinks it to a 1-column... :-/)
    
    Copyright Xerox(C) 2019  Jean-Luc Meunier
    
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
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
import scipy.sparse

class LabelBinarizer2(LabelBinarizer):
    """
    1-hot encoding
    """
    
    def __init__(self, sparse_output=False):
        """
        if sparse_output=True then the returned array from transform is in sparse CSR format.
        """
        super(LabelBinarizer2, self).__init__(sparse_output=sparse_output)
        
    def transform(self, y):
        Y = super(LabelBinarizer2, self).transform(y)
        if self.y_type_ == 'binary':
            if self.sparse_output:
                # subtracting a sparse matrix from a nonzero scalar is not supported...
                # failed to find a elegant way to do that...
                YY = Y.todense()
                return scipy.sparse.csr_matrix(np.hstack((1-YY, YY)))
            else:
                return np.hstack((1-Y, Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super(LabelBinarizer2, self).inverse_transform(Y[:, 1], threshold)
        else:
            return super(LabelBinarizer2, self).inverse_transform(Y, threshold)


# ---  TESTs  ------------------------------------------------------------------
        
def test_many():
    lb = LabelBinarizer()
    lb.fit(["yes", "maybe", "no", "yes", "no", "no", "yes"])
    lb2 = LabelBinarizer2()
    lb2.fit(["yes", "maybe", "no", "yes", "no", "no", "yes"])
    res = lb.transform(["yes", "no", "maybe", "yes", "no", "maybe"])
    res2 = lb2.transform(["yes", "no", "maybe", "yes", "no", "maybe"])
    assert (res == res2).all()
    assert len(lb2.classes_) == 3

def test_many_sparse():
    lb = LabelBinarizer(sparse_output=True)
    lb.fit(["yes", "maybe", "no", "yes", "no", "no", "yes"])
    lb2 = LabelBinarizer2(sparse_output=True)
    lb2.fit(["yes", "maybe", "no", "yes", "no", "no", "yes"])
    res = lb.transform(["yes", "no", "maybe", "yes", "no", "maybe"])
    res2 = lb2.transform(["yes", "no", "maybe", "yes", "no", "maybe"])
    assert (res != res2).nnz == 0
    assert len(lb2.classes_) == 3


def test_binary():
    lb2 = LabelBinarizer2()
    lb2.fit(["yes", "no", "yes", "no", "no", "yes"])
    data = ["yes", "no", "yes", "no", "no"]
    res2 = lb2.transform(data)
    assert (res2 == np.array([  [1,0]
                              , [0,1]
                              , [1,0]
                              , [0,1]
                              , [0,1]
        ])).all() or (res2 == np.array([
                                [0,1]
                              , [1,0]
                              , [0,1]
                              , [1,0]
                              , [1,0]
        ])).all(), res2
    assert (lb2.inverse_transform(res2) == data).all()
    assert len(lb2.classes_) == 2

def test_binary_sparse():
    lb2 = LabelBinarizer2(sparse_output=True)
    lb2.fit(["yes", "no", "yes", "no", "no", "yes"])
    data = ["yes", "no", "yes", "no", "no"]
    res2 = lb2.transform(data)
    assert (res2 == np.array([  [1,0]
                              , [0,1]
                              , [1,0]
                              , [0,1]
                              , [0,1]
        ])).all() or (res2 == np.array([
                                [0,1]
                              , [1,0]
                              , [0,1]
                              , [1,0]
                              , [1,0]
        ])).all()
    assert (lb2.inverse_transform(res2) == data).all()
    assert len(lb2.classes_) == 2


if __name__ == "__main__":
    test_many()
    test_many_sparse
    test_binary()
    test_binary_sparse()
   