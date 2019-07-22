# -*- coding: utf-8 -*-

"""
    Jaccard distance and index of lists or sets
    
    https://en.wikipedia.org/wiki/Jaccard_index

    Copyright Naver Labs Europe(C) 2019 H. Déjean, JL. Meunier

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

def jaccard_distance(x,y):
    """
        intersection over union
        x and y are of list or set or mixed of
        returns a cost (1-similarity) in [0, 1]
    """
    try:    
        setx = set(x)
        return  1 - (len(setx.intersection(y)) / len(setx.union(y)))
    except ZeroDivisionError:
        return 0.0

def jaccard_index(x,y):
    """
        intersection over union
        x and y are of list or set or mixed of
        returns a similarity  in [0, 1]
    """
    try:    
        setx = set(x)
        return  len(setx.intersection(y)) / len(setx.union(y))
    except ZeroDivisionError:
        return 1.0
    

# ----  tests  ---------------------------------------------------------
def test_jaccard():
    def assert_dist_and_index(a, b, ref_distance):
        assert jaccard_distance(a, b) ==  ref_distance
        assert jaccard_index   (a, b) == (1.0 - ref_distance)
        
    for fun in (list, set):
        assert_dist_and_index(fun([]), fun([])             , 0)
        assert_dist_and_index(fun([1, 2]), fun([1, 2])     , 0)
        assert_dist_and_index(fun([1]), fun([1, 2])        , 0.5)
        assert_dist_and_index(fun([3]), fun([1, 2])        , 1.0)
