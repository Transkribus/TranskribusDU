# -*- coding: utf-8 -*-

"""
    Utility: split a list of things in n distinct parts, at random
   
    Similar in spirit to sklearn's ShuffleSplit, but empty intersection between parts.
       
    Copyright Xerox(C)  2019  Jean-Luc Meunier
    
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

def getSplitIndexList(N, n, funTrace=None):
    """
    split N objects to n non-intersecting parts
    return for each onbject the index of the part it is assigned to.
    
    Build even parts, each will be of of length N // n  or N // n + 1

    The output of this function is deterministic.
    
    Provide an output function to get some verbose message
    
    return a list of N part index (each index is in [0, n-1])
    """
    assert N >= 0
    assert n > 0
#     if N % n == 0:
#         q = N // n
#         # ok no pb!
#         print("%d part(s) of %d files" %(N // n, n))
#         ld = [i // q for i in range(N)]
#     else: 
    q = N // n
    r = N % n
    # N == n * q + r
    # N == (n-r) * q + r * (q+1)
    # CQFD
    if not funTrace is None:
        funTrace("%d part(s) of %d files, %d part(s) of %d files" %(n-r, q, r, q+1))
    
    A = (n-r) * q
    
    ld = [i//q for i in range(A)] + [ n - r + (i - A) // (q+1) for i in range(A, N)]
    
    assert len(ld) == N
    return ld


def test_even():
    assert getSplitIndexList(0, 1) == []
    assert getSplitIndexList(1, 1) == [0]
    assert getSplitIndexList(0, 2) == []
    assert getSplitIndexList(2, 2) == [0, 1]
    assert getSplitIndexList(10, 2) == [0,0,0,0,0, 1,1,1,1,1]
    assert getSplitIndexList(3, 3) == [0, 1, 2]
    assert getSplitIndexList(9, 3) == [0,0,0, 1,1,1 ,2,2,2]
    
def test_uneven():
    assert getSplitIndexList(2, 1) == [0, 0]
    assert getSplitIndexList(3, 1) == [0, 0, 0]
    assert getSplitIndexList(1, 2) == [1]
    assert getSplitIndexList(3, 2) == [0, 1,1]
    assert getSplitIndexList(5, 2) == [0,0, 1,1,1]
    assert getSplitIndexList(10, 4) == [0,0, 1,1, 2,2,2, 3,3,3]
 