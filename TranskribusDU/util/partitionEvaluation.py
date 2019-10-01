# -*- coding: utf-8 -*-
"""


    Evaluation of two partitions
    
    copyright Naver Labs 2018
    READ project 

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
    
    @author: H. Déjean
    DTW: https://github.com/pierre-rouanet/dtw
    """
from numpy import array, zeros, argmin, inf
from shapely.ops import cascaded_union

def dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r - 1)
                j_k = min(j + k, c - 1)
                min_list += [D0[i_k, j], D0[i, j_k]]
            D1[i, j] += min(min_list)
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def jaccard(x,y):
    """
        intersection over union (set)
        returns a cost (1-distance)!
    """
    try:    
        return  1 - (len(set(x).intersection(y)) / len(set(x+y)))
    except ZeroDivisionError:
        return 0.0

def jaccard_set(setX,setY):
    """
        intersection over union (set)
        returns a cost (1-distance)!
    """
    try:    
        return  1 - (len(setX.intersection(setY)) / len(setX.union(setY)))
    except ZeroDivisionError:
        return 0.0

def iuo(x,y):
    """
        intersection over union (area)
        returns a cost (1-distance)
    """ 
#     print (x.bounds,y.bounds,x.intersection(y).area,cascaded_union([x,y]).area , x.intersection(y).area /cascaded_union([x,y]).area)
    try:    
        return  1 - x.intersection(y).area /cascaded_union([x,y]).area
    except ZeroDivisionError:
        return 0.0
 
def evalPartitions(x,y,th,distf):
    """
        Compare two lists of lists(partitions)
        Use DTW for matching, jaccard/iuo distance as dist(i,j)
        :param list x: generated list of partitions
        :param list y: reference list of partitions
        :param th: int
        Returns cntOk , cntErr , cntMissed, , lFound,lErr,lMissed
        
        for each pair: take pair those score >= TH. If several, take the first(sic)
        compute cntOk , cntErr , cntMissed from these pairs

    """
#     print ("xx",len(x),len(y))
    if x == []: 
        cntOk = 0
        cntErr = 0
        cntMissed = len(y)
        lFound = []
        lErr= []
        lMissed = y
        return cntOk,cntErr,cntMissed, lFound,lErr,lMissed
    elif len(x) == 1:
        ltemp = [(x[0],yy,1-distf(x[0],yy)) for yy in y]
#         print (ltemp)
    elif len(y) == 1:
        ltemp = [(xx,y[0],1-distf(xx,y[0])) for xx in x]
    else:
        _, cost, _, path = dtw(x, y, distf)
        ltemp=[]
        [ltemp.append((x[i],y[j],1 - cost[i][j])) for i,j in zip(path[0],path[1]) ]
    
    lFound = []
#     ltemp.sort(key=lambda xyc:xyc[2],reverse=True)
    for i,j,c in ltemp :
#         print (i[:2],j[:2],c)
        # when multi matching for a ref: take the best score (first taken if same score) 
        if c >= th and j not in map(lambda x:x[1],lFound) and i not in map(lambda x:x[0],lFound):
            lFound.append((i,j))
    lMissed  = list(filter (lambda e: e not in map(lambda x:x[1],lFound),y))
    lErr  = list(filter (lambda e: e not in  map(lambda x:x[0],lFound),x))
    cntOk = len(lFound)
    cntErr = abs(len(lFound)- len(x))
    cntMissed = abs(len(lFound)-len(y))
#         ss
    return cntOk,cntErr,cntMissed, lFound,lErr,lMissed


def matchingPairs(x,y,distf):
    """
            Compare two sequences
            Use DTW for matching, jaccard/iuo distance as dist(i,j)
            :param list x: generated list of partitions
            :param list y: reference list of partitions
            :param th: int
            Returns matched elements 
            
        """
    if x == []: 
        return []
    elif len(x) == 1:
        ltemp = [(x[0],yy,1-distf(x[0],yy)) for yy in y]
    elif len(y) == 1:
        ltemp = [(xx,y[0],1-distf(xx,y[0])) for xx in x]
    else:
        _, cost, _, path = dtw(x, y, distf)
        ltemp=[]
        [ltemp.append((x[i],y[j],1 - cost[i][j])) for i,j in zip(path[0],path[1]) ]
    
    lmatches = []
    curleft, curright = [],[]
    for i,j,c in ltemp:
#         print (i,j,c) #, curleft,curright)
        if i not in [x[0] for x in curleft] and j not in curright:
            if (curleft,curright) != ([],[]):
                lmatches.append((curleft,curright))
            curleft = [(i,c)]
            curright = [j]
        else:
            if i not in [x[0] for x in curleft]:   curleft.append((i,c))
            if j not in curright: curright.append(j)
    lmatches.append((curleft,curright))
    return lmatches
    

def test_jaccard():
    assert jaccard([], []) == 0.0
    assert jaccard([1], [1]) == 0.0
    assert jaccard([1], [3]) == 1.0

def test_iuo():
    import shapely.geometry as geom
    assert iuo(geom.Polygon([]), geom.Polygon([])) == 0.0
    assert iuo(geom.Polygon([(0,0), (1,1), (0,1)]), geom.Polygon([(10,0), (11,1), (10,1)])) == 1.0

def test_jaccard_set():
    assert jaccard_set(set([]) , set([])) == 0.0
    assert jaccard_set(set([1]), set([1])) == 0.0
    assert jaccard_set(set([1]), set([3])) == 1.0

def test_samples():
    
    ref = [[0,1,2,3] ,[4,5,6,7],[8,9,10],[11,12],[13,14,15,16,17]]
    run = [[0,1,2],[3],[4,5,6,7],[8,9,10],[11,12],[13,14],[15,16,17]]
#     run = [[0,1,2,3,4,5,6,7,8,9]]
#     for th in [ x*0.01 for x in  range(50,105,5)]:
#         ok, err, miss = evalPartitions(run, ref, th)
#         print (th, ok,err,miss)
    #1.0 3 4 2   
    assert (3,4,2) ==  evalPartitions(run, ref, 0.8, jaccard)[:3]
    # 0.8 3 4 2


    ref= [['a','b'],['c'],['d','e']]
    run= [['a','b'],['c','e'],['d','e'],[]]
#     ok, err, miss = evalPartitions(run, ref, 0.8)
    assert  (2,2,1) == evalPartitions(run, ref, 0.8,jaccard)[:3]

def test_samples_2():
    
    ref = [[0,1,2,3]              ,[4,5,6,7],[8,9,10],[11,12],[13,14,15,16,17]]
    run = [[0,1,2,3], [0,1,2,3,4],[4,5,6,7],[8,9,10],[11,12],[13,14],[15,16,17]]
    assert (4,3,1) ==  evalPartitions(run, ref, 0.8, jaccard)[:3]

def test_samples_set():
    ref = [set([0,1,2,3])  , set([4,5,6,7]), set([8,9,10]), set([11,12]), set([13,14,15,16,17])]
    run = [set([0,1,2]), set([3]), set([4,5,6,7]), set([8,9,10]), set([11,12]), set([13,14]), set([15,16,17])]
#     run = [[0,1,2,3,4,5,6,7,8,9]]
#     for th in [ x*0.01 for x in  range(50,105,5)]:
#         ok, err, miss = evalPartitions(run, ref, th)
#         print (th, ok,err,miss)
    #1.0 3 4 2   
    assert (3,4,2) ==  evalPartitions(run, ref, 0.8, jaccard_set)[:3]
    # 0.8 3 4 2


    ref= [set(['a','b']),set(['c']),set(['d','e'])]
    run= [set(['a','b']),set(['c','e']),set(['d','e']),set([])]
#     ok, err, miss = evalPartitions(run, ref, 0.8)
    assert  (2,2,1) == evalPartitions(run, ref, 0.8,jaccard_set)[:3]
    
def test_samples_frozenset():
    ref = [frozenset([0,1,2,3])  , frozenset([4,5,6,7]), frozenset([8,9,10]), frozenset([11,12]), frozenset([13,14,15,16,17])]
    run = [frozenset([0,1,2]), frozenset([3]), frozenset([4,5,6,7]), frozenset([8,9,10]), frozenset([11,12]), frozenset([13,14]), frozenset([15,16,17])]
#     run = [[0,1,2,3,4,5,6,7,8,9]]
#     for th in [ x*0.01 for x in  range(50,105,5)]:
#         ok, err, miss = evalPartitions(run, ref, th)
#         print (th, ok,err,miss)
    #1.0 3 4 2   
    assert (3,4,2) ==  evalPartitions(run, ref, 0.8, jaccard_set)[:3]
    # 0.8 3 4 2


    ref= [frozenset(['a','b']),frozenset(['c']),frozenset(['d','e'])]
    run= [frozenset(['a','b']),frozenset(['c','e']),frozenset(['d','e']),set([])]
#     ok, err, miss = evalPartitions(run, ref, 0.8)
    assert  (2,2,1) == evalPartitions(run, ref, 0.8,jaccard_set)[:3]

def test_samples_set_emptyness():
    ref = [set([0,1,2,3])]
    run = [set([0,1,2]), set([3])]
    assert (1,1,0) ==  evalPartitions(run, ref, 0.75, jaccard_set)[:3]

    ref = [set([0,1,2,3])]
    run = [set([0,1,2]), set([])]
    assert (1,1,0) ==  evalPartitions(run, ref, 0.75, jaccard_set)[:3]

def test_samples_emptyness():
    ref = [[0,1,2,3], [3]]
    run = [[0,1,2], [3]]
    assert (2,0,0) ==  evalPartitions(run, ref, 0.75, jaccard)[:3]

    ref = [[0,1,2,3]]
    run = [[0,1,2], [3]]
    assert (1,1,0) ==  evalPartitions(run, ref, 0.75, jaccard)[:3]

    ref = [[0,1,2,3], []]
    run = [[0,1,2]]
    assert (1,0,1) ==  evalPartitions(run, ref, 0.75, jaccard)[:3]

    ref = [[0,1,2,3], []]
    run = [[0,1,2], []]
    assert (2,0,0) ==  evalPartitions(run, ref, 0.75, jaccard)[:3]

    ref = [[0,1,2,3]]
    run = [[0,1,2], []]
    assert (1,1,0) ==  evalPartitions(run, ref, 0.75, jaccard)[:3]

    ref = [[0,1,2,3], []]
    run = [[0,1,2]]
    assert (1,0,1) ==  evalPartitions(run, ref, 0.75, jaccard)[:3]

if __name__ == "__main__":
    test_samples()
    
    
