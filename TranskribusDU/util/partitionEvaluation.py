# -*- coding: utf-8 -*-
"""


    Samples of layout generators
    
    generate Layout annotated data 
    
    copyright Naver Labs 2018
    READ project 


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
    @author: H. Déjean
    DTW: https://github.com/pierre-rouanet/dtw
    """

from .dtw import  dtw
from .jaccard import jaccard_distance
from .iou import iou_distance

 
def evalPartitions(x, y, th, distf):
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
    ltemp.sort(key=lambda xyc:xyc[2],reverse=True)
    for i,j,c in ltemp :
        # when multi matching for a ref: take the best score (first taken if same score) 
        if c >= th and j not in map(lambda x:x[1],lFound) and i not in map(lambda x:x[0],lFound):
#             print (i,j,c)
            lFound.append((i,j))
    lMissed  = list(filter (lambda e: e not in map(lambda x:x[1],lFound),y))
    lErr  = list(filter (lambda e: e not in  map(lambda x:x[0],lFound),x))
    cntOk = len(lFound)
    cntErr = abs(len(lFound)- len(x))
    cntMissed = abs(len(lFound)-len(y))
#         ss
    return cntOk,cntErr,cntMissed, lFound,lErr,lMissed


# ----  tests  ---------------------------------------------------------

def test_jaccard():
    assert jaccard_distance([], []) == 0.0
    assert jaccard_distance([1], [1]) == 0.0
    assert jaccard_distance([1], [3]) == 1.0

def test_iuo():
    import shapely.geometry as geom
    assert iou_distance(geom.Polygon([]), geom.Polygon([])) == 0.0
    assert iou_distance(geom.Polygon([(0,0), (1,1), (0,1)]), geom.Polygon([(10,0), (11,1), (10,1)])) == 1.0

def test_jaccard_set():
    assert jaccard_distance(set([]) , set([])) == 0.0
    assert jaccard_distance(set([1]), set([1])) == 0.0
    assert jaccard_distance(set([1]), set([3])) == 1.0

def test_samples():
    
    ref = [[0,1,2,3] ,[4,5,6,7],[8,9,10],[11,12],[13,14,15,16,17]]
    run = [[0,1,2],[3],[4,5,6,7],[8,9,10],[11,12],[13,14],[15,16,17]]
#     run = [[0,1,2,3,4,5,6,7,8,9]]
#     for th in [ x*0.01 for x in  range(50,105,5)]:
#         ok, err, miss = evalPartitions(run, ref, th)
#         print (th, ok,err,miss)
    #1.0 3 4 2   
    assert (3,4,2) ==  evalPartitions(run, ref, 0.8, jaccard_distance)[:3]
    # 0.8 3 4 2


    ref= [['a','b'],['c'],['d','e']]
    run= [['a','b'],['c','e'],['d','e'],[]]
#     ok, err, miss = evalPartitions(run, ref, 0.8)
    assert  (2,2,1) == evalPartitions(run, ref, 0.8,jaccard_distance)[:3]

def test_samples_2():
    
    ref = [[0,1,2,3]              ,[4,5,6,7],[8,9,10],[11,12],[13,14,15,16,17]]
    run = [[0,1,2,3], [0,1,2,3,4],[4,5,6,7],[8,9,10],[11,12],[13,14],[15,16,17]]
    assert (4,3,1) ==  evalPartitions(run, ref, 0.8, jaccard_distance)[:3]

def test_samples_set():
    ref = [set([0,1,2,3])  , set([4,5,6,7]), set([8,9,10]), set([11,12]), set([13,14,15,16,17])]
    run = [set([0,1,2]), set([3]), set([4,5,6,7]), set([8,9,10]), set([11,12]), set([13,14]), set([15,16,17])]
#     run = [[0,1,2,3,4,5,6,7,8,9]]
#     for th in [ x*0.01 for x in  range(50,105,5)]:
#         ok, err, miss = evalPartitions(run, ref, th)
#         print (th, ok,err,miss)
    #1.0 3 4 2   
    assert (3,4,2) ==  evalPartitions(run, ref, 0.8, jaccard_distance)[:3]
    # 0.8 3 4 2


    ref= [set(['a','b']),set(['c']),set(['d','e'])]
    run= [set(['a','b']),set(['c','e']),set(['d','e']),set([])]
#     ok, err, miss = evalPartitions(run, ref, 0.8)
    assert  (2,2,1) == evalPartitions(run, ref, 0.8,jaccard_distance)[:3]
    
def test_samples_frozenset():
    ref = [frozenset([0,1,2,3])  , frozenset([4,5,6,7]), frozenset([8,9,10]), frozenset([11,12]), frozenset([13,14,15,16,17])]
    run = [frozenset([0,1,2]), frozenset([3]), frozenset([4,5,6,7]), frozenset([8,9,10]), frozenset([11,12]), frozenset([13,14]), frozenset([15,16,17])]
#     run = [[0,1,2,3,4,5,6,7,8,9]]
#     for th in [ x*0.01 for x in  range(50,105,5)]:
#         ok, err, miss = evalPartitions(run, ref, th)
#         print (th, ok,err,miss)
    #1.0 3 4 2   
    assert (3,4,2) ==  evalPartitions(run, ref, 0.8, jaccard_distance)[:3]
    # 0.8 3 4 2


    ref= [frozenset(['a','b']),frozenset(['c']),frozenset(['d','e'])]
    run= [frozenset(['a','b']),frozenset(['c','e']),frozenset(['d','e']),set([])]
#     ok, err, miss = evalPartitions(run, ref, 0.8)
    assert  (2,2,1) == evalPartitions(run, ref, 0.8,jaccard_distance)[:3]

def test_samples_set_emptyness():
    ref = [set([0,1,2,3])]
    run = [set([0,1,2]), set([3])]
    assert (1,1,0) ==  evalPartitions(run, ref, 0.75, jaccard_distance)[:3]

    ref = [set([0,1,2,3])]
    run = [set([0,1,2]), set([])]
    assert (1,1,0) ==  evalPartitions(run, ref, 0.75, jaccard_distance)[:3]

def test_samples_emptyness():
    ref = [[0,1,2,3], [3]]
    run = [[0,1,2], [3]]
    assert (2,0,0) ==  evalPartitions(run, ref, 0.75, jaccard_distance)[:3]

    ref = [[0,1,2,3]]
    run = [[0,1,2], [3]]
    assert (1,1,0) ==  evalPartitions(run, ref, 0.75, jaccard_distance)[:3]

    ref = [[0,1,2,3], []]
    run = [[0,1,2]]
    assert (1,0,1) ==  evalPartitions(run, ref, 0.75, jaccard_distance)[:3]

    ref = [[0,1,2,3], []]
    run = [[0,1,2], []]
    assert (2,0,0) ==  evalPartitions(run, ref, 0.75, jaccard_distance)[:3]

    ref = [[0,1,2,3]]
    run = [[0,1,2], []]
    assert (1,1,0) ==  evalPartitions(run, ref, 0.75, jaccard_distance)[:3]

    ref = [[0,1,2,3], []]
    run = [[0,1,2]]
    assert (1,0,1) ==  evalPartitions(run, ref, 0.75, jaccard_distance)[:3]

if __name__ == "__main__":
    test_samples()
    
    
