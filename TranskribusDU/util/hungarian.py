# -*- coding: utf-8 -*-

"""
    Hungarian algorithm

    Copyright Naver Labs Europe(C) 2019 H. Déjean, JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import numpy as np

from scipy.optimize import linear_sum_assignment

from util.jaccard import jaccard_distance


def evalHungarian_v0(lX,lY,th, costFun=jaccard_distance):
    """
    https://en.wikipedia.org/wiki/Hungarian_algorithm
    
    return ok, err, miss
    """
                    
    ## cost_matrix=np.zeros((len(lX),len(lY)),dtype=float)
    ## for a,x in enumerate(lX):
    ##     for b,y in enumerate(lY):
    ##         cost_matrix[a,b] = costFun(x,y)
    ## cc = cost_matrix
    
    lCost = [costFun(x,y) for x in lX for y in lY]
    cost_matrix = np.array(lCost, dtype=float).reshape((len(lX), len(lY)))
    
    ## assert (cc == cost_matrix).all()
    
    r1,r2 = linear_sum_assignment(cost_matrix)
    ltobeDel=[]
    for a,i in enumerate(r2):
        # print (r1[a],ri)      
        if 1 - cost_matrix[r1[a],i] < th :
            ltobeDel.append(a)
            # if bt:print(lX[r1[a]],lY[i],1- cost_matrix[r1[a],i])              
            # else: print(lX[i],lY[r1[a]],1-cost_matrix[r1[a],i])           
        # else:
            # if bt:print(lX[r1[a]],lY[i],1-cost_matrix[r1[a],i])
            # else:print(lX[i],lY[r1[a]],1-cost_matrix[r1[a],i])                        
    r2 = np.delete(r2,ltobeDel)
    r1 = np.delete(r1,ltobeDel)
    # print("wwww",len(lX),len(lY),len(r1),len(r2))
            
    return len(r1), len(lX)-len(r1), len(lY)-len(r1)


def evalHungarian_v1(lX, lY, th_or_lth, costFun=jaccard_distance):
    """
    optimal assignment given the cost function, ignoring assignments strictly above
    a certain cost (computed as 1 - threshold)
    https://en.wikipedia.org/wiki/Hungarian_algorithm
    
    Take :
        list of clusters (sequence of sequences)
        list of reference clusters
        either one threshold or a list of thresholds
        optionnaly a cost function, which returns values in [0, 1]
    
    return either    (ok, err, miss)
        or a list of (ok, err, miss), one per threshold in the list
    """
    cost_matrix = np.array( [[costFun(x,y) for y in lY] for x in lX]
                            , dtype = float
                            )
    assert cost_matrix.shape == (len(lX), len(lY))
    
    r1,r2 = linear_sum_assignment(cost_matrix)
    
    if isinstance(th_or_lth, list):
        o = []
        for th in th_or_lth:
            nOk = (cost_matrix[r1,r2] <= (1.0 - th))       .sum()
            o.append( (nOk, len(lX)-nOk, len(lY)-nOk) )
    else:
        nOk =     (cost_matrix[r1,r2] <= (1.0 - th_or_lth)).sum()
        o = (nOk, len(lX)-nOk, len(lY)-nOk)
        
    return o


evalHungarian = evalHungarian_v1


# ----  tests  ---------------------------------------------------------

def test_simple():
    
    lref = [ (3,4), (1,2), (99,6) ]

    l1 = [ (1,2), (3,4), ( 5,6) ]
    
    
    assert evalHungarian(l1, l1, 0.4) == (3, 0, 0)
    assert evalHungarian(l1, lref, 0.3) == (3, 0, 0)
    assert evalHungarian(l1, lref, 0.6) == (2, 1, 1)
    
    l2 = [ (3,4), (1,2), (66,6), (99, 999)]
    assert evalHungarian(l2, lref, 0.6) == (2, 2, 1)
    
    assert evalHungarian([(1,)], lref, 0.6) == (0, 1, 3)
    
def test_simple_unordered():
    
    lref = [ (3,4), (1,2), (99,6) ]

    l1 = [ (2,1), (4,3), ( 5,6) ]
    
    assert evalHungarian(l1, l1, 0.4) == (3, 0, 0)
    assert evalHungarian(l1, lref, 0.3) == (3, 0, 0)
    assert evalHungarian(l1, lref, 0.6) == (2, 1, 1)
    
    l2 = [ (3,4), (1,2), (66,6), (99, 999)]
    assert evalHungarian(l2, lref, 0.6) == (2, 2, 1)
    
    assert evalHungarian([(1,)], lref, 0.6) == (0, 1, 3)


def test_v1_by_v0():
    """
    generally ok, apart roundign problems with v0
    
    Because:
       if (cost_matrix[i,j] <= (1 - th)):
       assert th <= (1 - cost_matrix[i,j]), (th, 1 - th, cost_matrix[i,j], 1 - cost_matrix[i,j])
​
    AssertionError: (0.1, 0.9, 0.9, 0.09999999999999998)
    """
    import random
    
    def get_random_clusters():
        nc_max   = 15 
        n_max    = 30
        
        # draw some elements from a population
        lN = random.sample(list(range(n_max+1)), random.randint(1, n_max))
        
        nC = random.randint(1, nc_max) # number of clusters
        lC = [list() for _ in range(nC)]
        
        # assign at random
        for n in lN:
            lC[random.randrange(nC)].append(n)

        return [C for C in lC if bool(C)] # only non-empty clusters
    
    random.seed(1970)
    # for _i in range(100):  # to get into trouble with rounding in v0 ...
    lth = [0, 0.1, 0.33, 0.5, 0.66, 0.8, 1.0]
    for _i in range(59):
        print(_i)
        lref = get_random_clusters()
        l    = get_random_clusters()
        
        print(l)
        print(lref)
        lo = []
        for th in lth:
            print(th, evalHungarian_v1(l, lref, th))
            print(th, evalHungarian_v0(l, lref, th))
            o = evalHungarian_v0(l, lref, th)
            assert evalHungarian_v1(l, lref, th) == o, (l, lref, th)
            lo.append(o)
        assert evalHungarian_v1(l, lref, lth) == lo


if __name__ == "__main__": 
    
    if False:
        # show BAD ROUNDING
        
        # why v0 does not match the 1st cluster to GT cluster???
        l = [[23, 14, 13], [18]]
        lref = [[14, 8, 1, 11, 16, 17, 27, 21, 24, 7, 15, 9, 4, 20, 12, 3, 19, 13, 30]]
        th = 0.1
        print(jaccard_distance(l[0], lref[0]), 1-th)
        print("evalHungarian_v0", evalHungarian_v0(l, lref, th))
        print("evalHungarian_v1", evalHungarian_v1(l, lref, th))
        
        # while here it matches the 100% to the 100%???
        l = [[23, 14, 13], [18]]
        lref = l
        th = 1.0
        print(jaccard_distance(l[0], lref[0]), 1-th)
        print("evalHungarian_v0", evalHungarian_v0(l, lref, th))
        print("evalHungarian_v1", evalHungarian_v1(l, lref, th))
        
    
