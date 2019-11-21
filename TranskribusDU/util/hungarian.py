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

from .jaccard import jaccard_distance


def evalHungarian(lX,lY,th, costFun=jaccard_distance):
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
    