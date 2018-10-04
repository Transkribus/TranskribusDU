# -*- coding: utf-8 -*-
"""


    Samples of layout generators
    
    generate Layout annotated data 
    
    copyright Naver Labs 2017
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
    J = 1 - (len(set(x).intersection(y)) /(len((set(x+y)))))
    return J


def evalPartitions(x,y,th):
    """
        Compare two lists of lists(partitions)
        Use DTW for matching, jaccard distance as dist(i,j)
        :param list x: generated list of partitions
        :param list y: reference list of partitions
        :param th: int
        Returns cntOk , cntErr , cntMissed
        
        for each pair: take pair those score >= TH. If several, take the first(sic)
        compute cntOk , cntErr , cntMissed from these pairs
    """
    _, cost, _, path = dtw(x, y, jaccard)
    lRun=[]
    lRef = []
    ltemp=[]
    [ltemp.append((x[i],y[j],1 - cost[i][j])) for i,j in zip(path[0],path[1]) ]
    ltemp.sort(key=lambda xyc:xyc[2],reverse=True)
    for i,j,c in ltemp :
        # when multi matching for a ref: take the best score (first taken if same score) 
        if c >= th and j not in lRef:
            lRun.append(i)
            lRef.append(j)

    cntOk = len(lRef)
    cntErr = abs(len(lRef)- len(x))
    cntMissed = abs(len(lRef)-len(y))
    return cntOk,cntErr,cntMissed
    

if __name__ == '__main__':
    ref = [[0,1,2,3] ,[4,5,6,7],[8,9,10],[11,12],[13,14,15,16,17]]
    run = [[0,1,2],[3],[4,5,6,7],[8,9,10],[11,12],[13,14],[15,16,17]]
#     ref = [[0,1],[2,3],[4,5]]
#     run = ref
#     run = [[0,1],[2],[3],[4,5]]
    
    for th in [ x*0.01 for x in  range(50,105,5)]:
        ok, err, miss = evalPartitions(run, ref, th)
        print (th, ok,err,miss)
    
    #1.0 3 4 2   
    assert ((3,4,2) ==  evalPartitions(run, ref, 0.8))
    # 0.8 3 4 2
    