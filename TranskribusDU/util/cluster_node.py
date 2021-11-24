# coding: utf8

"""
Analyse edge break/continue accuracy as a fonction of edge characteristics

JL Meunier
Dec. 2020

Copyright NAVER France
"""

import sys, os
from optparse import OptionParser
from collections import defaultdict
import bisect
import random

import numpy as np
import shapely.geometry as geom
from   shapely.prepared import prep

from common.trace       import traceln

try: 
    from PIL import Image, ImageDraw 
    from PIL import ImagePath
except ImportError: 
    traceln("Library pillow missing: cannot plot")
    Image, ImageDraw, ImagePath = None, None, None
    
try:
    import cv2
except ImportError:
    traceln("Library opencv missing: cannot generate video")
    cv2 = None
    
from common.chrono      import chronoOn, chronoOff
from util.gzip_pkl      import gzip_pickle_load
from util.hungarian     import evalHungarian
from util.metrics       import evalAdjustedRandScore, average, stddev
from util.jaccard       import jaccard_distance

from graph.Cluster import ClusterList, Cluster

iPLOT       = 0             # debug mode: show a plot of the solution

lfSimil = [0.66, 0.80, 1.00]


def computePRF(nOk, nErr, nMiss):
    eps = 0.00001
    fP = 100 * nOk / (nOk + nErr + eps)
    fR = 100 * nOk / (nOk + nMiss + eps)
    fF = 2 * fP * fR / (fP + fR + eps)
    return fP, fR, fF


# ------------------------------------------------------------------------------
class Clusterer:
    
    def __init__(self, nn, D, aE, aAB, aYp):
        self.nn  = nn               # nb nodes
        self.D   = D                # Depth of the hierarchy. 0 is top-level
        self.aE  = aE               # either None or edge data : bOk, Y_GT, Y, iEdgeType, fProba, fLength
        self.aAB = aAB
        self.aYp = aYp              # proba distribution
        self.ne  = aAB.shape[0]     # nb edge
        
        assert self.ne == self.aYp.shape[0], "ERROR: aAB and aYp must have same first dimension"
        
    def cluster(self, lvl):
        """
        return a list of clusters at given level
        """
        raise Exception("Internal error: class must be specialized")
        
    def getY_proba_at_level(self, lvl):
        """
        convert a Y_level into a Y of type continue/break
        """
        Ylvl = np.zeros(shape=(self.ne, 2))  # make it a continue/break probability
        sptr = lvl+2
        Ylvl[:,0] = self.aYp[:,0] + self.aYp[:,sptr:].sum(axis=1)
        Ylvl[:,1] = self.aYp[:,1:sptr].sum(axis=1)
        return Ylvl

    def doMetric(self, lvl, clusters, GTclusters, lfSimil):
        dOkErrMiss = { fSimil:(0,0,0) for fSimil in lfSimil }
        lsRpt = []
        for fSimil in lfSimil:
                _nOk, _nErr, _nMiss = evalHungarian(clusters, GTclusters, fSimil, jaccard_distance)
                _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
                
                #traceln("simil:%.2f  P %6.2f  R %6.2f  F1 %6.2f   ok=%6d  err=%6d  miss=%6d" %(
                lsRpt.append("@simil %.2f   P %6.2f  R %6.2f  F1 %6.2f   ok=%6d  err=%6d  miss=%6d" %(
                      fSimil
                    , _fP, _fR, _fF
                    , _nOk, _nErr, _nMiss
                    ))
                
                # global count
                nOk, nErr, nMiss = dOkErrMiss[fSimil]
                nOk   += _nOk
                nErr  += _nErr
                nMiss += _nMiss
                dOkErrMiss[fSimil] = (nOk, nErr, nMiss)

        lSummary = []
        for fSimil in lfSimil:
            nOk, nErr, nMiss = dOkErrMiss[fSimil]
            fP, fR, fF = computePRF(nOk, nErr, nMiss)
            sLine = "lvl_%d  @simil %.2f   P %6.2f  R %6.2f  F1 %6.2f " % (lvl, fSimil, fP, fR, fF ) \
                    + "        "                                                                    \
                    +"ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss)
            lSummary.append(sLine)
        
        sRpt = "\n".join(lSummary) + "\n\n" + "\n".join(lsRpt) + "\n\n" + "\n".join(lSummary)
        
        return nOk, nErr, nMiss, sRpt

    def clusterscore(self, clusters, lvl):
        # debugging....
        Y_proba = self.getY_proba_at_level(lvl) # binarizing in case of hierarchy

        aN2C  = np.arange(self.nn                , dtype=np.int32)   # 1 cluster per node
        for j, c in enumerate(clusters):
            for iA in c:
                aN2C[iA] = j
        nC = len(clusters)        
        awC2C = np.zeros(shape=(nC, nC), dtype=np.float32)
        
        for k in range(self.ne):
            iA, iB = self.aAB[k]
            cA, cB = aN2C[iA], aN2C[iB]
            
            p = Y_proba[k,0] - 0.5
            awC2C[cA, cB] += p
            if cB != cA: awC2C[cB, cA] += p
        return NBestClusterer._score(awC2C)

# ------------------------------------------------------------------------------
class OrigClusterer(Clusterer):
    """
    New style clustering...
    """
    def __init__(self, nn, D, aAB, aYp):
        """
        This class serves for experimenting. It should not be used for real...
        Actually, it create circular import...
        """
        global I_GraphBinaryConjugateClusterer
        
        super(OrigClusterer, self).__init__(nn, D, None, aAB, aYp)
        
        # dynamic import to avoid import loops...
        import graph.pkg_GraphBinaryConjugateSegmenter.I_GraphBinaryConjugateClusterer as I_GraphBinaryConjugateClusterer
        
    def cluster(self, lvl):
        """
        return a list of clusters at given level
        """
        Y_proba = self.getY_proba_at_level(lvl)
        ClusterList = self.connected_component(Y_proba[:,1], 1-0.99) # 0=cont 1=brk

        dEdges = self.getEdges(Y_proba)
        doer = I_GraphBinaryConjugateClusterer.I_GraphBinaryConjugateClusterer()
        lCluster = doer.clusterPlus(ClusterList,dEdges)

        return lCluster
        
    def connected_component(self, Y, fThres):
        """
        Y            array of proba of break
        """
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(2*recursion_limit)

        # create clusters of node based on edge binary labels
        try:
            def DFS(i):
                visited[i] = 1
                visited_index.append(i)
                for j in range(nb_node):
                    if visited[j] != 1 and ani[i][j]==1:
                        visited_index.append(j)
                        DFS(j)
                return visited_index
    
            lCluster = ClusterList([], "CC")  # CC stands for COnnected Component
            
            # create an adjacency matrix
            if False:
                # truely original method??
                nb_node = self.aAB.max() + 1
            else:
                #print(self.nn)
                nb_node = self.nn
            ani = np.zeros(shape=(nb_node, nb_node), dtype='int64')
            fltr = Y < fThres
            ani[self.aAB[fltr,1], self.aAB[fltr,0]] = 1
            ani[self.aAB[fltr,0], self.aAB[fltr,1]] = 1
            
            visited = np.zeros(nb_node, dtype='int64')
            for i in range(nb_node):
                visited_index = []
                if visited[i] == 0:
                    lCluster.append(Cluster(DFS(i)))
        finally:  
            sys.setrecursionlimit(recursion_limit)
            
        return lCluster

    def getEdges(self,Y_proba):
        """
        Y_proba is cont,break proba 2 columns
        return a dictionary of 
            {
            0 -> { (index, index) -> Y_proba_of_0 }
            1 -> { (index, index) -> Y_proba_of_1 }
            }
        """
        #lLabels=['continue','break']
        dEdges={0:{},1:{} }
        Y = Y_proba.argmax(axis=1)
        for i, (iA, iB) in enumerate(self.aAB):
            #type(edge) in [HorizontalEdge, VerticalEdge]:
            #cls = Y[i]
            dEdges[ Y[i] ] [(iA, iB)]= Y_proba[i,  Y[i]]
            dEdges[ Y[i] ] [(iB, iA)]= Y_proba[i,  Y[i]]
        return dEdges


# ------------------------------------------------------------------------------
class NBestClusterer_NoChange(Exception):
    """
    an exception to indicate that no change were done
    """
    pass

class NBestClusterer(Clusterer):
    """
    Clustering that explores multiple solutions
    """
    nBEST = 5
    
    def __init__(self, nn, D, aE, aAB, aYp
                 , nBest=None              # beam size
                 , bByLength=False
                 ):
        super(NBestClusterer, self).__init__(nn, D, aE, aAB, aYp)
        
        self.nBest     = nBest if bool(nBest) else self.nBEST  
        self.bByLength = bByLength  # edge will be sorted by length instead of by probability
        
        self.iNbStateEvaluated = 0

    def cluster(self, lvl):
        """
        return a list of clusters at given level

        here we sort edge by hisgest probability of 'continue', considering that
        each node is in its own cluster.
        And we merge clusters iteratively, keeping H best hypothesis
        return the clustering produced by best hypothesis
        """
        _score, lCluster = self.clusterNBest(lvl)[0] 
        
        return lCluster

    def _clusterNBest_init(self, lvl):
        """
        preparing data
        return 
        - ordered array of index of edges to consider
        - ordered array of edge A,B (same order!)
        - array N2C
        - array C2C
        """
        Y_proba = self.getY_proba_at_level(lvl) # binarizing in case of hierarchy

        # Data
        # aN2C  = Node to cluster membership
        # awC2C = cluster adjacency weighted matrix. (sum of continue probability)
        aN2C  = np.arange(self.nn                , dtype=np.int32)   # 1 cluster per node
        awC2C = np.zeros(shape=(self.nn, self.nn), dtype=np.float32)
#         awC2C[ self.aAB[:,0], self.aAB[:,1] ] = Y_proba[:, 0] - 0.5
#         awC2C[ self.aAB[:,1], self.aAB[:,0] ] = Y_proba[:, 0] - 0.5
        aAB0, aAB1 = self.aAB[:,0], self.aAB[:,1]
        awC2C[ aAB0, aAB1 ] = Y_proba[:, 0] - 0.5
        awC2C[ aAB1, aAB0 ] = Y_proba[:, 0] - 0.5
        
        # keep only the continue edges 
        aIdx = (Y_proba[:,0]>0.5)
        # aIdx = (Y_proba[:,0]>0.0)  # why not looking at wrong prediction 'break'?
        cont_aAB     = self.aAB[aIdx,:]
        if self.bByLength:
            # sort by length
            cont_aE_length = self.aE[aIdx,5]
            aIdx = np.argsort(cont_aE_length[:,])
            del cont_aE_length
        else:
            # sort them by confidence
            cont_Y_proba = Y_proba [aIdx,:]
#             # first sort criteria: edge type => nothing good
#             cont_aE_type = self.aE[aIdx,3]
#             cont_Y_proba[:,1] -= (2*cont_aE_type)
            aIdx = np.argsort(cont_Y_proba[:,1]) # index ordered by proba of 'continue' (increasing 'break' proba!) 
            del cont_Y_proba
        aABContSorted = cont_aAB[aIdx,:]
        del cont_aAB
        
        return aIdx, aABContSorted, aN2C, awC2C
    
    def clusterNBest(self, lvl):
        """
        return a list of nbest clustering with their score:
        return [ (score, lCluster), ... ]
        """

        
        aIdx, aABContSorted, aN2C, awC2C = self._clusterNBest_init(lvl)
        
        # find best clustering
        self._do_cluster_init(aIdx, aABContSorted, aN2C, awC2C)
        
        ltScorelCluster = self._do_cluster() 
        
        self._do_cluster_finish()
        
        del aIdx, aABContSorted
        
        return ltScorelCluster

    def getNumberOfExploredState(self):
        return self.iNbStateEvaluated
    
    @classmethod
    def _score(cls, awC2C):
        """
        positive attractiveness along the diagonal are good (because inside clusters)
        negative values elsewhere are good, but counted twice as the matrix is symmetric
        """
        #  awC2C.trace() - (awC2C.sum() - awC2C.trace()) / 2
        #return 1.5 * awC2C.trace() - awC2C.sum() / 2.0
        return  1.5 * awC2C.trace() - awC2C.sum() / 2.0

    def _makeClusters(self, aN2C): 
        # make it a list of cluster
        dCluster = dict()  # dict  cluster_id -> Cluster object
        for iA in range(self.nn):
            try:
                dCluster[ aN2C[iA] ].add(iA)
            except KeyError:
                dCluster[ aN2C[iA] ] = Cluster( [iA] )
        return list(dCluster.values())

    @classmethod
    def _mergeCluster(cls, jA, jB, h_awC2C):
        """
        merge clusters jA and jB
        return 
            jA (index of merged cluster)
            jB index of discarded cluster
            new cluster adjacency matrix
        """
        n,m = h_awC2C.shape
        assert n==m
        jA, jB = min(jA, jB), max(jA, jB)

        # *** we will transfer everything from cluster jB to cluster jA ***
        awC2C = np.zeros(shape=(n-1, m-1), dtype=np.float32)
    
        # keep everything everything before jB
        awC2C[:jB,:jB] = h_awC2C[:jB,:jB] 

        # keep everything everything after jB
        awC2C[jB:,:jB] = h_awC2C[jB+1:,:jB] 
        awC2C[:jB,jB:] = h_awC2C[:jB,jB+1:] 
        awC2C[jB:,jB:] = h_awC2C[jB+1:,jB+1:] 
        
        # merge edge within the new cluster
        awC2C[jA,jA] += h_awC2C[jB,jB] 
        
        awC2C[jA,:jB] += h_awC2C[jB,:jB]
        awC2C[jA,jB:]  = h_awC2C[jA,jB+1:] + h_awC2C[jB,jB+1:]
        
        awC2C[:jB,jA] += h_awC2C[:jB  ,jB]
        awC2C[jB:,jA]  = h_awC2C[jB+1:,jA] + h_awC2C[jB+1:,jB]
        
        awC2C[jA,jA] -= h_awC2C[jA,jB] # was summed twice...
        
        return jA, jB, awC2C

    @classmethod
    def _updateN2C(cls, jA, jB, h_aN2C):
        # B goes with A, or A goes with B, but jA is the smallest cluster index, which we keep
        aN2C  = np.copy(h_aN2C)
        # and all clusters above jB get moved by 1 ...
        aN2C[ aN2C[:] == jB ] = jA
        aN2C[ aN2C[:] > jB ] -= 1
        return aN2C

    @classmethod
    def _asContinue(cls, iA, iB, h_aN2C, h_awC2C):
        """
        considering Ith edge as a continue, from node iA to node iB
        let's merge concerned clusters, if source and target node not already in same cluster
        return 
        - new cluster index
        - new node2cluster assignment             (freshly allocated!)
        - new cluster adjacency weighted matrix   (freshly allocated!)
        """
        
        # they have been assigned to a cluster
        jA, jB = h_aN2C[iA], h_aN2C[iB]     # cluster index

        if jA == jB: raise NBestClusterer_NoChange()
        
        jA, jB, awC2C = cls._mergeCluster(jA, jB, h_awC2C)
            
        # B goes with A, or A goes with B, but jA is the smallest cluster index, which we keep
        aN2C = cls._updateN2C(jA, jB, h_aN2C)
            
        return jA, aN2C, awC2C


# ------------------------------------------------------------------------------
class HypothesisClusterer(NBestClusterer):
    """
    Here we keep N best hypothesis
    """
    def __init__(self, nn, D, aE, aAB, aYp
                 , nBest=None              # number of parallel hypothesis
                 , bByLength=False
                 ):
        super(HypothesisClusterer, self).__init__(nn, D, aE, aAB, aYp
                                                  , nBest=nBest, bByLength=bByLength)

    def _do_cluster_init(self, _aIdx, aABContSorted, aN2C, awC2C):
        """
        prepare for beam clustering
        """
        self.fBestWorst  = -1e8                         # worst score of N best 
        self.lHypothesis = []                           # sync list of state
        self.aScore      = np.zeros(shape=(self.nBest,))     # sync list of score
        self.aScore[:]   = self.fBestWorst
        self.aABSorted   = aABContSorted
        # starting point
        f = self._score(awC2C)
        self._store_hypothesis(f, aN2C, awC2C)

        return
    
    def _makeClustersByHypothesis(self, laN2C):
        """
        return a list of (score, clusters)
        """
        ltRet = []
        for i, aN2C in enumerate(laN2C):
            score = self.aScore[i]
            lCluster = self._makeClusters(aN2C)
            ltRet.append((score, lCluster))
        ltRet.sort(reverse=True)
        return ltRet
        
    def _do_cluster(self):
        """
        given the current passed state, explore iBeamSize child hypothesis and
        return sorted list[ (score, lCluster), ...]
        """
        #print("FIRST SCORE", self._score(self.lHypothesis[0][1]))
        ne = self.aABSorted.shape[0]
        bGo = True
        while bGo:
            bGo = False
            #print('-'*50)
            for i in range(ne):
                # taking each edge in turn...
                # either we take it as a continue or as a break
                lCurHypothesis = [o for o in self.lHypothesis]
                for h_aN2C, h_awC2C in lCurHypothesis:
                    
                    # imagine edge i is a continue
                    iA, iB = self.aABSorted[i]
                    self.iNbStateEvaluated += 1
                    try:
                        _jA, aN2C, awC2C = self._asContinue(iA, iB, h_aN2C, h_awC2C)
                        f = self._score(awC2C)
                        if f > self.fBestWorst and self._store_hypothesis(f, aN2C, awC2C):
                            bGo = True 
                    except NBestClusterer_NoChange:
                        pass

        ltRet = self._makeClustersByHypothesis([aN2C for (aN2C, _) in self.lHypothesis])

        return ltRet

    def _store_hypothesis(self, fScore, aN2C, stuff, iEdge=None):
        """
        replace worse hypothesis by this one(supposedly a better one!)
        aN2C = node to cluster assignment
        stuff = cluster attractiveness matrix, in this class
        iEdge = index of the edge that was applied as 'continue' edge, if any
        """
        # let's check it is not already in it!!
        for _aN2C, _ in self.lHypothesis:
            # ZZZZZZZZZZZZ
            #TODO can find faster code probably!
            if self._same_aN2C(_aN2C, aN2C): return False
            
        i = np.argmin(self.aScore)   # hypothesis i will be replaced by new one
        assert self.aScore[i] < fScore
        self.aScore[i]  = fScore
        self.fBestWorst = self.aScore.min()
        try: 
            self.lHypothesis[i] = (aN2C, stuff)
        except IndexError:
            assert i == len(self.lHypothesis), "Internal error: nup.argmin not behaving as expected"
            self.lHypothesis.append((aN2C, stuff))
        return True

    def _same_aN2C(self, a1, a2):
        if (a1 == a2).all(): return True
        d1 = {}
        d2 = {}
        for i,j in zip(a1, a2):
            try:
                if d1[i] != j: return False
            except KeyError:
                d1[i] = j
            try:
                if d2[j] != i: return False
            except:
                d2[j] = i
#         print(a1)
#         print(" ==", a2)
        return True    
                
    def _do_cluster_finish(self):
        """
        clean data
        """
        del self.lHypothesis
        del self.aScore
        del self.aABSorted
        return


# ------------------------------------------------------------------------------

class NoNextEdgeException(Exception): pass

class LocalHypothesisClusterer(HypothesisClusterer):
    """
    Here we keep N best hypothesis
    But when exploring, we try to explore each cluster fully before jumping on 
    another one. 
    """
    bRandomStart = False
    
    def __init__(self, nn, D, aE, aAB, aYp
                 , nBest=None              # number of parallel hypothesis
                 , bByLength=False
                 ):
        super(LocalHypothesisClusterer, self).__init__(nn, D, aE, aAB, aYp
                                                  , nBest=nBest, bByLength=bByLength)

    def _do_cluster_init(self, _aIdx, aABContSorted, aN2C, awC2C):
        o = super(LocalHypothesisClusterer, self)._do_cluster_init(_aIdx, aABContSorted, aN2C, awC2C)
        self.local_init()
        return o
    
    def local_init(self):
        """
        Initialize
        - the edge stack (first edges to look at)
        - the edge set (all edges to process)
        """
        ne = self.aABSorted.shape[0]
        self._eEdgeStack = set([])
        if self.bRandomStart:
            # insert a random start point!
            self._eEdgeStack.add(random.randrange(ne))
            #self._eEdgeStack.add(600)
        self._eEdgeTodo  = set([iE for iE in range(ne)])
        
        # node2edge list of list
        self._llN2E = [list() for _ in range(self.nn)]
        for iE in range(ne):
            self._llN2E[self.aABSorted[iE][0]].append(iE)  # iE departs from this node
            self._llN2E[self.aABSorted[iE][1]].append(iE)  # iE arrives on   this node
        
        return
    
    def local_next_edge(self):
        """
        return 
        - the index of next edge to process
        - iA
        - iB
        raises NoNextEdgeException if no more edge to process
        """
        try:
            iE = min(self._eEdgeStack)      # take "best" one (highest confidence, or shorter?)
            self._eEdgeStack.remove(iE)
            self._eEdgeTodo .remove(iE)
        except ValueError:                  # min on empty list
            try:
                iE = min(self._eEdgeTodo)   # take "best" one or raise ValueError
            except ValueError:
                raise NoNextEdgeException()
            self._eEdgeTodo.remove(iE)
        
        iA, iB = self.aABSorted[iE]

        return iE, iA, iB
    
    def _do_cluster(self):
        """
        given the current passed state, explore iBeamSize child hypothesis and
        return sorted list[ (score, lCluster), ...]
        """
        while self._eEdgeTodo:
            try:    
                _iE, iA, iB = self.local_next_edge()
            except NoNextEdgeException: 
                break
                    
            # imagine edge i is a continue
            lCurHypothesis = [o for o in self.lHypothesis]
            for h_aN2C, h_awC2C in lCurHypothesis:
                
                self.iNbStateEvaluated += 1
                try:
                    jA, aN2C, awC2C = self._asContinue(iA, iB, h_aN2C, h_awC2C)
                    f = self._score(awC2C)
                    if f > self.fBestWorst and self._store_hypothesis(f, aN2C, awC2C):
                        # record next edges that will be processed in priority
                        # that is all edges related to the nodes in the new cluster
                        for iN in np.asarray(aN2C==jA).nonzero()[0]:
                            self._eEdgeStack = self._eEdgeStack.union(self._llN2E[iN])
                except NBestClusterer_NoChange:
                    pass
            self._eEdgeStack = self._eEdgeStack.intersection(self._eEdgeTodo)

        ltRet = self._makeClustersByHypothesis([aN2C for (aN2C, _) in self.lHypothesis])

        return ltRet


# ------------------------------------------------------------------------------
class Clusterer_NegativeLink(Exception): pass


class ClustererBB:
    """
    offering class methods to deal with clusters' BB
    """
    bForbidOverlap          = True  # set to False to perform as HypothesisClusterer
    fWidthReductionFactor   = 2.0  # reduction factor
    fHeightReductionFactor  = 10.0  # reduction factor
#     fHeightReductionFactor  = -1  # force height to 2

#     fWidthReductionFactor   = 1.0  # reduction factor
#     fHeightReductionFactor  = 2.0  # reduction factor

    @classmethod
    def _resize_NBB(cls, aNBB):
        """
        resize the node BB
        """
        if cls.fWidthReductionFactor != 1.0:
            #resizing the BB width    
            xm = (aNBB[:,2] + aNBB[:,0]) / 2  #x_middle
            xw = (aNBB[:,2] - aNBB[:,0]) / cls.fWidthReductionFactor / 2  #reduced half width
            aNBB[:,0] = xm - xw
            aNBB[:,2] = xm + xw
        if cls.fHeightReductionFactor != 1.0:
            #resizing the BB height
            ym = (aNBB[:,3] + aNBB[:,1]) / 2  #y_middle
            yh = abs(cls.fHeightReductionFactor) if cls.fHeightReductionFactor < 0 else (aNBB[:,3] - aNBB[:,1]) / cls.fHeightReductionFactor / 2  #reduced half height
            aNBB[:,1] = ym - yh
            aNBB[:,3] = ym + yh
        return aNBB

    @classmethod            
    def _aNBB2lBB(cls, nn, aNBB, aN2C):
        """
        transform the array of BB into a list of shapely polygons
        """
        # now the BB of clusters
        lCBB = [None] * nn

        for i, (x1,y1,x2,y2) in enumerate(aNBB[:]):
            plg = geom.Polygon([(x1, y1), (x2, y1), (x2,y2), (x1,y2)])
            assert lCBB[aN2C[i]] == None
            lCBB[aN2C[i]] = plg   # just in case nodes were not process in natural order by superclass...
        return lCBB

    @classmethod
    def _mergeCluster(cls, jA, jB, h_awC2C, h_lCBB):
        """
        merge cluster jA and jB
        compute new cluster adjacency and BB of merged cluster
        return
        - the index of merged cluster (the smallest of the two in fact)
        - the index of deleted cluster (not to be used with new returned data!)
        - new awC2C
        - new lCBB
        """
        jA, jB, awC2C = HypothesisClusterer._mergeCluster(jA, jB, h_awC2C)
        
        # Now the bounding box...
        min_x_A, min_y_A, max_x_A, max_y_A = h_lCBB[jA].bounds
        min_x_B, min_y_B, max_x_B, max_y_B = h_lCBB[jB].bounds
        min_x, min_y = min(min_x_A,min_x_B), min(min_y_A, min_y_B)
        max_x, max_y = max(max_x_A,max_x_B), max(max_y_A, max_y_B)
        plg = geom.Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y),(min_x, max_y)])
        
        lCBB = list(h_lCBB)
        lCBB[jA] = plg
        del lCBB[jB]
        
        return (jA          # index of merged  cluster
                , jB        # index of deleted cluster
                , awC2C
                , lCBB)

    @classmethod
    def plgBB(cls, min_x_A, min_y_A, max_x_A, max_y_A
              , min_x_B, min_y_B, max_x_B, max_y_B):
#         return (   min(min_x_A,min_x_B), min(min_y_A, min_y_B)
#                  , max(max_x_A,max_x_B), max(max_y_A, max_y_B) )
        min_x, min_y = min(min_x_A,min_x_B), min(min_y_A, min_y_B)
        max_x, max_y = max(max_x_A,max_x_B), max(max_y_A, max_y_B)
        return geom.Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y),(min_x, max_y)])
         
    @classmethod
    def _asContinueBB(cls, iA, iB, h_aN2C, h_awC2C, h_lCBB):
        """
        considering Ith edge as a continue, from node iA to node iB
        let's merge concerned clusters, if source and target node not already in same cluster
        return new node2cluster assignment and new cluster adjacency weighted matrix
            and new clusters BB
            ... and track of merges (needed by the Hierarchical BB stuff)
            ... and inter-score : summed score of edges in-between the merged clusters 
        """
        # they have been assigned to a cluster
        jA, jB = h_aN2C[iA], h_aN2C[iB]     # cluster index
        
        if jA == jB: raise NBestClusterer_NoChange()

        # what if it overlaps some other cluster? => we merge them as well!
        if cls.bForbidOverlap:
            # compute list of clusters formed by overlap (in addition to jA and jB
            ljMerge = []
            # and the fInterScore 
            fInterScoreA = h_awC2C[jA, jB]
            fInterScoreB = fInterScoreA
            plg   = cls.plgBB(*h_lCBB[jA].bounds, *h_lCBB[jB].bounds)
            prepO = prep(plg)
            nC = h_awC2C.shape[0]
            lj = list(range(nC)) # todo list
            lj.remove(jA)
            lj.remove(jB)
            bGo = True
            while len(lj) and bGo:
                bGo = False
                lTodo = list([_j for _j in lj])
                for j in lTodo:
                    if prepO.intersects(h_lCBB[j]):
                        # need to merge
                        fInterScoreA += h_awC2C[jA, j]
                        fInterScoreB += h_awC2C[jB, j]
                        lj.remove(j)
                        ljMerge.append(j)
                        plg   = cls.plgBB(*plg.bounds, *h_lCBB[j].bounds)
                        prepO = prep(plg)
                        bGo = True
            fInterScore = min(fInterScoreA, fInterScoreB) # take worse case

            if fInterScore <= 0.0:
                raise Clusterer_NegativeLink()
        
        jA, aN2C, awC2C, lCBB = cls._asContinueBB_orig(iA, iB, h_aN2C, h_awC2C, h_lCBB)
        
        return jA, aN2C, awC2C, lCBB
#         # merge jB cluster into jA
#         jA, jB, awC2C, lCBB = cls._mergeCluster(jA, jB, h_awC2C, h_lCBB)
#         
#         # jA is the index of the new merged cluster, still valid with old data
#         aN2C = cls._updateN2C(jA, jB, h_aN2C)
#         
#         for j in ljMerge:
#             # TODO optimize code (final BB already computed)
#             jA, jB, awC2C, lCBB = cls._mergeCluster(j, jA, awC2C, lCBB)  
#             aN2C = cls._updateN2C(jA, jB, aN2C)
# 
#         return jA, aN2C, awC2C, lCBB, fInterScore

    @classmethod
    def _asContinueBB_orig(cls, iA, iB, h_aN2C, h_awC2C, h_lCBB):
        """
        considering Ith edge as a continue, from node iA to node iB
        let's merge concerned clusters, if source and target node not already in same cluster
        return new node2cluster assignment and new cluster adjacency weighted matrix
            and new clusters BB
            ... and track of merges (needed by the Hierarchical BB stuff)
            ... and inter-score : summed score of edges in-between the merged clusters 
        """
        # they have been assigned to a cluster
        jA, jB = h_aN2C[iA], h_aN2C[iB]     # cluster index
        
        if jA == jB: raise NBestClusterer_NoChange()

        # merge jB cluster into jA
        jA, jB, awC2C, lCBB = cls._mergeCluster(jA, jB, h_awC2C, h_lCBB)
        
        # jA is the index of the new merged cluster, still valid with old data
        aN2C = cls._updateN2C(jA, jB, h_aN2C)
        
        # what if it overlaps some other cluster? => we merge them as well!
        if cls.bForbidOverlap:
            bGo = True
            while bGo:
                bGo = False
                prepO = prep(lCBB[jA])
                (n, _) = awC2C.shape
                for j in range(n):
                    if j == jA: continue
                    if prepO.intersects(lCBB[j]):
                        # need to merge j and jA
                        jA, jB, awC2C, lCBB = cls._mergeCluster(j, jA, awC2C, lCBB)  
                        aN2C = cls._updateN2C(jA, jB, aN2C)
                        bGo = True
                        break

        return jA, aN2C, awC2C, lCBB

    @classmethod
    def plot_get_dir(cls):
        try:
            return cls.plt_dir
        except:
            cls.plt_dir = "clustering.%s" % os.getpid()
            return cls.plt_dir
     
    def plot_init(self, lCBB):            
        self.lCBB0 = list(lCBB)
        
        # to make a video
        try:                    os.mkdir(self.plot_get_dir())
        except FileExistsError: pass
        for img in os.listdir(self.plt_dir):
            if img.endswith(".frm.png"): os.unlink(os.path.join(self.plt_dir, img))
            
        with open("%s/argv.txt"%self.plt_dir, "w")         as fd: fd.write(str(sys.argv))
        sSynth = ".".join(map(str, [_s for _s in sys.argv[1:] if _s not in ["--plot", "--video"]]))
        sSynth = sSynth.replace("--", "")
        with open("%s/%s.txt"%(self.plt_dir, sSynth), "w") as fd: fd.write(str(sys.argv))
        
        
        self.plt_xmax = np.max(self.aNBB[:,[0,2]])
        self.plt_ymax = np.max(self.aNBB[:,[1,3]])

        self.plt_i = 0
        try:
            Clusterer.plt_lvl += 1
        except:
            Clusterer.plt_lvl = 0

    def plot_finish(self):
        
        images = [os.path.join(self.plt_dir, img) for img in os.listdir(self.plt_dir) if img.endswith(".frm.png")]
        images.sort()
        
        frame = cv2.imread(images[0])
        height, width, _layers = frame.shape
        
        # generate videos
        for fps in [2, 10]:
            video_name = 'video_%d.fps%d.avi' % (Clusterer.plt_lvl, fps)
            # fourcc = cv2.VideoWriter_fourcc( *"MJPG" )
            fourcc = cv2.VideoWriter_fourcc( *'mp4v' )
            
            video = cv2.VideoWriter(os.path.join(self.plt_dir, video_name)
                                    , fourcc
                                    , fps   #fps
                                    , (width,height))
           
            for img in images:
                video.write(cv2.imread(img))
            
            cv2.destroyAllWindows()
            video.release()

        for img in images: os.unlink(img)        

    # --- pillow
    def _get_hyp_offset(self, iHyp):
        return (self.plt_xmax + 15) * iHyp
    
    def _plot_offset(self, iHyp, x1, y1, x2, y2):
        x0 = self._get_hyp_offset(iHyp)
        return (x0+x1+1, y1+1, x0+x2+1, y2+1)
    
    def plot_open(self):

        self.img = Image.new("RGB", (self._get_hyp_offset(self.nBest)+2, self.plt_ymax + 4), "#ffffff")  
        self.imgdrw = ImageDraw.Draw(self.img)   
                             
    def plot_hyp(self, iHyp, lCBB):
        self.imgdrw.rectangle(self._plot_offset(iHyp
                                                , 1, 1, self.plt_xmax + 2, self.plt_ymax + 2)
                                                , fill=None, outline="green")
        for o in self.lCBB0: self.imgdrw.rectangle(self._plot_offset(iHyp, *o.bounds), fill=None, outline="black")  
        for o in lCBB      : self.imgdrw.rectangle(self._plot_offset(iHyp, *o.bounds), fill=None, outline='blue')   
        #self.img.show()
 
    def plot_hierhyp(self, iHyp, llCBB):
        self.imgdrw.rectangle(self._plot_offset(iHyp
                                                , 1, 1, self.plt_xmax + 2, self.plt_ymax + 2)
                                                , fill=None, outline="green")
        for o in self.lCBB0: self.imgdrw.rectangle(self._plot_offset(iHyp, *o.bounds), fill=None, outline="black")
        
        for i in range(self.D-1, -1, -1):
            try:               col = ["blue", "red", "orange"
                                      , "green", "maroon", "pink", "slategray", "violet"][i]
            except IndexError: col = "blue"   
            for o in llCBB[i]: 
                self.imgdrw.rectangle(self._plot_offset(iHyp, *o.bounds), fill=None, outline=col)   
        
    def plot_store(self, bFinal=False):  
        if bFinal:
            self.img.save(os.path.join(self.plt_dir, 'final_%d.png' % Clusterer.plt_lvl))
        else:
            self.img.save(os.path.join(self.plt_dir, 'z%06d.frm.png' % self.plt_i))
            self.plt_i += 1
        #plt.show()
        
    def plot_close(self):
        del self.img, self.imgdrw


class HypothesisClustererBB(ClustererBB, HypothesisClusterer):
    """
    Here we keep N best hypothesis, and force clusters not to overlap
    """
    
    def __init__(self, nn, D, aE, aAB, aYp, aNBB
                 , nBest=None              # number of parallel hypothesis
                 , bByLength=False
                 ):
        super(HypothesisClustererBB, self).__init__(nn, D, aE, aAB, aYp
                                                  , nBest=nBest, bByLength=bByLength)
        self.aNBB = aNBB
    
    
    def _do_cluster_init(self, _aIdx, aABContSorted, aN2C, awC2C):
        """
        prepare for beam clustering
        """
        self.fBestWorst  = -1e8                         # worst score of N best 
        self.lHypothesis = []                           # sync list of state
        self.aScore      = np.zeros(shape=(self.nBest,))     # sync list of score
        self.aScore[:]   = self.fBestWorst
        self.aABSorted   = aABContSorted
        
        # now resize BB and 
        self.aNBB = self._resize_NBB(self.aNBB)
        
        # build the BB of initial clusters
        lCBB = self._aNBB2lBB(self.nn, self.aNBB, aN2C)
            
        if iPLOT: self.plot_init(lCBB)
                
        assert None not in lCBB
            
        # starting point
        f = self._score(awC2C)
        self._store_hypothesis(f, aN2C, (awC2C, lCBB))

        return f, aN2C, awC2C

    def _do_cluster(self):
        """
        given the current passed state, explore iBeamSize child hypothesis and
        return sorted list[ (score, lCluster), ...]
        """
        #print("FIRST SCORE", self._score(self.lHypothesis[0][1]))
        ne = self.aABSorted.shape[0]
        bGo = True
        while bGo:
            bGo = False
            #print('-'*50)
            for i in range(ne):
                # taking each edge in turn...
                # either we take it as a continue or as a break
                bNewHyp = False
                lCurHypothesis = [o for o in self.lHypothesis]
                for (h_aN2C, (h_awC2C, h_lCBB)) in lCurHypothesis:
                    
                    # imagine edge i is a continue
                    iA, iB = self.aABSorted[i]
                    self.iNbStateEvaluated += 1
                    try:
                        _jA, aN2C, awC2C, lCBB = self._asContinueBB(iA, iB, h_aN2C, h_awC2C, h_lCBB)
                        f = self._score(awC2C)
                        if f > self.fBestWorst and self._store_hypothesis(f, aN2C, (awC2C, lCBB)):
                            bNewHyp = True
                    except NBestClusterer_NoChange:
                        pass
                    except Clusterer_NegativeLink:
                        pass
                bGo = bGo or bNewHyp
                if iPLOT>1 and bNewHyp: 
                    self.plot_open()
                    for i, (_aN2C, (_awC2C, lCBB)) in enumerate(self.lHypothesis):
                        self.plot_hyp(i, lCBB)
                    self.plot_store()
                    self.plot_close()
        
        if iPLOT >= 1:
            self.plot_open()
            for i, (_aN2C, (_awC2C, lCBB)) in enumerate(self.lHypothesis):
                self.plot_hyp(i, lCBB)
                self.plot_store(bFinal=True)
            self.plot_close()
                
        ltRet = self._makeClustersByHypothesis([aN2C for (aN2C, _) in self.lHypothesis])

        return ltRet
        
    def _do_cluster_finish(self):
        """
        clean data
        """
        del self.lHypothesis
        del self.aScore
        del self.aABSorted
        
        if iPLOT > 1: self.plot_finish()

        return

class LocalHypothesisClustererBB(HypothesisClustererBB, LocalHypothesisClusterer):
    """
    Here we keep N best hypothesis, and force clusters not to overlap
    """
    def __init__(self, nn, D, aE, aAB, aYp, aNBB
                 , nBest=None              # number of parallel hypothesis
                 , bByLength=False
                 ):
        super(LocalHypothesisClustererBB, self).__init__(nn, D, aE, aAB, aYp, aNBB
                                                  , nBest=nBest, bByLength=bByLength)
        self.aNBB = aNBB
    
    def _do_cluster_init(self, _aIdx, aABContSorted, aN2C, awC2C):
        """
        prepare for beam clustering
        """
        f, aN2C, awC2C = super(LocalHypothesisClustererBB, self)._do_cluster_init(_aIdx, aABContSorted, aN2C, awC2C)
        self.local_init()
        return f, aN2C, awC2C

    def _do_cluster(self):
        """
        given the current passed state, explore iBeamSize child hypothesis and
        return sorted list[ (score, lCluster), ...]
        """
        #print("FIRST SCORE", self._score(self.lHypothesis[0][1]))
        while self._eEdgeTodo:
            try:    
                _iE, iA, iB = self.local_next_edge()
            except NoNextEdgeException: 
                break
  
            # imagine edge iE is a continue
            bNewHyp = False
            lCurHypothesis = [o for o in self.lHypothesis]
            for iHyp, (h_aN2C, (h_awC2C, h_lCBB)) in enumerate(lCurHypothesis):
                
                self.iNbStateEvaluated += 1
                try:
                    jA, aN2C, awC2C, lCBB = self._asContinueBB(iA, iB, h_aN2C, h_awC2C, h_lCBB)
                    f = self._score(awC2C)
                    if f > self.fBestWorst and self._store_hypothesis(f, aN2C, (awC2C, lCBB)):
                        # record next edges that will be processed in priority
                        # that is all edges related to the nodes in the new cluster
                        bNewHyp = True
                        for iN in np.asarray(aN2C==jA).nonzero()[0]:
                            self._eEdgeStack = self._eEdgeStack.union(self._llN2E[iN])
#                         self._eEdgeStack.union(np.asarray(aN2C==jA).nonzero()[0])
                except NBestClusterer_NoChange:
                    pass
                except Clusterer_NegativeLink:
                    pass
            self._eEdgeStack = self._eEdgeStack.intersection(self._eEdgeTodo)
            
            if iPLOT>1 and bNewHyp: 
                self.plot_open()
                for iHyp, (_aN2C, (_awC2C, lCBB)) in enumerate(lCurHypothesis):
                    self.plot_hyp(iHyp, lCBB)
                self.plot_store()
                self.plot_close()
        
        if iPLOT >= 1:
            self.plot_open()
            for i, (_aN2C, (_awC2C, lCBB)) in enumerate(self.lHypothesis):
                self.plot_hyp(i, lCBB)
                self.plot_store(bFinal=True)
            self.plot_close()
                
        ltRet = self._makeClustersByHypothesis([aN2C for (aN2C, _) in self.lHypothesis])

        return ltRet
        
    def _do_cluster_finish(self):
        """
        clean data
        """
        del self.lHypothesis
        del self.aScore
        del self.aABSorted
        
        if iPLOT > 1: self.plot_finish()

        return


# ------------------------------------------------------------------------------
class HierarchicalHypothesisClusterer(HypothesisClusterer):
    """
    Here we keep N best hypothesis, accross the D levels of the hierarchy
    
    aYp is no longer a 2-column array but a D+1 columns array
    """
    def __init__(self, nn, D, aE, aAB, aYp
                 , nBest=None              # number of hypothesis
                 , bByLength=False
                 ):
        super(HierarchicalHypothesisClusterer, self).__init__(nn, D, aE, aAB, aYp, nBest=nBest, bByLength=bByLength)
        
    def _clusterNBest_init(self):
        """
        select 'continue' edge across levels

        return:
        - aABContSorted selected edges, sorted
        - aContLvl      level at which edges were selected
        - laN2C         list per level of aN2C
        - lawC2C        list per level of awC2C
        """
        # convert Y_proba: cont, brk_0, brk_1, ... brk_D-1
        #   to             brk_0, brk_1, ... brk_D-1, cont 
        aZ_proba = np.copy(self.aYp)
        # 1st column becomes last
        aZ_proba = aZ_proba[:, list(range(1,self.D+1))+[0]]
        
        # Data
        # aN2C  = Node to cluster membership
        # awC2C = cluster adjacency weighted matrix. (sum of continue probability)
        laN2C, lawC2C = list(), list()
        for lvl in range(self.D):
            aN2C  = np.arange(self.nn                , dtype=np.int32)   # 1 cluster per node
            awC2C = np.zeros(shape=(self.nn, self.nn), dtype=np.float32)
            # sum the proba of cont of current level with cont of below levels
            _acc_aZ_score = np.sum(aZ_proba[:, lvl+1:], axis=1) - 0.5
            aAB0, aAB1 = self.aAB[:,0], self.aAB[:,1]
            awC2C[ aAB0, aAB1 ] = _acc_aZ_score
            awC2C[ aAB1, aAB0 ] = _acc_aZ_score
#             _a = awC2C + 0.5
#             awC2C += np.sqrt(np.matmul(_a, _a)) - 0.5
            laN2C .append(aN2C)
            lawC2C.append(awC2C)
        aN2C, awC2C, _acc_aZ_score = None, None, None
        
        # remove pure break edge, i.e. not a continue at any level
        aIdx     = (aZ_proba[:,0]<=0.5)  # proba of break lower than 0.5, must be some sort of continue
        cont_aAB = self.aAB[aIdx,:]

        # let's keep all possible non negative scores
        cont_Z_proba = aZ_proba[aIdx,:]
        lt = list() # a list of [score, Lvl, Idx], which we will sort...
        for i in range(cont_Z_proba.shape[0]):
            for lvl in range(self.D):
                score = cont_Z_proba[i,lvl+1]
                # NOTE: because > 0.5, an edge can be selected only at ONE level!!
                if score > 0.5: lt.append((score, lvl, i)) # let's consider this edge as a possible continue for that level!
        if self.bByLength:
            aELen = self.aE[aIdx,5]
            lt = [(score - aELen[i], lvl, i) for (score, lvl, i) in lt]
        lt.sort(reverse=True)
        aContIdx = np.array([i for (_, _, i) in lt], dtype=np.int32)
        aContLvl = np.array([l for (_, l, _) in lt], dtype=np.int32)        
        del lt, aIdx, cont_Z_proba, aZ_proba

        aABContSorted   = cont_aAB  [aContIdx,:]
        del aContIdx, cont_aAB
        
        return aABContSorted, aContLvl, laN2C, lawC2C

    def clusterNBest(self):
        """
        return a list of score + list, per level, of nbest clustering:
        return [ (score, llCluster), ... ]
        """
        
        aABContSorted, aContLvl, laN2C, lawC2C = self._clusterNBest_init()
        
        self._do_cluster_init(aABContSorted, aContLvl, laN2C, lawC2C)
               
        ltScorellCluster = self._do_cluster() 
        
        self._do_cluster_finish()
        
        del aABContSorted, aContLvl, laN2C, lawC2C
        
        return ltScorellCluster

    def _do_cluster_init(self, aABContSorted, aABLevel, laN2C, lawC2C):
        """
        prepare for clustering
        We get 
        - a list of edges, sorted by their "max proba accross level"
        - the list of edge level (level where the continue was most probable)
        - list, per level, of array of node-to-cluster assignment
        - list, per level, of array cluster-to-cluster sum of continue(for this level) proba        
        """
        self.fBestWorst  = -1e8                             # worst score of N best 
        self.lHypothesis = []                               # sync list of state
        self.aScore      = np.zeros(shape=(self.nBest,))    # sync list of score, per hypothesis
        self.aScore[:]   = self.fBestWorst
        self.aABSorted   = aABContSorted    # edges Continue proba sorted by highest Continue proba
        self.aABLevel    = aABLevel         # edges Level          in sync with aABContSorted
        
        # starting point
        f = self._score(lawC2C)
        self._store_hypothesis(f, laN2C, lawC2C)

        return

    def _makeClustersByHypothesisByLevel(self, llaN2C):
        """
        return [(score, lCluster), ...] by level
        """
        ltRet = []
        for i, laN2C in enumerate(llaN2C):
            score = self.aScore[i]
            llCluster = list()
            for lvl in range(self.D):
                lCluster = self._makeClusters(laN2C[lvl])
                llCluster.append(lCluster)
            ltRet.append((score, llCluster))
        ltRet.sort(reverse=True)
        return ltRet
    
    def _do_cluster(self):
        """
        given the current passed state, explore iBeamSize child hypothesis and
        return sorted list[ (score, lCluster), ...]
        """
        #print("FIRST SCORE", self._score(self.lHypothesis[0][1]))
        ne = self.aABSorted.shape[0]   # nb "edges"  (might be more than actual edges!)
        bGo = True
        while bGo:
            bGo = False
            #print('-'*50)
            for i in range(ne):
                # taking each edge in turn...
                # either we take it as a continue or as a break
                
                lCurHypothesis = [o for o in self.lHypothesis]
                for h_laN2C, h_lawC2C in lCurHypothesis:
                    
                    # imagine edge i is a continue at the given level
                    iA, iB  = self.aABSorted[i]
                    lvl     = self.aABLevel [i] 
                    self.iNbStateEvaluated += 1
                    try:
                        _ljA, laN2C, lawC2C = self._asContinueAtLevel(iA, iB, lvl, h_laN2C, h_lawC2C)
                        f = self._score(lawC2C)
                        if f > self.fBestWorst and self._store_hypothesis(f, laN2C, lawC2C):
                            bGo = True 
                    except NBestClusterer_NoChange:
                        pass
                    except Clusterer_NegativeLink:
                        pass
        ltRet = self._makeClustersByHypothesisByLevel([laN2C for (laN2C, _) in self.lHypothesis])
        
        return ltRet

    def _store_hypothesis(self, fScore, laN2C, stuff, iEdge=None):
        """
        replace worse hypothesis by this one(supposedly a better one!)
        laN2C = node to cluster assignment, per level
        stuff is lawC2C in this class = cluster attractiveness matrix, per level
        iEdge = index of the edge that was applied as 'continue' edge, if any
        """
        # let's check it is not already in it!!
        for _laN2C, _ in self.lHypothesis:
            #TODO can find faster code probably!
            if all(self._same_aN2C(_aN2C, aN2C) for _aN2C,aN2C in zip(_laN2C, laN2C)): 
                return False
            
        # print(aN2C)
        i = np.argmin(self.aScore)   # hypothesis i will be replaced by new one
        assert self.aScore[i] < fScore
        self.aScore[i]  = fScore
        self.fBestWorst = self.aScore.min()
        try: 
            self.lHypothesis[i] = (laN2C, stuff)
        except IndexError:
            assert i == len(self.lHypothesis), "Internal error: nup.argmin not behaving as expected"
            self.lHypothesis.append((laN2C, stuff))
        return True
    
    def _do_cluster_finish(self):
        """
        clean data
        """
        del self.lHypothesis
        del self.aScore
        del self.aABSorted
        del self.aABLevel
        return

    @classmethod
    def _score(cls, lawC2C):
        """
        positive attractiveness along the diagonal are good (because inside clusters)
        negative values elsewhere are good, but counted twice as the matrix is symmetric
        """
        return  sum(1.5 * awC2C.trace() - awC2C.sum() / 2.0 for awC2C in lawC2C)

    @classmethod
    def _asContinueAtLevel(cls, iA, iB, lvl, h_laN2C, h_lawC2C, bRaiseNegLink=True):
        """
        considering Ith edge as a continue, from node iA to node iB
        let's merge concerned clusters, if source and target node not already in same cluster
        return 
        - reversed list per level of index of merged cluster 
        - new node2cluster assignment 
        - new cluster adjacency weighted matrix
        raise Clusterer_NegativeLink
        """
        # shallow copies, items will be replaced by new stuff
        ljA           = list()
        laN2C, lawC2C = list(h_laN2C), list(h_lawC2C)

        # i.e. total score of edges between clusters merged and  current one, per level
        
        # start from level lvl and move up
        # as soon as not change occur, we can stop
        # because if means iA and iB were already in the same cluster
        # so also in same cluster at parent level.
        for i in range(lvl, -1, -1):
            # the edge is considered as continue, from top-level to lvl
            if bRaiseNegLink:
                aN2C, awC2C = h_laN2C[i], h_lawC2C[i]
                jA, jB = aN2C[iA], aN2C[iB]
                if jA != jB and awC2C[jA, jB] < 0: raise Clusterer_NegativeLink()
            try:
                jA, aN2C, awC2C = cls._asContinue(iA, iB, aN2C, awC2C)
                ljA.append(jA)
            except NBestClusterer_NoChange:
                aN2C, awC2C = aN2C.copy(), awC2C.copy()
                ljA.append(None)
            laN2C [i] = aN2C
            lawC2C[i] = awC2C
        
        # lower levels are unchanged
        
        return ljA, laN2C, lawC2C


class LocalHierarchicalHypothesisClusterer(HierarchicalHypothesisClusterer, LocalHypothesisClusterer):
    """
    Here we keep N best hypothesis, accross the D levels of the hierarchy
    
    aYp is no longer a 2-column array but a D+1 columns array
    """
    def __init__(self, nn, D, aE, aAB, aYp
                 , nBest=None              # number of hypothesis
                 , bByLength=False
                 ):
        super(LocalHierarchicalHypothesisClusterer, self).__init__(nn, D, aE, aAB, aYp
                                                                   , nBest=nBest, bByLength=bByLength)

    def _do_cluster_init(self, _aIdx, aABContSorted, aN2C, awC2C):
        # edge stack and edge todo list
        o = super(LocalHierarchicalHypothesisClusterer, self)._do_cluster_init(_aIdx, aABContSorted, aN2C, awC2C)
        self.local_init()
        return o

    def _do_cluster(self):
        """
        given the current passed state, explore iBeamSize child hypothesis and
        return sorted list[ (score, lCluster), ...]
        """
        while self._eEdgeTodo:
            try:    
                iE, iA, iB = self.local_next_edge()
            except NoNextEdgeException: 
                break
            
            lCurHypothesis = [o for o in self.lHypothesis]
            for h_laN2C, h_lawC2C in lCurHypothesis:
                
                # imagine edge i is a continue at the given level
                lvl     = self.aABLevel [iE] 
                self.iNbStateEvaluated += 1
                try:
                    ljA, laN2C, lawC2C = self._asContinueAtLevel(iA, iB, lvl, h_laN2C, h_lawC2C)
                    f = self._score(lawC2C)
                    if f > self.fBestWorst and self._store_hypothesis(f, laN2C, lawC2C):
                        ljA.reverse()
                        for jA, aN2C in zip(ljA, laN2C):  # ljA is shorter
                            if jA != None:
                                for iN in np.asarray(aN2C==jA).nonzero()[0]:
                                    self._eEdgeStack = self._eEdgeStack.union(self._llN2E[iN])
                except NBestClusterer_NoChange:
                    pass
            # some already processed edges can appear in stack...
            self._eEdgeStack = self._eEdgeStack.intersection(self._eEdgeTodo)
                
        ltRet = self._makeClustersByHypothesisByLevel([laN2C for (laN2C, _) in self.lHypothesis])
        
        return ltRet  


# ------------------------------------------------------------------------------
class HierarchicalHypothesisClustererBB(ClustererBB, HierarchicalHypothesisClusterer):
    """
    Here we keep N best hypothesis, accross the D levels of the hierarchy
    
    aYp is no longer a 2-column array but a D+1 columns array
    """
    bForbidOverlap = True
    
    def __init__(self, nn, D, aE, aAB, aYp, aNBB
                 , nBest=None              # number of hypothesis
                 , bByLength=False
                 ):
        super(HierarchicalHypothesisClustererBB, self).__init__(nn, D
                                                                , aE, aAB, aYp
                                                                , nBest=nBest
                                                                , bByLength=bByLength)
        self.aNBB = aNBB     
 
    def clusterNBest(self):
        """
        return a list of score + list, per level, of nbest clustering:
        return [ (score, llCluster), ... ]
        """
        
        aABContSorted, aContLvl, laN2C, lawC2C = self._clusterNBest_init()

        # now resize BB and 
        self.aNBB = self._resize_NBB(self.aNBB)

        # build the BB of initial clusters
        aN2C0 = np.arange(self.nn, dtype=np.int32)   # 1 cluster per node
        lCBB0 = self._aNBB2lBB(self.nn, self.aNBB, aN2C0)
        llCBB = [list(lCBB0) for _ in range(self.D)] # list of copies
        
        self._do_cluster_init(aABContSorted, aContLvl, laN2C, lawC2C, llCBB)
               
        ltScorellCluster = self._do_cluster() 
        
        self._do_cluster_finish()
        
        del aABContSorted, aContLvl, laN2C, lawC2C
        
        return ltScorellCluster
    
    def _do_cluster_init(self, aABContSorted, aABLevel, laN2C, lawC2C, llCBB):
        """
        prepare for clustering
        We get 
        - a list of edges, sorted by their "max proba accross level"
        - the list of edge level (level where the continue was most probable)
        - list, per level, of array of node-to-cluster assignment
        - list, per level, of array cluster-to-cluster sum of continue(for this level) proba   
        - list, per level, of list of cluster BB     
        """
        self.fBestWorst  = -1e8                             # worst score of N best 
        self.lHypothesis = []                               # sync list of state
        self.aScore      = np.zeros(shape=(self.nBest,))    # sync list of score, per hypothesis
        self.aScore[:]   = self.fBestWorst
        self.aABSorted   = aABContSorted    # edges Continue proba sorted by highest Continue proba
        self.aABLevel    = aABLevel         # edges Level          in sync with aABContSorted

        # starting point
        f = self._score(lawC2C)
        
        self._store_hypothesis(f, laN2C, (lawC2C, llCBB))

        if iPLOT: self.plot_init(llCBB[0])

        return

    def _do_cluster(self):
        """
        given the current passed state, explore iBeamSize child hypothesis and
        return sorted list[ (score, lCluster), ...]
        """
        #print("FIRST SCORE", self._score(self.lHypothesis[0][1]))
        ne = self.aABSorted.shape[0]   # nb "edges"  (might be more than actual edges!)
        bGo = True
        while bGo:
            bGo = False
            #print('-'*50)
            for i in range(ne):
                # taking each edge in turn...
                # either we take it as a continue or as a break
                bNewHyp = False
                lCurHypothesis = [o for o in self.lHypothesis]
                for (h_laN2C, (h_lawC2C, h_llCBB)) in lCurHypothesis:
                    
                    # imagine edge i is a continue at the given level
                    iA, iB  = self.aABSorted[i]
                    lvl     = self.aABLevel [i] 
                    self.iNbStateEvaluated += 1
                    try:
                        _ljA, laN2C, lawC2C, llCBB = self._asContinueAtLevel(iA, iB, lvl
                                                                       , h_laN2C, h_lawC2C, h_llCBB
                                                                       )
                        f = self._score(lawC2C)
                        if f > self.fBestWorst and self._store_hypothesis(f
                                                                          , laN2C
                                                                          , (lawC2C, llCBB)
                                                                         ):
                            bNewHyp = True 
                    except NBestClusterer_NoChange:
                        pass
                    except Clusterer_NegativeLink:
                        pass
                    
                bGo = bGo or bNewHyp
                if iPLOT>1 and bNewHyp: 
                    self.plot_open()
                    for i, (_aN2C, (_awC2C, llCBB)) in enumerate(self.lHypothesis):
                        self.plot_hierhyp(i, llCBB)
                    self.plot_store()
                    self.plot_close()
        
        if iPLOT >= 1:
            self.plot_open()
            for i, (_aN2C, (_awC2C, llCBB)) in enumerate(self.lHypothesis):
                self.plot_hierhyp(i, llCBB)
                self.plot_store(bFinal=True)
            self.plot_close()
 
        ltRet = self._makeClustersByHypothesisByLevel([laN2C for (laN2C, _) in self.lHypothesis])

        return ltRet

    
    def _do_cluster_finish(self):
        """
        clean data
        """
        super(HierarchicalHypothesisClustererBB, self)._do_cluster_finish()       
        if iPLOT > 1: self.plot_finish()

    @classmethod
    def _asContinueAtLevel(cls, iA, iB, lvl, h_laN2C, h_lawC2C, h_llCBB):
        """
        considering Ith edge as a continue, from node iA to node iB
        let's merge concerned clusters, if source and target node not already in same cluster
        return 
        - reversed list per level of index of merged cluster 
        - new node2cluster assignment 
        - new cluster adjacency weighted matrix
        - inter-score : summed score of edges in-between the merged clusters 
        raises Clusterer_NegativeLink
        """
        # shallow copies, items will be replaced by new stuff
        ljA           = list()
        laN2C, lawC2C, llCBB = list(h_laN2C), list(h_lawC2C), list(h_llCBB)
        # start from level lvl and move up
        # as soon as not change occur, we can stop
        # because if means iA and iB were already in the same cluster
        # so also in same cluster at parent level.
        for i in range(lvl, -1, -1):
            # the edge is considered as continue, from top-level to lvl
            aN2C, awC2C, lCBB = h_laN2C[i], h_lawC2C[i], h_llCBB[i]
            try:
                jA, aN2C, awC2C, lCBB = cls._asContinueBB(iA, iB, aN2C, awC2C, lCBB)
                ljA.append(jA)
            except NBestClusterer_NoChange:
                aN2C, awC2C, lCBB = aN2C.copy(), awC2C.copy(), list(lCBB)
                ljA.append(None)

            laN2C [i] = aN2C
            lawC2C[i] = awC2C
            llCBB [i] = lCBB
            
        # lower levels are unchanged
        
        return ljA, laN2C, lawC2C, llCBB

    
class LocalHierarchicalHypothesisClustererBB(HierarchicalHypothesisClustererBB, LocalHypothesisClusterer):
    """
    Here we keep N best hypothesis, accross the D levels of the hierarchy
    
    aYp is no longer a 2-column array but a D+1 columns array
    """
    
    def __init__(self, nn, D, aE, aAB, aYp, aNBB
                 , nBest=None              # number of hypothesis
                 , bByLength=False
                 ):
        super(LocalHierarchicalHypothesisClustererBB, self).__init__(nn, D
                                                                , aE, aAB, aYp, aNBB
                                                                , nBest=nBest
                                                                , bByLength=bByLength)

    def _do_cluster_init(self, aABContSorted, aABLevel, laN2C, lawC2C, llCBB):
        # edge stack and edge todo list
        o = super(LocalHierarchicalHypothesisClustererBB, self)._do_cluster_init(aABContSorted
                                                                                 , aABLevel, laN2C, lawC2C, llCBB)
        self.local_init()
        return o

    def _do_cluster(self):
        """
        given the current passed state, explore iBeamSize child hypothesis and
        return sorted list[ (score, lCluster), ...]
        """
        while self._eEdgeTodo:
            try:    
                iE, iA, iB = self.local_next_edge()
            except NoNextEdgeException:
                break
            
            bNewHyp = False
            lCurHypothesis = [o for o in self.lHypothesis]
            for (h_laN2C, (h_lawC2C, h_llCBB)) in lCurHypothesis:
                
                # imagine edge i is a continue at the given level
                lvl     = self.aABLevel [iE] 
                self.iNbStateEvaluated += 1
                try:
                    ljA, laN2C, lawC2C, llCBB = self._asContinueAtLevel(iA, iB, lvl
                                                                   , h_laN2C, h_lawC2C, h_llCBB
                                                                   )
                    f = self._score(lawC2C)
                    if f > self.fBestWorst and self._store_hypothesis(f
                                                                      , laN2C
                                                                      , (lawC2C, llCBB)
                                                                     ):
                        bNewHyp = True 
                        ljA.reverse()
                        for jA, aN2C in zip(ljA, laN2C):  # ljA is shorter
                            if jA != None:
                                for iN in np.asarray(aN2C==jA).nonzero()[0]:
                                    self._eEdgeStack = self._eEdgeStack.union(self._llN2E[iN])
                        
                except NBestClusterer_NoChange:
                    pass
                except Clusterer_NegativeLink:
                    pass
                
            # some already processed edges can appear in stack...
            self._eEdgeStack = self._eEdgeStack.intersection(self._eEdgeTodo)
            
            if iPLOT>1 and bNewHyp: 
                self.plot_open()
                for i, (_aN2C, (_awC2C, llCBB)) in enumerate(self.lHypothesis):
                    self.plot_hierhyp(i, llCBB)
                self.plot_store()
                self.plot_close()
        
        if iPLOT >= 1:
            self.plot_open()
            for i, (_aN2C, (_awC2C, llCBB)) in enumerate(self.lHypothesis):
                self.plot_hierhyp(i, llCBB)
                self.plot_store(bFinal=True)
            self.plot_close()
 
        ltRet = self._makeClustersByHypothesisByLevel([laN2C for (laN2C, _) in self.lHypothesis])

        return ltRet
        
     
# ------------------------------------------------------------------------------
class BeamClusterer(NBestClusterer):
    """
    run N beam in parallel
    """
    def __init__(self, nn, D, aAB, aYp
                 , nBest=None              # beam size
                 ):
        super(BeamClusterer, self).__init__(nn, D, aAB, aYp, nBest)

    def _do_cluster_init(self, aIdx, aABContSorted, aN2C, awC2C):
        """
        prepare for beam clustering
        """
        #self.fBestWorst  = -1e8                         # worst score of N best 
        #self.lHypothesis = []                           # sync list of state
        #self.aScore      = np.zeros(shape=(self.nBest,))     # sync list of score
        #self.aScore[:]   = self.fBestWorst
        self.aABSorted   = aABContSorted
        
        # starting point
        f = self._score(awC2C)
        liEdge = aIdx.tolist()  # sorted list of index of continue edges in aAB
        self.lBeam = [ Beam(f, aN2C, awC2C, liEdge) for _i in range(self.nBest) ]

        return

    def _do_cluster(self):
        """
        given the current passed state, explore iBeamSize child hypothesis and
        return sorted list[ (score, lCluster), ...]
        """
        Beam.init()
        
        bGo = True
        while bGo:
            bGo = False
            
            for iBeam, beam in enumerate(self.lBeam):
                if beam.bAlive:
                    iEdge_best              = None 
                    fScore_best             = -1e8
                    for iEdge in beam.getTodoList():
                        # taking each edge in turn... what if we apply it as a continue?
                        path = beam.extendPath(iEdge)
                        if Beam.isNewPath(path):
                            # imagine edge i is a continue
                            iA, iB = self.aABSorted[iEdge]
                            self.iNbStateEvaluated += 1
                            _jA, _aN2C, _awC2C, _fInter = self._asContinue(iA, iB, beam.aN2C, beam.awC2C)
                            f = self._score(_awC2C)       
                            if f > fScore_best:
                                iEdge_best = iEdge
                                fScore_best, aN2C_best, awC2C_best, path_best = f, _aN2C, _awC2C, path
                    if iEdge_best is None:
                        # beam is at a dead end, cannot progress anymore       
                        beam.bAlive = False
                        #print("Beam %d   finished" % (iBeam))
                    else:
                        beam.update(fScore_best, aN2C_best, awC2C_best, iEdge_best, path_best)
                        bGo = True
                        
        ltRet = []
        for beam in self.lBeam:
            lCluster = self._makeClusters(beam.aN2C)
            ltRet.append((beam.fScore, lCluster))
        ltRet.sort(reverse=True)

        return ltRet

    def _do_cluster_finish(self):
        """
        clean data
        """
        del self.aABSorted
        del self.lBeam
        return




# ------------------------------------------------------------------------------
class Beam:
    """
    one Beam is a state
    all Beam come from a different path
    """        
    dUnseenPath = defaultdict(lambda :True)
    
    @classmethod
    def init(cls):
        cls.dUnseenPath = defaultdict(lambda :True)
        
    def __init__(self, fScore, aN2C, awC2C, liEdge):
        """
        A beam is created from
            fScore = score
            aN2C   = node to cluster assignment
            awC2C  = cluster attractiveness matrix
            iEdge  = index of the edge that was applied as 'continue' edge, if any
        """
        self.fScore = fScore
        self.aN2C   = aN2C
        self.awC2C  = awC2C
        self.bAlive = True
        self.liEdgeTodo = list(liEdge)   # index of edges to explore in aABSorted
        self.tPast      = tuple()  # sorted tuple: path taken so far of edge index
    
    def getTodoList(self):
        return self.liEdgeTodo
    
    def update(self, fScore, aN2C, awC2C, iEdge, tPast):
        """
        we apply this edge and get this new clustering state
        """
        self.fScore = fScore
        self.aN2C   = aN2C
        self.awC2C  = awC2C
        self.liEdgeTodo.remove(iEdge)
        self.tPast = tPast
        # we can probably delete previous path from path dic...
        self.dUnseenPath[tPast] = False

    def extendPath(self, iEdge):
        l = list(self.tPast)
#         l.append(iEdge)
#         l.sort()
        bisect.insort(l, iEdge)
        return tuple(l)
    
    @classmethod        
    def isNewPath(cls, tPast):
        return cls.dUnseenPath[tPast]


# -----------------------------------------------------------------------------
def do_byLevel_ByPage_bySimil(D, lltEdgeAB, lltNodeClustersIdx, lltNodeBB, nBest
                              , aE, ADistr
                              , options
                              , ClustererClass
                              , lsFilename
                              , bByLength=False
                              , bVerbose=False
                              , bBestOnly=False
                              ):
    """
    for historical, beam, hypothesis clustering
    aE  is the edge data : bOk, Y_GT, Y, iEdgeType, fProba, fLength

    #TODO: cut in smaller pieces...
    """    
    if lltNodeBB is None:
        lltNodeBB = [None] * len(lltNodeClustersIdx)  # ...for the zip ... :-/
    assert len(lltNodeClustersIdx) == len(lltNodeBB)
    
    # by level, then by page
    for lvl in range(D):
        chronoOn(lvl)
        traceln(" ----- level %d -----" % lvl)
        
        lARI = []
        ldOkErrMiss = [ { fSimil:(0,0,0) for fSimil in lfSimil } for _ in range(nBest+1)] # last one will be accumulated best ones
        lsRpt = []

        i_edge_start = 0  # first edge belonging to current page
        for iPage, (ltEdgeAB, ltNodeClustersIdx, ltNodeBB, sFilename) in enumerate(zip(lltEdgeAB, lltNodeClustersIdx, lltNodeBB, lsFilename)):
            
            aAB = np.array(ltEdgeAB, dtype=np.int)
            nEdge, iTwo = aAB.shape
            assert iTwo == 2
            
            #aY  = A     [i_edge_start:i_edge_start+nEdge, 2]    # Y
            aYp = ADistr[i_edge_start:i_edge_start+nEdge, :]    # Y_proba
            aEd = aE    [i_edge_start:i_edge_start+nEdge, :]
            nn = len(ltNodeClustersIdx) # number of nodes on this page
                
            if options.bVerbose:
                traceln("   %d 'continue' edges  (%s)" % (np.count_nonzero(aYp[:,0]>0.5), sFilename))
            
            # convert GT cluster to our data structure
            lClusterPerNode = [_l[lvl] for _l in ltNodeClustersIdx]     # [clusterIdx, clusterIdx, ...]
            nbCluster = max(lClusterPerNode)+1
            GTclusters = ClusterList([Cluster() for _ in range(nbCluster)])
            for i,iCluster in enumerate(lClusterPerNode): GTclusters[iCluster].add(i)

            # *** clustering !! 
            if options.bOrig:
                doer = OrigClusterer(nn, D, aAB, aYp)
                clusters = doer.cluster(lvl)
                lt = [(None, clusters)]
            else:
                if ltNodeBB is None:
                    doer = ClustererClass(nn, D, aEd, aAB, aYp
                                          , nBest=nBest, bByLength=bByLength)
                else:
                    assert len(ltNodeBB) == nn
                    aNBB = np.array(ltNodeBB)
                    doer = ClustererClass(nn, D, aEd, aAB, aYp, aNBB
                                          , nBest=nBest, bByLength=bByLength)

                lt = doer.clusterNBest(lvl)
                if options.bVerbose:
                    traceln("...%d states explored" % doer.getNumberOfExploredState())

            GTScore = doer.clusterscore(GTclusters, lvl)
                
            lARI.append(evalAdjustedRandScore(lt[0][1], GTclusters))
                
            dBestF1BySimil = { fSimil:-1        for fSimil in lfSimil } # to select best beam per simil accross files
            dBestNNBySimil = { fSimil:(0, 0, 0) for fSimil in lfSimil } # to select best beam per simil accross files
            dtScoreBySol = {}
            for iBeam, (score, clusters) in enumerate(lt):
                dtScoreBySol[iBeam] = (-1111, GTScore) if score is None else (score, GTScore)
                _score = doer.clusterscore(clusters, lvl)
#                 if abs(score - _score) > 0.1 and abs(score-_score)/max(abs(score),abs(_score))>0.05:
#                     traceln("is GT score fun wrong?: score=%.2f  score_fun=%.2f" % (score, _score))
#                 assert abs(score - _score) < 0.1 or (abs(score-_score)/max(0.0001,abs(score))<0.01), (score, _score)

                if options.bVerbose and score is not None:          # beam clusterer
                    s = "%s  score = %.2f      (GT score = %.2f)" % ( "" if options.bOrig else "* beam %2d"%iBeam
                                                                    , score
                                                                    , doer.clusterscore(GTclusters, lvl))
                    lsRpt.append(s)
#                 elif iBeam == 0:
#                     traceln("\t\t%s  score = %.2f      (GT score = %.2f)" % ( "" if options.bOrig else "* beam %2d"%iBeam
#                                                                     , score
#                                                                     , doer.clusterscore(GTclusters, lvl)))
                    
                # metric by simil
                for fSimil in lfSimil:
                    _nOk, _nErr, _nMiss = evalHungarian(clusters, GTclusters, fSimil, jaccard_distance)
                    _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)

#                     lsRpt.append("@simil %.2f   P %6.2f  R %6.2f  F1 %6.2f   ok=%6d  err=%6d  miss=%6d" %(
                    if bVerbose: traceln("%04d sol=%02d lvl=%02d @simil %.2f   P %6.2f  R %6.2f  F1 %6.2f   ok=%6d  err=%6d  miss=%6d     Ajusted_Rand_Index=%.2f" %(iPage, iBeam, lvl
                        , fSimil
                        , _fP, _fR, _fF
                        , _nOk, _nErr, _nMiss
                        , evalAdjustedRandScore(clusters, GTclusters) # bug with --orig
                        ))
                    
                    if _fF > dBestF1BySimil[fSimil]: 
                        dBestNNBySimil[fSimil] = _nOk, _nErr, _nMiss
                        dBestF1BySimil[fSimil] = _fF

                    # global count
                    nOk, nErr, nMiss = ldOkErrMiss[iBeam][fSimil]
                    nOk   += _nOk
                    nErr  += _nErr
                    nMiss += _nMiss
                    ldOkErrMiss[iBeam][fSimil] = (nOk, nErr, nMiss)
            # end for iBeam ... 
             
            # accumulate best beam output      
            for fSimil in lfSimil:
                nOk, nErr, nMiss    = ldOkErrMiss[-1][fSimil]
                _nOk, _nErr, _nMiss = dBestNNBySimil [fSimil]
                nOk    += _nOk
                nErr   += _nErr
                nMiss  += _nMiss
                ldOkErrMiss[-1][fSimil] = (nOk, nErr, nMiss) 
                        
            
            # next page
            i_edge_start += nEdge

            lSummary = []
            # for iBeam in range(nBest+1 if nBest > 1 else nBest):
            liToShow = ([0, nBest] if nBest > 1 else [0]) if bBestOnly else range(nBest+1 if nBest > 1 else nBest)
            sPref = "   "
            
            for iBeam in liToShow:
                if score is not None: 
                    if iBeam < nBest:
                        lSummary.append("* solution %d"%iBeam \
                                        + "\t(score=%.2f   GT=%.2f"%dtScoreBySol[iBeam] )
                        sPref = "   "
                    else:
                        lSummary.append("* best of solutions accumulated")
                        sPref = "  ."
                        
                for fSimil in lfSimil:
                    nOk, nErr, nMiss = ldOkErrMiss[iBeam][fSimil]
                    fP, fR, fF = computePRF(nOk, nErr, nMiss)
                    sLine = "%slvl_%d  @simil %.2f   P %6.2f  R %6.2f  F1 %6.2f " % (sPref, lvl
                                                                                                  , fSimil
                                                                                                  , fP, fR, fF ) \
                            + "        "                                                                    \
                            + "ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss)
                    lSummary.append(sLine)
            #END OF for iBeam in range(nBest+1):
            
        #END OF for i, (ltEdgeAB, ltNodeClustersIdx)            
                    
#         sRpt = "\n".join(lsRpt) + "\n\n" + "\n".join(lSummary)
    
        traceln("\n".join(lSummary))
        if len(lARI) > 1:
            traceln("ARI average: %.2f   (± %.3f)  (#%d)" % (average(lARI), stddev(lARI), len(lARI)))
        else:
            traceln("ARI : %.2f" % average(lARI))
        traceln(" [%.1fs]"%chronoOff(lvl))
        try:
            with open(os.path.join(doer.plot_get_dir(), "eval.txt"), "a") as fd:
                fd.write("\n".join(lSummary))
        except:
            pass
    # END OF     for lvl in range(D):

    return i_edge_start


# ------------------------------------------------------------------------------
def do_ByPage_byLevel_bySimil(D, lltEdgeAB, lltNodeClustersIdx, lltNodeBB, nBest, aE, ADistr, ClustererClass
                              , lsFilename
                              , bByLength=False
                              , bVerbose=False
                              , bBestOnly=False
                              ):
    """
    for the hierarchical clustering..
    """   
    dldOkErrMiss = {lvl : [ { fSimil:(0,0,0) for fSimil in lfSimil } for _ in range(nBest+1)] for lvl in range(D)}# last one will be accumulated best ones
    lsRpt = []

    chronoOn()
    i_edge_start = 0  # first edge belonging to current page
    llARI = [list() for _l in range(D)] # list of ARI per level
    for iPage, (ltEdgeAB, ltNodeClustersIdx, sFilename) in enumerate(zip(lltEdgeAB, lltNodeClustersIdx, lsFilename)):
        if bVerbose: chronoOn(iPage)
        ltNodeBB = None if lltNodeBB is None else lltNodeBB[iPage]
        aAB = np.array(ltEdgeAB)
        nEdge, iTwo = aAB.shape
        assert iTwo == 2
        
        #aY  = A     [i_edge_start:i_edge_start+nEdge, 2]    # Y
        aYp = ADistr[i_edge_start:i_edge_start+nEdge, :]    # Y_proba
        aEd = aE    [i_edge_start:i_edge_start+nEdge, :]
        
        nn = len(ltNodeClustersIdx) # number of nodes on this page
        if bVerbose:
            traceln("   %d 'continue' edges  (%s) " % (np.count_nonzero(aYp[:,0]>0.5), sFilename))
        
        # convert GT cluster to our data structure, per level
        lGTclusters = list()
        for lvl in range(D):
            lClusterPerNode = [_l[lvl] for _l in ltNodeClustersIdx]     # [clusterIdx, clusterIdx, ...]
            nbCluster = max(lClusterPerNode)+1
            GTclusters = ClusterList([Cluster() for _ in range(nbCluster)])
            for i,iCluster in enumerate(lClusterPerNode): GTclusters[iCluster].add(i)
            lGTclusters.append(GTclusters)
            
        # *** clustering !! 
        if ltNodeBB is None:
            doer = ClustererClass(nn, D, aEd, aAB, aYp, nBest=nBest, bByLength=bByLength)
        else:
            assert len(ltNodeBB) == nn
            aNBB = np.array(ltNodeBB)
            doer = ClustererClass(nn, D, aEd, aAB, aYp, aNBB, nBest=nBest, bByLength=bByLength)
        llt = doer.clusterNBest()
        if bVerbose: traceln("...%d states explored" % doer.getNumberOfExploredState())

        for lvl in range(D):
            llARI[lvl].append(evalAdjustedRandScore(llt[0][1][lvl], lGTclusters[lvl]))
            
        ddBestF1BySimil = {lvl:{ fSimil:-1        for fSimil in lfSimil } for lvl in range(D)}# to select best beam per simil accross files
        ddBestNNBySimil = {lvl:{ fSimil:(0, 0, 0) for fSimil in lfSimil } for lvl in range(D)}# to select best beam per simil accross files
        dtScoreByBeamByLvl = defaultdict(dict)
        for iBeam, (score, lclusters) in enumerate(llt):
            #if bVerbose and score is not None:          # beam clusterer
#             if score is not None:          # beam clusterer
#                 s = "%s  score = %.2f      (GT score = %.2f)" % ( "* solution %2d"%iBeam
#                                                                 , score
#                                                                 , doer.clusterscore(lGTclusters))
#                 lsRpt.append(s)
            # metric by simil
            for lvl in range(D):
                if lvl > 0: lsRpt.append("")
                dtScoreByBeamByLvl[iBeam][lvl] = (  doer.clusterscore(lclusters  [lvl], lvl)
                                                  , doer.clusterscore(lGTclusters[lvl], lvl) )         
                       
                for fSimil in lfSimil:
                    _nOk, _nErr, _nMiss = evalHungarian(lclusters[lvl], lGTclusters[lvl], fSimil, jaccard_distance)
                    _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
    
#                     lsRpt.append("@simil %.2f   P %6.2f  R %6.2f  F1 %6.2f   ok=%6d  err=%6d  miss=%6d" %(
#                     traceln("@simil %.2f   P %6.2f  R %6.2f  F1 %6.2f   ok=%6d  err=%6d  miss=%6d" %(
                    if bVerbose: traceln("%04d sol=%02d lvl=%02d @simil %.2f   P %6.2f  R %6.2f  F1 %6.2f   ok=%6d  err=%6d  miss=%6d     Ajusted_Rand_Index=%.2f" %(iPage, iBeam, lvl
                        , fSimil
                        , _fP, _fR, _fF
                        , _nOk, _nErr, _nMiss
                        , evalAdjustedRandScore(lclusters[lvl], lGTclusters[lvl])
                        ))
                    
                    if _fF > ddBestF1BySimil[lvl][fSimil]: 
                        ddBestNNBySimil[lvl][fSimil] = _nOk, _nErr, _nMiss
                        ddBestF1BySimil[lvl][fSimil] = _fF
    
                    # global count
                    nOk, nErr, nMiss = dldOkErrMiss[lvl][iBeam][fSimil]
                    nOk   += _nOk
                    nErr  += _nErr
                    nMiss += _nMiss
                    dldOkErrMiss[lvl][iBeam][fSimil] = (nOk, nErr, nMiss)
        # end for iBeam ...  
        
        # accumulate best beam output      
        for lvl in range(D):
            for fSimil in lfSimil:
                nOk, nErr, nMiss    = dldOkErrMiss[lvl][-1][fSimil]
                _nOk, _nErr, _nMiss = ddBestNNBySimil[lvl] [fSimil]
                nOk    += _nOk
                nErr   += _nErr
                nMiss  += _nMiss
                dldOkErrMiss[lvl][-1][fSimil] = (nOk, nErr, nMiss) 
                    
        if bVerbose: traceln("  ... page %d done [%.1fs]" % (iPage+1, chronoOff(iPage)))

        # next page
        i_edge_start += nEdge
        lSummary = []
        liToShow = ([0, nBest] if nBest > 1 else [0]) if bBestOnly else range(nBest+1 if nBest > 1 else nBest)
        sPref = "   "
        for iBeam in liToShow:
            for lvl in range(D):
                if score is not None:
                    if lvl == 0: 
                        if iBeam < nBest:
                            lSummary.append("* solution %d" % iBeam)
                            sPref = "   "
                        else:
                            lSummary.append("* best of solutions accumulated")
                            sPref = "  ."
                    if iBeam < nBest: 
                        lSummary.append("\t(score=%.2f  GT=%.2f)" % dtScoreByBeamByLvl[iBeam][lvl])
                    elif lvl > 0:
                        lSummary.append("")
                
                for fSimil in lfSimil:
                    nOk, nErr, nMiss = dldOkErrMiss[lvl][iBeam][fSimil]
                    fP, fR, fF = computePRF(nOk, nErr, nMiss)
                    sLine = "%slvl_%d  @simil %.2f   P %6.2f  R %6.2f  F1 %6.2f " % (sPref, lvl
                                                                                                  , fSimil
                                                                                                  , fP, fR, fF ) \
                            + "        "                                                                    \
                            + "ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss)
                    lSummary.append(sLine)
                
#     if bVerbose:
#         sRpt = "\n".join(lsRpt) + "\n\n" + "\n".join(lSummary)
#     else:
#         sRpt = "\n".join(lSummary)

    traceln("\n".join(lSummary))
    for lvl in range(D):
        lARI = llARI[lvl]
        if len(lARI) > 1:
            traceln("lvl_%d  ARI average: %.2f   (± %.3f)  (#%d)" % (lvl, average(lARI), stddev(lARI), len(lARI)))
        else:
            traceln("lvl_%d  ARI : %.2f" % (lvl, average(lARI)))
            
    traceln(" [%.1fs]"%chronoOff())
    
    try:
        with open(os.path.join(doer.plot_get_dir(), "eval.txt"), "a") as fd:
            fd.write("\n".join(lSummary))
    except:
        pass

    return i_edge_start

# -----------------------------------------------------------------------------

def test_Hier_1():
    nn = 4
    D = 2
    aAB = np.array([ [0,1], [0,2] ]) 
    ne = aAB.shape[0]
    Y_p = np.array([  [0.7, 0.1, 0.2]
                    , [0.1, 0.1, 0.8] 
                    ])
    assert Y_p.shape  == (ne, D+1)
    
    doer = HierarchicalHypothesisClusterer(nn, D, aAB, Y_p, nBest=2)
    ltScorellCluster = doer.clusterNBest()
    return ltScorellCluster

def test_Hier_2():
    nBest = 1
    ltNodeClustersIdx = [  [0, 0]
                         , [0, 0] 
                         , [0, 1]   # node 2 is in cluster0 at level 0, cluster 1 at level 1 
                         , [1, 2] 
                         ]
    D = 2
    laAB = [ [0,1], [0,2] ] 
    Y_p = np.array([  [0.7, 0.1, 0.2]
                    , [0.1, 0.1, 0.8] 
                    ])
    
    ADistr = Y_p
    ret = do_ByPage_byLevel_bySimil(D, [laAB], [ltNodeClustersIdx]
                         , nBest, ADistr, HierarchicalHypothesisClusterer
                         , bVerbose=False)
# test_Hier_2()
# sys.exit(1)

    
def test_BeamClusterer_asContinue_1():
    print("*********  test_BeamClusterer_asContinue_1")
    nn = 4
    D = 1
    aAB = np.array([ [0,1], [1,2], [2,3], [3,0] ]) # a square
    ne = aAB.shape[0]
    Y_proba = np.zeros(shape=aAB.shape)
    Y_proba[:,0] = [0.01, 0.12, 0.23, 0.30]
    Y_proba[:,1] = 1.0 - Y_proba[:,0]
    
    doer = BeamClusterer(nn, D, aAB, Y_proba)
    
    lvl = 0
    aN2C = np.arange(nn)
    nC = aN2C.max() + 1
    
    awC2C = np.zeros(shape=(nC,nC))
    for k in range(ne):
        iA, iB = aAB[k]
        cA, cB = aN2C[iA], aN2C[iB]
        
        p = Y_proba[k,0] - 0.5
        awC2C[cA, cB] += p
        if cB != cA: awC2C[cB, cA] += p
    
    print(repr(aN2C))
    print(repr(awC2C), "  score=", doer._score(awC2C))
    print("iA=", iA, "  iB=", iB)
    _jA, h_aN2C, h_awC2C, _fInter = doer._asContinue(iA, iB, aN2C, awC2C)
    print(repr(h_aN2C))
    print(repr(h_awC2C), "  score=", doer._score(h_awC2C))
    assert np.allclose(h_aN2C, [0 ,1 ,2 ,0])
    assert np.allclose(h_awC2C, np.array([ [-0.2 , -0.49, -0.27]
                                         , [-0.49,  0.  , -0.38]
                                         , [-0.27, -0.38,  0.  ]]))
    
    iA = 3
    iB = 1
    print("iA=", iA, "  iB=", iB)
    _jA, h_aN2C, h_awC2C, _fInter = doer._asContinue(iA, iB, h_aN2C, h_awC2C)
    print(repr(h_aN2C))
    print(repr(h_awC2C), "  score=", doer._score(h_awC2C))
    assert np.allclose(h_aN2C, [0 ,0 ,1 ,0])
    assert np.allclose(h_awC2C, np.array([ [-0.69, -0.65]
                                         , [-0.65,  0.  ] ]))
    

    

def test_BeamClusterer_asContinue_2():
    print("*********  test_BeamClusterer_asContinue_2")
    nn = 4
    D = 1
    aAB = np.array([ [0,1], [1,2], [2,3], [3,0] ]) # a square
    ne = aAB.shape[0]
    Y_proba = np.zeros(shape=aAB.shape)
    Y_proba[:,0] = [0.01, 0.88, 0.23, 0.30]
    Y_proba[:,1] = 1.0 - Y_proba[:,0]
    
    doer = BeamClusterer(nn, D, aAB, Y_proba)
    
    lvl = 0
    aN2C = np.arange(nn)
    nC = aN2C.max() + 1
    
    awC2C = np.zeros(shape=(nC,nC))
    for k in range(ne):
        iA, iB = aAB[k]
        cA, cB = aN2C[iA], aN2C[iB]
        
        p = Y_proba[k,0] - 0.5
        awC2C[cA, cB] += p
        if cB != cA: awC2C[cB, cA] += p
    
    print(repr(aN2C))
    print(repr(awC2C), "  score=", doer._score(awC2C))
    iA, iB = 1, 2
    print("iA=", iA, "  iB=", iB)
    _jA, h_aN2C, h_awC2C, _fInter = doer._asContinue(iA, iB, aN2C, awC2C)
    print(repr(h_aN2C))
    print(repr(h_awC2C), "  score=", doer._score(h_awC2C))
    assert np.allclose(h_aN2C, [0 ,1 ,1 ,2])
    assert np.allclose(h_awC2C, np.array([ [ 0.  , -0.49, -0.2]
                                         , [-0.49,  0.38, -0.27]
                                         , [-0.2 , -0.27,  0.  ]]))
    

if 0:
    test_BeamClusterer_asContinue_1()
    test_BeamClusterer_asContinue_2()
    sys.exit(0)

# -----------------------------------------------------------------------------
def oracle(aE, ADistr, nb_edge, D):
    """
    return edge proba distribution reflecting GT
    """
    assert ADistr.shape == (nb_edge, D+1)
    Ygt = np.rint(aE[:,1])
    assert Ygt.shape == (nb_edge, )
    
    Ygtint = np.rint(Ygt).astype(np.int)
    assert Ygtint.shape == (nb_edge, ), Ygtint.shape
    assert min(Ygtint) == 0
    assert max(Ygtint) == D
    newADistr = np.zeros(shape=ADistr.shape, dtype=ADistr.dtype)
    newADistr[np.arange(nb_edge), Ygtint] = 1.0
    traceln("       edge accuracy= %.2f %%" % (100 * np.count_nonzero( np.argmax(   ADistr, axis=1) == Ygt ) / nb_edge) )
    traceln("oracle edge accuracy= %.2f %%" % (100 * np.count_nonzero( np.argmax(newADistr, axis=1) == Ygt ) / nb_edge) )
    
    return newADistr

    
# ------------------------------------------------------------------------------
if __name__ == "__main__":
     
    parser = OptionParser(usage="PKL_FILE+", version=0.1)
     
    parser.add_option("--verbose"    , dest='bVerbose'    ,  action="store_true")
#     parser.add_option("--best_only"  , dest='bBestOnly'   ,  action="store_true"
#                       , help="show only best solution")
    parser.add_option("--oracle"  , dest='bOracle'   ,  action="store_true"
                      , help="Use GT edge label (1 hot encoding as proba distribution.)")
    parser.add_option("--show_all"  , dest='bShowAll'   ,  action="store_true"
                      , help="show all solutions")
    parser.add_option("--plot"  , dest='bPlot'   ,  action="store_true"
                      , help="plot the solution(s)")
    parser.add_option("--video" , dest='bVideo'  ,  action="store_true"
                      , help="video of the exploration of the solution(s)")
    parser.add_option("--bb"        , dest='bBB',  action="store_true"
                      , help="Use bounding box of nodes to force geometrically non-overlapping clusters")    
    parser.add_option("--local"    , dest='bLocalExplore' ,  action="store_true"
                      , help="Edge are explored locally")
    parser.add_option("--length"    , dest='bByLength'    ,  action="store_true"
                      , help="Edges are sorted by length instead of by confidence")
    parser.add_option("--orig"       , dest='bOrig'       ,  action="store_true"
                      , help="historical method")
    parser.add_option("--hypothesis_size"   , dest='iHypoSize'   
                      ,  action="store", type="int"
                      , help="number of hypothesis")
    parser.add_option("--div_hypothesis"    , dest='bDivHypo',  action="store_true"
                      , help="Generate more diverse hypothesis")
    parser.add_option("--hierarchical_hypothesis_size"   , dest='iHierHypoSize'   
                      ,  action="store", type="int"
                      , help="number of hypothesis for hierarchical clustering")
    parser.add_option("--beam_size"         , dest='iBeamSize'   
                      ,  action="store", type="int"
                      , help="beam size")
     
    (options, args) = parser.parse_args()
     
    if [1 if options.bOrig else None
        , options.iBeamSize
        , options.iHypoSize
        , options.iHierHypoSize].count(None) != 3:
        traceln("ERROR: specify one method from: --orig --hypothesis_size --beam_size --hierarchical_hypothesis_size")
        traceln(parser.print_help())
        sys.exit(1)
                 
    if options.bOrig:
        nBest = 0
        traceln("* Historical clustering")
        options.bByLength = False
        ClustererClass = OrigClusterer
    elif bool(options.iHypoSize):
        nBest = options.iHypoSize
        if options.bDivHypo:
            ClustererClass = DiverseHypothesisClusterer
            traceln("** Diverse hypothesis clustering |  # hypothesis=%d"%nBest)
            traceln(" THIS IS CRAPPY STUFF WHY DO YOU USE IT??")
        else:
            if options.bBB:
                if options.bLocalExplore:
                    ClustererClass = LocalHypothesisClustererBB
                    traceln("** Hypothesis clustering, local mode, using BB |  # hypothesis=%d"%nBest)
                else:
                    ClustererClass =      HypothesisClustererBB
                    traceln("** Hypothesis clustering, using BB |  # hypothesis=%d"%nBest)
            else:
                if options.bLocalExplore:
                    ClustererClass = LocalHypothesisClusterer
                    traceln("** Hypothesis clustering, local mode |  # hypothesis=%d"%nBest)
                else:
                    ClustererClass =      HypothesisClusterer
                    traceln("** Hypothesis clustering |  # hypothesis=%d"%nBest)
                    
                if options.bPlot or options.bVideo:
                    ClustererClass =      HypothesisClustererBB \
                            if ClustererClass == HypothesisClusterer \
                            else     LocalHypothesisClustererBB
                    traceln("** plot => using %s (but authorizing overlap, and no reduction factor)" % ClustererClass.__name__)
                    ClustererClass.bForbidOverlap            = False
                    ClustererClass.fHeightReductionFactor    = 1
                    
    elif bool(options.iHierHypoSize):
        nBest = options.iHierHypoSize
        if options.bBB:
            if options.bLocalExplore:
                ClustererClass = LocalHierarchicalHypothesisClustererBB
                traceln("** Hierarchical Hypothesis clustering, local mode, using BB |  # hypothesis=%d"%nBest)
            else:
                ClustererClass =      HierarchicalHypothesisClustererBB
                traceln("** Hierarchical Hypothesis clustering, using BB |  # hypothesis=%d"%nBest)
        else:
            if options.bLocalExplore:
                ClustererClass = LocalHierarchicalHypothesisClusterer
                traceln("** Hierarchical Hypothesis clustering, local mode |  # hypothesis=%d"%nBest)
            else:
                ClustererClass =      HierarchicalHypothesisClusterer
                traceln("** Hierarchical Hypothesis clustering |  # hypothesis=%d"%nBest)
            
            if options.bPlot or options.bVideo:
                traceln("** plot => using %s (but authorizing overlap, and no reduction factor)" % ClustererClass.__name__)
                ClustererClass =      HierarchicalHypothesisClustererBB \
                        if ClustererClass == HierarchicalHypothesisClusterer \
                        else     LocalHierarchicalHypothesisClustererBB
                ClustererClass.bForbidOverlap            = False
                ClustererClass.fHeightReductionFactor    = 1
    else:
        nBest = options.iBeamSize
        ClustererClass = BeamClusterer
        traceln("*** Beam clustering | # beam = %d"%nBest)
        
    if options.bByLength: traceln("Edges will be sorted by length")
    if options.bBB      : 
        traceln("Clusters must not overlap each other, geometrically")
        assert HypothesisClustererBB.bForbidOverlap
    if options.bShowAll : traceln("Showing all solutions")
    if options.bPlot    : traceln("Plotting solution(s) in ", ClustererClass.plot_get_dir())
    if options.bVideo   : traceln("Video of solution(s) in ", ClustererClass.plot_get_dir())

    if options.bPlot : 
        assert ImageDraw , "pillow missing"
        iPLOT = 1
    if options.bVideo: 
        assert cv2       , "opencv missing"
        iPLOT = 2
    
    traceln("\tclass = %s" % ClustererClass.__name__)
        
    chronoOn("main")
    for sPklFile in args:
        traceln("=========== ", sPklFile)
        tt = gzip_pickle_load(sPklFile)
        if len(tt) == 10:
            ( nb_edge                   # global: nb edges
            , nb_label                  # global: nb edge labels (e.g. 3 for cont, brk_0, brk_1)
            , aE                        # global: aE is the edge data : bOk, Y_GT, Y, iEdgeType, fProba, fLength
            , ADistr                    # global: array: probability distribution per edge
            , D                         # global: depth (same as nb_label-1, e.g. 2
            , lltEdgeAB                 # per page: [(edge.A.idx, edge.B.idx), ...]
            , lltNodeClustersIdx        # per page, per node: [cluster_idx, ...]   of length D
            , lltNodeBB                 # par page, per node: (x1, y1, x2, y2)
            , lsFilename
            , sLog                      # global: computation log
            ) = tt
        else:
            assert not options.bBB, "Cannot do --bb because the pickle does not contain node BB"
            ( nb_edge                   # global: nb edges
            , nb_label                  # global: nb edge labels (e.g. 3 for cont, brk_0, brk_1)
            , aE                        # global: aE is the edge data : bOk, Y_GT, Y, iEdgeType, fProba, fLength
            , ADistr                    # global: array: probability distribution per edge
            , D                         # global: depth (same as nb_label-1, e.g. 2
            , lltEdgeAB                 # per page: [(edge.A.idx, edge.B.idx), ...]
            , lltNodeClustersIdx        # per page, per node: [cluster_idx, ...]   of length D
            , lsFilename
            , sLog                      # global: computation log
            ) = tt
            lltNodeBB = None
            assert type(lsFilename) == type(list()), "Recreate the pkl. Probably it was created before Jan 28, 2021 and do not include the filenames"
        
        if options.bOracle:
            traceln("Oracle: using GT to set edge prediction as a 'Dirac function'.")
            ADistr = oracle(aE, ADistr, nb_edge, D)
            
        if ClustererClass.__name__.endswith("BB"): 
            assert lltNodeBB is not None, "BB info missing from pickle file"
            traceln("\t%s.fWidthReductionFactor  = %s" % (ClustererClass.__name__, ClustererClass.fWidthReductionFactor))
            traceln("\t%s.fHeightReductionFactor = %s" % (ClustererClass.__name__, ClustererClass.fHeightReductionFactor))
            traceln("\t%s.bForbidOverlap         = %s" % (ClustererClass.__name__, ClustererClass.bForbidOverlap))
        else:
            lltNodeBB = None
            
        traceln("- %d labels for depth of %d" % (nb_label, D))
        assert nb_label == (1+D)
 
        nn = sum(len(_l) for _l in lltNodeClustersIdx) # number of nodes
         
        traceln("- %d pages totalizing %d edges  %d nodes " % (len(lltEdgeAB)
                                                               , nb_edge
                                                               , nn))
        assert len(lltNodeClustersIdx) == len(lltEdgeAB)
         
        if bool(options.iHierHypoSize):
            i_edge = do_ByPage_byLevel_bySimil(D, lltEdgeAB, lltNodeClustersIdx
                                               , lltNodeBB
                                             , nBest, aE, ADistr
                                             , ClustererClass
                                             , lsFilename
                                             , bByLength=options.bByLength
                                             , bVerbose=options.bVerbose
                                             , bBestOnly=not(options.bShowAll)
                                             )
        else:
            i_edge = do_byLevel_ByPage_bySimil(D, lltEdgeAB, lltNodeClustersIdx
                                               , lltNodeBB
                                             , nBest, aE, ADistr
                                             , options
                                             , ClustererClass
                                             , lsFilename
                                             , bByLength=options.bByLength
                                             , bVerbose=options.bVerbose
                                             , bBestOnly=not(options.bShowAll)
                                             )
                 
        assert i_edge == aE.shape[0]  # total number of edges
             
    traceln("Done  [%.1fs]" % chronoOff("main"))
        
        