# coding: utf8

'''
This code is about the graph we build - edges and nodes   (nodes are called Blocks)

JL Meunier
March 3rd, 2016
2019
2020


Copyright Xerox 2016
Copyright NLE 2020

'''
import copy
import math 

import numpy as np

import shapely.geometry as geom
import shapely.ops as ops
from rtree import index

from common.trace import traceln
from util.Shape import ShapeLoader

DEBUG=0
#DEBUG=1

# --- Edge CLASSE -----------------------------------------
class Edge:
    iGKNN = None     # in --gknn mode, this is K
        
    def __init__(self, A, B):
        """
        An edge from A to B
        """
        self.A = A
        self.B = B
        
    def __str__(self):
        n = 15  #show only the n leading characters of each node
        if True:
            return "Edge %s p%d-p%d %s --> %s" %(self.__class__, self.A.pnum, self.B.pnum, self.A.getText(n), self.B.getText(n))
        else:
            return "Edge %s p%d-p%d %s -->\n\t %s" %(self.__class__, self.A.pnum, self.B.pnum, self.A, self.B)

    def getCoords(self):
        """
        return the coordinates of the edge, if applicable
        """
        return None
    
    def revertDirection(self):
        """
        revert the direction of the edge
        """
        self.A, self.B = self.B, self.A
    

    def computeOverlap(self):
        """
        compute the overlap between the two nodes
        return 0 or a positive number in case of overlap
        """
        return 0
    
    def computeBB(self):
        """
        compute the  BB between the two nodes
        return x,y,w,h
        """
        x1 = min(self.A.x1, self.B.x1)
        x2 = max(self.A.x2, self.B.x2)
        x1, x2 = min(x1, x2), max(x1, x2)
        
        y1 = min(self.A.y1, self.B.y1)
        y2 = max(self.A.y2, self.B.y2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        return x1, y1, abs(x2-x1), abs(y2-y1)
    
    def computeOverlapBB(self):
        """
        compute the overlap BB between the two nodes
        return x,y,w,h
        """
        raise Exception("Internal Error: method to be specialized")

    def computeOverlapPosition(self):
        """
        compute the overlap between the two nodes and its position 
            relative to each node.
        The overlap is a length
        The position is a number in [-1, +1] relative to the center of a node.
            -1 denote left or top, +1 denotes right or bottom.
            The position is the position of the center of overlap.
        return a tuple: (overlap, pos_on_A, pos_on_B)
        """
        return 0, 0, 0
    
    # ------------------------------------------------------------------------------------------------------------------------------------        
    #specific code for the CRF graph
    def computeEdges(cls, lPrevPageEdgeBlk, lPageBlk, iGraphMode, bShortOnly=False):
        """
        we will compute the edge between those nodes, some being on previous "page", some on "current" page.
        
        if bShortOnly, we filter intra-page edge and keep only "short" ones
        """
        from . import Block

        if iGraphMode == 6: # k nearest neighbors
            lAllEdge = Block.Block.findKNNEdges(lPageBlk, Edge.iGKNN)
            traceln("\t\tEDGES: %4d reflecting %d nearest neighbors" % (  len(lAllEdge)
                                                                        , Edge.iGKNN))
        elif iGraphMode == 3: # DELAUNAY
            lAllEdge = EdgeInterCentroid.findPageDelaunayEdges(lPageBlk)
            traceln("\t\tEDGES: %4d Delaunay" % len(lAllEdge))
        else:
            lAllEdge = list()
            
            #--- horizontal and vertical neighbors
            lHEdge, lVEdge = Block.Block.findPageNeighborEdges(lPageBlk, bShortOnly, iGraphMode=iGraphMode)
            lAllEdge.extend(lHEdge)
            lAllEdge.extend(lVEdge)
            if DEBUG: 
                cls.dbgStorePolyLine("neighbors", lHEdge)
                cls.dbgStorePolyLine("neighbors", lVEdge)
            
            #--- Celtic Edges
            lCelticEdge = CelticEdge.findPageCelticEdges(lPageBlk, lAllEdge)
            lAllEdge.extend(lCelticEdge)
            traceln("\t\tEDGES: %4d horizontal   %4d vertical   %4d celtic" % (len(lHEdge), len(lVEdge), len(lCelticEdge)))
                
            
        #--- overlap with previous page
        if lPrevPageEdgeBlk:
            lConseqOverEdge = Block.Block.findConsecPageOverlapEdges(lPrevPageEdgeBlk, lPageBlk)
            lAllEdge.extend(lConseqOverEdge) 
            
        return lAllEdge
    computeEdges = classmethod(computeEdges)

    def dbgStorePolyLine(cls, sAttr, lEdge):
        """
        Store a polyline in the given attribute
        """
        for e in lEdge:
            xA, yA = e.A.getCenter()
            xB, yB = e.B.getCenter()
            ndA = e.A.node
            sPolyLine = "%.1f,%.1f,%.1f,%.1f,%.1f,%.1f"%(xA,yA,xB,yB,xA,yA)
            if ndA.hasProp(sAttr):
                ndA.set(sAttr, ndA.prop(sAttr) + "," + sPolyLine)
            else:
                ndA.set(sAttr,                         sPolyLine)
        return
    dbgStorePolyLine = classmethod(dbgStorePolyLine)


    
    # --- Edge SUB-CLASSES ------------------------------------
class SamePageEdge(Edge):
    def __init__(self, A, B, length, overlap):
        Edge.__init__(self, A, B)
        self.length  = length
        self.overlap = overlap
        self.iou     = 0

    def getCoords(self):
        """
        return the coordinates of the edge, if applicable
        return x1, y1, x2, y2
        """
        return self.A.x2, self.A.y2, self.B.x1, self.B.y1


class CrossPageEdge(Edge): pass    


class CrossMirrorPageEdge(Edge): pass    


class CrossMirrorContinuousPageVerticalEdge(Edge): 
    def __init__(self, A, B, length, overlap):
        Edge.__init__(self, A, B)
        self.length  = length
        self.overlap = overlap
        self.iou = 0


class CelticEdge(SamePageEdge):
    """
    an edge that has no overlap (so cannot link nodes linked by an Horizontal or Vertical edge)
    an edge that links nodes close by from each other
    """
    iViewRange   = 0
    iMaxBySector = 0
        
    def __init__(self, A, B, length):
        SamePageEdge.__init__(self, A, B, length, 0) # 0 is the overlap
            
    def getCoords(self):
        """
        return the coordinates of the edge, if applicable
        return x1, y1, x2, y2
        """

        # *** code not tested!! *** (JLM Dec 2020)
        Ax1, Ay1, Ax2, Ay2 = self.A.getBB()
        Bx1, By1, Bx2, By2 = self.B.getBB()
        
        _x1 = min(Ax2, Bx2)
        _x2 = max(Ax1, Bx1)
        x1 = min(_x1, _x2)
        x2 = max(_x1, _x2)
        
        _y1 = min(Ay2, By2)
        _y2 = max(Ay1, By1)
        y1 = min(_y1, _y2)
        y2 = max(_y1, _y2)
        
        return x1, y1, x2, y2

    @classmethod
    def setViewRange(cls, i):   cls.iViewRange   = i
    @classmethod
    def getViewRange(cls):   return cls.iViewRange
    
    @classmethod
    def setMaxBySector(cls, i): cls.iMaxBySector = i
    @classmethod
    def getMaxBySector(cls): return cls.iMaxBySector
    
    # --- Celtic Edges
    @classmethod
    def distanceDiagonal(cls
                         , Ax1, Ax2, Ay2
                         , B
                         , dInf):
        """
        a diagonal distance between 2 blocks
        
        
        if horizontal or vertical overlap, distance is dInf
        
        Assumption: A.y2 <= B.y2
        
        return slash-distance, backslash distance
        """
        # Ax1, _Ay1, Ax2, Ay2 = A.getBB()
        Bx1, By1, Bx2, By2 = B.getBB()
        assert Ay2 <= By2, "Assumption not respected. Internal error in calling code"
        if Ay2 < By1:      # B is strictly below of A
            if Ax2 < Bx1:        # B is strictly below and on right of A
                    dx, dy = Bx1-Ax2, By1-Ay2
                    return (dInf                    , math.sqrt(dx*dx + dy*dy)) 
            elif Bx2 < Ax1:      # B strictly below and on left of A
                    dx, dy = Ax1-Bx2, By1-Ay2
                    return (math.sqrt(dx*dx + dy*dy), dInf)
        # no need to test if B is above A:
        # since A.y2 <= B.y2, it means B ends below A
        # so if B starts above, there is overl
        return (dInf, dInf)
        
                
    @classmethod
    def findPageCelticEdges_better_true(cls, lPageBlk, lAllEdge):
        """
        Find at most nbMaxCelticNeighbors celtic edges
        , with viewrange=iCelticEdge
        , per block
        , per sector (diagonal down+left to top-right    # called slash
                    , diagonal down+right to top-left)   # called backslash
        """
        # np.uint64  uint64_t   Unsigned integer (0 to 18446744073709551615)
        dInf = 18446744073709551614

        if (cls.iMaxBySector <= 0 
            or not lPageBlk   
            or cls.iViewRange <= 0): return []  # at least this is simple...
        
        # shallow copy before sorting by y2
        _lBlk = copy.copy(lPageBlk)
        _lBlk.sort(key=lambda o: o.y2)  # sorted from top to bottom, considering the bottom of the BB
        
        # indexing
        for i, A in enumerate(_lBlk): 
            A._ii = i

        # now we compute the upper triangular distance matrices for / and \
        N = len(_lBlk)
        aDistSlash     = np.full((N, N), dInf, dtype=np.uint64)
        aDistBackslash = np.full((N, N), dInf, dtype=np.uint64)
        for i, A in enumerate(_lBlk): 
            Ax1, _Ay1, Ax2, Ay2 = A.getBB()
            for j, B in ((_j, _lBlk[_j]) for _j in range(i+1, N)):
                (aDistSlash[i,j], aDistBackslash[i,j]) = cls.distanceDiagonal(Ax1, Ax2, Ay2, B, dInf) 

        # now find N closest for each block, upward and downward
        aCntSlashUp       = np.zeros(shape=(N,), dtype=np.uint32)
        aCntSlashDown     = np.zeros(shape=(N,), dtype=np.uint32)
        aCntBackslashUp   = np.zeros(shape=(N,), dtype=np.uint32)
        aCntBackslashDown = np.zeros(shape=(N,), dtype=np.uint32)
        lSlashEdge, lBackslashEdge = [], []
        
        iMaxBySector = cls.iMaxBySector
        iViewRange   = cls.iViewRange
        Ashape = (N, N)
        for lEdge, aCntUp, aCntDown, aDist in [
              (lSlashEdge    , aCntSlashUp    , aCntSlashDown    , aDistSlash)
            , (lBackslashEdge, aCntBackslashUp, aCntBackslashDown, aDistBackslash)]:
            cnt = 0
            cntMax = 4 * N * iMaxBySector / 2 # When we have this number of edge, we are done
            while True:
                i,j = np.unravel_index(aDist.argmin(), Ashape)
                dst = aDist[i,j]
                if dst > iViewRange   : break  # no short enough distance left
                if dst == dInf        : break  # all distances exploited??
                #ok assert i < j  
                #ok assert _lBlk[i].y2 <= _lBlk[j].y2
                aDist[i,j] = dInf # we do not want to see it again
                if aCntDown[i] < iMaxBySector and aCntUp[j] < iMaxBySector:
                    e = CelticEdge(_lBlk[i], _lBlk[j], dst)
                    lEdge.append(e)
                    cnt += 1
                    if cnt == cntMax: break
                    
                # we count this potential edge, since reflect the currently closest
                # neighbour for both of the blocks
                aCntDown[i] += 1
                aCntUp[j] += 1
                
                # if either node i or j sector is saturated, then we can
                # update the distance matrix to discard other candidates
                # node i is "above" node j
                if aCntDown[i] >= iMaxBySector: aDist[i,:] = dInf
                if aCntUp  [j] >= iMaxBySector: aDist[:,j] = dInf
            
        traceln("\t\t\t Celtic : %d /    %d \\" % (len(lSlashEdge), len(lBackslashEdge)))
        
        # cleaning
        del aCntSlashUp, aCntSlashDown, aCntBackslashUp, aCntBackslashDown
        del aDistSlash, aDistBackslash
        for i, _blk in enumerate(lPageBlk): 
            del _blk._ii
                    
        lSlashEdge.extend(lBackslashEdge)
        return lSlashEdge    
        
    @classmethod
    def findPageCelticEdges_true(cls, lPageBlk, lAllEdge):
        """
        Find at most nbMaxCelticNeighbors celtic edges
        , with viewrange=iCelticEdge
        , per block
        , per sector (down+left, down+right)
        """

        lEdge = []
        if cls.iMaxBySector <= 0 or not lPageBlk: return []
        
        # create viewrange circles
        lO = []
        for _blk in lPageBlk:
            # seems to return 9 points...
            # len(p.buffer(42, resolution=2).simplify(1).exterior.coords))
            # making a simplified circle around the center of the block
            x, y = _blk.getCenter()
            # OK assert _blk.y1 <= _blk.y2
            _blk.oBottomCenter = geom.Point((x, _blk.y2))
            buf =  _blk.oBottomCenter.buffer(cls.iViewRange, resolution=2).simplify(1)
            # now make it half circle, downward
            _x1,_y1,_x2,_y2 = buf.bounds
            # ok assert _y1 <= _y2
            # ok assert _y1 <= y and y <= _y2
            msk = geom.Polygon([(_x1, _blk.y2), (_x1, _y2), (_x2, _y2), (_x2, _blk.y2)])
            
            half_circle = buf.intersection(msk)

            lO.append(half_circle)

        # make an indexed rtree
        idx = index.Index()
        for i, o in enumerate(lO):
            idx.insert(i, o.bounds)
        
        # UGLY
        # compute current adjacency matrix....
        # +1 indicates a classical adjacency relation (e.g. Horizontal edge)
        # -1 indicates a Celtic edge (created by this algo)
        aAdj = np.zeros(shape=(len(lPageBlk), len(lPageBlk)))
        for i, _blk in enumerate(lPageBlk): 
            _blk._ii = i
        for e in lAllEdge:
            i, j = e.A._ii, e.B._ii
            aAdj[i,j] = 1
            aAdj[j,i] = 1
        np.fill_diagonal(aAdj, 1)   # to avoid edge to self
        
        # take each block in turn and find close-by blocks!
        for iBlk, blk in enumerate(lPageBlk):
            x,y = blk.getCenter()
            oTopCenter = geom.Point((x, blk.y1))
            # find N closest blocks (as tuple (<distance>, <blk>)
            ltCloseLeftBlk, ltCloseRightBlk = [], []
            for i in idx.intersection(oTopCenter.bounds):
                # this block in in the other half-circle!
                other = lPageBlk[i]
                if iBlk == other._ii: continue

                # make sure no overlap may exist, by forcing us to be strictly below other
                # ok assert other.y2 <= blk.y1, (str(blk), str(other))
                adj = aAdj[iBlk, other._ii]
                if adj == 0:  # no edge
                    # on left and no overlap
                    if blk.x2 <= other.x1:
                        ltCloseLeftBlk.append((oTopCenter.distance(other.oBottomCenter), other))
                    # on right and no overlap
                    elif other.x2 <= blk.x1:
                        ltCloseRightBlk.append((oTopCenter.distance(other.oBottomCenter), other))
            
            for lt in [ltCloseLeftBlk, ltCloseRightBlk]:
                # cap the length and sort (always sort, do get stable output)
                lt.sort(key=lambda o:o[0])
                lt = lt[:cls.iMaxBySector]
                    
                for d, other in lt:
                    e = CelticEdge(blk, other, d)
                    aAdj[iBlk       , other._ii] = -1
                    aAdj[other._ii  , iBlk     ] = -1
                    lEdge.append(e)

        # cleaning
        del aAdj
        del idx
        for i, _blk in enumerate(lPageBlk): 
            del _blk._ii
            del _blk.oBottomCenter
            
        return lEdge

    @classmethod
    def findPageCelticEdges(cls, lPageBlk, lAllEdge):
        return cls.findPageCelticEdges_better_true(lPageBlk, lAllEdge)


class VirtualEdge(Edge): pass    


# --- SamePageEdge SUB-CLASSES ----------------------------
class HorizontalEdge(SamePageEdge):
    def __init__(self, A, B, length, overlap):
        SamePageEdge.__init__(self, A, B, length, overlap)
        try: 
            self.iou = max(0, self.overlap) / (abs(A.y2-A.y1) + abs(B.y2-B.y1) - abs(self.overlap))
        except ZeroDivisionError:
            self.iou = 0    

    def getCoords(self):
        """
        return the coordinates of the edge, if applicable
        return x1, y1, x2, y2
        """
        x,y,w,h = self.computeOverlapBB()
        ym = y + h/2.0
        return x, ym, x+w, ym

    def computeOverlap(self):
        """
        compute the vertical overlap between the two nodes
        return 0 or a positive number in case of overlap
        """
        return max(0, min(self.A.y2, self.B.y2) - max(self.A.y1, self.B.y1))

    def computeOverlapBB(self):
        """
        compute the overlap BB between the two nodes
        return x,y,w,h
        """
        # X and Width
        x1 = min(self.A.x2, self.B.x2)
        x2 = max(self.A.x1, self.B.x1)
        # x1<x2 means overlapping blocks
        # x = x1 if x1 < x2 else x2
        x = min(x1, x2)
        
        # Y and height 
        y1 = max(self.A.y1, self.B.y1)
        y2 = min(self.A.y2, self.B.y2)
        h = y2 - y1
        assert h >= 0, ("Some invalid horizontal edge BB", h)
        
        return x, y1, abs(x1-x2), h
        

    def computeOverlapPosition(self):
        """
        compute the vertical overlap between the two nodes and its position 
            relative to each node.
        The overlap is a length
        The position is a number in [-1, +1] relative to the center of a node.
            -1 denote left or top, +1 denotes right or bottom.
            The position is the position of the center of overlap.
        return a tuple: (overlap, pos_on_A, pos_on_B)
        """
        y1 = max(self.A.y1, self.B.y1)
        y2 = min(self.A.y2, self.B.y2)
        ovrl = max(0, y2 - y1)
        if ovrl > 0:
            m  = (y1 + y2) / 2.0
            pA = (m + m - self.A.y1 - self.A.y2) / abs(self.A.y2 - self.A.y1)
            pB = (m + m - self.B.y1 - self.B.y2) / abs(self.B.y2 - self.B.y1)
            return (m, pA, pB)
        else:
            return 0, 0, 0

    def getExtremeLength(self):
        """
        compute total length from begin of first block to end of second block
        """
        return max(0,  max(self.A.x2, self.B.x2) - min(self.A.x1, self.B.x1)) 

    def getOverlapRatio(self):
        return max(0, self.overlap) / (abs(self.A.y2 - self.A.y1) + abs(self.B.y2 - self.B.y1))
    
    def computeOverlapPositionAndRatio(self):
        """
        compute the vertical overlap between the two nodes and its position 
            relative to each node, as well as the ratio to total possible overlap
        The overlap is a length
        The position is a number in [-1, +1] relative to the center of a node.
            -1 denote left or top, +1 denotes right or bottom.
            The position is the position of the center of overlap.
        return a tuple: (overlap, pos_on_A, pos_on_B, ratio)
        """
        A, B = self.A, self.B
        y1 = max(A.y1, B.y1)
        y2 = min(A.y2, B.y2)
        ovrl = max(0, y2 - y1)
        if ovrl > 0:
            m  = (y1 + y2) / 2.0
            hA = abs(A.y2 - A.y1)
            hB = abs(B.y2 - B.y1)
            py1A = (y1 - A.y1) / hA
            py1B = (y1 - B.y1) / hB
            pA = (m + m - A.y1 - A.y2) / hA
            pB = (m + m - B.y1 - B.y2) / hB
            py2A = (A.y2 - y2) / hA
            py2B = (B.y2 - y2) / hB
            rO  = ovrl / (ovrl + hA + hB)
            return (m, py1A, py1B, pA, pB, py2A, py2B, rO)
        else:
            return 0 , 0   , 0   , 0 , 0 , 0   , 0   , 0

class VerticalEdge(SamePageEdge):
    def __init__(self, A, B, length, overlap):
        SamePageEdge.__init__(self, A, B, length, overlap)
        try: 
            self.iou = max(0, self.overlap) / (abs(A.x2-A.x1) + abs(B.x2-B.x1) - abs(self.overlap))
        except ZeroDivisionError:
            self.iou = 0    

    def getCoords(self):
        """
        return the coordinates of the edge, if applicable
        return x1, y1, x2, y2
        """
        x,y,w,h = self.computeOverlapBB()
        xm = x + w/2.0
        return xm, y, xm, y+h

    def computeOverlap(self):
        """
        compute the horizontal overlap between the two nodes
        return 0 or a positive number in case of overlap
        """
        return max(0, min(self.A.x2, self.B.x2) - max(self.A.x1, self.B.x1))

    def computeOverlapBB(self):
        """
        compute the overlap BB between the two nodes
        return x,y,w,h
        """
        # X and Width
        x1 = max(self.A.x1, self.B.x1)
        x2 = min(self.A.x2, self.B.x2)
        w = x2 - x1
        assert w >= 0, ("Some invalid vertical edge BB", w)
        
        # Y and height 
        y1 = min(self.A.y2, self.B.y2)
        y2 = max(self.A.y1, self.B.y1)
        y = min(y1, y2)
        
        return x1, y, w, abs(y1-y2)
        
    def computeOverlapPosition(self):
        """
        compute the horizontal overlap between the two nodes and its position 
            relative to each node.
        The overlap is a length
        The position is a number in [-1, +1] relative to the center of a node.
            -1 denote left or top, +1 denotes right or bottom.
            The position is the position of the center of overlap.
        return a tuple: (overlap, pos_on_A, pos_on_B)
        """
        x1 = max(self.A.x1, self.B.x1)
        x2 = min(self.A.x2, self.B.x2)
        ovrl = max(0, x2 - x1)
        if ovrl > 0:
            m  = (x1 + x2) / 2.0
            pA = (m + m - self.A.x1 - self.A.x2) / abs(self.A.x2 - self.A.x1)
            pB = (m + m - self.B.x1 - self.B.x2) / abs(self.B.x2 - self.B.x1)
            return (m, pA, pB)
        else:
            return 0, 0, 0

    def getExtremeLength(self):
        """
        compute total length from begin of first block to end of second block
        """
        return max(0,  max(self.A.y2, self.B.y2) - min(self.A.y1, self.B.y1)) 

    def computeOverlapPositionAndRatio(self):
        """
        compute the vertical overlap between the two nodes and its position 
            relative to each node, as well as the ratio to total possible overlap
        The overlap is a length
        The position is a number in [-1, +1] relative to the center of a node.
            -1 denote left or top, +1 denotes right or bottom.
            The position is the position of the center of overlap.
        return a tuple: (overlap, pos_on_A, pos_on_B, ratio)
        """
        A, B = self.A, self.B
        x1 = max(A.x1, B.x1)
        x2 = min(A.x2, B.x2)
        ovrl = max(0, x2 - x1)
        if ovrl > 0:
            m  = (x1 + x2) / 2.0
            wA = abs(A.x2 - A.x1)
            wB = abs(B.x2 - B.x1)
            px1A = (x1 - A.x1) / wA
            px1B = (x1 - B.x1) / wB
            pmA = (m + m - A.x1 - A.x2) / wA
            pmB = (m + m - B.x1 - B.x2) / wB
            px2A = (A.x2 - x2) / wA
            px2B = (B.x2 - x2) / wB
            rO = ovrl / (ovrl + wA + wB)
            return (m, px1A, px1B, pmA, pmB, px2A, px2B, rO)
        else:
            return 0 , 0   , 0   , 0 , 0 , 0   , 0   , 0


class EdgeInterCentroid(SamePageEdge):
    """
    a class of edge very general from a centroid to another
    
    JLM Friday 28/02/2020 
    """
    def __init__(self, A, cA, B, cB, length):
        """
        we compute ourself our meaning of overlap and overlap_max

        the overlap is:
        0 if no overlap (neither H nor V)
        negative if either H or V overlap
        positive is both H and V overlap
        """
        SamePageEdge.__init__(self, A, B, length, 0)
        # computing a centroid may be costly, I guess
        self.cA = cA  # (x,y)
        self.cB = cB  # (x,y)
        
        # neg=overlap, pos = no overlap, null=touching
        ovr_H = max(self.A.x1, self.B.x1) - min(self.A.x2, self.B.x2)
        ovr_V = max(self.A.y1, self.B.y1) - min(self.A.y2, self.B.y2)
        if ovr_H >= 0 and ovr_V >= 0:
            self.overlap = 0
            self.ovrl_max = 0
            self.iou = -1
        else:
            self.overlap = ovr_H * ovr_V
            self.ovrl_max = abs(self.A.area()) - abs(self.overlap) + abs(self.B.area())
            self.iou = -min(0, self.ovrl_max)

    def computeOverlapPosition(self):
        """
        in pA and pB we encode the edge direction
        
        """
        cax, cay = self.cA
        cbx, cby = self.cB
        pA = (cbx - cax) / self.A.page.w
        pB = (cby - cay) / self.A.page.h
        return self.ovrl_max, pA, pB
    
    @classmethod
    def findPageDelaunayEdges(cls, lPageBlk):
        lEdge = []
        
        # some polygon do not have a valid centroid....
        lC = []
        d = {}
        for blk in lPageBlk:
            try:
                o = ShapeLoader.node_to_Polygon(blk.node)
                c = o.centroid
                d[(c.x, c.y)] = blk
                lC.append(c)
            except: # have seen IndexError and ValueError due to bad data
                # on c.x   :-0
                pass
        lEdgeLineString = ops.triangulate(geom.collection.GeometryCollection(lC), edges=True)

        # NOTE: if 2 block have same centroid, then one is ignored!
        for line in lEdgeLineString:
            ptA, ptB = line.coords
            if ptB < ptA: ptA, ptB = ptB, ptA  # I want to control this
            A,B = d[ptA], d[ptB]
            e = EdgeInterCentroid(A, ptA, B, ptB, line.length)
            lEdge.append(e)
        return lEdge
 
    def computeOverlapBB(self):
        """
        compute the overlap BB between the two nodes
        here, we may have no overlap...
        if overlap => as in VerticalEdge or HorizontalEdge
        if no overlap, then we compute a rectangle defined by the closest corners of each
        return x,y,w,h
        """
        x1 = max(self.A.x1, self.B.x1)
        x2 = min(self.A.x2, self.B.x2)
        x1, x2 = min(x1, x2), max(x1, x2)
        
        y1 = max(self.A.y1, self.B.y1)
        y2 = min(self.A.y2, self.B.y2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        return x1, y1, (x2-x1), (y2-y1)

#     @classmethod
#     def findPageDeJLEdges_centroid(cls, lPageBlk):
#         """
#         Design:
#         - symetry: an edge created from A to B 
#         is the same as if it was created from B to A. (so that we consider
#         only n*(n-1)/2 pairs)
#         - no direction is priviledged (like horizontal and vertical compared
#         to e.g. 12.5 degree)
#         - reflect each object by a point
#         - reflect each edge by a line that does not cross any other object
#         INCLUDING OTHER EDGES
#         
#         
#         TOOOOOO SLOOOOOOW
#         """
#         lEdge = []
#         
#         ltCO = []
#         for blk in lPageBlk:
#             c = blk.shape.centroid
#             ltCO.append( (c, blk.shape))
#         n = len(ltCO)
#     
#         # make an indexed rtree
#         idx = index.Index()
#         for i, (_c, o) in enumerate(ltCO):
#             idx.insert(i, o.bounds)
#     
#         # distance matrix (upper triangular)
#         aDst = np.full((n,n), np.inf)
#         for i, (cA, A) in enumerate(ltCO):
#                 for j in range(i+1, n):
#                     cB, B = ltCO[j]
#                     aDst[i,j] = cA.distance(cB)
#         
#         while True:
#             i,j = np.unravel_index(aDst.argmin(), (n,n))
#             
#             if aDst[i,j] == np.inf: break
#             aDst[i,j] = np.inf  #make sure we use once
#             
#             cA, A = ltCO[i]
#             cB, B = ltCO[j]
#             e = geom.LineString([(cA.x, cA.y), (cB.x,cB.y)])
#                 
#             bIntersect = False
#             for k in idx.intersection(e.bounds):
#                 if k == i or k == j: continue
#                 _cC, C = ltCO[k]
#                 bIntersect = e.crosses(C)
#                 if bIntersect: break
#     
#             if not bIntersect:
#                 ltCO.append((None, e))
#                 idx.insert(len(ltCO)-1, e.bounds) #insert in last position
#                 e = EdgeInterCentroid(lPageBlk[i], lPageBlk[j], e.length)
#                 lEdge.append(e)
#                     
#         return lEdge    

