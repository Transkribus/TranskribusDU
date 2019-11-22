# coding: utf8

'''
This code is about the graph we build - edges and nodes   (nodes are called Blocks)

JL Meunier
March 3rd, 2016


Copyright Xerox 2016

'''




DEBUG=0
#DEBUG=1

# --- Edge CLASSE -----------------------------------------
class Edge:
        
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

        lAllEdge = list()
        
        #--- horizontal and vertical neighbors
        lHEdge, lVEdge = Block.Block.findPageNeighborEdges(lPageBlk, bShortOnly, iGraphMode=iGraphMode)
        lAllEdge.extend(lHEdge)
        lAllEdge.extend(lVEdge)
        if DEBUG: 
            cls.dbgStorePolyLine("neighbors", lHEdge)
            cls.dbgStorePolyLine("neighbors", lVEdge)
        
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

class CrossPageEdge(Edge): pass    

class CrossMirrorPageEdge(Edge): pass    

class CrossMirrorContinuousPageVerticalEdge(Edge): 
    def __init__(self, A, B, length, overlap):
        Edge.__init__(self, A, B)
        self.length  = length
        self.overlap = overlap
        self.iou = 0

class VirtualEdge(Edge): pass    


# --- SamePageEdge SUB-CLASSES ----------------------------
class HorizontalEdge(SamePageEdge):
    def __init__(self, A, B, length, overlap):
        SamePageEdge.__init__(self, A, B, length, overlap)
        try: 
            self.iou = max(0, self.overlap) / (abs(A.y2-A.y1) + abs(B.y2-B.y1) - abs(self.overlap))
        except ZeroDivisionError:
            self.iou = 0    

    def computeOverlap(self):
        """
        compute the vertical overlap between the two nodes
        return 0 or a positive number in case of overlap
        """
        return max(0, min(self.A.y2, self.B.y2) - max(self.A.y1, self.B.y1))

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


class VerticalEdge(SamePageEdge):
    def __init__(self, A, B, length, overlap):
        SamePageEdge.__init__(self, A, B, length, overlap)
        try: 
            self.iou = max(0, self.overlap) / (abs(A.x2-A.x1) + abs(B.x2-B.x1) - abs(self.overlap))
        except ZeroDivisionError:
            self.iou = 0    

    def computeOverlap(self):
        """
        compute the horizontal overlap between the two nodes
        return 0 or a positive number in case of overlap
        """
        return max(0, min(self.A.x2, self.B.x2) - max(self.A.x1, self.B.x1))

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

 
