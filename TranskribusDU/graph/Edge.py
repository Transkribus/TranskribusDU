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

class VerticalEdge(SamePageEdge):
    def __init__(self, A, B, length, overlap):
        SamePageEdge.__init__(self, A, B, length, overlap)
        try: 
            self.iou = max(0, self.overlap) / (abs(A.x2-A.x1) + abs(B.x2-B.x1) - abs(self.overlap))
        except ZeroDivisionError:
            self.iou = 0    

 
