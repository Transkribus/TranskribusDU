# coding: utf8

'''
This code is about the graph we build - edges and nodes   (nodes are called Blocks)

JL Meunier
March 3rd, 2016


Copyright Xerox 2016

'''





from past.builtins import cmp

import collections

from . import Edge
# from Edge import CrossPageEdge, HorizontalEdge, VerticalEdge
from util.masking import applyMask, applyMask2

DEBUG=0
DEBUG=1

"""
We have 2 ways for computing edges: simplified and g2
- in simplified, we consider that the overlapping first box downward (or rightward) blocks the visibility.
- in g2 mode: we deal properly with the masking (as explained in our publications)
"""

class Block:
    
    # when creating edges, blocks are aligned on a grid. This is the grid size
    iGRID = 2
    
    def __init__(self, page, tXYWH, text, orientation, cls, nodeType, domnode=None, domid=None):
        """
        pnum is an int
        orientation is an int, usually in [0-3]
        cls is the node label, is an int in N+
        """
        (x, y, w, h) = tXYWH
        self.page = page
        self.pnum = int(page.pnum)
        self.setBB(self.xywhtoxyxy(x, y, w, h))
        self.text = text
        try:
            self.orientation = int(orientation)
        except TypeError:
            assert orientation == None
            self.orientation = 0
        assert 0 <= self.orientation and self.orientation <= 3
        self.node = domnode
        self.domid = domid
        #try:
        #    print (self.node.attrib['DU_sem'])
        #except:
        #    pass
        #for attr in dir(self.node):
        #    print("obj.%s = %r" % (attr, getattr(self.node, attr)))
        #input()
        self.cls = cls #the class of the block, in [0, N]
        #Node type
        self.type = nodeType
        self.fontsize = 0.0 #new in loader v04
        self.sconf = "" #new in v08
        
        #neighbouring relationship
        self.lHNeighbor     = list()
        self.lVNeighbor     = list()
        self.lCPNeighbor    = list()
        self.lCMPNeighbor   = list()

    def setFontSize(self, fFontSize):
        self.fontsize = fFontSize
    
    @classmethod
    def setGrid(cls, iGrid):
        """
        Blocks are aligned on a grid when computing the edges
        """
        assert iGrid > 0
        cls.iGRID = iGrid
        
    def setShape(self,s):
        self.shape =  s
    
    def getShape(self): return self.shape
    
    def detachFromDOM(self): 
        """
        Erase any pointer to the DOM so that we can free it.
        """
        return   		
        self.node.clear()
        self.node = None
        self.page.node = None
    
    def getText(self, iTruncate=None):
        if iTruncate:
            return self.text[:min(len(self.text), iTruncate)]
        else:
            return self.text
        
    def getPage(self): return self.page
    
    ##Bounding box methods: getter/setter + geometrical stuff
    def getBB(self):
        return self.x1, self.y1, self.x2, self.y2
    def setBB(self, t_x1_y1_x2_y2):
        self.x1, self.y1, self.x2, self.y2 = t_x1_y1_x2_y2
 
    def getWidthHeight(self):
        return self.x2-self.x1, self.y2-self.y1
    def getCenter(self):
        """return the (x,y) of the geometric center of the image"""
        return (self.x1+self.x2)/2, (self.y1+self.y2)/2
    def area(self):
        return(self.x2-self.x1) * (self.y2-self.y1)

    def scale(self, scale_h, scale_v):
        dx = (self.x2-self.x1) * (1-scale_h) / 2
        self.x1, self.x2 = self.x1 + dx, self.x2 - dx
        dy = (self.y2-self.y1) * (1-scale_v) / 2
        self.y1, self.y2 = self.y1 + dy, self.y2 - dy
        
    def setThickBox(self, f):
        """make the box border thicker """
        self.x1 = self.x1 - f
        self.x2 = self.x2 + f
        self.y1 = self.y1 - f
        self.y2 = self.y2 + f
    
    def translate(self, deltaX, deltaY):
        """
        Translate this block
        """
        self.x1 = self.x1 + deltaX
        self.x2 = self.x2 + deltaX
        self.y1 = self.y1 + deltaY
        self.y2 = self.y2 + deltaY

    def mirrorHorizontally(self, fPageWidth):
        """
        Mirror horizontally with respect to the page
        """                
        self.x1, self.x2 = fPageWidth - self.x2, fPageWidth - self.x1
        
    def equalGeo(self, tb):
        """ Return True if both objects are geometrically identical
        accepts also a tule (x1,y1,x2,y2)
        """
        try:
            return tb and self.x1 == tb.x1 and self.y1 == tb.y1 and self.x2 == tb.x2 and self.y2 == tb.y2
        except AttributeError:
            (x1, y1, x2, y2) = tb
            return tb and self.x1 == x1 and self.y1 == y1 and self.x2 == x2 and self.y2 == y2
            
    def overlap(self, b, epsilon=0):
        """
        Return True if the two objects overlap each other, at epsilon-precision (opverlap of at each epsilon)
        """
        #Any horizontal overlap?
        return (min(self.x2, b.x2) - max(self.x1, b.x1)) >= epsilon or (min(self.y2, b.y2) - max(self.y1, b.y1)) >= epsilon

    def significantXOverlap(self, b, fThreshold=0.25):
        """
        Return True if the two objects significantly overlap each other, on the X axis.
        The ratio of the overlap length to the sum of length must be aove the threshold.
        """
        #Any horizontal overlap?
        a = self
        overlap = float(min(a.x2, b.x2) - max(a.x1, b.x1))
        if overlap > 0:
            la = abs(a.x2 - a.x1)
            lb = abs(b.x2 - b.x1)
            ratio = overlap / (la + lb)
            return ratio >= fThreshold 
        else:
            return False

    def significantYOverlap(self, b, fThreshold=0.25):
        """
        Return True if the two objects significantly overlap each other, on the X axis.
        The ratio of the overlap length to the sum of length must be aove the threshold.
        """
        #Any horizontal overlap?
        a = self
        overlap = float(min(a.y2, b.y2) - max(a.y1, b.y1))
        if overlap > 0:
            la = abs(a.y2 - a.y1)
            lb = abs(b.y2 - b.y1)
            ratio = overlap / (la + lb)
            return ratio >= fThreshold 
        else:
            return False

#         a = self
#         w = min(a.x2, b.x2) - max(a.x1, b.x1)
#         if w >= epsilon:
#             #any vertical overlap?
#             h = min(a.y2, b.y2) - max(a.y1, b.y1)
#             if h >= epsilon:
#                 #ok, overlap or inclusion of one text box into the other
#                 return True
#         return False

    def significantOverlap(self, b, fThreshold=0.25):
        """
        The significance of an overlap is the ratio of the area of overlap divided by the area of union
        Return the percentage of overlap if above threshold or 0.0   (which is False)
        """
        #any overlap?
        ovr = 0.0
        a = self
        #Any horizontal overlap?
        w = min(a.x2, b.x2) - max(a.x1, b.x1)
        if w > 0:
            #any vertical overlap?
            h = min(a.y2, b.y2) - max(a.y1, b.y1)
            if h > 0:
                #ok, overlap or inclusion of one text box into the other
                ovArea = h * w
                aArea = (a.x2-a.x1) * (a.y2-a.y1)
                bArea = (b.x2-b.x1) * (b.y2-b.y1)
                unionArea = aArea + bArea - ovArea
#                 if self.text == "S2.0":
#                     print self.text, b.text, float(ovArea)/unionArea 
                ovr = float(ovArea)/unionArea
            else:
                return 0.0
        else:
            return 0.0
        
        if ovr >= fThreshold:
            return ovr
        else:
            return 0.0       

    def significantOverlapMirror(self, b, pageWidth=None, fThreshold=0.25):
        """
        The significance of an overlap is the ratio of the area of overlap divided by the area of union
        Return the percentage of overlap if above threshold or 0.0   (which is False)
        """
        if pageWidth is None: pageWidth = self.page.w
        
        #any overlap?
        ovr = 0.0
        a = self
        #MIRROR EFFECT
        ax2, ax1 = pageWidth-a.x1, pageWidth-a.x2
        #Any horizontal overlap?
        w = min(ax2, b.x2) - max(ax1, b.x1)
        if w > 0:
            #any vertical overlap?
            h = min(a.y2, b.y2) - max(a.y1, b.y1)
            if h > 0:
                #ok, overlap or inclusion of one text box into the other
                ovArea = h * w
                aArea = (ax2-ax1) * (a.y2-a.y1)
                bArea = (b.x2-b.x1) * (b.y2-b.y1)
                unionArea = aArea + bArea - ovArea
#                 if self.text == "S2.0":
#                     print self.text, b.text, float(ovArea)/unionArea 
                ovr = float(ovArea)/unionArea
            else:
                return 0.0
        else:
            return 0.0
        
        if ovr >= fThreshold:
            return ovr
        else:
            return 0.0       

    def fitIn(self, t_x1_y1_x2_y2):
        """
        return true if this object fits in the given bounding box
        """
        (x1, y1, x2, y2) = t_x1_y1_x2_y2
        return self.x1>=x1 and self.x2<=x2 and self.y1>=y1 and self.y2<=y2


    def getXOverlap(self, tb):
        """
        return a positive value is there is an overlap on the X axis
        """
        return min(self.x2, tb.x2) - max(self.x1, tb.x1)

    def getYOverlap(self, tb):
        """
        return a positive value is there is an overlap on the Y axis
        """
        return min(self.y2, tb.y2) - max(self.y1, tb.y1)

    def getXYOverlap(self, tb):
        """
        Return the horizontal and vertical distances between the closest corners of the object.
        if both value are POSITIVE, the objects overlap each other.
        return (x-distance, y-distance)
        """
        return min(self.x2, tb.x2) - max(self.x1, tb.x1), min(self.y2, tb.y2) - max(self.y1, tb.y1)
    
    #-- Comparison methods
    def cmpX1(tb1, tb2):
        return cmp(tb1.x1, tb2.x1)
    cmpX1 = staticmethod(cmpX1)
 
    def cmpX1Y1(tb1, tb2):
        ret = cmp(tb1.x1, tb2.x1)
        if ret == 0:
            ret = cmp(tb1.y1, tb2.y1)
        return ret
    cmpX1Y1 = staticmethod(cmpX1Y1)
                 
    # -- utility to convert from x,y,width,height to (x1,y1, x2,y2)
    def xywhtoxyxy(cls, x, y, w, h):
        """
        convert from x,y,width,height to (x1,y1, x2,y2)
        accepts string
        convert to float
        """
        x1 = float(x)
        y1 = float(y)
        #we must accept 0 width or 0 height blocks
        if w < 0:
            #traceln("WARNING: negative width textbox - x taken as right-x")
            x2 = x1
            x1 = x2 + w
        else:
            x2 = x1+w            
        if h < 0:
            #traceln("WARNING: negative height textbox - y taken as bottom-y")
            y2 = y1
            y1 = y2+h
        else:
            y2 = y1+h
        return x1, y1, x2, y2
    xywhtoxyxy = classmethod(xywhtoxyxy)

    def __str__(self):
        return "Block id=%s page=%d (%f, %f) (%f, %f) '%s'" %(self.domid, self.pnum, self.x1, self.y1, self.x2, self.y2, self.text)
    
    # --- Neighboring relationship to build graph------------------------------------------------------------------------------------- 
    @classmethod       
    def findPageNeighborEdges(cls, lBlk, bShortOnly=False, iGraphMode=1):
        """
        find neighboring edges, horizontal and vertical ones
        """
        if iGraphMode == 1:
            fun = cls._findVerticalNeighborEdges_g1
        elif iGraphMode == 2:
            fun = cls._findVerticalNeighborEdges_g2
        elif iGraphMode == 4:
            fun = cls._findVerticalNeighborEdges_g1o
        else:
            raise ValueError("Unkown graph mode '%s'")% iGraphMode
            
        #look for vertical neighbors
        lVEdge = fun(lBlk, Edge.VerticalEdge, bShortOnly)
        
        #look for horizontal neighbors
        for blk in lBlk: blk.rotateMinus90deg()          #rotate by 90 degrees and look for vertical neighbors :-)
        lHEdge = fun(lBlk, Edge.HorizontalEdge, bShortOnly)
        for blk in lBlk: blk.rotatePlus90deg()         #restore orientation :-)
        
        return lHEdge, lVEdge
    
    # ---- Internal stuff ---
    def findConsecPageOverlapEdges(cls, lPrevPageEdgeBlk, lPageBlk, bMirror=True, epsilon = 1):
        """
        find block that overlap from a page to the other, and have same orientation
        """
        
        #N^2 brute force
        lEdge = list()
        for prevBlk in lPrevPageEdgeBlk:
            orient = prevBlk.orientation
            prevBlkPageWidth = prevBlk.page.w
            for blk in lPageBlk:
                if orient == blk.orientation:
                    if             prevBlk.significantOverlap(blk)                                  : lEdge.append( Edge.CrossPageEdge      (prevBlk, blk) )
                    if bMirror and prevBlk.significantOverlapMirror(blk, pageWidth=prevBlkPageWidth): lEdge.append( Edge.CrossMirrorPageEdge(prevBlk, blk) )
        
#         #maybe faster on LAAARGE bnumber of blocks but complicated and in practice no quantified time advantage... :-/
#         
#         """
#         we will sort the blocks by their value x1 - y1
#         
#         considering block A, lets say A.x2 - A.y2 = alpha
#         
#         considering block B, if B.x1 - B.y1 > alpha, then B cannot overlap A 
#         (just take a paper and draw a line at 45° from A botton-right corner!)
#         """
#         import time
#         t0 = time.time()
#         
#         lEdge = list()
#         
#         dBlk_by_alpha = collections.defaultdict(list)   # x1-y1 --> [ list of block having that x1-y1 ]
#         for blk in lPageBlk:
#             rx1 =  cls.gridRound(blk.x1, epsilon)
#             ry1 =  cls.gridRound(blk.y1, epsilon)
#             #rx1, ry1 = blk.x1, blk.y1
#             #OK assert abs(ry1-b.y1) < epsilon
#             dBlk_by_alpha[rx1 - ry1].append(blk)
#         lAlpha = dBlk_by_alpha.keys(); lAlpha.sort() #increasing alphas
#         
#         for prevBlk in lPrevPageEdgeBlk:
#             prevBlkAlpha = prevBlk.x2 - prevBlk.y2  #alpha value of bottom-right corner
#             for alpha in lAlpha:
#                 if alpha > prevBlkAlpha:
#                     for blk in dBlk_by_alpha[alpha]:
#                         if prevBlk.overlap(blk): 
#                             print  "PB ----------"
#                             print prevBlk.getXOverlap(blk), prevBlk.getYOverlap(blk)
#                             print prevBlk, " -- ", prevBlk.x2 - prevBlk.y2, prevBlkAlpha
#                             print blk    , " -- ",     blk.x1 -     blk.y1, alpha
#                             print 
#                     break
#                 for Blk in dBlk_by_alpha[alpha]:
#                     if prevBlk.overlap(Blk): lEdge.append( (prevBlk, Blk) )
#         timeAlgo = time.time() - t0
#         
#         cls.checkThisAlgo(timeAlgo, lEdge, lPrevPageEdgeBlk, lPageBlk, epsilon)
        
        return lEdge 
    findConsecPageOverlapEdges = classmethod(findConsecPageOverlapEdges)

    # ------------------------------------------------------------------------------------------------------------------------------------        
    def rotateMinus90deg(self):
        #assert self.x1 < self.x2 and self.y1 < self.y2
        self.x1, self.y1,  self.x2, self.y2 = -self.y2, self.x1,  -self.y1, self.x2
        #assert self.x1 < self.x2 and self.y1 < self.y2        

    def rotatePlus90deg(self):
        self.x1, self.y1,  self.x2, self.y2 = self.y1, -self.x2,  self.y2, -self.x1

    def gridRound(cls, f, iGrid): 
        return int(round(f / iGrid, 0)*iGrid)
    gridRound = classmethod(gridRound)
    
    def XXOverlap(cls, tAx1_Ax2, tBx1_Bx2): #overlap if the max is smaller than the min
        Ax1, Ax2 = tAx1_Ax2
        Bx1, Bx2 = tBx1_Bx2
        return max(Ax1, Bx1), min(Ax2, Bx2)

    XXOverlap = classmethod(XXOverlap)


    @classmethod
    def _findVerticalNeighborEdges_init(cls, lBlk, iGrid):
        assert type(iGrid) is int, repr(iGrid)
        
        #index along the y axis based on y1 and y2
        dBlk_Y1 = collections.defaultdict(list)     # y1 --> [ list of block having that y1 ]
        setY2 = set()                               # set of (unique) y2
        for blk in lBlk:
            ry1 =  cls.gridRound(blk.y1, iGrid)
            ry2 =  cls.gridRound(blk.y2, iGrid)
            #OK assert abs(ry1-b.y1) < iGrid
            dBlk_Y1[ry1].append(blk)
            setY2.add(ry2)
        
        #lY1 and lY2 are sorted list of unique values
        lY1 = list(dBlk_Y1.keys()); lY1.sort(); n1 = len(lY1)
        lY2 = list(setY2) 
        lY2.sort(); 
                
        di1_by_y2 = dict() #smallest index i1 of Y1 so that lY1[i1] >= Y2, where Y2 is in lY2, if no y1 fit, set it to n1
        i1, y1 = 0, lY1[0]
        for y2 in lY2:
            while y1 < y2 and i1 < n1-1:
                i1 += 1
                y1 = lY1[i1]
            di1_by_y2[y2] = i1
        return n1, lY1, dBlk_Y1, di1_by_y2

    @classmethod
    def _findVerticalNeighborEdges_init_y1(cls, lBlk, iGrid):
        """
        Same as _findVerticalNeighborEdges_init, but ignoring y2
        """
        assert type(iGrid) is int, repr(iGrid)
        
        #index along the y axis based on y1 and y2
        dBlk_Y1 = collections.defaultdict(list)     # y1 --> [ list of block having that y1 ]
        for blk in lBlk:
            ry1 =  cls.gridRound(blk.y1, iGrid)
            #OK assert abs(ry1-b.y1) < iGrid
            dBlk_Y1[ry1].append(blk)
        
        #lY1 and lY2 are sorted list of unique values
        lY1 = list(dBlk_Y1.keys()); lY1.sort(); n1 = len(lY1)
        return n1, lY1, dBlk_Y1

    @classmethod
    def _findVerticalNeighborEdges_g1(cls, lBlk, EdgeClass, bShortOnly=False, iGrid = None):
        """
        any dimension smaller than 5 is zero, we assume that no block are narrower than this value
        
        ASSUMTION: BLOCKS DO NOT OVERLAP EACH OTHER!!!
        
        return a list of pair of block
        
        """
        if not lBlk: return []
        if iGrid is None: iGrid = Block.iGRID
        
        #look for vertical neighbors
        lVEdge = list()
        
        n1, lY1, dBlk_Y1, di1_by_y2 = cls._findVerticalNeighborEdges_init(lBlk, iGrid)

        epsilon = 2*iGrid
        epsilon = 0  # back to old version for being able to compare results
        
        for i1,y1 in enumerate(lY1):
            #start with the block(s) with lowest y1
            #  (they should not overlap horizontally and cannot be vertical neighbors to each other)
            for A in dBlk_Y1[y1]:
                Ax1,Ay1, Ax2,Ay2 = map(cls.gridRound, A.getBB(), [iGrid, iGrid, iGrid, iGrid])
                A_height = A.y2 - A.y1   #why were we accessing the DOM?? float(A.node.prop("height"))
                assert Ay2 >= Ay1
                lOx1x2 = list() #list of observed overlaps for current block A
                leftWatermark, rightWatermark = Ax1, Ax2    #optimization to see when block A has been entirely "covered"
                jstart = di1_by_y2[Ay2]                 #index of y1 in lY1 of next block below A (because its y1 is larger than A.y2)
                jstart = jstart - 1                     #because some block overlap each other, we try the previous index (if it is not the current index)
                jstart = max(jstart, i1+1)              # but for sure we want the next group of y1          
                for j1 in range(jstart, n1):            #take in turn all Y1 below A
                    By1 = lY1[j1]
                    for B in dBlk_Y1[By1]:          #all block starting at that y1
                        Bx1,By1, Bx2,_ = map(cls.gridRound, B.getBB(), [iGrid, iGrid, iGrid, iGrid])
                        #ovABx1, ovABx2 = cls.XXOverlap( (Ax1,Ax2), (Bx1, Bx2) )
                        ovABx1, ovABx2 = max(Ax1,Bx1), min(Ax2, Bx2)
                        if ovABx2 - ovABx1 > epsilon: # significantoverlap
                            #we now check if that B block is not partially hidden by a previous overlapping block
                            bVisible = True
                            for ovOx1, ovOx2 in lOx1x2:
                                #oox1, oox2 = cls.XXOverlap( (ovABx1, ovABx2), (ovOx1, ovOx2) )
                                oox1, oox2 = max(ovABx1, ovOx1), min(ovABx2, ovOx2)
                                if oox1 < oox2:
                                    bVisible = False  
                                    break
                            if bVisible: 
                                length = abs(B.y1 - A.y2)
                                if bShortOnly:
                                    #we need to measure how far this block is from A
                                    #we use the height attribute (not changed by the rotation)
                                    if length < A_height: 
                                        lVEdge.append( EdgeClass(A, B, length, ovABx2 - ovABx1) )
                                else:
                                    lVEdge.append( EdgeClass(A, B, length, ovABx2 - ovABx1) )
                                
                            lOx1x2.append( (ovABx1, ovABx2) ) #an hidden object may hide another one
                            #optimization to see when block A has been entirely "covered"
                            #(it does not account for all cases, but is fast and covers many common situations)
                            if Bx1 < Ax1: leftWatermark =  max(leftWatermark, Bx2)
                            if Ax2 < Bx2: rightWatermark = min(rightWatermark, Bx1)
                            if leftWatermark >= rightWatermark: break #nothing else below is visible anymore
                    if leftWatermark >= rightWatermark: break #nothing else below is visible anymore
                            
        return lVEdge

    @classmethod
    def _findVerticalNeighborEdges_g2(cls, lBlk, EdgeClass, bShortOnly=False, iGrid = None):
        """
        the masking is done properly.
        
        ASSUMTION (??? JL June'19): BLOCKS DO NOT OVERLAP EACH OTHER!!!
        
        return a list of pair of block
        """
        if not lBlk: return []
        if iGrid is None: iGrid = Block.iGRID
        
        #look for vertical neighbors
        lVEdge = list()

        n1, lY1, dBlk_Y1, _di1_by_y2 = cls._findVerticalNeighborEdges_init(lBlk, iGrid)
        # we do not use _di1_by_y2 because we want to include vertically overlapping block in our search
        
        for i1,y1 in enumerate(lY1):
            #start with the block(s) with lowest y1
            #  (they should not overlap horizontally and cannot be vertical neighbors to each other)
            for A in dBlk_Y1[y1]:
                Ax1,Ay1, Ax2,Ay2 = map(cls.gridRound, A.getBB(), [iGrid, iGrid, iGrid, iGrid])
                A_height = A.y2 - A.y1   #why were we accessing the DOM?? float(A.node.prop("height"))
                lViewA = [(Ax1, Ax2)]   # what A can view (or what it covers horizontally)
                assert Ay2 >= Ay1
                jstart = i1 + 1   # consider all block slightly below the current one
                for j1 in range(jstart, n1):            #take in turn all Y1 below A
                    By1 = lY1[j1]
                    for B in dBlk_Y1[By1]:          #all block starting at that y1
                        Bx1,By1, Bx2,_ = map(cls.gridRound, B.getBB(), [iGrid, iGrid, iGrid, iGrid])
                        
                        lNewViewA, ovrl = applyMask2(lViewA, [(Bx1, Bx2)]) # what remains of A views...
                        if lNewViewA == lViewA:
                            # no overlap between what A can still view and B
                            pass
                        else:
                            # B is visible
                            length = abs(B.y1 - A.y2)
                            if bShortOnly:
                                #we need to measure how far this block is from A
                                #we use the height attribute (not changed by the rotation)
                                if length < A_height: 
                                    lVEdge.append( EdgeClass(A, B, length, ovrl) )
                            else:
                                lVEdge.append( EdgeClass(A, B, length, ovrl) )
                            lViewA = lNewViewA    
                    if not lViewA: break
                            
        return lVEdge
  

    @classmethod
    def _findVerticalNeighborEdges_g1o(cls, lBlk, EdgeClass, bShortOnly=False, iGrid = None):
        """
        Blocks can overlap
        return a list of pair of block      
        """
        if not lBlk: return []
        if iGrid is None: iGrid = Block.iGRID
        
        #look for vertical neighbors
        lVEdge = list()
        
        n1, lY1, dBlk_Y1 = cls._findVerticalNeighborEdges_init_y1(lBlk, iGrid)

        epsilon = 2*iGrid
        epsilon = 0  # back to old version for being able to compare results
        
        for i1,y1 in enumerate(lY1):
            #start with the block(s) with lowest y1
            #  (they should not overlap horizontally and cannot be vertical neighbors to each other)
            for A in dBlk_Y1[y1]:
                Ax1,Ay1, Ax2,Ay2 = map(cls.gridRound, A.getBB(), [iGrid, iGrid, iGrid, iGrid])
                A_height = A.y2 - A.y1   #why were we accessing the DOM?? float(A.node.prop("height"))
                assert Ay2 >= Ay1
                lOx1x2 = list() #list of observed overlaps for current block A
                leftWatermark, rightWatermark = Ax1, Ax2    #optimization to see when block A has been entirely "covered"
                jstart = i1 + 1                         #index of next y1 
                for j1 in range(jstart, n1):            #take in turn all Y1 below A.y1
                    By1 = lY1[j1]
                    for B in dBlk_Y1[By1]:          #all block starting at that y1
                        Bx1,By1, Bx2,By2 = map(cls.gridRound, B.getBB(), [iGrid, iGrid, iGrid, iGrid])
                        #ovABx1, ovABx2 = cls.XXOverlap( (Ax1,Ax2), (Bx1, Bx2) )
                        ovABx1, ovABx2 = max(Ax1,Bx1), min(Ax2, Bx2)
                        if ovABx2 - ovABx1 > epsilon: # significantoverlap
                            #we now check if that B block is not partially hidden by a previous overlapping block
                            bVisible = True
                            for ovOx1, ovOx2 in lOx1x2:
                                #oox1, oox2 = cls.XXOverlap( (ovABx1, ovABx2), (ovOx1, ovOx2) )
                                oox1, oox2 = max(ovABx1, ovOx1), min(ovABx2, ovOx2)
                                if oox1 < oox2:
                                    bVisible = False  
                                    break
                            if bVisible: 
                                # length = abs(B.y1 - A.y2)
                                length = By1 - Ay2  # in g1o, it can be negative!
                                if length < 0:
                                    # let's decide if we create an horizontal or a vertical edge, but not both
                                    # we decide that by looking at the orientation of the rectangular shape of the overlap
                                    # if the horizontal overlap is larger than the vertical overlap, we keep as vertical edge
                                    bVisible = (ovABx2 - ovABx1) > (-length)
                            if bVisible:
                                if bShortOnly:
                                    #we need to measure how far this block is from A
                                    #we use the height attribute (not changed by the rotation)
                                    if 0 < length and length < A_height: 
                                        lVEdge.append( EdgeClass(A, B, length, ovABx2 - ovABx1) )
                                else:
                                    lVEdge.append( EdgeClass(A, B, length, ovABx2 - ovABx1) )
                                
                            lOx1x2.append( (ovABx1, ovABx2) ) #an hidden object may hide another one
                            #optimization to see when block A has been entirely "covered"
                            #(it does not account for all cases, but is fast and covers many common situations)
                            if Bx1 < Ax1: leftWatermark =  max(leftWatermark, Bx2)
                            if Ax2 < Bx2: rightWatermark = min(rightWatermark, Bx1)
                            if leftWatermark >= rightWatermark: break #nothing else below is visible anymore
                    if leftWatermark >= rightWatermark: break #nothing else below is visible anymore
                            
        return lVEdge

class BlockShallowCopy(Block):
    """
    A shallow copy of a block
    """    
    
    def __init__(self, blk):
        self.page       = blk.page
        self.pnum       = blk.pnum
        self.setBB(blk.getBB())
        self.text       = blk.text
        self.orientation = blk.orientation
        self.node       = blk.node
        self.domid      = blk.domid
        self.cls        = blk.cls #the class of the block, in [0, N]
        #Node type
        self.type       = blk.type
        self.fontsize   = blk.fontsize
        self.sconf      = blk.sconf
        
        #neighbouring relationship
        self.lHNeighbor     = blk.lHNeighbor
        self.lVNeighbor     = blk.lVNeighbor
        self.lCPNeighbor    = blk.lCPNeighbor
        self.lCMPNeighbor   = blk.lCMPNeighbor      
        
        #finally
        self._blk        = blk
        
    def getOrigBlock(self): return self._blk
    



def test_scale():
    class Page:
        pnum = 0
    b = Block(Page(), (1, 10, 100, 1000), "", 0, None, None)
    ref = b.getBB()
    b.scale(0.1, 0.1)
    assert b.getBB() == (46, 460, 56, 560)
    b.scale(10, 10)
    assert b.getBB() == ref
    
    