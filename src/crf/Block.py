# coding: utf8

'''
This code is about the graph we build - edges and nodes   (nodes are called Blocks)

JL Meunier
March 3rd, 2016


Copyright Xerox 2016

'''


import collections, types

from common.trace import traceln

import Edge
# from Edge import CrossPageEdge, HorizontalEdge, VerticalEdge

DEBUG=0
DEBUG=1


class Block:
        
    def __init__(self, page, (x, y, w, h), text, orientation, cls, nodeType, domnode=None, domid=None):
        """
        pnum is an int
        orientation is an int, usually in [0-3]
        cls is the node label, is an int in N+
        """
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
    
    def detachFromDOM(self): 
        """
        Erase any pointer to the DOM so that we can free it.
        """
        self.node = None
        self.page.node = None
    
    def getText(self, iTruncate=None):
        if iTruncate:
            return self.text[:min(len(self.text), iTruncate)]
        else:
            return self.text
    ##Bounding box methods: getter/setter + geometrical stuff
    def getBB(self):
        return self.x1, self.y1, self.x2, self.y2
    def setBB(self, (x1, y1, x2, y2)):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
 
    def getWidthHeight(self):
        return self.x2-self.x1, self.y2-self.y1
    def getCenter(self):
        """return the (x,y) of the geometric center of the image"""
        return (self.x1+self.x2)/2, (self.y1+self.y2)/2
    def area(self):
        return(self.x2-self.x1) * (self.y2-self.y1)


    def setThickBox(self, f):
        """make the box border thicker """
        self.x1 = self.x1 - f
        self.x2 = self.x2 + f
        self.y1 = self.y1 - f
        self.y2 = self.y2 + f
        
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

    def fitIn(self, (x1, y1, x2, y2)):
        """
        return true if this object fits in the given bounding box
        """
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
    def findPageNeighborEdges(cls, lBlk, bShortOnly=False):
        """
        find neighboring edges, horizontal and vertical ones
        """
        #look for vertical neighbors
        lVEdge = cls._findVerticalNeighborEdges(lBlk, Edge.VerticalEdge, bShortOnly)
        
        #look for horizontal neighbors
        for blk in lBlk: blk.rotateMinus90deg()          #rotate by 90 degrees and look for vertical neighbors :-)
        lHEdge = cls._findVerticalNeighborEdges(lBlk, Edge.HorizontalEdge, bShortOnly)
        for blk in lBlk: blk.rotatePlus90deg()         #restore orientation :-)
        
        return lHEdge, lVEdge
    findPageNeighborEdges = classmethod(findPageNeighborEdges)
    
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
#             rx1 =  cls.epsilonRound(blk.x1, epsilon)
#             ry1 =  cls.epsilonRound(blk.y1, epsilon)
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

    def epsilonRound(cls, f, epsilon): 
        return int(round(f / epsilon, 0)*epsilon)
    epsilonRound = classmethod(epsilonRound)
    
    def XXOverlap(cls, (Ax1,Ax2), (Bx1, Bx2)): #overlap if the max is smaller than the min
        return max(Ax1, Bx1), min(Ax2, Bx2)
    XXOverlap = classmethod(XXOverlap)

    def _findVerticalNeighborEdges(cls, lBlk, EdgeClass, bShortOnly=False, epsilon = 2):
        """
        any dimension smaller than 5 is zero, we assume that no block are narrower than this value
        
        ASSUMTION: BLOCKS DO NOT OVERLAP EACH OTHER!!!
        
        return a list of pair of block
        
        """
        assert type(epsilon) == types.IntType
        
        if not lBlk: return []
        
        #look for vertical neighbors
        lVEdge = list()
        
        #index along the y axis based on y1 and y2
        dBlk_Y1 = collections.defaultdict(list)     # y1 --> [ list of block having that y1 ]
        setY2 = set()                               # set of (unique) y2
        for blk in lBlk:
            ry1 =  cls.epsilonRound(blk.y1, epsilon)
            ry2 =  cls.epsilonRound(blk.y2, epsilon)
            #OK assert abs(ry1-b.y1) < epsilon
            dBlk_Y1[ry1].append(blk)
            setY2.add(ry2)
        
        #lY1 and lY2 are sorted list of unique values
        lY1 = dBlk_Y1.keys(); lY1.sort(); n1 = len(lY1)
        lY2 = list(setY2) 
        lY2.sort(); 
                
        di1_by_y2 = dict() #smallest index i1 of Y1 so that lY1[i1] >= Y2, where Y2 is in lY2, if no y1 fit, set it to n1
        i1, y1 = 0, lY1[0]
        for y2 in lY2:
            while y1 < y2 and i1 < n1-1:
                i1 += 1
                y1 = lY1[i1]
            di1_by_y2[y2] = i1
        
        for i1,y1 in enumerate(lY1):
            #start with the block(s) with lowest y1
            #  (they should not overlap horizontally and cannot be vertical neighbors to each other)
            for A in dBlk_Y1[y1]:
                Ax1,Ay1, Ax2,Ay2 = map(cls.epsilonRound, A.getBB(), [epsilon, epsilon, epsilon, epsilon])
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
                        Bx1,By1, Bx2,_ = map(cls.epsilonRound, B.getBB(), [epsilon, epsilon, epsilon, epsilon])
                        ovABx1, ovABx2 = cls.XXOverlap( (Ax1,Ax2), (Bx1, Bx2) )
                        if ovABx1 < ovABx2: #overlap
                            #we now check if that B block is not partially hidden by a previous overlapping block
                            bVisible = True
                            for ovOx1, ovOx2 in lOx1x2:
                                oox1, oox2 = cls.XXOverlap( (ovABx1, ovABx2), (ovOx1, ovOx2) )
                                if oox1 < oox2:
                                    bVisible = False  
                                    break
                            if bVisible: 
                                length = abs(B.y1 - A.y2)
                                if bShortOnly:
                                    #we need to measure how far this block is from A
                                    #we use the height attribute (not changed by the rotation)
                                    if length < A_height: 
                                        lVEdge.append( EdgeClass(A, B, length) )
                                else:
                                    lVEdge.append( EdgeClass(A, B, length) )
                                
                            lOx1x2.append( (ovABx1, ovABx2) ) #an hidden object may hide another one
                            #optimization to see when block A has been entirely "covered"
                            #(it does not account for all cases, but is fast and covers many common situations)
                            if Bx1 < Ax1: leftWatermark =  max(leftWatermark, Bx2)
                            if Ax2 < Bx2: rightWatermark = min(rightWatermark, Bx1)
                            if leftWatermark >= rightWatermark: break #nothing else below is visible anymore
                    if leftWatermark >= rightWatermark: break #nothing else below is visible anymore
                            
        return lVEdge
    _findVerticalNeighborEdges = classmethod(_findVerticalNeighborEdges)
    
    
    
    
    