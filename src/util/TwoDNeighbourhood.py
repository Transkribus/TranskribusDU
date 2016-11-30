"""
    compute left, right, top, bootm neighborhood
"""

# ------------------------------------------------------------------------------------------------------------------------------------
#OLD

def rotateMinus90degOLD(e):
#     self.x1, self.y1,  self.x2, self.y2 = -self.y2, self.x1,  -self.y1, self.x2
    x = e.getX()
    e.setX(-e.getY2())
    e.setY(x)
    h= e.getHeight()
    e.setHeight(e.getWidth())
    e.setWidth(h) 

def rotatePlus90degOLD(e):
    x = e.getX()
    e.setX(e.getY())
    e.setY(-(x+e.getWidth()))
    h= e.getHeight()
    e.setHeight(e.getWidth())
    e.setWidth(h) 
    
## new sequenceAPI elements
def rotateMinus90deg(e):
#     self.x1, self.y1,  self.x2, self.y2 = -self.y2, self.x1,  -self.y1, self.x2
    x = e.getX()
    e.addAttribute('x',-e.getY2())
    e.addAttribute('y',x)
    h= e.getHeight()
    e.addAttribute('height',e.getWidth())
    e.addAttribute('width',h) 

def rotatePlus90deg(e):
    x = e.getX()
    e.addAttribute('x',e.getY())
    e.addAttribute('y',- (x+e.getWidth()))
    h= e.getHeight()
    e.addAttribute('height',e.getWidth())
    e.addAttribute('width',h) 
#     e.x1, e.y1,  e.x2, e.y2 = e.y1, -e.x2,  e.y2, -e.x1

def epsilonRound( f, epsilon): 
        return int(round(f / epsilon, 0)*epsilon)
def XXOverlap( (Ax1,Ax2), (Bx1, Bx2)): #overlap if the max is smaller than the min
    return max(Ax1, Bx1), min(Ax2, Bx2)

def findVerticalNeighborEdges( lBlk, bShortOnly=False, epsilon = 2):
    """
    any dimension smaller than 5 is zero, we assume that no block are narrower than this value
    
    ASSUMTION: BLOCKS DO NOT OVERLAP EACH OTHER!!!
    
    return a list of pair of block
    
    """
    import collections
    
    if not lBlk: return []
    
    #look for vertical neighbors
    lVEdge = list()
    
    #index along the y axis based on y1 and y2
    dBlk_Y1 = collections.defaultdict(list)     # y1 --> [ list of block having that y1 ]
    setY2 = set()                               # set of (unique) y2
    for blk in lBlk:
        ry1 =  epsilonRound(blk.getY(), epsilon)
        ry2 =  epsilonRound(blk.getY2(), epsilon)
        #OK assert abs(ry1-b.y1) < epsilon
        dBlk_Y1[ry1].append(blk)
        setY2.add(ry2)
    
    #lY1 and lY2 are sorted list of unique values
    lY1 = dBlk_Y1.keys(); lY1.sort(); n1 = len(lY1)
    lY2 = list(setY2) 
    lY2.sort(); n2 = len(lY2)
            
    di1_by_y2 = dict() #smallest index i1 of Y1 so that lY1[i1] >= Y2, where Y2 is in lY2, if no y1 fit, set it to n1
    i1, y1 = 0, lY1[0]
    for i2, y2 in enumerate(lY2):
        while y1 < y2 and i1 < n1-1:
            i1 += 1
            y1 = lY1[i1]
        di1_by_y2[y2] = i1
    
    for i1,y1 in enumerate(lY1):
        #start with the block(s) with lowest y1
#         print 'start:',i1, y1
        #  (they should not overlap horizontally and cannot be vertical neighbors to each other)
        for A in dBlk_Y1[y1]:
#             print 'A:', A
#             Ax1,Ay1, Ax2,Ay2 = map(epsilonRound, A.getBB(), [epsilon, epsilon, epsilon, epsilon])
            Ax1,Ay1, Ax2,Ay2 = epsilonRound(A.getX(),epsilon), epsilonRound(A.getY(),epsilon), epsilonRound(A.getX2(),epsilon), epsilonRound(A.getY2(),epsilon)

            A_height = A.getHeight()
            assert Ay2 >= Ay1
            lOx1x2 = list() #list of observed overlaps for current block A
            leftWatermark, rightWatermark = Ax1, Ax2    #optimization to see when block A has been entirely "covered"
            jstart = di1_by_y2[Ay2]                 #index of y1 in lY1 of next block below A (because its y1 is larger than A.y2)
            jstart = jstart - 1                     #because some block overlap each other, we try the previous index (if it is not the current index)
            jstart = max(jstart, i1+1)              # but for sure we want the next group of y1          
            for j1 in range(jstart, n1):            #take in turn all Y1 below A
                By1 = lY1[j1]
                for B in dBlk_Y1[By1]:          #all block starting at that y1
                    Bx1,By1, Bx2,By2 = epsilonRound(B.getX(),epsilon), epsilonRound(B.getY(),epsilon), epsilonRound(B.getX2(),epsilon), epsilonRound(B.getY2(),epsilon)
                    ovABx1, ovABx2 = XXOverlap( (Ax1,Ax2), (Bx1, Bx2) )
#                     print '\tB', Bx1,By1, Bx2,By2, ovABx1, ovABx2, ovABx1 < ovABx2
                    if ovABx1 < ovABx2: #overlap
                        #we now check if that B block is not partially hidden by a previous overlapping block
                        bHidden = False
                        oox1, oox2 = XXOverlap( (ovABx1, ovABx2), (leftWatermark, rightWatermark) )
                        bHidden = oox1 >= oox2
#                         print 'hidden?', bHidden
                         
#                         for ovOx1, ovOx2 in lOx1x2:
#                             oox1, oox2 = XXOverlap( (ovABx1, ovABx2), (ovOx1, ovOx2) )
#                             print '\too', oox1,oox2
#                             if oox1 < oox2:
#                                 print oox1, oox2
#                                 bHidden = True  
#                                 break
                        if not bHidden: 
                            if bShortOnly:
                                #we need to measure how far this block is from A
                                #we use the height attribute (not changed by the rotation)
                                if abs(B.y1 - A.y2) < A_height: 
                                    lVEdge.append( (A, B) )
                            else:
                                lVEdge.append( (A, B) )
#                                 print '\t\tadded:', B
#                                 A.getNode().setProp('nextItem',B.getID())
                            
                        lOx1x2.append( (ovABx1, ovABx2) ) #an hidden object may hide another one
                        #optimization to see when block A has been entirely "covered"
                        #(it does not account for all cases, but is fast and covers many common situations)
                        if Bx1 <= Ax1: leftWatermark =  max(leftWatermark, Bx2)
                        if Ax2 <= Bx2: rightWatermark = min(rightWatermark, Bx1)
                        if leftWatermark >= rightWatermark: break #nothing else below is visible anymore
#                         print 'watermark', leftWatermark, rightWatermark
                if leftWatermark >= rightWatermark:
#                     print 'BREAK!', leftWatermark, rightWatermark
                    break #nothing else below is visible anymore
    
    return lVEdge
    
def findNeighborhood(selfx,list):
    """
        structure: Dict[elt][direction]=(elt,s)
        
        rename X in Y and Y in x !!! (06/07/2011)
    """

    for elt1 in list:
        # get nearest left/right
        print elt1.getID(), elt1.getY(),  elt1.getY() + elt1.getHeight(),"---"
        leftElt = filter(lambda x: x < elt1.getX(),selfx.X1Tree.keys())
        lOverlap= []
        for x in leftElt:
            for elt2 in selfx.X1Tree[x]:
                if elt1.overlapY(elt2):
                    lOverlap.append(elt2)
#                        print elt2.getID(), elt2.getY(), elt.getY() + elt.getHeight() 
        lOverlap.sort(key=lambda x:x.getX())
        print 'x-left overY', lOverlap
        if lOverlap !=[]:
            value  = lOverlap[-1].getX()+lOverlap[-1].getWidth()
            filterList= filter(lambda x: x.getX()+x.getWidth() >= value,lOverlap)
#                print value, filterList
#                print "\t" , map(lambda x: x.getID(),lOverlap), value,map(lambda x: x.getID(),filterList) 

            for e in filterList:
                if e not in elt1.Ynearest[0]:
                    elt1.Ynearest[0].append(e)
                if elt1 not in e.Ynearest[1]:
                    e.Ynearest[1].append(elt1)  

        
        # get nearest top
        topElt = filter(lambda x: x < elt1.getY(),selfx.Y1Tree.keys())
        ## do not start after the end of elt1
        leftequalElt =  filter(lambda x: x < elt1.getX()+elt1.getWidth(),selfx.X1Tree.keys())
        lOverlap= []
        for y in topElt:
            for elt2 in selfx.Y1Tree[y]:
                if elt2.getX() in leftequalElt and elt1.overlapX(elt2):
#                        print elt1,elt2
                    lOverlap.append(elt2)
#            lOverlap.sort(key=lambda x:x.getBaseline())
        lOverlap.sort(key=lambda x:x.getY())
        print 'over y', lOverlap
        if lOverlap !=[]:
            # how to define a near (est?) object
            # take the nearest Y and add elements which end after it
            ## getY or getY+getHeight ?
            ### take the baseline !!!
            value  = lOverlap[-1].getY()
#                 print  value,  lOverlap[-1].getY(), lOverlap
#                 filterList= filter(lambda x: x.getY()+x.getHeight() >= value,lOverlap)
            filterList= filter(lambda x: x.getY() >= value,lOverlap)

#                value  = lOverlap[-1].getBaseline()
#                filterList= filter(lambda x: x.getBaseline() >= value,lOverlap)
#                 print elt1, filterList
            for e in filterList:
                if e not in elt1.Xnearest[0]:
                    elt1.Xnearest[0].append(e)
                if elt1 not in e.Xnearest[1]:
                    e.Xnearest[1].append(elt1)        

    print
    
def storeInKTree(selfx,listx):
    selfx.X1Tree = {}
    selfx.Y1Tree = {}
    selfx.X2Tree = {}
    selfx.Y2Tree = {}   
    for elt in listx:
        try:selfx.X1Tree[elt.getX()].append(elt)
        except KeyError: selfx.X1Tree[elt.getX()] = [elt]
        try:selfx.X2Tree[elt.getX()+elt.getHeight()].append(elt)
        except KeyError: selfx.X2Tree[elt.getX()+elt.getHeight()] = [elt]
        try:selfx.Y1Tree[elt.getY()].append(elt)
        except KeyError: selfx.Y1Tree[elt.getY()] = [elt]
        try:selfx.Y2Tree[elt.getY()+elt.getWidth()].append(elt)
        except KeyError: selfx.Y2Tree[elt.getX()+elt.getWidth()] = [elt]




def getBoundingBoxOLD(l):
        minbx = 9e9
        minby = 9e9
        maxbx = 0
        maxby = 0
        for elt in l:
            print elt
            if elt.getX()>=0 and elt.getX() < minbx: minbx = elt.getX()
            if elt.getY()>=0 and elt.getY() < minby: minby = elt.getY()
            if elt.getX() + elt.getWidth() > maxbx: maxbx = elt.getX() + elt.getWidth()
            if elt.getY() + elt.getHeight()  > maxby: maxby = elt.getY() + elt.getHeight()
        return minbx,minby,maxby-minby,maxbx-minbx    
