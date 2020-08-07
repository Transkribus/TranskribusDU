# -*- coding: utf-8 -*-

"""
    Utilities to deal with the PageXMl polygon
    

    Copyright Xerox(C) 2016 H. Déjean, JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""


class Polygon:
    
    def __init__(self, lXY):
        assert lXY, "ERROR: empty list of points"
        self.lXY = [(x,y) for x,y in lXY]
    
    @classmethod
    def parsePoints(cls, sPoints):
        """
        Parse a string containing a space-separated list of points, like:
        "3466,3342 3512,3342 3512,3392 3466,3392"
        returns a Polygon object
        """
        it_sXsY = (sPair.split(',') for sPair in sPoints.split(' '))
        return Polygon((int(sx), int(sy)) for sx, sy in it_sXsY)

    def lX(self): return [x for x,_y in self.lXY]
    def lY(self): return [y for _x,y in self.lXY]
    
    def getBoundingBox(self):
        """
        return x1,y1,x2,y2    (top-left, bottom-right)
        """
        lX, lY = self.lX(), self.lY()
        return min(lX), min(lY), max(lX), max(lY)
        
    def getArea(self):
        """
        https://fr.wikipedia.org/wiki/Aire_et_centre_de_masse_d'un_polygone
        return a positive float
        """
        if len(self.lXY) < 2: raise ValueError("Only one point: polygon area is undefined.")
        fA = 0.0    #float!
        xprev, yprev = self.lXY[-1]
        for x, y in self.lXY:
            fA += xprev*y - yprev*x
            xprev, yprev = x, y
        return abs(fA / 2)
    
    def getArea_and_CenterOfMass(self):
        """
        https://fr.wikipedia.org/wiki/Aire_et_centre_de_masse_d'un_polygone
        
        return A, (Xg, Yg) which are the area and the coordinates (float) of the center of mass of the polygon
        """
        if len(self.lXY) < 2: raise ValueError("Only one point: polygon area is undefined.")
        
        fA = 0.0
        xSum, ySum = 0, 0
        
        
        xprev, yprev = self.lXY[-1]
        for x, y in self.lXY:
            iTerm = xprev*y - yprev*x
            fA   += iTerm
            xSum += iTerm * (xprev+x)
            ySum += iTerm * (yprev+y)
            xprev, yprev = x, y
        if fA == 0.0: raise ValueError("surface == 0.0")
        fA = fA / 2
        xg, yg = xSum/6/fA, ySum/6/fA
        
        if fA <0:
            return -fA, (xg, yg)
        else:
            return fA, (xg, yg)
        assert fA >0 and xg >0 and  yg >0, "%s\t%s"%(self.lXY (fA, (xg, yg)))
        return fA, (xg, yg)
        
#     def fitRectangleByBaseline(self, lBaselineXY):
#         """
#         Fit a rectangle at best to match the polygone (polyline) of an object, using the object baseline
#         
#         The resulting rectangle has same area and center of mass as the polygon
#         
#         Return, x, y, w, h of the rectangle   (int values)
#         """
#         
#         #TODO MAKE SOMETHING MORE GENERIC!!
#         
#         #average y of the baseline
#         lBaselineY = [y for x,y in lBaselineXY]
#         avgYBaseline = sum(lBaselineY) / float(len(lBaselineY))
        
        
    def fitRectangle(self, bPreserveWidth=True):
        """
        Fit a rectangle at best to match the polygone (polyline) of an object
        
        The resulting rectangle has same area and center of mass as the polygon
        
        If bKeepWidth is True, then only the height is adjusted.
        
        Return x1,y1, x2,y2 of the rectangle   (int values)  (top-left, bottom-right)
        """
        
        #TODO MAKE SOMETHING MORE GENERIC!!
        
        fA, (fXg, fYg) = self.getArea_and_CenterOfMass()
        
        x1,_y1, x2,_y2 = self.getBoundingBox()
        #build a rectangle with same "width" as the polygon...    is-it good enough??
        w = x2 - x1
        
        #but this width should not lead to go out of the bounding box!
        if bPreserveWidth:
            fW = float(abs(x2-x1))
            fH = fA / fW
            x1,y1, x2,y2 = [ int(round(v)) for v in [  x1, fYg - fH/2
                                                     , x2, fYg + fH/2 ]]
        else:
            #historical code
            fW = min(w, (x2-fXg)*2, (fXg-x1)*2)
        
        #same area
        fH = fA / fW
        
        x1,y1, x2,y2 = [ int(round(v)) for v in [  fXg - fW/2.0, fYg - fH/2
                                                 , fXg + fW/2.0, fYg + fH/2 ]]
        
        return x1,y1, x2,y2
        
    def partitionSegmentTopRightBottomLeft(self):
        """
        partition the polygon segment into those on "top", on left, on bottom,
         on left of the centre of gravity of the polygon
        return 4 lists of segments
        """
        _fA, (fXg, fYg) = self.getArea_and_CenterOfMass()

        if len(self.lXY) < 2: raise ValueError("Only one point: wrong polygon.")

        #lists top, right, bottom, left segments
        lT, lR, lB, lL = [],[],[],[]
        xprev, yprev = self.lXY[-1]
        for x, y in self.lXY:
            segment = (xprev, yprev, x, y)
            dx = x - xprev
            dy = y - yprev
            xm = (x + xprev) / 2.0
            ym = (y + yprev) / 2.0
            
            if abs(dx) > abs(dy):
                # "horizontal segment" :)
                if fYg > ym:    # Y axis goes downward!!!
                    #left one
                    lT.append(segment)
                else:
                    lB.append(segment)
            else:
                # "vertical" segment
                if fXg < xm:
                    lR.append(segment)
                else:
                    lL.append(segment)
            xprev, yprev = x, y    

        return lT, lR, lB, lL

    
def test_trigo():    
    print([(3673, 1721), (3744, 1742), (3944, 1729), (3946, 1764), (3740, 1777), (3664, 1755)])
    p = Polygon([(3673, 1721), (3744, 1742), (3944, 1729), (3946, 1764), (3740, 1777), (3664, 1755)])
    fA, (xg, yg) =p.getArea_and_CenterOfMass()
    assert fA and xg > 0 and yg >0
    ## trigo   
    
    print( [(253, 129), (356, 108), (363, 142), (260, 163)])
    p = Polygon([(253, 129), (356, 108), (363, 142), (260, 163)])
    fA, (xg, yg) =p.getArea_and_CenterOfMass()
    assert fA and xg > 0 and yg >0
    
    #non trigo
    print([(4140, 2771), (4140, 3400), (4340, 3400), (4340, 2771)])
    p = Polygon([(4140, 2771), (4140, 3400), (4340, 3400), (4340, 2771)])
    fA, (xg, yg) =  p.getArea_and_CenterOfMass()
    assert fA and xg > 0 and yg >0      
    

def test_parse():
    s = "3466,3342 3512,3342"  
    p = Polygon.parsePoints(s)
    assert p.lXY == [ (3466,3342), (3512,3342) ]
        
def test_partition():
    s = "0,0 1,3 4,4 6,3 5,1 3,0"  
    p = Polygon.parsePoints(s)
    lT, lR, lB, lL = p.partitionSegmentTopRightBottomLeft()
    assert lB == [(1, 3, 4, 4), (4, 4, 6, 3)]
    assert lR == [(6, 3, 5, 1)]
    assert lT == [(3, 0, 0, 0), (5, 1, 3, 0)]
    assert lL == [(0, 0, 1, 3)]

    s = "0,0 1,3 4,4 6,3 5,1 6,-1 3,0"  #added one before last  
    p = Polygon.parsePoints(s)
    lT, lR, lB, lL = p.partitionSegmentTopRightBottomLeft()
    assert lB == [(1, 3, 4, 4), (4, 4, 6, 3)]
    assert lR == [(6, 3, 5, 1), (5,1,6,-1)]
    assert lT == [(3, 0, 0, 0), (6,-1, 3,0)]
    assert lL == [(0, 0, 1, 3)]

        
# s = "0,0 1,3 4,4 6,3 5,1 3,0"  
# s = "0,0 1,3 4,4 6,3 5,1 6,-1 3,0"  #added one before last  
# s = "65,41 67,322 485,323 480,42"
# p = Polygon.parsePoints(s)
# print(p.lXY)
# (lT, lR, lB, lL) =  p.partitionSegmentTopRightBottomLeft()
# print("\t", lT)
# print(lL, "   ", lR)
# print("\t", lB)
         
