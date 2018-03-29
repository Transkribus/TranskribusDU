# -*- coding: utf-8 -*-

"""
    Utilities to deal with the PageXMl polygon
    

    Copyright Xerox(C) 2016 H. Déjean, JL. Meunier

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
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

class Polygon:
    
    def __init__(self, lXY):
        assert lXY, "ERROR: empty list of points"
        self.lXY = [(x,y) for x,y in lXY]
    
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
        
        
    def fitRectangle(self):
        """
        Fit a rectangle at best to match the polygone (polyline) of an object
        
        The resulting rectangle has same area and center of mass as the polygon
        
        Return x1,y1, x2,y2 of the rectangle   (int values)  (top-left, bottom-right)
        """
        
        #TODO MAKE SOMETHING MORE GENERIC!!
        
        fA, (fXg, fYg) = self.getArea_and_CenterOfMass()
        
        x1,_y1, x2,_y2 = self.getBoundingBox()
        #build a rectangle with same "width" as the polygon...    is-it good enough??
        w = x2 - x1
        
        #but this width should not lead to go out of the bounding box!
        fW = min(w, (x2-fXg)*2, (fXg-x1)*2)
        
        #same area
        fH = fA / fW
        
        x1,y1, x2,y2 = [ int(round(v)) for v in [  fXg - fW/2.0, fYg - fH/2
                                                 , fXg + fW/2.0, fYg + fH/2 ]]
        
        return x1,y1, x2,y2
        
        
        
    def clipping(self,clipPolygon):
        """
        Weiler–Atherton clipping algorithm
        
        Candidate polygons need to be oriented clockwise.
        Candidate polygons should not be self-intersecting (i.e., re-entrant).
        
        Input: clipping region list of 2plets (x,y)
        Output: clipped region of self (list of 2plets (x,y))
        
        """
        
        def inside(p):
            return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
     
        def computeIntersection():
            dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
            dp = [ s[0] - e[0], s[1] - e[1] ]
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
            n2 = s[0] * e[1] - s[1] * e[0] 
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
            return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
        
        outputList = self.lXY
        cp1 = clipPolygon[-1]
        for clipVertex in clipPolygon:
            cp2 = clipVertex
            inputList = outputList
            outputList = []
            s = inputList[-1]
     
            for subjectVertex in inputList:
                e = subjectVertex
                if inside(e):
                    if not inside(s):
                        outputList.append(computeIntersection())
                    outputList.append(e)
                elif inside(s):
                    outputList.append(computeIntersection())
                s = e
            cp1 = cp2
        return(outputList)        
        
        
    def signedOverlap(self,p2):
        
        def overlapX(p1,p2):
            x21,y21,x22,y22 = p2
            x11,y11,x12,y12 = p1
            [a1,a2] = x11,x12
            [b1,b2] = x21,x22
            return min(a2, b2) >=   max(a1, b1) 
        
        def overlapY(p1,p2):
            x21,y21,x22,y22 = p2
            x11,y11,x12,y12 = p1
            [a1,a2] = y11,y12
            [b1,b2] = y21,y22          
            return min(a2, b2) >=  max(a1, b1)             
        
        
        x21,y21,x22,y22 = p2
        x11,y11,x12,y12 = self.getBoundingBox()
        
        w1 = x12 - x11
        h1 = y12 - y11
        
        fOverlap = 0.0
        
        if overlapX(self.getBoundingBox(),p2) and overlapY(self.getBoundingBox(),p2):
            s1 = w1 * h1
            
            # possible ?
            if s1 == 0: s1 = 1.0
            
            #intersection
            nx1 = max(x11,x21)
            nx2 = min(x12,x22)
            ny1 = max(y11,y21)
            ny2 = min(y12,y22)
            h = abs(nx2 - nx1)
            w = abs(ny2 - ny1)
            
            inter = h * w
            if inter > 0 :
                fOverlap = inter/s1
            else:
                # if overX and Y this is not possible !
                fOverlap = 0.0
            
        return  fOverlap  
        
def test_clipping(capsys):
    p1 = Polygon([(50,150),(200,50),(350,150),(350,300),(250,250),(150,350),(100,250),(100,200)])
    p2 = Polygon([(100,100),(300,100),(300,300),(100,300)])
    with capsys.disabled(): #         
        lres = p1.clipping(p2.lXY)
        print (lres)
    
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
        
        
        
        
        
        
        