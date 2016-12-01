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


class Polygon:
    
    def __init__(self, lXY):
        assert lXY, "ERROR: empty list of points"
        self.lXY = [(x,y) for x,y in lXY]
    
    def lX(self): return [x for x,y in self.lXY]
    def lY(self): return [y for x,y in self.lXY]
    
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
        fA = abs(fA / 2)
        xg, yg = xSum/6/fA, ySum/6/fA
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
        
        x1,y1, x2,y2 = self.getBoundingBox()
        #build a rectangle with same "width" as the polygon...    is-it good enough??
        w = x2 - x1
        
        #but this width should not lead to go out of the bounding box!
        fW = min(w, (x2-fXg)*2, (fXg-x1)*2)
        
        #same area
        fH = fA / fW
        
        x1,y1, x2,y2 = [ int(round(v)) for v in [  fXg - fW/2.0, fYg - fH/2
                                                 , fXg + fW/2.0, fYg + fH/2 ]]
        
        return x1,y1, x2,y2
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        