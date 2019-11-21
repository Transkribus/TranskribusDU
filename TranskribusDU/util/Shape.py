# -*- coding: utf-8 -*-

"""
    Utilities to deal with the PageXMl 2D objects using shapely
    

    Copyright Xerox(C) 2018 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import shapely.geometry as geom
from shapely.prepared import prep
from shapely.ops import cascaded_union        
from rtree import index

import numpy as np

from xml_formats.PageXml import MultiPageXml 

class ShapeLoader:

    @classmethod
    def getCoordsString(cls, o, bFailSafe=False):
        """
        Produce the usual content of the "Coords" attribute, e.g.:
            "3162,1205 3162,1410 126,1410 126,1205 3162,1205"
        may raise an exception
        """
        try:
            lt2 = o.exterior.coords  # e.g. [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)]
        except:
            if bFailSafe:
                try:
                    lt2 = o.coords
                except:
                    return ""
            else:
                lt2 = o.coords
        return " ".join("%d,%d" % (a,b) for a,b in lt2)

    
    @classmethod
    def contourObject(cls, lNd):
        """
        return the stringified list of coordinates of the contour 
        for the list of PageXml node.
            e.g. "3162,1205 3162,1410 126,1410 126,1205 3162,1205"
        return "" upon error
        
        if bShapelyObjecy is True, then return the Shapely object
        raise an Exception upon error
        """
        lp = []
        for nd in lNd:
            try:
                lp.append(ShapeLoader.node_to_Polygon(nd))
            except:
                pass
            
        o = cascaded_union([p if p.is_valid else p.convex_hull for p in lp ]) 
        return o

    @classmethod
    def minimum_rotated_rectangle(cls, lNd, bShapelyObject=False):
        """
        return the stringified list of coordinates of the minimum rotated 
        rectangle for the list of PageXml node.
            e.g. "3162,1205 3162,1410 126,1410 126,1205 3162,1205"
        return "" upon error
        
        if bShapelyObjecy is True, then return the Shapely object
        raise an Exception upon error
        """
        contour = cls.contourObject(lNd) 
        o = contour.minimum_rotated_rectangle
        return o if bShapelyObject else cls.getCoordsString(o, bFailSafe=True)

    @classmethod
    def convex_hull(cls, lNd, bShapelyObject):    
        """
        return the stringified list of coordinates of the minimum rotated 
        rectangle for the list of PageXml node.
            e.g. "3162,1205 3162,1410 126,1410 126,1205 3162,1205"
        return "" upon error
        
        if bShapelyObjecy is True, then return the Shapely object
        raise an Exception upon error
        """
        contour = cls.contourObject(lNd) 
        o = contour.convex_hull
        return o if bShapelyObject else cls.getCoordsString(o, bFailSafe=True)
        
    @classmethod
    def node_to_Point(cls, nd):
        """
        Find the points attribute (either in the DOM node itself or in a 
        children Coord node)
        Parse the points series as a LineString
        Return its centroid 
        """
        return cls._shapeFromNodePoints(nd, geom.LineString).centroid

    @classmethod
    def node_to_LineString(cls, nd):
        """
        Find the points attribute (either in the DOM node itself or in a 
        children Coord node)
        Parse the points series
        Return a LineString shapely object
        """
        return cls._shapeFromNodePoints(nd, geom.LineString)

    @classmethod
    def node_to_SingleLine(cls, nd):
        """
        Find the points attribute (either in the DOM node itself or in a 
        children Coord node)
        Parse the points series
        Do a linear regression to create a single-segment LineString
        Return a LineString shapely object with a single segment
        """
        o = cls._shapeFromNodePoints(nd, geom.LineString)
        if len(o.coords) == 2:
            # speed-up
            return o
        else:
            return cls.LinearRegression(o)

    @classmethod
    def node_to_Polygon(cls, nd, bValid=True):
        """
        Find the points attribute (either in the DOM node itself or in a 
        children Coord node)
        Parse the points series
        Return a Polygon shapely object
        """
        p = cls._shapeFromNodePoints(nd, geom.Polygon)
        if bValid and not p.is_valid: 
            # making sure it is a valid shape
            p = p.buffer(0)
        return p

    @classmethod
    def children_to_LineString(cls, node, name, fun=None):
        """
        do a MultiPageXml.getChildByName from a node to get some nodes
            e.g. "SeparatorRegion"
        construct a shapely LineString from the coordinates of each node
            e.g.  <SeparatorRegion orient="horizontal 373.8 0.006" row="1">
                <Coords points="3324,392 3638,394"/>
              </SeparatorRegion>
        if fun is given applies fun, with arguments: shape, current_node
        return the list of shape objects in same order as retrived by getChildByName
        """
        return cls._doShape_getChildByName(node, name
                                           , geom.LineString
                                           , fun)
        
    @classmethod
    def children_to_LinearRing(cls, node, name, fun=None):
        """
        do a MultiPageXml.getChildByName from a node to get some nodes
            e.g. "SeparatorRegion"
        construct a shapely LinearRing from the coordinates of each node
            e.g.  <SeparatorRegion orient="horizontal 373.8 0.006" row="1">
                <Coords points="3324,392 3638,394"/>
              </SeparatorRegion>
        if fun is given applies fun, with arguments: shape, current_node
        return the list of shape objects in same order as retrived by getChildByName
        """
        return cls._doShape_getChildByName(node, name
                                           , geom.LinearRing
                                           , fun)

#     @classmethod
#     def LinearRegression_bad(cls, oLineString, step=10):
#         """
#         make a single-segment LineString out of the input LineString
#         do a LinearRegression on a subsampling of the LineString
#         """
#         lCoord = list(oLineString.coords)      
#         lX, lY = [], []  
#         x0, y0 = lCoord.pop(0)
#         xx00 = x0
#         for x1, y1 in lCoord:
#             try:
#                 a = (y1 - y0) / (x1 - x0)
#                 #for x in range(x0, x1, step):
#                 x = x0
#                 while x < x1:
#                     y = y0 + (x - x0) * a
#                     lX.append(x)
#                     lY.append(y)
#                     x += step
#             except ZeroDivisionError:
#                 # vertical segment?
#                 try:
#                     a = (x1 - x0) / (y1 - y0)
#                     while y < y1:
#                         x = x0 + (y - y0) * a
#                         lX.append(x)
#                         lY.append(y)
#                         y += step
#                 except:
#                     pass # empty segment
#             x0, y0 = x1, y1
#         lX.append(x0)
#         lY.append(y0)
#         z = np.polyfit(np.array(lX), np.array(lY), 1)
#         y0 = round(z[1] + z[0] * xx00, 2)
#         y1 = round(z[1] + z[0] * x0, 2)
#         
#         return geom.LineString([(xx00, y0), (x0, y1)])
    
    @classmethod
    def LinearRegression(cls, oLineString):
        """
        make a single-segment LineString out of the input LineString
        do a LinearRegression of the LineString segments
        """
        return geom.LineString(cls._LinearRegression(oLineString.coords))
    
    @classmethod
    def _LinearRegression(cls, lCoords):
        """
        weighted LinearRegression of the segments
        """
        XY = np.array(lCoords)
        # Norm of each segment
        Norm = np.linalg.norm(np.roll(XY, -1, axis=0) - XY, axis=1)
        Norm[-1] = 0
        # a point gets the sum of norm of the adjacent segment (1 at each end)
        W = Norm + np.roll(Norm, 1)
        W = np.sqrt(W / W.sum())
        z = np.polyfit(XY[:,0], XY[:,1], 1, w=W)
        x0, x1 = XY[0,0], XY[-1,0]
        y0 = round(z[1] + z[0] * x0, 2)
        y1 = round(z[1] + z[0] * x1, 2)
        return ((x0, y0), (x1, y1))

    @classmethod
    def _shapeFromPoints(cls, sPoints, ShapeClass):
        """
        Parse a string containing a space-separated list of points, like:
        "3466,3342 3512,3342 3512,3392 3466,3392"
        returns a shape of given class
        """
        # line = LineString([(0, 0), (1, 1)])
        return ShapeClass(tuple(int(_v) for _v in _sPair.split(','))
                            for _sPair in sPoints.split(' '))
                                

    @classmethod
    def _shapeFromNodePoints(cls, nd, ShapeClass):
        """
        Find the Coords child of the node and parse its points
         e.g. <SeparatorRegion orient="horizontal 373.8 0.006" row="1">
                <Coords points="3324,392 3638,394"/>
              </SeparatorRegion>
        returns a shape of given class
        """
        sPoints = nd.get('points')
        if sPoints is None:
            sPoints  = MultiPageXml.getChildByName(nd, 'Coords')[0].get('points')
        return cls._shapeFromPoints(sPoints, ShapeClass)
        
    @classmethod
    def _doShape_getChildByName(cls, node, name, ShapeClass, fun=None):
        """
        do a MultiPageXml.getChildByName from a node to get some nodes
            e.g. "SeparatorRegion"
        construct a shape of given Shapely class from the coordinates of each node
            e.g.  <SeparatorRegion orient="horizontal 373.8 0.006" row="1">
                <Coords points="3324,392 3638,394"/>
              </SeparatorRegion>
        if fun is given applies fun, with arguments: shape, current_node
        return the list of shape objects in same order as retrived by getChildByName
        """
        lO = []
        for _nd in MultiPageXml.getChildByName(node, name):
            try:
                o = cls._shapeFromNodePoints(_nd, ShapeClass)
            except Exception as e:
                print('ERROR: cannot load this element "%s"' % str(_nd))
                print('  because "%s"' % e)
                continue
            if not fun is None: fun(o, _nd)
            lO.append(o)
            
        return lO


class ShapePartition:
    """
    Given a list of shapely objects, deal with partition of those objects
    """
    CollectionClass = geom.collection
    INF = 99999 # INFINITY...
    
    def __init__(self, lo):
        """
        Initialize the list of objects
        """
        self.lo = lo
        self.idx = index.Index()
        
        # Populate R-tree index with bounds of grid cells
        for pos, o in enumerate(self.lo):
            self.idx.insert(pos, o.bounds)

    def isValidCut(self, oLine):
        """
        Does this line passes between the objects?
        """
        prepO = prep(oLine)
        for pos in self.idx.intersection(oLine.bounds):
            if prepO.intersects(self.lo[pos]): return False
        return True

    def isValidRibbonCut(self, oLine, h):
        """
        Does this ribbon of height h passes between the objects?
        """
        (xa, ya), (xb, yb) = oLine.coords
        oRibbon = geom.Polygon([  (xa, ya)    , (xb, yb)
                                , (xb, yb + h), (xa, ya + h)])
        prepO = prep(oRibbon)
        for pos in self.idx.intersection(oRibbon.bounds):
            if prepO.intersects(self.lo[pos]): return False
        return True
        
    def getObjectAboveLineByIds(self, oLine):
        """
        return a tuple of the index of the objects above the line
        The index is based on the initial list of objects passed to __init__
        """
        xa, ya = oLine.coords[0]
        xb, yb = oLine.coords[-1]
        if xa > xb: xa, ya, xb, yb = xb, yb, xa, ya
        half_plan = geom.Polygon([(xa, -self.INF) , (xb, -self.INF)
                                , (xb, yb)      , (xa, ya)
                                ])
        bounds = half_plan.bounds
        half_plan_prep = prep(half_plan) # speed up!!
        return tuple(pos for pos in self.idx.intersection(bounds)
                        if half_plan_prep.intersects(self.lo[pos]))
        # NOTE: intersects is better than contains because some Baseline may go beyond page limit...

    def getObjectBelowLineByIds(self, oLine):
        """
        return a tuple of the index of the objects below the line
        The index is based on the initial list of objects passed to __init__
        """
        xa, ya = oLine.coords[0]
        xb, yb = oLine.coords[-1]
        if xa > xb: xa, ya, xb, yb = xb, yb, xa, ya
        half_plan = geom.Polygon([(xa, +self.INF) , (xb, +self.INF)
                                , (xb, yb)      , (xa, ya)
                                ])
        bounds = half_plan.bounds
        half_plan_prep = prep(half_plan) # speed up!!
        return tuple(pos for pos in self.idx.intersection(bounds)
                        if half_plan_prep.intersects(self.lo[pos]))
        # NOTE: intersects is better than contains because some Baseline may go beyond page limit...

    def getObjectLeftOfLineByIds(self, oLine):
        """
        return a tuple of the index of the objects on left of the line
        The index is based on the initial list of objects passed to __init__
        """
        xa, ya = oLine.coords[0]
        xb, yb = oLine.coords[-1]
        if ya > yb: xa, ya, xb, yb = xb, yb, xa, ya
        half_plan = geom.Polygon([(-self.INF, ya) , (-self.INF, yb)
                                , (xb, yb)      , (xa, ya)
                                ])
        bounds = half_plan.bounds
        half_plan_prep = prep(half_plan) # speed up!!
        return tuple(pos for pos in self.idx.intersection(bounds)
                        if half_plan_prep.intersects(self.lo[pos]))
        # NOTE: intersects is better than contains because some Baseline may go beyond page limit...

    def getObjectRightOfLineByIds(self, oLine):
        """
        return a tuple of the index of the objects on left of the line
        The index is based on the initial list of objects passed to __init__
        """
        xa, ya = oLine.coords[0]
        xb, yb = oLine.coords[-1]
        if ya > yb: xa, ya, xb, yb = xb, yb, xa, ya
        half_plan = geom.Polygon([(+self.INF, ya) , (+self.INF, yb)
                                , (xb, yb)      , (xa, ya)
                                ])
        bounds = half_plan.bounds
        half_plan_prep = prep(half_plan) # speed up!!
        return tuple(pos for pos in self.idx.intersection(bounds)
                        if half_plan_prep.intersects(self.lo[pos]))

    def getObjects(self, lId):
        """
        Return the list of corresponding objects
        """
        return [self.lo[pos] for pos in lId]
    
    def getObjectAboveLine(self, oLine):
        """
        Given a list of objects above the given line
        """
        return tuple(self.lo[pos] for pos in self.getObjectAboveLineByIds(oLine))
    
    def getObjectBelowLine(self, oLine):
        """
        Given a list of objects below the given line
        """
        return tuple(self.lo[pos] for pos in self.getObjectBelowLineByIds(oLine))

    def getObjectOnLeftOfLine(self, oLine):
        """
        Given a list of objects on left of the given line
        """
        return tuple(self.lo[pos] for pos in self.getObjectLeftOfLineByIds(oLine))

    def getObjectOnRightOfLine(self, oLine):
        """
        Given a list of objects on left of the given line
        """
        return tuple(self.lo[pos] for pos in self.getObjectRightOfLineByIds(oLine))

    def free(self):
        """
        disposal of any internal resource of self
        """
        del self.lo, self.idx


class PolygonPartition(ShapePartition):
    """
    Partition of LineString objects
    """
    CollectionClass = geom.MultiPolygon
    
    def __init__(self, lo):
        """
        Initialize the list of objects
        """
        ShapePartition.__init__(self, lo)

    
# -----------------------------------------------------------------------
def test_ShapeLoader():
    o = ShapeLoader._shapeFromPoints("0,0 0,9", geom.LineString)
    assert o.length == 9
    assert o.area == 0.0

def test_ShapeLoader_Coords():
    s = "3162,1205 3162,1410 126,1410 3162,1205"
    o = ShapeLoader._shapeFromPoints(s, geom.Polygon)
    assert ShapeLoader.getCoordsString(o) == s

# -----------------------------------------------------------------------
def test_ShapePartition_object_above(capsys):
        with capsys.disabled():
    
            for cls in [ShapePartition, PolygonPartition]:
                b1 = geom.Polygon([(1,2), (2,2), (2,3), (1,3)]) 
                b2 = geom.Polygon([(3,7), (4,7), (4,8), (3,8)])
                o = cls([b1,b2])
            
                l0 = geom.LineString([(0,0), (10,0)])
                assert o.getObjectAboveLineByIds(l0) == ()
                assert o.getObjectAboveLine(l0) == ()
                
                l1 = geom.LineString([(0,6), (10,5)])
                assert o.getObjectAboveLineByIds(l1) == (0,)
                assert o.getObjectAboveLine(l1) == (b1,)
                
                l2 = geom.LineString([(0,60), (10,50)])
                assert o.getObjectAboveLineByIds(l2) == (0,1)
                assert o.getObjectAboveLine(l2) == (b1, b2)
            
                l3 = geom.LineString([(0,8), (10,7)])
                # above if intersect with half-plan
                assert o.getObjectAboveLineByIds(l3) == (0,1)
                assert o.getObjectAboveLine(l3) == (b1,b2)

                o.free()
                
                
def test_ShapePartition_cut():
                
    for cls in [ShapePartition, PolygonPartition]:
        b1 = geom.Polygon([(1,2), (2,2), (2,3), (1,3)]) 
        b2 = geom.Polygon([(3,7), (4,7), (4,8), (3,8)])
        o = cls([b1,b2])

        assert o.isValidCut(geom.LineString([(0,0), (0,50)]))
        assert o.isValidCut(geom.LineString([(0,0), (50,0)]))
        assert o.isValidCut(geom.LineString([(0,4), (50,4)]))
        assert o.isValidCut(geom.LineString([(0,40), (50,40)]))
        assert o.isValidCut(geom.LineString([(2.5,40), (2.5,40)]))
        assert o.isValidCut(geom.LineString([(0,10), (5,0)]))
        
        assert not o.isValidCut(geom.LineString([(0,0), (3,5)]))
        assert not o.isValidCut(geom.LineString([(0,0), (5,10)]))
        
def test_ShapePartition_cut_ribbon():
                
    for cls in [ShapePartition, PolygonPartition]:
        b1 = geom.Polygon([(1,2), (2,2), (2,3), (1,3)]) 
        b2 = geom.Polygon([(3,7), (4,7), (4,8), (3,8)])
        o = cls([b1,b2])

        # height of 0 => same as before
        h = 0
        assert o.isValidRibbonCut(geom.LineString([(0,0), (0,50)]), h)
        assert o.isValidRibbonCut(geom.LineString([(0,0), (50,0)]), h)
        assert o.isValidRibbonCut(geom.LineString([(0,4), (50,4)]), h)
        assert o.isValidRibbonCut(geom.LineString([(0,40), (50,40)]), h)
        assert o.isValidRibbonCut(geom.LineString([(2.5,40), (2.5,40)]), h)
        assert o.isValidRibbonCut(geom.LineString([(0,10), (5,0)]), h)
        
        assert not o.isValidRibbonCut(geom.LineString([(0,0), (3,5)]), h)
        assert not o.isValidRibbonCut(geom.LineString([(0,0), (5,10)]), h)
         
        # heigt of 1 => same
        h = 1
        assert o.isValidRibbonCut(geom.LineString([(0,0), (0,50)]), h)
        assert o.isValidRibbonCut(geom.LineString([(0,0), (50,0)]), h)
        assert o.isValidRibbonCut(geom.LineString([(0,4), (50,4)]), h)
        assert o.isValidRibbonCut(geom.LineString([(0,40), (50,40)]), h)
        assert o.isValidRibbonCut(geom.LineString([(2.5,40), (2.5,40)]), h)
        assert o.isValidRibbonCut(geom.LineString([(0,10), (5,0)]), h)
        
        assert not o.isValidRibbonCut(geom.LineString([(0,0), (3,5)]), h)
        assert not o.isValidRibbonCut(geom.LineString([(0,0), (5,10)]), h)
         
        # heigt of 1 => Noo
        h = 10
        assert o.isValidRibbonCut(geom.LineString([(0,0), (0,50)]), h)
        assert not o.isValidRibbonCut(geom.LineString([(0,0), (50,0)]), h)
        assert not o.isValidRibbonCut(geom.LineString([(0,4), (50,4)]), h)
        assert o.isValidRibbonCut(geom.LineString([(0,40), (50,40)]), h)
        assert o.isValidRibbonCut(geom.LineString([(2.5,40), (2.5,40)]), h)
        assert not o.isValidRibbonCut(geom.LineString([(0,10), (5,0)]), h)
        
        assert not o.isValidRibbonCut(geom.LineString([(0,0), (3,5)]), h)
        assert not o.isValidRibbonCut(geom.LineString([(0,0), (5,10)]), h)


def test_LineStringToLine(capsys):
    "1113,1281 1318,1274 1320,1324 1115,133"
    
#     # OLD BAD EMPIRICAL METHOD
#     with capsys.disabled():
#         for step in [1, 2, 10]:
#             o05 = geom.LineString([(0,0), (5,5)])
#             o = ShapeLoader.LinearRegression_bad(o05, step)
#             assert o05 == o, str(o)
#             
#             o39 = geom.LineString([(3,2), (9,10)])
#             o = ShapeLoader.LinearRegression_bad(o39, step)
#             assert o39 == o, str(o)
#             
#             o08 = geom.LineString([(0,0), (5,5), (8,8)])
#             o = ShapeLoader.LinearRegression_bad(o08, step)
#             assert o == geom.LineString([(0,0), (8,8)]), str(o)
#         
#         o10 = geom.LineString([(0,0), (5,5), (10,0)])
#         o = ShapeLoader.LinearRegression_bad(o10, 1)
#         assert o == geom.LineString([(0,2.27), (10,2.27)]), str(o)
#         o = ShapeLoader.LinearRegression_bad(o10, 0.001)
#         assert o == geom.LineString([(0,2.5), (10,2.5)]), str(o)
#         
#         oBL = geom.LineString([(1734,654),(2013,640),(2197,646),(2439,611)])
#         o = ShapeLoader.LinearRegression_bad(oBL)
#         assert o == geom.LineString([(1734,655.64), (2439,623.28)]), str(o)
#         
#         oBL = geom.LineString([(10,10),(20,10),(20,20),(30,20)])
#         o = ShapeLoader.LinearRegression_bad(oBL, 0.001)
#         assert o == geom.LineString([(10,7.5), (30,22.5)]), str(o)
        
    # GOOD METHOD
    with capsys.disabled():
        o05 = geom.LineString([(0,0), (5,5)])
        o = ShapeLoader.LinearRegression(o05)
        assert o05 == o, str(o)
        
        o39 = geom.LineString([(3,2), (9,10)])
        o = ShapeLoader.LinearRegression(o39)
        assert o39 == o, str(o)
        
        o08 = geom.LineString([(0,0), (5,5), (8,8)])
        o = ShapeLoader.LinearRegression(o08)
        assert o == geom.LineString([(0,0), (8,8)]), str(o)
        
        o10 = geom.LineString([(0,0), (5,5), (10,0)])
        o = ShapeLoader.LinearRegression(o10)
        assert o == geom.LineString([(0,2.5), (10,2.5)]), str(o)
        
        oBL = geom.LineString([(10,10),(20,10),(20,20),(30,20)])
        o = ShapeLoader.LinearRegression(oBL)
        assert o == geom.LineString([(10,10), (30,20)]), str(o)

        oBL = geom.LineString([(1734,654),(2013,640),(2197,646),(2439,611)])
        o = ShapeLoader.LinearRegression(oBL)
        assert o == geom.LineString([(1734,657), (2439,622.21)]), str(o)
        
        