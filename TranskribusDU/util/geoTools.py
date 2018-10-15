from rtree import index
import numpy as np


def polygon2points(p):
    """
        convert a polygon to a sequence of points for DS documents
        :param p: shapely.geometry.Polygon
        returns a string representing the set of points
    """
    return ",".join(list("%s,%s"%(x,y) for x,y in p.exterior.coords))

def points2polygon(s):
    """
        convert a string (from DSxml) to a polygon
    """
    
def iuo(z1,z2):
    """
        intersection over union 
        :param z1: polygon
        :param z2: polygon
        returns z1.intersection(z2) / z1.union(z2)
        
    """
    assert z1.isvalid
    assert z2.isvalid
    return z1.intersection(z2) / z1.union(z2)


def populateGeo(lZones:list(),lElements:list()):
    """
        affect lElements i  to lZones using  argmax(overlap(elt,zone)
    """

    lIndElements =   index.Index()
    dPopulated = {}
    for pos, cell  in enumerate(lZones):
#         lIndElements.insert(pos, cell.toPolygon().bounds)
        lIndElements.insert(pos, cell.bounds)


    aIntersection = np.zeros((len(lZones),len(lElements)),dtype=float)
    for j,elt in enumerate(lElements):
#         ll  = lIndElements.intersection(elt.toPolygon().bounds)
#         for x in ll: aIntersection[x][j] =  elt.toPolygon().intersection(lZones[x].toPolygon()).area
        ll  = lIndElements.intersection(elt.bounds)
        for x in ll: 
#             print (elt,x,lZones[x], elt.intersection(lZones[x]).area)
            aIntersection[x][j] =  elt.intersection(lZones[x]).area
        
    for i,z in enumerate(lZones):
        best = np.argmax(aIntersection[i])
        print (z, lElements[best],best, aIntersection[i][best])
        try: dPopulated[best].append(z)
        except KeyError:dPopulated[best] = [z]
    
    return dPopulated

if __name__ == "__main__":
# def test_geo():
    from shapely.geometry import Polygon
    lP= []
    for i in range(0,100,10):
        lP.append(Polygon(((i,i),(i,i+10),(i+10,i+10),(i+10,i))))
#         print (lP[-1])
    lE= []
    for i in range(0,100,5):
        lE.append(Polygon(((i,i),(i,i+9),(i+9,i+9),(i+9,i))))
#         print (lE[-1])
    dres = populateGeo(lP,lE)
#     for item in dres:
#         print (lE[item],[str(x) for x in dres[item]])
    print(polygon2points(lP[0]))
      
        