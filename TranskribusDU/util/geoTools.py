from rtree import index
import numpy as np
#from shapely.prepared import prep
import shapely 
def polygon2points(p):
    """
        convert a polygon to a sequence of points for DS documents
        :param p: shapely.geometry.Polygon
        returns a string representing the set of points
    """
    return ",".join(list("%s,%s"%(x,y) for x,y in p.exterior.coords))

def sPoints2tuplePoints(s):
    """
        convert a string (from DSxml) to a polygon
        :param s: string = 'x,y x,y...'
        returns a Geometry
    """
#    lList = s.split(',') 
#    return [(float(x),float(y)) for x,y in  zip(lList[0::2],lList[1::2])]
    
    return [ (float(x),float(y)) for sxy in s.split(' ') for (x,y)  in sxy.split(',') ]

    
    
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
    for pos, z  in enumerate(lZones):
#         lIndElements.insert(pos, cell.toPolygon().bounds)
#         print (cell,cell.is_valid,cell.bounds)
        lIndElements.insert(pos, z.bounds)


    aIntersection = np.zeros((len(lElements),len(lZones)),dtype=float)
    for j,elt in enumerate(lElements):
        ll  = lIndElements.intersection(elt.bounds)
        for x in ll: 
            try:aIntersection[j][x] =  elt.intersection(lZones[x]).area
            except shapely.errors.TopologicalError: pass #This operation could not be performed. Reason: unknown

        
    for i,e in enumerate(lElements):
        best = np.argmax(aIntersection[i])
        # aIntersection == np.zeros : empty
        if aIntersection[i][best]>0:
            try: dPopulated[best].append(i)
            except KeyError:dPopulated[best] = [i]
    
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
    for item in dres:
        print (lE[item],[lE[x].wkt for x in dres[item]])

#     print(polygon2points(lP[0]))
      
    