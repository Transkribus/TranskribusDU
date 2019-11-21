# -*- coding: utf-8 -*-

"""
    geometric intersection over union of shapely objects
    
    https://en.wikipedia.org/wiki/Jaccard_index

    Copyright Naver Labs Europe(C) 2019 H. Déjean, JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

def iou_distance(geomX, geomY):
    """
    intersection over union (area) of teh shapely geometric objects
    returns a cost (1-distance)
    """ 
#     print (x.bounds,y.bounds,x.intersection(y).area,cascaded_union([x,y]).area , x.intersection(y).area /cascaded_union([x,y]).area)
    try:    
        return  1.0 - geomX.intersection(geomY).area / geomX.union(geomY).area
    except ZeroDivisionError:
        return 0.0 if geomX == geomY else 1.0

def iou(geomX, geomY):
    """
    intersection over union (area) of teh shapely geometric objects
    returns a cost (1-distance)
    """ 
#     print (x.bounds,y.bounds,x.intersection(y).area,cascaded_union([x,y]).area , x.intersection(y).area /cascaded_union([x,y]).area)
    try:    
        return  geomX.intersection(geomY).area / geomX.union(geomY).area
    except ZeroDivisionError:
        return 1.0 if geomX == geomY else 0.0


# ----  tests  ---------------------------------------------------------
def test_iou():
    import shapely.geometry as geom
    
    def assert_iou_and_dist(a, b, ref_iou):
        assert iou(a, b) ==  ref_iou
        assert iou_distance   (a, b) == (1.0 - ref_iou)
        
    # empty areas    
    oo = geom.Polygon([(0,0), (1,1), (2,2)])
    oo2 = geom.Polygon([(0,0), (1,1), (2,2)])
    oo3 = geom.Polygon([(0,0), (1,1), (2,2), (3,3)])

    assert_iou_and_dist(oo, oo, 1.0)
    assert_iou_and_dist(oo, oo2, 1.0)
    assert_iou_and_dist(oo, oo3, 0.0)
        
    oo4 = geom.Polygon([(0,0), (0, 1), (1,1), (1, 0)])
    assert_iou_and_dist(oo4, oo4, 1.0)
    
    oo42 = geom.Polygon([(0,0), (0, 0.5), (1,0.5), (1, 0)])
    assert_iou_and_dist(oo42, oo42, 1.0)
    assert_iou_and_dist(oo4, oo42, 0.5)
    
    # inconsistent data... :-/
    assert iou_distance(geom.Polygon([]), geom.Polygon([])) == 0.0

