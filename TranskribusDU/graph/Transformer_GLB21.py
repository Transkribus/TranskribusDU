# -*- coding: utf-8 -*-

"""
    Node and edge feature transformers to extract features from a graph
          
    Copyright Naver(C) 2019, 2021 JL. Meunier
"""
import math

import numpy as np

from .Edge          import HorizontalEdge, VerticalEdge
from .FeatureDType  import dtype as Feat_dtype
from .Transformer   import Transformer

class NodeTransformerXYXY_GLB21(Transformer):
    """
    Simplifieds NodeTransformerXYWH_vPeri
    """
    nbFEAT = 4
    
    def transform(self, lNode):
        a = np.empty( ( len(lNode), self.nbFEAT) , dtype=Feat_dtype)
        
        # Computing the min and max, x for each page
        for nd in lNode:
            assert nd.x1 <= nd.x2
            assert nd.y1 <= nd.y2
            pg = nd.page
            try:
                pg.min_x = min(nd.x1, pg.min_x)
                pg.min_y = min(nd.y1, pg.min_y)
            except AttributeError:
                pg.min_x = nd.x1
                pg.min_y = nd.y1
        # computing page max width and height
        for nd in lNode:
            pg = nd.page
            try:
                pg.max_w = max(pg.max_w, nd.x2 - pg.min_x)
                pg.max_h = max(pg.max_h, nd.y2 - pg.min_y)
            except AttributeError:
                pg.max_w =               nd.x2 - pg.min_x
                pg.max_h =               nd.y2 - pg.min_y
        
        # normalized coords
        for i, blk in enumerate(lNode):
            pg = blk.page
            a[i, :] = (  (blk.x1 - pg.min_x) / pg.max_w
                       , (blk.x2 - pg.min_x) / pg.max_w
                       , (blk.y1 - pg.min_y) / pg.max_h
                       , (blk.y2 - pg.min_y) / pg.max_h
                   )
        assert (a>=0).all() and (a<=1).all()
        return a


class EdgeNumericalSelector_noText_GLB21(Transformer):
    # Bug found in July 2020 by JLM: last feature was not used...
    # Bug found by JLM March 2021 ovr was wrongly used
    # :-//
    nbFEAT = 14
    
    def transform(self, lEdge):
        a = np.zeros( ( len(lEdge), self.nbFEAT ) , dtype=Feat_dtype)

        for i, edge in enumerate(lEdge):
            A,B = edge.A, edge.B     
            pgA = A.page
            pgB = B.page
            assert pgA == pgB, "INTERNAL ERROR: cross page edges unexpected"
            
            # UNnormalized manhattan
            dH, dV = A.manhattanDistance(B)
            
            #overlap
            areaInter = A.interSpaceSignedArea(B)    # UNnormalized A.interSpaceArea(B)
            areaOuter = A.outerSpaceArea(B)    # UNnormalized area of bounding box of A and B
            areaA     = A.area()               # UNnormalized area
            areaB     = B.area() 
            # ratio (normalized on [-1, 1])
            try:
                rInter      = areaInter / (areaA + abs(areaInter) + areaB)
            except ZeroDivisionError:
                rInter      = 0.0
            # ratio (normalized on [ -1, 1])
            try:
                rOuter      = (areaOuter - areaA - areaB) / areaOuter  # proportion of blank space
            # ratio (normalized on [-1, 1])
                rInterOuter = areaInter                   / areaOuter
            except ZeroDivisionError:
                rOuter      = -1.0
                rInterOuter = -1.0
                
            #             assert abs(rInter) <= 1.0       , (rInter,rOuter, rInterOuter) 
            #             assert abs(rOuter) <= 1.0       , (rInter,rOuter, rInterOuter) 
            #             assert abs(rInterOuter) <= 1.0  , (rInter,rOuter, rInterOuter) 
            
            # normalized manhattan and euclidian  (WRONG if not on same page!!!)
            dH = dH / float(pgA.max_w)
            dV = dV / float(pgA.max_h)
            #             assert 0 <= dH and dH <= 1, (str(A), str(B), pgA.max_w)
            #             assert 0 <= dV and dV <= 1, (str(A), str(B), pgA.max_H)
            dE = math.sqrt(dH*dH + dV*dV) / 1.5  # upper bound of sqrt(2)
            #             assert dE <= 1, (dH, dV, dE)
            
            a[i, : ] = ( dH, dV, dE         
                       , rInter, rOuter, rInterOuter
                       , (A.x1 - pgA.min_x) / pgA.max_w
                       , (A.x2 - pgA.min_x) / pgA.max_w
                       , (A.y1 - pgA.min_y) / pgA.max_h
                       , (A.y2 - pgA.min_y) / pgA.max_h
                       , (B.x1 - pgB.min_x) / pgB.max_w
                       , (B.x2 - pgB.min_x) / pgB.max_w
                       , (B.y1 - pgB.min_y) / pgB.max_h
                       , (B.y2 - pgB.min_y) / pgB.max_h
                       )  
        assert (-1 <= a).all() and (a<=1).all(), a
        return a  

