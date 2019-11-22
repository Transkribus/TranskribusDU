# -*- coding: utf-8 -*-

"""
    A class to load the SeparatorRegion of a PageXml to add features to the
    edges of a graph conjugate used for segmentation.
    
    It specialises the _index method to add specific attributes to the edges
    , so that the specific feature transformers can be used.
    
    Copyright NAVER(C) 2019 
    
    2019-08-20    JL. Meunier
"""

import numpy as np

import shapely.geometry as geom
from shapely.prepared import prep
from rtree import index
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import QuantileTransformer

from common.trace import traceln
from util.Shape import ShapeLoader
from xml_formats.PageXml import PageXml

from .GraphBinaryConjugateSegmenter_DOM         import GraphBinaryConjugateSegmenter_DOM
from graph.Transformer                          import Transformer


class PageXmlSeparatorRegion(GraphBinaryConjugateSegmenter_DOM):
    """
    Extension of a segmenter conjugate graph to exploit graphical separator
    as additional edge features
    """
    bVerbose = True
    
    def __init__(self):
        super(PageXmlSeparatorRegion, self).__init__()
    
    def _index(self):
        """
        This method is called before computing the Xs
        We call it and right after, we compute the intersection of edge with SeparatorRegions
        Then, feature extraction can reflect the crossing of edges and separators 
        """
        bFirstCall = super(PageXmlSeparatorRegion, self)._index()
        
        if bFirstCall:
            # indexing was required
            # , so first call
            # , so we need to make the computation of edges crossing separators!
            self.addSeparatorFeature()

    def addSeparatorFeature(self):
        """
        We load the graphical separators
        COmpute a set of shapely object
        In turn, for each edge, we compute the intersection with all separators
        
        The edge features will be:
        - boolean: at least crossing one separator
        - number of crossing points
        - span length of the crossing points
        - average length of the crossed separators
        - average distance between two crossings
        """
        
        # graphical separators
        dNS = {"pc":PageXml.NS_PAGE_XML}
        someNode = self.lNode[0]
        ndPage = someNode.node.xpath("ancestor::pc:Page", namespaces=dNS)[0]
        lNdSep = ndPage.xpath(".//pc:SeparatorRegion", namespaces=dNS)
        loSep = [ShapeLoader.node_to_LineString(_nd) for _nd in lNdSep]
        
        if self.bVerbose: traceln(" %d graphical separators"%len(loSep))

        # make an indexed rtree
        idx = index.Index()
        for i, oSep in enumerate(loSep):
            idx.insert(i, oSep.bounds)
            
        # take each edge in turn and list the separators it crosses
        nCrossing = 0
        for edge in self.lEdge:
            # bottom-left corner to bottom-left corner
            oEdge = geom.LineString([(edge.A.x1, edge.A.y1), (edge.B.x1, edge.B.y1)])
            prepO = prep(oEdge)
            lCrossingPoints = []
            fSepTotalLen = 0
            for i in idx.intersection(oEdge.bounds):
                # check each candidate in turn
                oSep = loSep[i]
                if prepO.intersects(oSep):
                    fSepTotalLen += oSep.length
                    oPt = oEdge.intersection(oSep)
                    if type(oPt) != geom.Point:
                        traceln('Intersection in not a point: skipping it')
                    else:
                        lCrossingPoints.append(oPt)
                    
            if lCrossingPoints:
                nCrossing += 1
                edge.bCrossingSep = True
                edge.sep_NbCrossing = len(lCrossingPoints)
                minx, miny, maxx, maxy  = geom.MultiPoint(lCrossingPoints).bounds
                edge.sep_SpanLen = abs(minx-maxx) + abs(miny-maxy)
                edge.sep_AvgSpanSgmt = edge.sep_SpanLen / len(lCrossingPoints) 
                edge.sep_AvgSepLen = fSepTotalLen / len(lCrossingPoints)
            else:
                edge.bCrossingSep = False
                edge.sep_NbCrossing = 0
                edge.sep_SpanLen = 0
                edge.sep_AvgSpanSgmt = 0 
                edge.sep_AvgSepLen = 0
                
            #traceln((edge.A.domid, edge.B.domid, edge.bCrossingSep, edge.sep_NbCrossing, edge.sep_SpanLen, edge.sep_AvgSpanSgmt, edge.sep_AvgSepLen))
                
                
        if self.bVerbose: 
            traceln(" %d (/ %d) edges crossing at least one graphical separator"%(nCrossing, len(self.lEdge)))

       
class Separator_boolean(Transformer):
    """
    a boolean encoding indicating if the edge crosses a separator
    """
    def transform(self, lO):
        nb = len(lO)
        a = np.zeros((nb, 1), dtype=np.float64)
        for i, o in enumerate(lO):
            if o.bCrossingSep: a[i,0] = 1
        return a

    def __str__(self):
        return "- Separator_boolean %s (#1)" % (self.__class__)


class Separator_num(Pipeline):
    """
    Node neighbour count feature quantiled
    """
    nQUANTILE = 16

    class Selector(Transformer):
        """
        Characterising the neighborough by the number of neighbour before and after
        """
        def transform(self, lO):
            nb = len(lO)
            a = np.zeros((nb, 4), dtype=np.float64)
            for i, o in enumerate(lO):
                a[i,:] = (o.sep_NbCrossing, o.sep_SpanLen, o.sep_AvgSpanSgmt, o.sep_AvgSepLen)
            return a

    def __init__(self, nQuantile=None):
        self.nQuantile = Separator_num.nQUANTILE if nQuantile is None else nQuantile
        Pipeline.__init__(self, [ ('geometry' , Separator_num.Selector())
                                , ('quantiled', QuantileTransformer(n_quantiles=self.nQuantile, copy=False))  #use in-place scaling
                                ])

    def __str__(self):
        return "- Separator_num %s (#4)" % (self.__class__)

