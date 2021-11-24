# -*- coding: utf-8 -*-

"""
    Computing the graph for a json file, which is the output of OCR
    @author: Nitin Choudhary

"""
import io
import json

from common.trace import traceln

from .Graph import Graph
from . import Edge
from .Page import Page


class Graph_JsonOCR(Graph):
    '''
    Computing the graph for a json file

    '''
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels

    sIN_FORMAT  = "JSON_OCR"   # tell here which input format is expected
    sOUTPUT_EXT = ".json"

    def __init__(self, lNode=[], lEdge=[]):
        Graph.__init__(self, lNode, lEdge)

    @classmethod
    def loadGraphs(cls
                   , cGraphClass  # graph class (must be subclass)
                   , lsFilename
                   , bNeighbourhood=True
                   , bDetach=False
                   , bLabelled=False
                   , iVerbose=0
                   ):
        """
        Load one graph per file, and detach its DOM
        return the list of loaded graphs
        """
        lGraph = []
        for n, sFilename in enumerate(lsFilename):
            if iVerbose: traceln("\t%d - %s" % (n+1, sFilename))
            [g] = cls.getSinglePages(cGraphClass, sFilename, bNeighbourhood, bDetach, bLabelled,
                                                         iVerbose)
            g._index()
            if not g.isEmpty():
                if bNeighbourhood: g.collectNeighbors()
                if bLabelled: g.parseDocLabels()
                if bDetach: g.detachFromDoc()
            lGraph.append(g)
        return lGraph


    @classmethod
    def getSinglePages(cls
                       , cGraphClass  # graph class (must be subclass)
                       , sFilename
                       , bNeighbourhood=True
                       , bDetach=False
                       , bLabelled=False
                       , iVerbose=0):
        """
        load a json
        Return a Graph object
        """
        lGraph = []
        if isinstance(sFilename, io.IOBase):
            # we got a file-like object (e.g. in server mode)
            doc = json.load(sFilename)
        else:
            with open(sFilename, encoding='utf-8') as fd:
                doc = json.load(fd)
            
        g = cGraphClass()
        g.doc = doc

        # g.lNode, g.lEdge = list(), list()
        # now that we have the page, let's create the node for each type!
        assert len(g.getNodeTypeList()) == 1, "Not yet implemented"
        
        # we skip the loop on pages since we always have 1 page for now from the OCR
        g.lNode = [nd for nodeType in g.getNodeTypeList() for nd in nodeType._iter_GraphNode(g.doc, sFilename) ]
        g.lEdge = Edge.Edge.computeEdges(None, g.lNode, g.iGraphMode)

        # create a fake page object (for GLB2021 features in particular)
        # pg = Page(pnum, pagecnt, w, h)
        pg = Page(1, 1, 0, 0)
        for nd in g.lNode: nd.page = pg
            
        if iVerbose >= 2: traceln("\tPage %5d    %6d nodes    %7d edges" % (1, len(g.lNode), len(g.lEdge)))

        return [g]

    @classmethod
    def saveDoc(cls, sFilename, doc, lG, sCreator, sComment):
        print("SaveDoc not done ", sFilename)
