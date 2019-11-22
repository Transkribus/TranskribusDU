# -*- coding: utf-8 -*-

"""
    Computing the graph for a "Multi-singlePage-PageXml" document
    
    A POSTERIORI EXPLANATION  :))
    at some point, we had independent pages taht were stored together in a 
    single .mpxml file. SO this class can load this file, and build one graph
    per page (instead of one graph per file).
    2018/03/30 JL

    Copyright Xerox(C) 2017 H . Déjean


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




from lxml import etree

from common.trace import traceln
from xml_formats.PageXml import PageXml

from .Graph_MultiPageXml import Graph_MultiPageXml
from . import Edge
from .Page import Page


class Graph_MultiSinglePageXml(Graph_MultiPageXml):
    '''
    Computing the graph for a MultiPageXml document
    '''
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels
    sIN_FORMAT  = "MultiSinglePageXML"   # tell here which input format is expected

    #Namespace, of PageXml, at least
    dNS = {"pc":PageXml.NS_PAGE_XML}
    
    #How to list the pages of a (Multi)PageXml doc
    sxpPage     = "//pc:Page"

    def __init__(self, lNode = [], lEdge = []):
        Graph_MultiPageXml.__init__(self, lNode, lEdge)

    @classmethod
    def loadGraphs(cls
                   , cGraphClass          # graph class (must be subclass)
                   , lsFilename
                   , bNeighbourhood=True
                   , bDetach=False
                   , bLabelled=False
                   , iVerbose=0):
        """
        Load one graph per file, and detach its DOM
        return the list of loaded graphs
        """
        lGraph = []
        for sFilename in lsFilename:
            if iVerbose: traceln("\t%s"%sFilename)
            lG= Graph_MultiSinglePageXml.getSinglePages(cGraphClass, sFilename, bNeighbourhood,bDetach,bLabelled, iVerbose)
            for g in lG: g._index()
            lGraph.extend(lG)
        return lGraph
    
    @classmethod
    def getSinglePages(cls
                       , cGraphClass          # graph class (must be subclass)
                       , sFilename
                       , bNeighbourhood=True
                       , bDetach=False
                       , bLabelled=False
                       , iVerbose=0):
        """
        load a pageXml
        Return a CRF Graph object
        """
    
        lGraph=[]
        doc = etree.parse(sFilename)

        for pnum, page, domNdPage in cls._iter_Page_DocNode(doc):
            g = cGraphClass()
            g.doc= doc
            
            g.lNode,  g.lEdge = list(), list()
            #now that we have the page, let's create the node for each type!
            setPageNdDomId = set() #the set of DOM id
            # because the node types are supposed to have an empty intersection

            llPageNodeByType = [ [nd for nd in nodeType._iter_GraphNode(doc, domNdPage, page) ] for nodeType in g.getNodeTypeList()]
            for iType1, lNodeType1 in enumerate(llPageNodeByType):
                lEdge = Edge.Edge.computeEdges(None, lNodeType1, g.iGraphMode)
                traceln("\tType %d - %d    %d nodes            %d edges"%(iType1, iType1, len(lNodeType1), len(lEdge)))
                g.lEdge.extend(lEdge)
                g.lNode.extend( lNodeType1 )
                
                for iType2 in range(iType1+1, len(llPageNodeByType)):
                #for lNodeType2 in llPageNodeByType[iType1:]:
                    lNodeType2 = llPageNodeByType[iType2]
                    #lPageEdge = Edge.Edge.computeEdges(lPrevPageNode, lPageNode, g.iGraphMode)
                    lEdge = Edge.Edge.computeEdges(None, lNodeType1+lNodeType2, g.iGraphMode)
                    traceln("\tType %d - %d    %d nodes, %d nodes  %d edges"%(iType1, iType2, len(lNodeType1), len(lNodeType2), len(lEdge)))
                    g.lEdge.extend(lEdge)

            #lPageNode = [nd for nodeType in g.getNodeTypeList() for nd in nodeType._iter_GraphNode(doc, domNdPage, page) ]

            #check that each node appears once
            setPageNdDomId = set([nd.domid for nd in g.lNode])
            assert len(setPageNdDomId) == len(g.lNode), "ERROR: some nodes fit with multiple NodeTypes"
            
            if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(g.lNode), len(g.lEdge)))
            
            if not g.isEmpty() and len(g.lEdge) > 0:    
                if bNeighbourhood: g.collectNeighbors()            
                if bLabelled: g.parseDocLabels()
                if bDetach: g.detachFromDoc()
    
                lGraph.append(g)
            if iVerbose: traceln("\t\t (%d nodes,  %d edges)"%(len(g.lNode), len(g.lEdge)) )
        
        return lGraph     
       
    # ---------------------------------------------------------------------------------------------------------
    @classmethod
    def _iter_Page_DocNode(cls, doc):
        """
        Parse a Multi-pageXml DOM, by page

        iterator on the DOM, that returns per page:
            page-num (int), page object, page dom node
        
        """
        assert Graph_MultiSinglePageXml.sxpPage, "CONFIG ERROR: need an xpath expression to enumerate PAGE elements"
        lNdPage = doc.xpath(Graph_MultiSinglePageXml.sxpPage, namespaces=Graph_MultiSinglePageXml.dNS)   #all pages
        pnum = 0
        pagecnt = len(lNdPage)
        for ndPage in lNdPage:
            pnum += 1
            iPageWidth  = int( ndPage.get("imageWidth") )
            iPageHeight = int( ndPage.get("imageHeight") )
            page = Page(pnum, pagecnt, iPageWidth, iPageHeight, cls=None, domnode=ndPage, domid=ndPage.get("id"))
            yield (pnum, page, ndPage)
            
        return        
               
        
