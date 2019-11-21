# -*- coding: utf-8 -*-

"""
    Computing the graph for a MultiPageXml document

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

from  lxml import etree

from common.trace import traceln
from xml_formats.PageXml import PageXml

from .Graph_DOM import Graph_DOM
from .Transformer_PageXml import  EdgeTransformerClassShifter
from .Block import Block, BlockShallowCopy
from .Edge import Edge, HorizontalEdge, VerticalEdge, CrossPageEdge, CrossMirrorContinuousPageVerticalEdge
from .Page import Page


class Graph_MultiPageXml(Graph_DOM):
    '''
    Computing the graph for a MultiPageXml document
    '''
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels

    sIN_FORMAT  = "(Multi)PageXML"   # tell here which input format is expected

    #Namespace, of PageXml, at least
    dNS = {"pc":PageXml.NS_PAGE_XML}
    
    #How to list the pages of a (Multi)PageXml doc
    sxpPage     = "//pc:Page"

    def __init__(self, lNode = [], lEdge = []):
        Graph_DOM.__init__(self, lNode, lEdge)

    # ---------------------------------------------------------------------------------------------------------        
    def _iter_Page_DocNode(self, doc):
        """
        Parse a Multi-pageXml DOM, by page

        iterator on the DOM, that returns per page:
            page-num (int), page object, page dom node
        
        """
        assert self.sxpPage, "CONFIG ERROR: need an xpath expression to enumerate PAGE elements"
        
        lNdPage = doc.xpath(self.sxpPage, namespaces=self.dNS)   #all pages
        
        pnum = 0
        pagecnt = len(lNdPage)
        for ndPage in lNdPage:
            pnum += 1
            iPageWidth  = int( ndPage.get("imageWidth") )
            iPageHeight = int( ndPage.get("imageHeight") )
            page = Page(pnum, pagecnt, iPageWidth, iPageHeight, cls=None, domnode=ndPage, domid=ndPage.get("id"))
            yield (pnum, page, ndPage)
            
        return       

# ------------------------------------------------------------------------------------------------------------------------------------------------
        
class ContinousPage:
    def computeContinuousPageEdges(self, lPrevPageEdgeBlk, lPageBlk, bMirror=True):
        """
        Consider pages in contunous mode, next page possibly left<-->right mirrored.
        Create edge between horrizontally overlapping block of 2nd half of previous page and block of 1st half of next page
        
        Computation trick:
            Create a fake page with lower half of previous page and upper half of current page.
            Compute vertical links.
            Only keep link from one page to the other. 
        """
        
        lAllEdge = list()
        
        if lPrevPageEdgeBlk and lPageBlk:
            #empty pages lead to empty return!
            p0, p1  = lPrevPageEdgeBlk[0].getPage(), lPageBlk[0].getPage()
            _w0, h0 = p0.getWidthHeight()
            w1, h1  = p1.getWidthHeight()
            
            p0_vertical_middle, p1_vertical_middle = h0/2.0 , h1/2.0
            
            #Shallow copy of half of pages (resp. lower and upper halfs)
            lVirtualPageBlk  = [ BlockShallowCopy(blk) for blk in lPrevPageEdgeBlk if blk.getCenter()[1] >= p0_vertical_middle]
            lNextHalfPage    = [ BlockShallowCopy(blk) for blk in lPageBlk        if blk.getCenter()[1] <= p1_vertical_middle]
        
            if lVirtualPageBlk and lNextHalfPage:
                #if one half is empty, return no edge!
                
                #translate blocks to fit in a fake page
                for blk in lVirtualPageBlk: blk.translate(0, -p0_vertical_middle)
                for blk in lNextHalfPage  : blk.translate(0, +p0_vertical_middle) 
                
                #optionnaly mirroring them horizontally
                if bMirror:
                    for blk in lNextHalfPage: blk.mirrorHorizontally(w1)            
                
                lVirtualPageBlk.extend(lNextHalfPage)
                lEdge = Block._findVerticalNeighborEdges_g1(lVirtualPageBlk, CrossMirrorContinuousPageVerticalEdge)
                
                #keep only those edge accross pages, and make them to link original blocks!
                lAllEdge = [CrossMirrorContinuousPageVerticalEdge(edge.A.getOrigBlock(), edge.B.getOrigBlock(), edge.length) \
                            for edge in lEdge if edge.A.pnum != edge.B.pnum]
            
                if False and lAllEdge:
                    print("---Page ", lAllEdge[0].A.pnum)
                    for edge in lAllEdge: print(edge.A.node.prop("id"), " -->  ", edge.B.node.prop("id"))
                   
        return lAllEdge

# --------------------------------------------------------------------------------------
    
class Graph_MultiContinousPageXml(Graph_MultiPageXml, ContinousPage):
    '''
    Here we have edge between blocks of consecutives page as if they were in continuous arrangement
    '''

    def __init__(self, lNode = [], lEdge = []):
        Graph_MultiPageXml.__init__(self, lNode, lEdge)
        EdgeTransformerClassShifter.setDefaultEdgeClass([HorizontalEdge, VerticalEdge, CrossPageEdge, CrossMirrorContinuousPageVerticalEdge])

    def parseDocFile(self, sFilename, iVerbose=0):
        """
        Load that document as a CRF Graph.
        Also set the self.doc variable!
        
        Return a CRF Graph object
        """
    
        self.doc = etree.parse(sFilename)
        self.lNode, self.lEdge = list(), list()
        #load the block of each page, keeping the list of blocks of previous page
        lPrevPageNode = None

        for pnum, page, domNdPage in self._iter_Page_DocNode(self.doc):
            #now that we have the page, let's create the node for each type!
            lPageNode = list()
            setPageNdDomId = set() #the set of DOM id
            # because the node types are supposed to have an empty intersection
                            
            lPageNode = [nd for nodeType in self.getNodeTypeList() for nd in nodeType._iter_GraphNode(self.doc, domNdPage, page) ]
            
            #check that each node appears once
            setPageNdDomId = set([nd.domid for nd in lPageNode])
            assert len(setPageNdDomId) == len(lPageNode), "ERROR: some nodes fit with multiple NodeTypes"
            
        
            self.lNode.extend(lPageNode)
            
            lPageEdge = Edge.computeEdges(lPrevPageNode, lPageNode, self.iGraphMode)
            self.lEdge.extend(lPageEdge)

            lContinuousPageEdge = self.computeContinuousPageEdges(lPrevPageNode, lPageNode)
            self.lEdge.extend(lContinuousPageEdge)
                        
            if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            
            lPrevPageNode = lPageNode
        if iVerbose: traceln("\t\t (%d nodes,  %d edges)"%(len(self.lNode), len(self.lEdge)) )
        
        return self
        

