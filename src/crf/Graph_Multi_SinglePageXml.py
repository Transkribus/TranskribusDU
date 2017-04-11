# -*- coding: utf-8 -*-

"""
    Computing the graph for a MultiPageXml document

    Copyright Xerox(C) 2017 H . Déjean

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
import libxml2

from Graph_MultiPageXml import Graph_MultiPageXml
from common.trace import traceln
from xml_formats.PageXml import PageXml
import Edge
from Page import Page

class Graph_MultiSinglePageXml(Graph_MultiPageXml):
    '''
    Computing the graph for a MultiPageXml document

        USAGE:
        - call parseFile to load the DOM and create the nodes and edges
        - call detachFromDOM before freeing the DOM
    '''
    #Namespace, of PageXml, at least
    dNS = {"pc":PageXml.NS_PAGE_XML}
    
    #How to list the pages of a (Multi)PageXml doc
    sxpPage     = "//pc:Page"

    def __init__(self, lNode = [], lEdge = []):
        Graph_MultiPageXml.__init__(self, lNode, lEdge)



    @classmethod
    def loadGraphs(cls, lsFilename, bNeighbourhood=True, bDetach=False, bLabelled=False, iVerbose=0):
        """
        Load one graph per file, and detach its DOM
        return the list of loaded graphs
        """
        lGraph = []
        for sFilename in lsFilename:
            if iVerbose: traceln("\t%s"%sFilename)
            lG= Graph_MultiSinglePageXml.getSinglePages(sFilename, iVerbose)
#             if bNeighbourhood: g.collectNeighbors()            
#             if bLabelled: g.parseDomLabels()
#             if bDetach: g.detachFromDOM()
            lGraph.extend(lG)
        return lGraph
    
    @classmethod
    def getSinglePages(cls, sFilename, iVerbose=0):
        """
        load a pageXml
        Return a CRF Graph object
        """
    
        lGraph=[]
        doc = libxml2.parseFile(sFilename)
        #load the block of each page, keeping the list of blocks of previous page
        lPrevPageNode = None

        for pnum, page, domNdPage in cls._iter_Page_DomNode(doc):
            g = cls()
            g.lNode,  g.lEdge = list(), list()
            #now that we have the page, let's create the node for each type!
            lPageNode = list()
            setPageNdDomId = set() #the set of DOM id
            # because the node types are supposed to have an empty intersection
                            
            lPageNode = [nd for nodeType in g.getNodeTypeList() for nd in nodeType._iter_GraphNode(doc, domNdPage, page) ]
            
            #check that each node appears once
            setPageNdDomId = set([nd.domid for nd in lPageNode])
            assert len(setPageNdDomId) == len(lPageNode), "ERROR: some nodes fit with multiple NodeTypes"
            
        
            g.lNode.extend(lPageNode)
            
            lPageEdge = Edge.Edge.computeEdges(lPrevPageNode, lPageNode)
            
            g.lEdge.extend(lPageEdge)
            if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            
            lPrevPageNode = lPageNode
        if iVerbose: traceln("\t\t (%d nodes,  %d edges)"%(len(g.lNode), len(g.lEdge)) )
        
        return lGraph     
       
    def parseXmlFile(self, sFilename, iVerbose=0):
        """
        load a pageXml
        Return a CRF Graph object
        """
    
        lGraph=[]
        self.doc = libxml2.parseFile(sFilename)
        self.lNode, self.lEdge = list(), list()
        #load the block of each page, keeping the list of blocks of previous page
        lPrevPageNode = None

        for pnum, page, domNdPage in self._iter_Page_DomNode(self.doc):
            #now that we have the page, let's create the node for each type!
            lPageNode = list()
            setPageNdDomId = set() #the set of DOM id
            # because the node types are supposed to have an empty intersection
                            
            lPageNode = [nd for nodeType in self.getNodeTypeList() for nd in nodeType._iter_GraphNode(self.doc, domNdPage, page) ]
            
            #check that each node appears once
            setPageNdDomId = set([nd.domid for nd in lPageNode])
            assert len(setPageNdDomId) == len(lPageNode), "ERROR: some nodes fit with multiple NodeTypes"
            
        
            self.lNode.extend(lPageNode)
            
            lPageEdge = Edge.Edge.computeEdges(lPrevPageNode, lPageNode)
            
            self.lEdge.extend(lPageEdge)
            if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            
            lPrevPageNode = lPageNode
        if iVerbose: traceln("\t\t (%d nodes,  %d edges)"%(len(self.lNode), len(self.lEdge)) )
        
        return lGraph    

    # ---------------------------------------------------------------------------------------------------------
    @classmethod
    def _iter_Page_DomNode(cls, doc):
        """
        Parse a Multi-pageXml DOM, by page

        iterator on the DOM, that returns per page:
            page-num (int), page object, page dom node
        
        """
        #--- XPATH contexts
        ctxt = doc.xpathNewContext()
        for ns, nsurl in Graph_MultiSinglePageXml.dNS.items(): ctxt.xpathRegisterNs(ns, nsurl)

        assert Graph_MultiSinglePageXml.sxpPage, "CONFIG ERROR: need an xpath expression to enumerate PAGE elements"
        lNdPage = ctxt.xpathEval(Graph_MultiSinglePageXml.sxpPage)   #all pages
        
        pnum = 0
        pagecnt = len(lNdPage)
        for ndPage in lNdPage:
            pnum += 1
            iPageWidth  = int( ndPage.prop("imageWidth") )
            iPageHeight = int( ndPage.prop("imageHeight") )
            page = Page(pnum, pagecnt, iPageWidth, iPageHeight, cls=None, domnode=ndPage, domid=ndPage.prop("id"))
            yield (pnum, page, ndPage)
            
        ctxt.xpathFreeContext()       
        
        raise StopIteration()        
               
        
