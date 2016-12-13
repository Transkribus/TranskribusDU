# -*- coding: utf-8 -*-

"""
    Computing the graph for a MultiPageXml document
    

    Copyright Xerox(C) 2016 JL. Meunier

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

from common.trace import traceln

from Graph import Graph
from Page import Page
from Block import Block
from Edge  import Edge
from Label_PageXml import Label_PageXml

from xml_formats.PageXml import PageXml
from util.Polygon import Polygon

from common.trace import traceln

# from ..xml_formats.PageXml import PageXml 

TEST_getPageXmlBlock = None

class Graph_MultiPageXml(Graph, Label_PageXml):
    '''
    Computing the graph for a MultiPageXml document

        USAGE:
        - call parseFile to load the DOM and create the nodes and edges
        - call parseLabels to get the labels from the nodes (if labelled, otherwise be ready to get KeyError exceptions)
        - call detachFromDOM before freeing the DOM
    '''
    #Namespace, of PageXml, at least
    dNS = {"pc":PageXml.NS_PAGE_XML}
    
    #TASK SPECIFIC STUFF TO BE DEFINED IN A SUB-CLASS
    sxpPage     = "//pc:Page"
    sxpNode     = None
    sxpTextual  = None    #CAUTION redundant TextEquiv nodes! 

    def __init__(self, lNode = [], lEdge = []):
        Graph.__init__(self, lNode, lEdge)
        Label_PageXml.__init__(self)
        
    # --- Graph building --------------------------------------------------------
    def parseXmlFile(self, sFilename, iVerbose=0):
        """
        Load that document as a CRF Graph
        
        Return a CRF Graph object
        """
    
        self.doc = libxml2.parseFile(sFilename)
        self.lNode, self.lEdge = list(), list()
        #load the block of each page, keeping the list of blocks of previous page
        lPrevPageNode = None

        for (pnum, lPageNode) in self._iter_PageXml_Nodes(self.doc, self.dNS, self.sxpPage, self.sxpNode, self.sxpTextual):
        
            self.lNode.extend(lPageNode)
            
            lPageEdge = Edge.computeEdges(lPrevPageNode, lPageNode)
            
            self.lEdge.extend(lPageEdge)
            if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            
            lPrevPageNode = lPageNode
        if iVerbose: traceln("\t- %d nodes,  %d edges)"%(len(self.lNode), len(self.lEdge)) )
        
        return self

    #Need to force the inheritance
#     def setDomNodeLabel(self, node, sLabel):
#         """
#         Set the DOM node associated to this graph node to a certain label
#         """        
    setDomNodeLabel = Label_PageXml.setDomNodeLabel
    
    # ---------------------------------------------------------------------------------------------------------        
    def _iter_PageXml_Nodes(self, doc, dNS, sxpPage, sxpNode, sxpTextual):
        """
        Parse a Multi-pageXml DOM

        iterator on the DOM, that returns per page:
            page-num (int), list-of-page-block-objects
        
        """
        #--- XPATH contexts
        ctxt = doc.xpathNewContext()
        for ns, nsurl in dNS.items(): ctxt.xpathRegisterNs(ns, nsurl)

        assert sxpPage, "CONFIG ERROR: need an xpath expression to enumerate PAGE elements"
        assert sxpNode, "CONFIG ERROR: need an xpath expression to enumerate elements corresponding to graph nodes"
        lNdPage = ctxt.xpathEval(sxpPage)   #all pages
        
        pnum = 0
        for ndPage in lNdPage:
            pnum += 1
            bEven = (pnum%2==0)
            iPageWidth  = int( ndPage.prop("imageWidth") )
            iPageHeight = int( ndPage.prop("imageHeight") )
            page = Page(pnum, iPageWidth, iPageHeight, cls=None, domnode=ndPage, domid=ndPage.prop("id"))
            lNode = []
            ctxt.setContextNode(ndPage)
            lNdBlock = ctxt.xpathEval(sxpNode) #all relevant nodes of the page

            for ndBlock in lNdBlock:
                ctxt.setContextNode(ndBlock)
                domid = ndBlock.prop("id")
                lNdText = ctxt.xpathEval(sxpTextual)
                assert len(lNdText) == 1 , "STRANGE; I expected only one useful TextEquiv below this node..."
                
                sText = PageXml.makeText(lNdText[0])
                if sText==None:
                    sText = ""
                    traceln("Warning: no text in node %s"%domid) 
                    #raise ValueError, "No text in node: %s"%ndBlock 
                
                #now we need to infer the bounding box of that object
                lXY = PageXml.getPointList(ndBlock)  #the polygon
                plg = Polygon(lXY)
                x1,y1, x2,y2 = plg.fitRectangle()
                if True:
                    #we reduce a bit this rectangle, to ovoid overlap
                    w,h = x2-x1, y2-y1
                    dx = max(w * 0.066, min(20, w/3))  #we make sure that at least 1/"rd of te width will remain!
                    dy = max(h * 0.066, min(20, w/3))
                    x1,y1, x2,y2 = [ int(round(v)) for v in [x1+dx,y1+dy, x2-dx,y2-dy] ]
                
                #TODO
                orientation = 0
                cls = 0

                #and create a Block
                assert ndBlock
                blk = Block(pnum, (x1, y1, x2-x1, y2-y1), sText, orientation, cls, ndBlock, domid=domid)
                assert blk.node
                
                #link it to its page
                blk.page = page

                lNode.append(blk)

                if TEST_getPageXmlBlock:
                    #dump a modified XML to view the rectangles
                    import util.xml_utils
                    ndTextLine = util.xml_utils.addElement(doc, ndBlock, "PARAGRAPH")
                    ndTextLine.setProp("id", ndBlock.prop("id")+"_tl")
                    ndTextLine.setProp("x", str(x1))
                    ndTextLine.setProp("y", str(y1))
                    ndTextLine.setProp("width", str(x2-x1))
                    ndTextLine.setProp("height", str(y2-y1))
                    ndTextLine.setContent(sText)
                    #ndCoord = util.xml_utils.addElement(doc, ndTextLine, "Coords")
                    #PageXml.setPoints(ndCoord, PageXml.getPointsFromBB(x1,y1,x2,y2))
            yield (pnum, lNode)
            
        ctxt.xpathFreeContext()       
        if TEST_getPageXmlBlock:
            util.xml_utils.toFile(doc, "TEST_getPageXmlBlock.mpxml", True)
        
        raise StopIteration()        
        
