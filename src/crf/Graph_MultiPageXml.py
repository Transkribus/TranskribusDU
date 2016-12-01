# -*- coding: utf-8 -*-

"""
    Computing the graph for a MultiPageXml document
    

    Copyright Xerox(C) 2016 H. Déjean, JL. Meunier

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

from Graph import Graph
from Block import Block

from xml_formats.PageXml import PageXml
from util.Polygon import Polygon

# from ..xml_formats.PageXml import PageXml 

TEST_getPageXmlBlock = None

class Graph_MultiPageXml(Graph):
    '''
    Computing the graph for a MultiPageXml document
    '''

    def __init__(self, lNode = [], lEdge = []):
        Graph.__init__(self, lNode, lEdge)

    def setConfig(self):
        """
        Configure this Graph
        """
        self._sEltName = "TextRegion"   #Name of Xml element that should map to a node
     
    def selectNodeNodes(self, doc, ctxt):
        """
        Select all node of interest
        """
        
    def parseFile(self, sFilename):
        """
        Load that document as a CRF Graph
        
        Return a CRF Graph object
        """
    
        doc = libxml2.parseFile(sFilename)
        
        dNS = {"pc":PageXml.NS_PAGE_XML}
        sxpPage     = "//pc:Page"
        sxpBlock    = ".//pc:TextRegion"
        sxpTextual  = "./pc:TextEquiv"             #CAUTION redundant TextEquiv nodes! 
        
        self.lNode = self.getPageXmlBlocks(doc, dNS, sxpPage, sxpBlock, sxpTextual)
        
        #TODO the edges!!
        self.lEdge = []
        
        return self
        
    def getPageXmlBlocks(self, doc, dNS, sxpPage, sxpBlock, sxpTextual):
        """
        Parse a Multi-pageXml DOM
        
        return a CRF Graph Object
        """
        lNode = []
        
        #--- XPATH contexts
        ctxt = doc.xpathNewContext()
        for ns, nsurl in dNS.items(): ctxt.xpathRegisterNs(ns, nsurl)


        lNdPage = ctxt.xpathEval(sxpPage)   #all pages
        pnum = 0
        for ndPage in lNdPage:
            pnum += 1
            ctxt.setContextNode(ndPage)
            
            lNdBlock = ctxt.xpathEval(sxpBlock) #all relevant nodes of the page
            for ndBlock in lNdBlock:
                ctxt.setContextNode(ndBlock)
                
                lNdText = ctxt.xpathEval(sxpTextual)
                assert len(lNdText) == 1 , "STRANGE; I expected only one useful TextEquiv below this node..."
                sText = PageXml.makeText(lNdText[0])
                
                #now we need to infer the bounding box of that object
                lXY = PageXml.getPointList(ndBlock)  #the polygon
                plg = Polygon(lXY)
                
#                 #Baseline node
#                 ndBaseline = PageXml.getChildByName(ndBlock, "Baseline")
#                 sBaseline_points = ndBaseline.prop("points")
#                 x, y, w, h = plg.fitRectangleByBaseline( PageXml.getPointList(sBaseline_points) )
                x1,y1, x2,y2 = plg.fitRectangle()
                if True:
                    #we reduce a bit this rectangle, to ovoid overlap
                    w,h = x2-x1, y2-y1
                    dx = max(w * 0.066, 20)
                    dy = max(h * 0.066, 20)
                    x1,y1, x2,y2 = [ int(round(v)) for v in [x1+dx,y1+dy, x2-dx,y2-dy] ]
                
                #TODO
                orientation = 0
                cls = 0

                #and create a Block
                blk = Block(pnum, (x1, y1, x2-x1, y2-y1), sText, orientation, cls, ndBlock)
                
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
                    
        if TEST_getPageXmlBlock:
            util.xml_utils.toFile(doc, "TEST_getPageXmlBlock.mpxml", True)
            
        ctxt.xpathFreeContext()       
                
        return lNode
        
        
