# -*- coding: utf-8 -*-

"""
    Reading and setting the labels of a PageXml document

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
TEST_getPageXmlBlock = False

from common.trace import traceln

from NodeType import NodeType
from xml_formats.PageXml import PageXml
from util.Polygon import Polygon
from Block import Block


class NodeType_DS(NodeType):

    #Namespace, of PageXml, at least
    dNS = {}

    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True):
        NodeType.__init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel, bOther)
        
    def setXpathExpr(self, sxpNode): #, sxpTextual)):
        self.sxpNode    = sxpNode
#         self.sxpTextual = sxpTextual    #always in TOKENs in DS XML
        
    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = self.sDefaultLabel
        
        sXmlLabel = None
        if domnode.prop("title_oracle_best"):
            sXmlLabel = "title"
        elif domnode.prop("pnum_oracle"):
            sXmlLabel = "pnum"
        if sXmlLabel:
            try:
                sLabel = self.dXmlLabel2Label[sXmlLabel]
            except KeyError:
                #not a label of interest
                if self.lsXmlIgnoredLabel and sXmlLabel not in self.lsXmlIgnoredLabel: 
                    raise ValueError("Invalid label in node %s"%str(domnode))            

        if not sLabel: raise ValueError("Missing label in node %s"%str(domnode))
        
        return sLabel


    def setDomNodeLabel(self, domnode, sLabel):
        """
        Set the DOM node label in the format-dependent way
        """
        if sLabel != self.sDefaultLabel:
            domnode.setProp(sLabel, "yes")
        return sLabel


    # ---------------------------------------------------------------------------------------------------------        
    def _iter_GraphNode(self, doc, domNdPage, page):
        """
        Get the DOM, the DOM page node, the page object

        iterator on the DOM, that returns nodes  (of class Block)
        """    
        #--- XPATH contexts
        ctxt = doc.xpathNewContext()
        for ns, nsurl in self.dNS.items(): ctxt.xpathRegisterNs(ns, nsurl)
        assert self.sxpNode, "CONFIG ERROR: need an xpath expression to enumerate elements corresponding to graph nodes"
        ctxt.setContextNode(domNdPage)
        lNdBlock = ctxt.xpathEval(self.sxpNode) #all relevant nodes of the page

        for ndBlock in lNdBlock:
            ctxt.setContextNode(ndBlock)
            domid = ndBlock.prop("id")
            
            #TODO
            orientation = self.getAttributeInDepth(ndBlock, "orientation")
            classIndex = 0   #is computed later on

            x1 = float(ndBlock.prop("x"))
            y1 = float(ndBlock.prop("y"))
            w, h = float(ndBlock.prop("width")), float(ndBlock.prop("height"))
            
            #get the textual content
            if ndBlock.name == 'TOKEN':
                sText = ndBlock.get_content().decode('utf-8')
            else:
                lnTok = ctxt.xpathEval('.//TOKEN')
                sText = " ".join([nd.get_content().decode('utf-8') for nd in lnTok])
                
            #and create a Block
            assert ndBlock
            blk = Block(page.pnum, (x1, y1, w, h), sText, orientation, classIndex, self, ndBlock, domid=domid)
            assert blk.node
            
            #link it to its page
            blk.page = page

            yield blk
            
        ctxt.xpathFreeContext()       
        
        raise StopIteration()        
        
    def getAttributeInDepth(self, nd, attr):
        """
        look in depth for orientation, return "" if none found
        into BLOCK LINE TEXT TOKEN
        return a string
        """
        value = nd.prop(attr)
        if not value:                         #ok, not yet stored
            try:
                nd = nd.children
                value = nd.prop(attr)
                if not value:
                    nd = nd.children
                    value = nd.prop(attr) 
                if not value:
                    nd = nd.children
                    value = nd.prop(attr) 
            except IndexError:
                pass
            except AttributeError: # when no children (strange BTW)
                pass            
        return value               



# --- AUTO-TESTS --------------------------------------------------------------------------------------------
