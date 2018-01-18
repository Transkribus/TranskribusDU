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

from NodeType import NodeType
from Block import Block


class NodeType_DS(NodeType):

    #Namespace, of PageXml, at least
    dNS = {}

    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True):
        NodeType.__init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel, bOther)
        
    def setXpathExpr(self, sxpNode): #, sxpTextual)):
        self.sxpNode    = sxpNode
#         self.sxpTextual = sxpTextual    #always in TOKENs in DS XML

    def getXpathExpr(self):
        """
        get any Xpath related information to extract the nodes from an XML file
        """
        return self.sxpNode
    
    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = self.sDefaultLabel
        
        sXmlLabel = None
        if domnode.get("title_oracle_best"):
            sXmlLabel = "title"
        elif domnode.get("pnum_oracle"):
            sXmlLabel = "pnum"
        if sXmlLabel:
            try:
                sLabel = self.dXmlLabel2Label[sXmlLabel]
            except KeyError:
                #not a label of interest
                try:
                    self.checkIsIgnored(sXmlLabel)
                    #if self.lsXmlIgnoredLabel and sXmlLabel not in self.lsXmlIgnoredLabel: 
                except:
                    raise ValueError("Invalid label '%s' in node %s"%(sXmlLabel, str(domnode)))

        if not sLabel: raise ValueError("Missing label in node %s"%str(domnode))
        
        return sLabel


    def setDomNodeLabel(self, domnode, sLabel):
        """
        Set the DOM node label in the format-dependent way
        """
        if sLabel != self.sDefaultLabel:
            domnode.set(sLabel, "yes")
        return sLabel


    # ---------------------------------------------------------------------------------------------------------        
    def _iter_GraphNode(self, doc, domNdPage, page):
        """
        Get the DOM, the DOM page node, the page object

        iterator on the DOM, that returns nodes  (of class Block)
        """    
        assert self.sxpNode, "CONFIG ERROR: need an xpath expression to enumerate elements corresponding to graph nodes"
        lNdBlock = domNdPage.xpath(self.sxpNode, namespaces=self.dNS) #all relevant nodes of the page

        for ndBlock in lNdBlock:
            domid = ndBlock.get("id")
            
            #TODO
            orientation = self.getAttributeInDepth(ndBlock, "orientation")
            classIndex = 0   #is computed later on

            x1 = float(ndBlock.get("x"))
            y1 = float(ndBlock.get("y"))
            w, h = float(ndBlock.get("width")), float(ndBlock.get("height"))
            
            #get the textual content
            if ndBlock.name == 'TOKEN':
                sText = ndBlock.text
            else:
                lnTok = ndBlock.xpath('.//TOKEN', namespaces=self.dNS)
                sText = " ".join([nd.text for nd in lnTok])
                
            #and create a Block
            assert ndBlock
            blk = Block(page, (x1, y1, w, h), sText, orientation, classIndex, self, ndBlock, domid=domid)
            assert blk.node
            
            yield blk
            
        raise StopIteration()        
        
    def getAttributeInDepth(self, nd, attr):
        """
        look in depth for orientation, return "" if none found
        into BLOCK LINE TEXT TOKEN
        return a string
        """
        value = nd.get(attr)
        if not value:                         #ok, not yet stored
            try:
                nd = nd[0]
                value = nd.get(attr)
                if not value:
                    nd = nd[0]
                    value = nd.get(attr) 
                if not value:
                    nd = nd[0]
                    value = nd.get(attr) 
            except IndexError:
                pass
            except AttributeError: # when no children (strange BTW)
                pass            
        return value               



# --- AUTO-TESTS --------------------------------------------------------------------------------------------
