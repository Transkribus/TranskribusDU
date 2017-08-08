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
import types

from common.trace import traceln

from NodeType import NodeType
from xml_formats.PageXml import PageXml
from util.Polygon import Polygon
from Block import Block

def defaultBBoxDeltaFun(w):
    """
    When we reduce the width or height of a bounding box, we use this function to compute the deltaX or deltaY
    , which is applied on x1 and x2 or y1 and y2
    
    For instance, for horizontal axis
        x1 = x1 + deltaFun(abs(x1-x2))
        x2 = x2 + deltaFun(abs(x1-x2))
    """
    # "historically, we were doing:
    dx = max(w * 0.066, min(20, w/3))
    #for ABP table RV is doing: dx = max(w * 0.066, min(5, w/3)) , so this function can be defined by the caller.
    return dx 
    

class NodeType_PageXml(NodeType):
    #where the labels can be found in the data
    sCustAttr_STRUCTURE     = "structure"
    sCustAttr2_TYPE         = "type"

    #Namespace, of PageXml, at least
    dNS = {"pc":PageXml.NS_PAGE_XML}

    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun):
        NodeType.__init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel, bOther)
        
        self.BBoxDeltaFun = BBoxDeltaFun
        assert type(self.BBoxDeltaFun) == types.FunctionType, "Error: BBoxDeltaFun must be a function (or a lambda)"

    def setXpathExpr(self, (sxpNode, sxpTextual)):
        self.sxpNode    = sxpNode
        self.sxpTextual = sxpTextual
    
    
    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = self.sDefaultLabel
        try:
            sXmlLabel = PageXml.getCustomAttr(domnode, self.sCustAttr_STRUCTURE, self.sCustAttr2_TYPE)
            try:
                sLabel = self.dXmlLabel2Label[sXmlLabel]
            except KeyError:
                #not a label of interest
                try:
                    self.checkIsIgnored(sXmlLabel)
                    #if self.lsXmlIgnoredLabel and sXmlLabel not in self.lsXmlIgnoredLabel: 
                except:
                    raise ValueError("Invalid label '%s' in node %s"%(sXmlLabel, str(domnode)))
        except KeyError:
            #no label at all
            if not self.sDefaultLabel: raise ValueError("Missing label in node %s"%str(domnode))
        
        return sLabel


    def setDomNodeLabel(self, domnode, sLabel):
        """
        Set the DOM node label in the format-dependent way
        """
        if sLabel != self.sDefaultLabel:
            PageXml.setCustomAttr(domnode, self.sCustAttr_STRUCTURE, self.sCustAttr2_TYPE, self.dLabel2XmlLabel[sLabel])
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
            domid = ndBlock.prop("id")
            sText = self._get_GraphNodeText(doc, domNdPage, ndBlock, ctxt)
            if sText == None:
                sText = ""
                traceln("Warning: no text in node %s"%domid) 
                #raise ValueError, "No text in node: %s"%ndBlock 
            
            #now we need to infer the bounding box of that object
            lXY = PageXml.getPointList(ndBlock)  #the polygon
            if lXY == []:
                continue
            
            plg = Polygon(lXY)
            try:
                x1,y1, x2,y2 = plg.fitRectangle()
            except ZeroDivisionError:
#                 traceln("Warning: ignoring invalid polygon id=%s page=%s"%(ndBlock.prop("id"), page.pnum))
#                 continue
#             if True:
#                 #we reduce a bit this rectangle, to ovoid overlap
#                 w,h = x2-x1, y2-y1
#                 dx = max(w * 0.066, min(20, w/3))  #we make sure that at least 1/"rd of te width will remain!
#                 dy = max(h * 0.066, min(20, w/3))
#                 x1,y1, x2,y2 = [ int(round(v)) for v in [x1+dx,y1+dy, x2-dx,y2-dy] ]

                x1,y1,x2,y2 = plg.getBoundingBox()
                
            #we reduce a bit this rectangle, to ovoid overlap
            w,h = x2-x1, y2-y1
            dx = self.BBoxDeltaFun(w)
            dy = self.BBoxDeltaFun(h)
            x1,y1, x2,y2 = [ int(round(v)) for v in [x1+dx,y1+dy, x2-dx,y2-dy] ]
                
            
            #TODO
            orientation = 0  #no meaning for PageXml
            classIndex = 0   #is computed later on

            #and create a Block
            assert ndBlock
            blk = Block(page, (x1, y1, x2-x1, y2-y1), sText, orientation, classIndex, self, ndBlock, domid=domid)
            assert blk.node
            
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
                ndCoord = util.xml_utils.addElement(doc, ndTextLine, "Coords")
                PageXml.setPoints(ndCoord, PageXml.getPointsFromBB(x1,y1,x2,y2))
            yield blk
            
        ctxt.xpathFreeContext()       
        if TEST_getPageXmlBlock:
            util.xml_utils.toFile(doc, "TEST_getPageXmlBlock.mpxml", True)
        
        raise StopIteration()        
        
    def _get_GraphNodeText(self, doc, domNdPage, ndBlock, ctxt=None):
        """
        Extract the text of a DOM node
        
        Get the DOM, the DOM page node, the page object DOM node, and optionally an xpath context

        return a unicode string
        """    
        ctxt.setContextNode(ndBlock)
        lNdText = ctxt.xpathEval(self.sxpTextual)
        if len(lNdText) != 1:
            if len(lNdText) <= 0:
                raise ValueError("I found no useful TextEquiv below this node... \n%s"%str(ndBlock))
            else:
                raise ValueError("I expected exactly one useful TextEquiv below this node. Got many... \n%s"%str(ndBlock))
        
        return PageXml.makeText(lNdText[0])
        
        
        
class NodeType_PageXml_type(NodeType_PageXml):
    """
    In some PageXml, the label is in the type attribute! 
    
    Use this class for those cases!
    """
    
    sLabelAttr = "type"

    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun):
        NodeType_PageXml.__init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun)

    def setLabelAttribute(self, sAttrName="type"):
        """
        set the name of the Xml attribute that contains the label
        """
        self.sLabelAttr = sAttrName
                    
    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = self.sDefaultLabel
        
        sXmlLabel = domnode.prop(self.sLabelAttr)
        try:
            sLabel = self.dXmlLabel2Label[sXmlLabel]
        except KeyError:
            #not a label of interest
            try:
                self.checkIsIgnored(sXmlLabel)
                #if self.lsXmlIgnoredLabel and sXmlLabel not in self.lsXmlIgnoredLabel: 
            except:
                raise ValueError("Invalid label '%s' in node %s"%(sXmlLabel, str(domnode)))
        
        return sLabel


    def setDomNodeLabel(self, domnode, sLabel):
        """
        Set the DOM node label in the format-dependent way
        """
        if sLabel != self.sDefaultLabel:
            domnode.setProp(self.sLabelAttr, self.dLabel2XmlLabel[sLabel])
        return sLabel

class NodeType_PageXml_type_woText(NodeType_PageXml_type):
    """
            for document wo HTR: no text
    """
    def _get_GraphNodeText(self, doc, domNdPage, ndBlock, ctxt=None):
        return u""
   

#---------------------------------------------------------------------------------------------------------------------------    
class NodeType_PageXml_type_NestedText(NodeType_PageXml_type):
    """
    In those PageXml, the text is not always is in the type attribute! 
    
    Use this class for GTBooks!
    """
    

    def _get_GraphNodeText(self, doc, domNdPage, ndBlock, ctxt=None):
        """
        Extract the text of a DOM node
        
        Get the DOM, the DOM page node, the page object DOM node, and optionally an xpath context

        return a unicode string
        """    
        ctxt.setContextNode(ndBlock)
        lNdText = ctxt.xpathEval(self.sxpTextual)
        if len(lNdText) != 1:
            if len(lNdText) > 1: raise ValueError("More than 1 textual content for this node: %s"%str(ndBlock))
            
            #let's try to get th etext of the words, and concatenate...
            # traceln("Warning: no text in node %s => looking at words!"%ndBlock.prop("id")) 
            lsText = [ntext.content.decode('utf-8').strip() for ntext in ctxt.xpathEval('.//pc:Word/pc:TextEquiv//text()')] #if we have both PlainText and UnicodeText in XML, :-/
            return " ".join(lsText)
        
        return PageXml.makeText(lNdText[0])

# --- AUTO-TESTS --------------------------------------------------------------------------------------------
def test_getset():
    import libxml2
    sXml = """
            <TextRegion type="page-number" id="p1_region_1471502505726_2" custom="readingOrder {index:9;} structure {type:page-number;}">
                <Coords points="972,43 1039,43 1039,104 972,104"/>
            </TextRegion>
            """
    doc = libxml2.parseMemory(sXml, len(sXml))
    nd = doc.getRootElement()
    obj = NodeType_PageXml()
    assert obj.parseDomNodeLabel(nd) == 'page-number'
    assert obj.parseDomNodeLabel(nd, "toto") == 'page-number'
    assert obj.setDomNodeLabel(nd, "index") == 'index'
    assert obj.parseDomNodeLabel(nd, "toto") == 'index'
    doc.freeDoc()

def test_getset_default():
    import libxml2
    sXml = """
            <TextRegion type="page-number" id="p1_region_1471502505726_2" custom="readingOrder {index:9;} ">
                <Coords points="972,43 1039,43 1039,104 972,104"/>
            </TextRegion>
            """
    doc = libxml2.parseMemory(sXml, len(sXml))
    nd = doc.getRootElement()
    obj = NodeType_PageXml()
    assert obj.parseDomNodeLabel(nd) == None
    assert obj.parseDomNodeLabel(nd, "toto") == 'toto'
    assert obj.setDomNodeLabel(nd, "index") == 'index'
    assert obj.parseDomNodeLabel(nd, "toto") == 'index'
    doc.freeDoc()

