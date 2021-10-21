# -*- coding: utf-8 -*-

"""
    Reading and setting the labels of a PageXml document

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import types
from lxml import etree

from common.trace import traceln
from xml_formats.PageXml import PageXml, PageXmlException
from util.Polygon import Polygon

from .NodeType import NodeType
from .Block import Block

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

    nbNoTextWarning = 0

    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun
                 , bPreserveWidth=False):
        """
        if bPreserveWidth is True, the fitted rectangle gets the x1 and x2 of the polygon bounding box
        """
        NodeType.__init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel, bOther)
        
        self.BBoxDeltaFun = BBoxDeltaFun
        if  self.BBoxDeltaFun is not None:
            if type(self.BBoxDeltaFun) is tuple:
                xFun, yFun = self.BBoxDeltaFun
                if xFun is not None and type(xFun)!= types.FunctionType:
                    raise Exception("Error: first element of BBoxDeltaFun tuple must be None or a function (or a lambda)")
                if yFun is not None and type(yFun)!= types.FunctionType:
                    raise Exception("Error: second element of BBoxDeltaFun tuple must be None or a function (or a lambda)")
            else:
                if type(self.BBoxDeltaFun) != types.FunctionType:
                    raise Exception("Error: BBoxDeltaFun must be None or a function (or a lambda)")
        self.bPreserveWidth = bPreserveWidth

    def setXpathExpr(self, t_sxpNode_sxpTextual):
        self.sxpNode, self.sxpTextual = t_sxpNode_sxpTextual
    
    def getXpathExpr(self):
        """
        get any Xpath related information to extract the nodes from an XML file
        """
        return (self.sxpNode, self.sxpTextual)
    
    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = self.sDefaultLabel
        domnode = graph_node.node
        try:
            try:
                sXmlLabel = PageXml.getCustomAttr(domnode, self.sCustAttr_STRUCTURE, self.sCustAttr2_TYPE)
            except PageXmlException as e:
                if self.bOther:
                    return self.sDefaultLabel  #absence of label but bOther was True (I guess)
                else:
                    raise e
            try:
                sLabel = self.dXmlLabel2Label[sXmlLabel]
            except KeyError:
                #not a label of interest
                try:
                    self.checkIsIgnored(sXmlLabel)
                    #if self.lsXmlIgnoredLabel and sXmlLabel not in self.lsXmlIgnoredLabel: 
                except:
                    raise ValueError("Invalid label '%s' in node %s"%(sXmlLabel, etree.tostring(domnode)))
        except KeyError:
            #no label at all
            if not self.sDefaultLabel: raise ValueError("Missing label in node %s"%etree.tostring(domnode))
        
        return sLabel


    def setDocNodeLabel(self, graph_node, sLabel):
        """
        Set the DOM node label in the format-dependent way
        """
        if sLabel != self.sDefaultLabel:
            PageXml.setCustomAttr(graph_node.node, self.sCustAttr_STRUCTURE, self.sCustAttr2_TYPE, self.dLabel2XmlLabel[sLabel])
        return sLabel


    # ---------------------------------------------------------------------------------------------------------        
    def _iter_GraphNode(self, doc, domNdPage, page):
        """
        Get the DOM, the DOM page node, the page object

        iterator on the DOM, that returns nodes  (of class Block)
        """    
        #--- XPATH contexts
        assert self.sxpNode, "CONFIG ERROR: need an xpath expression to enumerate elements corresponding to graph nodes"
        lNdBlock = domNdPage.xpath(self.sxpNode, namespaces=self.dNS) #all relevant nodes of the page

        for ndBlock in lNdBlock:
            domid = ndBlock.get("id")
            sText = self._get_GraphNodeText(doc, domNdPage, ndBlock)
            if sText == None:
                sText = ""
                NodeType_PageXml.nbNoTextWarning += 1
                if NodeType_PageXml.nbNoTextWarning < 33:
                    traceln("Warning: no text in node %s"%domid) 
                elif NodeType_PageXml.nbNoTextWarning == 33:
                    traceln("Warning: no text in node %s  - *** %d repetition : I STOP WARNING ***" % (domid, NodeType_PageXml.nbNoTextWarning))
                #raise ValueError, "No text in node: %s"%ndBlock 
            
            #now we need to infer the bounding box of that object
            lXY = PageXml.getPointList(ndBlock)  #the polygon
            if lXY == []:
                continue
            
            plg = Polygon(lXY)
            try:
                x1,y1, x2,y2 = plg.fitRectangle(bPreserveWidth=self.bPreserveWidth)
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
            except ValueError:
                x1,y1,x2,y2 = plg.getBoundingBox()
                
                
            #we reduce a bit this rectangle, to ovoid overlap
            if not(self.BBoxDeltaFun is None):
                if type(self.BBoxDeltaFun) is tuple and len(self.BBoxDeltaFun) == 2:
                    xFun, yFun = self.BBoxDeltaFun
                    if xFun is not None:
                        dx = xFun(x2-x1)
                        x1, x2 = int(round(x1+dx)), int(round(x2-dx))
                    if yFun is not None:
                        dy = yFun(y2-y1)
                        y1, y2 = int(round(y1+dy)), int(round(y2-dy))
                else:
                    # historical code
                    w,h = x2-x1, y2-y1
                    dx = self.BBoxDeltaFun(w)
                    dy = self.BBoxDeltaFun(h)
                    x1,y1, x2,y2 = [ int(round(v)) for v in [x1+dx,y1+dy, x2-dx,y2-dy] ]
                
            # store the rectangle"            
            ndBlock.set("DU_points", " ".join( ["%d,%d"%(int(x), int(y)) for x,y in [(x1, y1), (x2,y1), (x2,y2), (x1,y2)]] ))
            
            #TODO
            orientation = 0  #no meaning for PageXml
            classIndex = 0   #is computed later on

            #and create a Block
            blk = Block(page, (x1, y1, x2-x1, y2-y1), sText, orientation, classIndex, self, ndBlock, domid=domid)
            
            yield blk
            
        return         
        
    def _get_GraphNodeText(self, doc, domNdPage, ndBlock):
        """
        Extract the text of a DOM node
        
        Get the DOM, the DOM page node, the page object DOM node, and optionally an xpath context

        return a unicode string
        """    
        lNdText = ndBlock.xpath(self.sxpTextual, namespaces=self.dNS)
        if len(lNdText) != 1:
            if len(lNdText) <= 0:
                # raise ValueError("I found no useful TextEquiv below this node... \n%s"%etree.tostring(ndBlock))
                return None
            else:
                raise ValueError("I expected exactly one useful TextEquiv below this node. Got many... \n%s"%etree.tostring(ndBlock))
        
        return PageXml.makeText(lNdText[0])
        
        
        
class NodeType_PageXml_type(NodeType_PageXml):
    """
    In some PageXml, the label is in the type attribute! 
    
    Use this class for those cases!
    """
    
    sLabelAttr = "type"

    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun
                 , bPreserveWidth=False):
        NodeType_PageXml.__init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun, bPreserveWidth)

    def setLabelAttribute(self, sAttrName="type"):
        """
        set the name of the Xml attribute that contains the label
        """
        self.sLabelAttr = sAttrName
                    
    def getLabelAttribute(self):
        return self.sLabelAttr

    def getLabel(self, domnode):
        return domnode.get(self.sLabelAttr)

    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = self.sDefaultLabel
        domnode = graph_node.node
        # sXmlLabel = domnode.get(self.sLabelAttr)
        sXmlLabel = self.getLabel(domnode)
        try:
            sLabel = self.dXmlLabel2Label[sXmlLabel]
        except KeyError:
            #not a label of interest
            try:
                if sXmlLabel:
                    self.checkIsIgnored(sXmlLabel)
                else:
                    return self.sDefaultLabel
                #if self.lsXmlIgnoredLabel and sXmlLabel not in self.lsXmlIgnoredLabel: 
            except:
                raise ValueError("Invalid label '%s'"
                                 " (from @%s or @%s) in node %s"%(sXmlLabel,
                                                           self.sLabelAttr,
                                                           self.sDefaultLabel,
                                                           etree.tostring(domnode)))
        
        return sLabel


    def setDocNodeLabel(self, graph_node, sLabel):
        """
        Set the DOM node label in the format-dependent way
        """
        if sLabel != self.sDefaultLabel:
            graph_node.node.set(self.sLabelAttr, self.dLabel2XmlLabel[sLabel])
        return sLabel

class NodeType_PageXml_type_woText(NodeType_PageXml_type):
    """
            for document wo HTR: no text
    """
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun):
        NodeType_PageXml_type.__init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun)

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
        lNdText = ndBlock.xpath(self.sxpTextual, namespaces=self.dNS)
        if len(lNdText) != 1:
            if len(lNdText) > 1: raise ValueError("More than 1 textual content for this node: %s"%etree.tostring(ndBlock))
            
            #let's try to get th etext of the words, and concatenate...
            # traceln("Warning: no text in node %s => looking at words!"%ndBlock.prop("id")) 
            # lsText = [ntext.content.decode('utf-8').strip() for ntext in ctxt.xpathEval('.//pc:Word/pc:TextEquiv//text()')] #if we have both PlainText and UnicodeText in XML, :-/
            lsText = [_nd.text.strip() for _nd in ctxt.xpathEval('.//pc:Word/pc:TextEquiv')] #if we have both PlainText and UnicodeText in XML, :-/
            return " ".join(lsText)
        
        return PageXml.makeText(lNdText[0])

# --- AUTO-TESTS --------------------------------------------------------------------------------------------
def test_getset():
    from lxml import etree
    from io import BytesIO
    
    sXml = b"""
            <TextRegion  id="p1_region_1471502505726_2" custom="readingOrder {index:9;} structure {type:page-number;}">
                <Coords points="972,43 1039,43 1039,104 972,104"/>
            </TextRegion>
            """
    class MyNode:
        def __init__(self, nd): self.node = nd
        
    doc = etree.parse(BytesIO(sXml))
    nd = doc.getroot()
    graph_node = MyNode(nd)
    obj = NodeType_PageXml("foo", ["page-number", "index"])
    assert obj.parseDocNodeLabel(graph_node) == 'foo_page-number', obj.parseDocNodeLabel(nd)
    assert obj.parseDocNodeLabel(graph_node) == 'foo_page-number'
    assert obj.setDocNodeLabel(graph_node, "foo_index") == 'foo_index'
    assert obj.parseDocNodeLabel(graph_node) == 'foo_index'

def test_getset2():
    from lxml import etree
    from io import BytesIO
    
    sXml = b"""
            <TextRegion type="page-number" id="p1_region_1471502505726_2" custom="readingOrder {index:9;} ">
                <Coords points="972,43 1039,43 1039,104 972,104"/>
            </TextRegion>
            """
    class MyNode:
        def __init__(self, nd): self.node = nd

    doc = etree.parse(BytesIO(sXml))
    nd = doc.getroot()
    graph_node = MyNode(nd)
    obj = NodeType_PageXml("foo", ["page-number", "index"], [""])
    assert obj.parseDocNodeLabel(graph_node) == 'foo_OTHER'
    assert obj.setDocNodeLabel(graph_node, "foo_index") == 'foo_index'
    assert obj.parseDocNodeLabel(graph_node) == 'foo_index'
    
    
def test_getset3():
    import pytest
    from lxml import etree
    from io import BytesIO
    
    sXml = b"""
            <TextRegion type="page-number" id="p1_region_1471502505726_2" custom="readingOrder {index:9;} ">
                <Coords points="972,43 1039,43 1039,104 972,104"/>
            </TextRegion>
            """
    class MyNode:
        def __init__(self, nd): self.node = nd

    doc = etree.parse(BytesIO(sXml))
    nd = doc.getroot()
    graph_node = MyNode(nd)
    obj = NodeType_PageXml("foo", ["page-number", "index"], [""], bOther=False)
    with pytest.raises(PageXmlException):
        assert obj.parseDocNodeLabel(graph_node) == 'foo_OTHER'
    assert obj.setDocNodeLabel(graph_node) == 'foo_index'
    assert obj.parseDocNodeLabel(graph_node) == 'foo_index'
