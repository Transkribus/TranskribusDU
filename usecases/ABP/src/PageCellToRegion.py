# -*- coding: utf-8 -*-
""" 
    H. Déjean

    cpy Xerox 2017

    simply replace cells by TextRegion
    this is a need for ABP transcribers
    
"""


import sys, os.path
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'src')

import  libxml2

import common.Component as Component
import config.ds_xml_def as ds_xml
from common.trace import traceln
from xml_formats.PageXml import PageXml

class primaAnalysis(Component.Component):
    #DEFINE the version, usage and description of this particular component
    usage = "[-f N.N] "
    version = "v1.23"
    description = "description: PAGE XML CELL 2 TEXTREGION"
    usage = " "
    version = "$Revision: 1.0 $"
        
    name="primaAnalysis"
    kDPI = "dpi"
    def __init__(self):

        Component.Component.__init__(self, "pageXMLconverter", self.usage, self.version, self.description) 
        self.sPttrn     = None
        self.dpi = 300
         
        self.xmlns='http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
        self.id=1
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if dParams.has_key(self.kDPI)    : self.dpi          = int(dParams[self.kDPI])

    
    def convertTableCells(self,document):
        """
        needed ?
         <ReadingOrder>
              <OrderedGroup id="ro_1493890373963" caption="Regions reading order">
                <RegionRefIndexed index="0" regionRef="region_1493009854544_3646"/>
                <RegionRefIndexed index="1" regionRef="region_1493009858356_3647"/>
              </OrderedGroup>
        </ReadingOrder>

        """
 
 
        ns= document.getRootElement().ns() 
 
        ctxt = document.xpathNewContext()
        ctxt.xpathRegisterNs("a", self.xmlns)
        
        
        xpath  = "//a:%s" % ("TableRegion")
        lTables = ctxt.xpathEval(xpath)
        
        
        # need
        xpath  = "//a:%s" % ("ReadingOrder")
        lRO = ctxt.xpathEval(xpath)
        if lRO == []:      
            ro = PageXml.createPageXmlNode('ReadingOrder', ns)
            #addPrevSibling
        else:
            ro =lRO[0]
        
        orderGroup = PageXml.createPageXmlNode('OrderedGroup', ns)
            
        for table in lTables:
            orderGroup = PageXml.createPageXmlNode('OrderedGroup', ns)
            ro.addChild(orderGroup)
            orderGroup.setNsProp(ns,'id',table.prop('id'))
            orderGroup.setNsProp(ns,'caption','Cell2TextRegion')

            xpath  = "./a:%s" % ("TableCell")
            ctxt.setContextNode(table)
            lCells = ctxt.xpathEval(xpath)
            ## sort cells by rows
#             nbRows = max(lambda x:int(x.prop('row')),lCells)
            lCells.sort(key=lambda x:int(x.prop('row')))
            for i,cell in enumerate(lCells):
                cell.unlinkNode()
                table.parent.addChild(cell)
                cell.setName('TextRegion')
                cell.setProp('custom',"readingOrder {index:%d;}"%i)
                # delete cell props
                for propname in ['row','col','rowSpan','colSpan']:
                    cell.unsetProp(propname)
                #del CornerPts
                xpath  = "./a:%s" % ("CornerPts")
                ctxt.setContextNode(cell)
                lCorner = ctxt.xpathEval(xpath)
                for c in lCorner:
                    c.unlinkNode()
                    c.freeNode()
                reind = PageXml.createPageXmlNode('RegionRefIndexed', ns)
                orderGroup.addChild(reind)
                reind.setNsProp(ns,'index',str(i))
                reind.setNsProp(ns,'regionRef',cell.prop('id'))
            table.unlinkNode()
            del(table)
        
        ctxt.xpathFreeContext()
        PageXml.validate(document)
        
    def run(self):
        
        docdom=self.loadDom()
        self.convertTableCells(docdom)
        PageXml.setMetadata( docdom, None, 'XRCE', Comments='TableCell 2 TextRegion')
        
        return docdom
                
if __name__ == "__main__":
    
    #command line
    traceln( "=============================================================================")
    
    cmp = primaAnalysis()

    #prepare for the parsing of the command line
    cmp.createCommandLineParser()
#     cmp.add_option("", "--"+cmp.kPTTRN, dest=cmp.kPTTRN, action="store", type="string", help="REQUIRED **: File name pattern, e.g. /tmp/*/to?o*.xml"   , metavar="<pattern>")
#     cmp.add_option("--"+cmp.kDOCID, dest=cmp.kDOCID, action="store", type='string', help="docId in col")
    
 
    #parse the command line
    dParams, args = cmp.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    cmp.setParams(dParams)
    doc = cmp.run()
    cmp.writeDom(doc, True)


