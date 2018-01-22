# -*- coding: utf-8 -*-
""" 
    H. Déjean

    cpy Naver Labs 2017

    simply replace cells by TextRegion
    this is a need for ABP transcribers
    
    
    ex: 
"""


import sys, os.path
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'src')

# import  libxml2

import common.Component as Component
import config.ds_xml_def as ds_xml
from common.trace import traceln
from xml_formats.PageXml import PageXml, MultiPageXml
from util.Polygon import Polygon

class table2TextRegion(Component.Component):
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
        
        self.HeightTH=0.5
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if dParams.has_key(self.kDPI)    : self.dpi          = int(dParams[self.kDPI])
#         if dParams.has_key("HTR")    : self.HeightTH         = int(dParams["HTR"])
        if dParams.has_key("vpadding")    : self.vpadding     = dParams["vpadding"]
        if dParams.has_key("hpadding")    : self.hpadding     = dParams["hpadding"]

    
    def resizeCell(self,cell,ns):
        """
            replace the cell region by a BB for textlines: better for transcriber
        """
        ctxt = cell.doc.xpathNewContext()
        ctxt.xpathRegisterNs("a", self.xmlns)
        xpath  = "./a:%s" % ("TextLine")
        ctxt.setContextNode(cell)
        lTextLines = ctxt.xpathEval(xpath)
        if lTextLines == []:
            return True
        
        ## get minx,maxy  for the cell
        lXY = PageXml.getPointList(cell)  #the polygon
        plg = Polygon(lXY)  
        x1,y1, x2,y2 = plg.fitRectangle()
        x1,y1,x2,y2 = plg.getBoundingBox()
        cellX1=x1
        cellX2=x2
        cellY1= y1
        cellY2=y2


        ## get all the textlines of the cell         
        minx,miny,maxx,maxy = 9e9,9e9,0,0
        for line in lTextLines:
            lXY = PageXml.getPointList(line)  #the polygon
            # in case empty cell
            if lXY == []:
                continue
            plg = Polygon(lXY)  
            try:
                x1,y1, x2,y2 = plg.fitRectangle()
            except ZeroDivisionError:
                continue
            x1,y1,x2,y2 = plg.getBoundingBox()
            minx = min(minx,x1)            
            miny = min(miny,y1)            
            maxx = max(maxx,x2)            
            maxy = max(maxy,y2)
        
        
        """
            finally: simply use the BB of the textlines + padding
        """
        
#         # new request: height= max(cell,text)
#         ## new new request: (12/10/2017): min (cell, text)!
#         HCell = cellY2 - cellY1
#         HBBText = maxy - miny
#   
        miny -= self.vpadding # vertical padding (top)       
        maxy += self.vpadding # vertical padding (bottom)
#         
#         # Height computation
#         ## if HBBText <= self.HeightTH * HCell: take HBBText as height for TextREgion
#         if HBBText > self.HeightTH * HCell:
#             miny = max(miny,cellY1)
#             maxy = min(maxy,cellY2)            
#         # else : don't touch miny, maxy  : use only Text for computing Height        
#         
#         # Width computation
        minx -= self.hpadding  # horizontal padding 
        maxx += self.hpadding  # horizontal padding 
        
#         minx = min(cellX1,minx)
#         maxx = max(cellX2, maxx)
#         print cellX2, maxx
        
        corner = cell.children
#         print minx,miny,maxx,miny,maxx,maxy,minx,maxy
        corner.setProp('points',"%d,%d %d,%d %d,%d %d,%d"%(minx,miny,maxx,miny,maxx,maxy,minx,maxy))            
        
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
#                 print cell
                table.parent.addChild(cell)
                cell.setName('TextRegion')
                cell.setProp('custom',"readingOrder {index:%d;}"%i)
                # delete cell props
                for propname in ['row','col','rowSpan','colSpan']:
                    cell.unsetProp(propname)
                # del leftBorderVisible, topBorderVisible,rightBorderVisible,bottomBorderVisible
                # to do
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
                ## resize cell/region:
                if self.resizeCell(cell,ns):
                    cell.unlinkNode()
            table.unlinkNode()
            del(table)
        
        ctxt.xpathFreeContext()
        PageXml.validate(document)
        
    def run(self):
        
        docdom=self.loadDom()
        self.convertTableCells(docdom)
        try:PageXml.setMetadata( docdom, None, 'NLE', Comments='TableCell 2 TextRegion')
        except ValueError:
            MultiPageXml.setMetadata(docdom, None, 'NLE', Comments='TableCell 2 TextRegion')
        return docdom
                
if __name__ == "__main__":
    
    #command line
    traceln( "=============================================================================")
    
    cmp = table2TextRegion()

    #prepare for the parsing of the command line
    cmp.createCommandLineParser()
#     cmp.add_option("", "--"+cmp.kPTTRN, dest=cmp.kPTTRN, action="store", type="string", help="REQUIRED **: File name pattern, e.g. /tmp/*/to?o*.xml"   , metavar="<pattern>")
#     cmp.add_option("--HeightCR", dest="HTH", action="store", type='string', help="Threshold for adapting TextRegion Height")
    cmp.add_option("--vpadding", dest="vpadding", action="store", type='int', default=20,help="vertical padding (default 20 pixels)")
    cmp.add_option("--hpadding", dest="hpadding", action="store", type='int', default=30,help="horizontal padding (default 30 pixels)")
    
 
    #parse the command line
    dParams, args = cmp.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    cmp.setParams(dParams)
    doc = cmp.run()
    cmp.writeDom(doc, True)


