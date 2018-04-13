# -*- coding: utf-8 -*-
""" 
    H. Déjean

    cpy Naver Labs 2017

    simply replace cells by TextRegion
    this is a need for ABP transcribers
    
    
    ex: 
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import sys, os.path
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'TranskribusDU')


import common.Component as Component
import config.ds_xml_def as ds_xml
from common.trace import traceln
from xml_formats.PageXml import PageXml, MultiPageXml
from util.Polygon import Polygon

class table2TextRegion(Component.Component):
    #DEFINE the version, usage and description of this particular component
    usage = " "
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
         
#         self.xmlns='http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
        self.ns={'a':PageXml.NS_PAGE_XML}
        self.id=1
        
        self.HeightTH=0.5
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if self.kDPI in dParams.keys()    : self.dpi          = int(dParams[self.kDPI])
#         if dParams.has_key("HTR")    : self.HeightTH         = int(dParams["HTR"])
        if "vpadding" in dParams: self.vpadding     = dParams["vpadding"]
        if "hpadding" in dParams: self.hpadding     = dParams["hpadding"]

    
    def resizeCell(self,cell,ns):
        """
            replace the cell region by a BB for textlines: better for transcriber
        """
        xpath  = "./a:%s" % ("TextLine")     
        lTextLines = cell.xpath(xpath, namespaces={'a':PageXml.NS_PAGE_XML})
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
        
        corner = cell[0]
#         print minx,miny,maxx,miny,maxx,maxy,minx,maxy
        corner.set('points',"%d,%d %d,%d %d,%d %d,%d"%(minx,miny,maxx,miny,maxx,maxy,minx,maxy))            
        
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
 
 
        
        xpath  = "//a:%s" % ("TableRegion")
        lTables = document.getroot().xpath(xpath,namespaces = self.ns)
        
        # need
        xpath  = "//a:%s" % ("ReadingOrder")
        lRO = document.getroot().xpath(xpath,namespaces = self.ns)
        if lRO == []:      
            ro = PageXml.createPageXmlNode('ReadingOrder')
            #addPrevSibling
        else:
            ro =lRO[0]
        
        for table in lTables:
            orderGroup = PageXml.createPageXmlNode('OrderedGroup')
            ro.append(orderGroup)
            orderGroup.set('{%s}id'%PageXml.NS_PAGE_XML,table.get('id'))
            orderGroup.set('{%s}caption'%PageXml.NS_PAGE_XML,'Cell2TextRegion')

            xpath  = "./a:%s" % ("TableCell")
            lCells = table.xpath(xpath,namespaces = self.ns)
            ## sort cells by rows
            lCells.sort(key=lambda x:int(x.get('row')))
            for i,cell in enumerate(lCells):
                #???
#                 cell.unlinkNode()
#                 print cell
                table.getparent().append(cell)
                cell.tag = '{%s}TextRegion'%(PageXml.NS_PAGE_XML)
                cell.set('custom',"readingOrder {index:%d;}"%i)
                # delete cell props
                for propname in ['row','col','rowSpan','colSpan']:
                    del cell.attrib[propname]
                # del leftBorderVisible, topBorderVisible,rightBorderVisible,bottomBorderVisible
                # to do
                #del CornerPts
                xpath  = "./a:%s" % ("CornerPts")
                lCorner = cell.xpath(xpath,namespaces = self.ns)
                for c in lCorner:
                    c.getparent().remove(c)
                reind = PageXml.createPageXmlNode('RegionRefIndexed')
                orderGroup.append(reind)
                reind.set('{%s}index'%PageXml.NS_PAGE_XML,str(i))
                reind.set('{%s}regionRef'%PageXml.NS_PAGE_XML,cell.get('id'))
                
                ## resize cell/region:
                if self.resizeCell(cell,self.ns):
                    cell.getparent().remove(cell)
#             table.unlinkNode()
            del(table)
        
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


