# -*- coding: utf-8 -*-
""" 
    H. Déjean

    cpy Naverlabs Europe 2017

    textregion 2  cells  (opposite of PageCellToRegion)
    
"""


import sys, os.path
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'src')

import  libxml2

import common.Component as Component
import config.ds_xml_def as ds_xml
from common.trace import traceln
from xml_formats.PageXml import PageXml
from util.Polygon import Polygon
from optparse import OptionParser

class region2cell(Component.Component):
    #DEFINE the version, usage and description of this particular component
    usage = " -i TEXTREGION.pxml --tabfile TABLE.pxml  -o MERGETABLETEXT.pxml"
    version = "v1.23"
    description = "description: Merge input textline with table version"
    usage = " "
    version = "$Revision: 1.0 $"
        
    name="region2cell"
    kDPI = "dpi"
    def __init__(self):

        Component.Component.__init__(self, "region2cell", self.usage, self.version, self.description) 
        self.sPttrn     = None
        self.dpi = 300
         
        self.tableFile =None
        self.xmlns='http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
        self.id=1
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if dParams.has_key(self.kDPI)    : self.dpi          = int(dParams[self.kDPI])
        if dParams.has_key("tablefile")    : self.tableFile  = dParams["tablefile"]

    
#     def createTableRegion(self,pagenode,ns):
#         """
#             find the BB of the table
#         """
#         print pagenode.name
#         ctxt = pagenode.doc.xpathNewContext()
#         ctxt.xpathRegisterNs("a", self.xmlns)
#         xpath  = ".//a:%s" % ("TableCell")
#         ctxt.setContextNode(pagenode)
#         lCells = ctxt.xpathEval(xpath)
#         if lCells == []:
#             return True
#         
#         table = PageXml.createPageXmlNode("TableRegion", ns)
#         pagenode.addChild(table)
#         ## get all the textlines of the cell         
#         minx,miny,maxx,maxy = 9e9,9e9,0,0
#         for cell in lCells:
#             if cell.prop('id')[0] =='T':
#                 lXY = PageXml.getPointList(cell)  #the polygon
#                 if lXY == []:
#                     continue
#                 plg = Polygon(lXY)  
#                 try:
#                     x1,y1, x2,y2 = plg.fitRectangle()
#                 except ZeroDivisionError:
#                     continue
#                 x1,y1,x2,y2 = plg.getBoundingBox()
#                 minx = min(minx,x1)            
#                 miny = min(miny,y1)            
#                 maxx = max(maxx,x2)            
#                 maxy = max(maxy,y2)
# 
#                 cell.unlinkNode()
#                 table.addChild(cell)
#         
#         """
#             finally: simply use the BB of the textlines + padding
#         """
#         coords = PageXml.createPageXmlNode("Coords", ns)
#         coords.setProp('points',"%d,%d %d,%d %d,%d %d,%d"%(minx,miny,maxx,miny,maxx,maxy,minx,maxy))
#         table.addChild(coords)
#         table.setNsProp(ns, "id", 'table_1')
            
                 
    def mergetextRegion2Cell(self,textdoc,tabledoc):
        """
            assume the textregions were orignially tablecell converted bt tableCell2Region
        """
        
        # get list of textLines for mtextdoc
#         ns= textdoc.getRootElement().ns() 
 
        ctxt = textdoc.xpathNewContext()
        ctxt.xpathRegisterNs("a", self.xmlns)
        xpath  = "//a:%s" % ("TextLine")
        lTextLinesTR = ctxt.xpathEval(xpath)

        # get textlines from tablecells
        ctxt2 = tabledoc.xpathNewContext()
        ctxt2.xpathRegisterNs("a", self.xmlns)
        xpath  = "//a:%s" % ("TextLine")
        lTextLinesTC = ctxt2.xpathEval(xpath)
        for texttb in lTextLinesTC:
            xpath  = ".//a:%s" % ("Unicode")
            ctxt2.setContextNode(texttb)
            ltbunicode = ctxt2.xpathEval(xpath)      
            for texttr in lTextLinesTR:
                if texttb.prop('id') == texttr.prop('id'):
                    """
                    <TextLine id="line_1501552365047_634" custom="readingOrder {index:1;}">
                    <Coords points="989,226 1341,222 1342,272 990,276"/>
                    <Baseline points="990,271 1342,267"/>
                    <TextEquiv>
                        <Unicode></Unicode>
                    """
                    xpath  = ".//a:%s" % ("Unicode")
                    ctxt.setContextNode(texttr)
                    ltrunicode = ctxt.xpathEval(xpath)
                    ltbunicode[0].setContent(ltrunicode[0].getContent())
                    if texttr.prop('custom') is not None:
                        texttb.setProp('custom',texttr.prop('custom') )
                    
        ctxt.xpathFreeContext()
        ctxt2.xpathFreeContext()
        return tabledoc
    
    def run(self):
        
        docdom=self.loadDom()
        tabledom=self.loadDom(filename=self.tableFile)
        tabledom =self.mergetextRegion2Cell(docdom,tabledom)
        PageXml.setMetadata( tabledom, None, 'NLE', Comments='TextRegion/TableCell Merging')
        docdom.free()
        
        return tabledom
                
if __name__ == "__main__":
    
    #command line
#     traceln( "=============================================================================")
    
    cmp = region2cell()
    
    #prepare for the parsing of the command line
#     cmp.createCommandLineParser()
    parser = OptionParser(usage="", version=cmp.versionComponent)
    parser.description = cmp.description    
    parser.add_option("-i", "--input", dest="input", default="-", action="store", type="string", help="TextRegion PageXML file", metavar="<file>")
    parser.add_option("-t","--tab", dest="tablefile", action="store", type="string", help="TableRegion PageXML file"   , metavar="<filename>")
    parser.add_option("-o", "--output", dest="output", default="-", action="store", type="string", help="output PageXML file", metavar="<file>")

    #parse the command line
    cmp.usage = "python %prog" + cmp.usageComponent
    (options, args) = parser.parse_args()
    cmp.inputFileName=options.input
    cmp.outputFileName = options.output
    cmp.tableFile = options.tablefile

    doc = cmp.run()
    cmp.writeDom(doc, True)
    doc.free()
    traceln("Merge done for %s and %s  in %s" % (options.input,options.tablefile,options.output))


