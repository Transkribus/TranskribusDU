# -*- coding: utf-8 -*-
""" 
    H. Déjean

    cpy Naverlabs Europe 2017

    textregion 2  cells  (opposite of PageCellToRegion)
    
"""

from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import sys, os.path
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'TranskribusDU')


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
        if self.kDP in dParams.keys()    : self.dpi          = int(dParams[self.kDPI])
        if "tablefile" in dParams.keys()    : self.tableFile  = dParams["tablefile"]

    
    def mergetextRegion2Cell(self,textdoc,tabledoc):
        """
            assume the textregions were orignially tablecell converted bt tableCell2Region
        """
        
        # get list of textLines for mtextdoc
#         ns= textdoc.getRootElement().ns() 
 
#         ctxt = textdoc.xpathNewContext()
#         ctxt.xpathRegisterNs("a", self.xmlns)

        xpath  = "//a:%s" % ("TextLine")
        lTextLinesTR = textdoc.getroot().xpath(xpath, namespaces={'a':self.xmlns})

        # get textlines from tablecells
        xpath  = "//a:%s" % ("TextLine")
        lTextLinesTC = tabledoc.getroot().xpath(xpath, namespaces={'a':self.xmlns})
        for texttb in lTextLinesTC:
            xpath  = ".//a:%s" % ("Unicode")
            ltbunicode = texttb.xpath(xpath, namespaces={'a':self.xmlns})      
            for texttr in lTextLinesTR:
                if texttb.get('id') == texttr.get('id'):
                    """
                    <TextLine id="line_1501552365047_634" custom="readingOrder {index:1;}">
                    <Coords points="989,226 1341,222 1342,272 990,276"/>
                    <Baseline points="990,271 1342,267"/>
                    <TextEquiv>
                        <Unicode></Unicode>
                    """
                    xpath  = ".//a:%s" % ("Unicode")
                    ltrunicode = texttr.xpath(xpath, namespaces={'a':self.xmlns})
                    ltbunicode[0].text = ltrunicode[0].text
                    if texttr.get('custom') is not None:
                        texttb.set('custom',texttr.get('custom') )
                    
        return tabledoc
    
    def run(self):
        
        docdom=self.loadDom()
        tabledom=self.loadDom(filename=self.tableFile)
        tabledom =self.mergetextRegion2Cell(docdom,tabledom)
        PageXml.setMetadata( tabledom, None, 'NLE', Comments='TextRegion/TableCell Merging')
        
        return tabledom
                
if __name__ == "__main__":
    
    #command line
#     traceln( "=============================================================================")
    
    cmp = region2cell()
    
    #prepare for the parsing of the command line
#     cmp.createCommandLineParser()
    parser = OptionParser(usage="python %prog" + cmp.usageComponent,version=cmp.versionComponent)
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
    traceln("Merge done for %s and %s  in %s" % (options.input,options.tablefile,options.output))


