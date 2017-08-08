# -*- coding: utf-8 -*-

"""
    DU task for BAR documents - see https://read02.uibk.ac.at/wiki/index.php/Document_Understanding_BAR
    
    Here we convert the human annotation into 2 kinds of annotations:
    - a semantic one: header, heading, page-number, resolution-marginalia, resolution-number, resolution-paragraph   (we ignore Marginalia because only 2 occurences)
    - a segmentation one: 2 complementary labels. We call them Heigh Ho. Could have been Yin Yang as well...
    - also, we store the resolution number in @DU_num
    
    These new annotations are stored in @DU_sem , @DU_sgm , @DU_num
    
    Copyright Naver Labs(C) 2017 JL Meunier

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You sSegmHould have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os, random, re
import libxml2

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln

from xml_formats.PageXml import PageXml, MultiPageXml, PageXmlException 
from crf.Graph_MultiPageXml import Graph_MultiPageXml
from util.Polygon import Polygon


class DU_BAR_Convert:
    """
    Here we convert the human annotation into 2 kinds of annotations:
    - a semantic one: header, heading, page-number, resolution-marginalia, resolution-number, resolution-paragraph   (we ignore Marginalia because only 2 occurences)
    - a segmentation one: 2 complementary labels. We call them Heigh Ho. Could have been Yin Yang as well...
    
    These new annotations are store in @DU_sem and @DU_sgm
    """
    sXml_HumanAnnotation_Extension   = ".mpxml"
    sXml_MachineAnnotation_Extension = ".du_mpxml"
    
    sMetadata_Creator  = "TranskribusDU/usecases/BAR/DU_ConvertGTAnnotation.py"
    sMetadata_Comments = "Converted human annotation into semantic and segmentation annotation. See attributes @DU_sem and @DU_sgm."

    dNS = {"pc":PageXml.NS_PAGE_XML}
    sxpNode = ".//pc:TextRegion"

    #Name of attributes for semantic / segmentation /resolution number
    sSemAttr = "DU_sem"
    sSgmAttr = "DU_sgm"
    sNumAttr = "DU_num"
    
    #Mapping to new semantic annotation
    dAnnotMapping = {"header"       :"header",
                     "heading"      :"heading",
                     "page-number"  :"page-number",
                     "marginalia"   : None,
                     "p"            :"resolution-paragraph",
                     "m"            :"resolution-marginalia",
                     ""             :"resolution-number"
                     }
    creResolutionHumanLabel = re.compile("([mp]?)([0-9]+.?)")  #e.g. p1 m23 456 456a 
    
    #The two complementary segmentation labels
    sSegmHeigh = "heigh"
    sSegmHo    = "ho"
    
    #=== CONFIGURATION ====================================================================
    def __init__(self): 

        random.seed()
        pass
    
        
    def convertDoc(self, sFilename):
        
        assert sFilename.endswith(self.sXml_HumanAnnotation_Extension)
        
        g = Graph_MultiPageXml()
        
        doc = libxml2.parseFile(sFilename)

        #the Heigh/Ho annotation runs over consecutive pages, so we keep those values accross pages
        self.prevResolutionNumber, self.prevSgmLbl = None, None
        self.lSeenResoNum = list()

        for pnum, page, domNdPage in g._iter_Page_DomNode(doc):
            self._convertPageAnnotation(pnum, page, domNdPage)
            
        MultiPageXml.setMetadata(doc, None, self.sMetadata_Creator, self.sMetadata_Comments)
        
        assert sFilename.endswith(self.sXml_HumanAnnotation_Extension)
        
        sDUFilename = sFilename[:-len(self.sXml_HumanAnnotation_Extension)] + self.sXml_MachineAnnotation_Extension
        doc.saveFormatFileEnc(sDUFilename, "utf-8", True)  #True to indent the XML
        doc.freeDoc()     
        
        return sDUFilename
           
    # -----------------------------------------------------------------------------------------------------------
    
    def _getNextSegmentationLabel(self, sPrevSegmLabel=None):
        """
        alternate beween HEIGH and HO, 1st at random 
        """
        if   sPrevSegmLabel == self.sSegmHeigh: return self.sSegmHo
        elif sPrevSegmLabel == self.sSegmHo:    return self.sSegmHeigh
        else:
            assert sPrevSegmLabel == None
            if random.random() < 0.5:   #Return the next random floating point number in the range [0.0, 1.0).
                return self.sSegmHeigh
            else:
                return self.sSegmHo

    def _iter_TextRegionNodeTop2Bottom(self, domNdPage, page):
        """
        Get the DOM, the DOM page node, the page object

        iterator on the DOM, that returns nodes
        """    
        #--- XPATH contexts
        ctxt = domNdPage.doc.xpathNewContext()
        for ns, nsurl in self.dNS.items(): ctxt.xpathRegisterNs(ns, nsurl)

        assert self.sxpNode, "CONFIG ERROR: need an xpath expression to enumerate elements corresponding to graph nodes"
        ctxt.setContextNode(domNdPage)
        lNdBlock = ctxt.xpathEval(self.sxpNode) #all relevant nodes of the page

        #order blocks from top to bottom of page
        lOrderedNdBlock = list()
        for ndBlock in lNdBlock:
            
            lXY = PageXml.getPointList(ndBlock)  #the polygon
            if lXY == []:
                raise ValueError("Node %x has invalid coordinates" % str(ndBlock))
            
            plg = Polygon(lXY)
            _, (xg, yg) = plg.getArea_and_CenterOfMass()
            
            lOrderedNdBlock.append( (yg, ndBlock))  #we want to order from top to bottom, so that TextRegions of different resolution are not interleaved
            
        lOrderedNdBlock.sort()
        
        for _, ndBlock in lOrderedNdBlock: yield ndBlock
            
        ctxt.xpathFreeContext()       
        
        raise StopIteration()        


    def _convertPageAnnotation(self, pnum, page, domNdPage):
        """
        
        """
        for nd in self._iter_TextRegionNodeTop2Bottom(domNdPage, page):
            
            try:
                lbl = PageXml.getCustomAttr(nd, "structure", "type")
            except PageXmlException:
                continue    #this node has no annotation whatsoever
            
            if lbl in ["heading", "header", "page-number", "marginalia"]:
                semLabel = lbl
                sgmLabel = None #those elements are not part of a resolution
                sResoNum = None
            else:
                o = self.creResolutionHumanLabel.match(lbl)
                if not o: raise ValueError("%s is not a valid human annotation" % lbl)
                semLabel = o.group(1)   #"" for the resolution number
                
                #now decide on the segmentation label
                sResoNum = o.group(2)
                if not sResoNum: raise ValueError("%s is not a valid human annotation - missing resolution number" % lbl)
                
                #now switch between heigh and ho !! :))
                if self.prevResolutionNumber == sResoNum:
                    sgmLabel = self.prevSgmLbl
                else:
                    sgmLabel = self._getNextSegmentationLabel(self.prevSgmLbl) 
                    assert sResoNum not in self.lSeenResoNum, "ERROR: the ordering of the block has not preserved resolution number contiguity"
                    self.lSeenResoNum.append(sResoNum)
                        
                self.prevResolutionNumber, self.prevSgmLbl = sResoNum, sgmLabel
                

            #always have a semantic label                
            sNewSemLbl = self.dAnnotMapping[semLabel]
            nd.setProp(self.sSemAttr, sNewSemLbl)       #DU annotation
            
            #resolution parts also have a segmentation label and a resolution number
            if sgmLabel: 
                nd.setProp(self.sSgmAttr, sgmLabel)     #DU annotation
                nd.setProp(self.sNumAttr, sResoNum)
            else:
                assert not sResoNum

    
def test_RE():
    cre = DU_BAR_Convert.creResolutionHumanLabel
    
    o = cre.match("m103a")
    assert  o.group(1) == 'm'
    assert  o.group(2) == '103a'

    o = cre.match("103a")
    assert  o.group(1) == ''
    assert  o.group(2) == '103a'

    o = cre.match("103")
    assert  o.group(1) == ''
    assert  o.group(2) == '103'
    
    o = cre.match("az103a")
    assert o == None
              
    
if __name__ == "__main__":
    from optparse import OptionParser
    
    #prepare for the parsing of the command line
    parser = OptionParser(usage="BAR annotation conversion", version="1.0")
    
#     parser.add_option("--tst", dest='lTst',  action="append", type="string"
#                       , help="Test a model using the given annotated collection.")    
#     parser.add_option("--fold-init", dest='iFoldInitNum',  action="store", type="int"
#                       , help="Initialize the file lists for parallel cross-validating a model on the given annotated collection. Indicate the number of folds.")    
#     parser.add_option("--jgjhg", dest='bFoldFinish',  action="store_true"
#                       , help="Evaluate by cross-validation a model on the given annotated collection.")    
#     parser.add_option("-w", "--warm", dest='warm',  action="store_true"
#                       , help="To make warm-startable model and warm-start if a model exist already.")   

    #parse the command line
    (options, args) = parser.parse_args()
    
    # --- 
    doer = DU_BAR_Convert()
    for sFilename in args:
        print "- Processing %s" % sFilename
        sOutputFilename = doer.convertDoc(sFilename)
        print "   done --> %s" %  sOutputFilename
        
    print "DONE."
        
