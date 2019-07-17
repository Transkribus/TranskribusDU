# -*- coding: utf-8 -*-

"""
    Check if the interpolated table separator are intersecting textlines
    
    Copyright Naver Labs Europe 2018
    JL Meunier

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
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""

import sys, os
from optparse import OptionParser

from lxml import etree
import shapely.geometry

from util.Shape import ShapeLoader

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from xml_formats.PageXml import MultiPageXml 

from common.TestReport import TestReport, TestReportConfusion

    
class SeparatorChecker:
    """
    checking if stable separators intersect TextLines
    """
    lsClassName = ["H", "V", "O"]
    
    def __init__(self):
        self.parser = etree.XMLParser(remove_blank_text=True)

    def check(self, sFilename, scale, bVerbose=False):
        """
        return Y_pred, YGT   (shape=(nb_node,) dtype=np.int)
        return a TestReport
        """         
        lY, lYGT = [], []
        
        #for the pretty printer to format better...
        assert os.path.isfile(sFilename), sFilename
        doc  = etree.parse(sFilename, self.parser)
        #doc  = etree.parse(sFilename)
        root = doc.getroot()
        

        #place each TextLine in the table rows and columns
        ndPage = MultiPageXml.getChildByName(root, 'Page')[0]
#         w, h = int(ndPage.get("imageWidth")), int(ndPage.get("imageHeight"))
        
        def storeNode(oShape, nd):
            oShape.duNode = nd

        if True:
            loTxt = ShapeLoader.children_to_LinearRing(ndPage, 'TextLine', storeNode)
        else:
            loTxt = ShapeLoader.children_to_LineString(ndPage, 'Baseline', storeNode)
            
        if not scale is None:
            scaled_loTxt = []
            for o in loTxt:
                scaled_o = shapely.affinity.scale(o, 1.0, scale)
                scaled_o.duNode = o.duNode
                scaled_loTxt.append(scaled_o)
            loTxt = scaled_loTxt
            
        if bVerbose: print("%d TextLines" % len(loTxt))
        loSep = ShapeLoader.children_to_LineString(ndPage, 'SeparatorRegion', storeNode)
        if bVerbose: print("%d SeparatorRegion" % len(loSep))
        
        if True:
            # brute-force code
            for oSep in loSep:
                if bVerbose: print("%35s %4s %4s %s" % (oSep
                                    , oSep.duNode.get("row"), oSep.duNode.get("col"), oSep.duNode.get("orient")))
                sInfo = oSep.duNode.get("orient")
                if sInfo.startswith("horizontal"):
                    YGT = 0
                elif sInfo.startswith("vertical"):
                    YGT = 1
                    continue
                else:
                    YGT = 2
                Y = None
                for oTxt in loTxt:
                    if oSep.crosses(oTxt):
                        Y = 2
                        nd = oTxt.duNode
                        sid = nd.get("id")
                        ndCell = nd.getparent()
                        if bVerbose: print("\tCrossing %s row=%s col=%s" %
                                           (sid, ndCell.get("row"), ndCell.get("col")))
                if Y is None: Y = YGT
                lY.append(Y)
                lYGT.append(YGT)           
        else:
            # Collection-based code
            moTxt = shapely.geometry.MultiLineString(loTxt)
            for oSep in loSep:
                print("%40s %4s %4s %s" % (oSep
                      , oSep.duNode.get("row"), oSep.duNode.get("col"), oSep.duNode.get("orient")))
                if oSep.crosses(moTxt):
                    print("NOO")
                lYGT.append("")
        oTstRpt = TestReport("SeparatorChecker", [lY] , [lYGT] , self.lsClassName, [sFilename])
        #return np.array(lY  , dtype=np.int), np.array(lYGT, dtype=np.int)
        fScore, sClassificationReport = oTstRpt.getClassificationReport()
        if fScore < 0.999: print("\t *** Accuracy score = %f" % fScore)
#         if fScore < 1: print(sFilename, sClassificationReport)
        return oTstRpt

# ------------------------------------------------------------------
def check(lsFilename, scale, bVerbose=False):

#     doer = SeparatorChecker()
#     lY, lYGT = [], []
#     for sFilename in lsFilename:
#         print(" - processing ", sFilename)
#         Y, YGT = doer.check(sFilename)
#         lY.append(Y)
#         lYGT.append(YGT)
#         print(TestReport("SeparatorChecker", [Y] , [YGT] , SeparatorChecker.lsClassName, [sFilename]))
#     oTstRpt = TestReport("SeparatorChecker", lY, lYGT, SeparatorChecker.lsClassName, lsFilename)
#     return oTstRpt        

    doer = SeparatorChecker()
    loTstRpt = []
    for sFilename in lsFilename:
        print(" - processing ", sFilename)
        oTstRpt = doer.check(sFilename, scale, bVerbose)
        loTstRpt.append(oTstRpt)
        if bVerbose: print(oTstRpt)
    oTstRpt = TestReportConfusion.newFromReportList("SeparatorChecker", loTstRpt)
    return oTstRpt        

    
# ------------------------------------------------------------------
if __name__ == "__main__":
    
    usage = "<file>+"
    parser = OptionParser(usage=usage, version="0.1")
    parser.add_option("--scale", dest='scale',  action="store"
                      , type=float
                      , help="scaling factor applied to TextLine") 
#     parser.add_option("--SIO"           , dest='bSIO'          ,  action="store_true", help="SIO labels") 

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    
    print("%d files to be checked: %s" % (len(args), args))
    
    oTstRpt = check(args, options.scale)
    print(oTstRpt)
        
    
    


