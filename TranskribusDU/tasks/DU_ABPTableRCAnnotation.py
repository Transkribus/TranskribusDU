# -*- coding: utf-8 -*-

"""
    Annotate textlines  for Table understanding (finding rows and columns)
    
    It tags the table header, vs data, vs other stuff.
    
    It ignore the binding cells (hack: rowspan >= 5 means binding...)
    It then reads the cell borders, and does a linear interpolation by row to produce 
    the horizontal graphical lines of the table.
    It adds a TableHLine elements in an element TableGraphicalLine of TableRegion.
    
    Copyright Naver Labs Europe 2017 
    H. Déjean
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

from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals
import sys, os
import collections
from lxml import etree

import numpy as np

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from xml_formats.PageXml import MultiPageXml 
from util.Polygon import Polygon
from common.trace import traceln

lLabelsBIEOS_R  = ['B', 'I', 'E', 'S', 'O']  #O?
lLabelsSM_C     = ['M', 'S', 'O']   # single cell, multicells
#lLabels_OI      = ['O','I']   # inside/outside a table           
#lLabels_SPAN    = ['rspan','cspan','nospan']
lLabels_HEADER  = ['D','CH', 'O']


sDURow    = "DU_row"
sDUCol    = 'DU_col'
sDUHeader = 'DU_header'

class TableAnnotationException(Exception):
    pass


def tag_DU_row_col_header(lCells, maxRowSpan):
    """
    Tag the XML nodes corresponding to those cells
    Modify the XML DOM
    """
    for cell in lCells:
    
        lText = MultiPageXml.getChildByName(cell,'TextLine')
         
        # HEADER WISE: D CH O
        if int(cell.get('row')) < maxRowSpan:
            [x.set(sDUHeader,lLabels_HEADER[1]) for x in lText]
        else:
            [x.set(sDUHeader,lLabels_HEADER[0]) for x in lText]
        
        # ROW WISE: B I E S O
        if len(lText) == 0:
            pass
        if len(lText) == 1:
            lText[0].set(sDURow,lLabelsBIEOS_R[3])
        elif len(lText) > 1:
    #         lText.sort(key=lambda x:float(x.prop('y')))
            lText[0].set(sDURow,lLabelsBIEOS_R[0])
            [x.set(sDURow,lLabelsBIEOS_R[1]) for x in lText[1:-1]]
            lText[-1].set(sDURow,lLabelsBIEOS_R[2])
    #         MultiPageXml.setCustomAttr(lText[0],"table","rtype",lLabelsBIEOS_R[0])
    #         MultiPageXml.setCustomAttr(lText[-1],"table","rtype",lLabelsBIEOS_R[2])
    #         [MultiPageXml.setCustomAttr(x,"table","rtype",lLabelsBIEOS_R[1]) for x in lText[1:-1]]    
        
        #COLUM WISE: M S O 
        lCoords = cell.xpath("./a:%s" % ("Coords"),namespaces={"a":MultiPageXml.NS_PAGE_XML})       
        coord= lCoords[0]
        sPoints=coord.get('points')
        plgn = Polygon.parsePoints(sPoints)
        (cx,cy,cx2,cy2) = plgn.getBoundingBox()     
        
        for txt in lText:
            lCoords = txt.xpath("./a:%s" % ("Coords"),namespaces={"a":MultiPageXml.NS_PAGE_XML})       
            coord= lCoords[0]
            sPoints=coord.get('points')
            lsPair = sPoints.split(' ')
            lXY = list()
            for sPair in lsPair:
                try:
                    (sx,sy) = sPair.split(',')
                    lXY.append( (int(sx), int(sy)) )
                except ValueError:
                    traceln("WARNING: invalid coord in TextLine id=%s  IGNORED"%txt.get("id"))
            ## HOW to define a CM element!!!!
            if lXY:
                (x1,y1,x2,y2) = Polygon(lXY).getBoundingBox()
                if x2> cx2 and (x2 - cx2) > 0.75 * (cx2 - x1):
                    txt.set(sDUCol,lLabelsSM_C[0])
                else:
                    txt.set(sDUCol,lLabelsSM_C[1])
            else:
                txt.set(sDUCol,lLabelsSM_C[-1])
                
    # textline outside table
    lRegions= MultiPageXml.getChildByName(root,'TextRegion')
    for region in lRegions:
        lText =  MultiPageXml.getChildByName(region,'TextLine')
        [x.set(sDURow,lLabelsBIEOS_R[-1]) for x in lText]
        [x.set(sDUCol,lLabelsSM_C[-1]) for x in lText]
        [x.set(sDUHeader,lLabels_HEADER[-1]) for x in lText]
        
    return

def addSeparator(root, lCells):
    """
    Add separator that correspond to cell boundaries
    modify the XML DOM
    """
    # let's collect the segment forming the separators
    dRowSep_lSgmt = collections.defaultdict(list)
    dColSep_lSgmt = collections.defaultdict(list)
    for cell in lCells:
        row, col, rowSpan, colSpan = [int(cell.get(sProp)) for sProp \
                                      in ["row", "col", "rowSpan", "colSpan"] ]
        coord = cell.xpath("./a:%s" % ("Coords"),namespaces={"a":MultiPageXml.NS_PAGE_XML})[0]
        sPoints = coord.get('points')
        plgn = Polygon.parsePoints(sPoints)
        try:
            lT, lR, lB, lL = plgn.partitionSegmentTopRightBottomLeft()
        except ZeroDivisionError:
            traceln("ERROR: cell %s row=%d col=%d has empty area and is IGNORED"
                    % (cell.get("id"), row, col))
            continue
        #now the top segments contribute to row separator of index: row
        dRowSep_lSgmt[row].extend(lT)
        #now the bottom segments contribute to row separator of index: row+rowSpan
        dRowSep_lSgmt[row+rowSpan].extend(lB)
        
        dColSep_lSgmt[col].extend(lL)
        dColSep_lSgmt[col+colSpan].extend(lR)
        
    #now make linear regression to draw relevant separators
    def getX(lSegment):
        lX = list()
        for x1,y1,x2,y2 in lSegment:
            lX.append(x1)
            lX.append(x2)
        return lX

    def getY(lSegment):
        lY = list()
        for x1,y1,x2,y2 in lSegment:
            lY.append(y1)
            lY.append(y2)
        return lY

    try:
        ndTR = MultiPageXml.getChildByName(root,'TableRegion')[0]
    except IndexError:
        raise TableAnnotationException("No TableRegion!!! ")

    lB = []
    for row, lSegment in dRowSep_lSgmt.items():
        X = getX(lSegment)
        Y = getY(lSegment)
        #sum(l,())
        lfNorm = [np.linalg.norm([[x1,y1], [x2,y2]]) for x1,y1,x2,y2 in lSegment]
        #duplicate each element 
        W = [fN for fN in lfNorm for _ in (0,1)]

        a, b = np.polynomial.polynomial.polyfit(X, Y, 1, w=W)

        xmin, xmax = min(X), max(X)
        y1 = a + b * xmin
        y2 = a + b * xmax
        lB.append(b*100)
        
        ndSep = MultiPageXml.createPageXmlNode("SeparatorRegion")
        ndSep.set("orient", "horizontal %.1f %.3f" % (a,b))
        ndSep.set("row", "%d" % row)
        ndTR.append(ndSep)
        ndCoord = MultiPageXml.createPageXmlNode("Coords")
        MultiPageXml.setPoints(ndCoord, [(xmin, y1), (xmax, y2)])
        ndSep.append(ndCoord)
    
    sStat = "\tHORIZONTAL: Average=%.1f%%  stdev=%.2f%%  min=%.1f%% max=%.1f%%" % (
        np.average(lB), np.std(lB), min(lB), max(lB)
        )
    ndTR.append(etree.Comment(sStat))
    print(sStat)
    
    lB = []
    for col, lSegment in dColSep_lSgmt.items():
        X = getX(lSegment)
        Y = getY(lSegment)
        #sum(l,())
        lfNorm = [np.linalg.norm([[x1,y1], [x2,y2]]) for x1,y1,x2,y2 in lSegment]
        #duplicate each element 
        W = [fN for fN in lfNorm for _ in (0,1)]

        # a * x + b
        a, b = np.polynomial.polynomial.polyfit(Y, X, 1, w=W)
        lB.append(b*100)

        ymin, ymax = min(Y), max(Y)
        x1 = a + b * ymin
        x2 = a + b * ymax 
        ndSep = MultiPageXml.createPageXmlNode("SeparatorRegion")
        ndSep.set("orient", "vertical %.1f %.3f" % (a,b))
        ndSep.set("col", "%d" % col)
        ndTR.append(ndSep)
        ndCoord = MultiPageXml.createPageXmlNode("Coords")
        MultiPageXml.setPoints(ndCoord, [(x1, ymin), (x2, ymax)])
        ndSep.append(ndCoord)
    sStat = "\tVERTICAL  : Average=%.1f%%  stdev=%.2f%%  min=%.1f%% max=%.1f%%" % (
        np.average(lB), np.std(lB), min(lB), max(lB)
        )
    ndTR.append(etree.Comment(sStat))
    print(sStat)
        
    return

def computeMaxRowSpan(lCells):
    """
        compute maxRowSpan for Row 0
        ignore cells for which rowspan = #row
    """
    nbRows = max(int(x.get('row')) for x in lCells)
    try: 
        return max(int(x.get('rowSpan')) for x in filter(lambda x: x.get('row') == "0" and x.get('rowSpan') != str(nbRows+1), lCells))
    except ValueError :return 1
    
# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        if len(sys.argv) == 3: 
            # COMPATIBILITY MODE
            #load mpxml 
            sFilename = sys.argv[1]
            sOutFilename = sys.argv[2]
            lsFilename = [sFilename]
            lsOutFilename = [sOutFilename]
        else:
            #we expect a folder 
            sInput = sys.argv[1]
            if os.path.isdir(sInput):
                lsFilename = [os.path.join(sInput, "col", s) for s in os.listdir(os.path.join(sInput, "col")) if s.endswith(".pxml") and s[-7] in "0123456789"]
                lsFilename.sort()
                lsOutFilename = [ os.path.dirname(s) + os.sep + "b_" + os.path.basename(s) for s in lsFilename]
            else:
                print("%s is not a folder"%sys.argv[1])
                raise IndexError()
    except IndexError:
        print("Usage: %s ( input-file output-file | folder )" % sys.argv[0])
        exit(1)
            
    print(lsFilename)
    print("%d files to be processed" % len(lsFilename))
    print(lsOutFilename)

    #for the pretty printer to format better...
    parser = etree.XMLParser(remove_blank_text=True)
    for sFilename, sOutFilename in zip(lsFilename, lsOutFilename):
        doc = etree.parse(sFilename, parser)
        root = doc.getroot()
        lPages = MultiPageXml.getChildByName(root,'Page')
        for page in lPages:

            lCells= MultiPageXml.getChildByName(page,'TableCell')
            if not lCells:
                traceln("ERROR: no TableCell - SKIPPING THIS FILE!!!")
                continue
            
            # default: O for all cells: all cells must have all tags!
            for cell in lCells:
                lText = MultiPageXml.getChildByName(cell,'TextLine')
                [x.set(sDURow,lLabelsBIEOS_R[-1]) for x in lText]
                [x.set(sDUCol,lLabelsSM_C[-1]) for x in lText]
                [x.set(sDUHeader,lLabels_HEADER[-1]) for x in lText]
                
            
            if False:
                # Oct' 2018 RV and JL decided that we keep the binding TextLine (if any!)
                # ignore "binding" cells
                # dirty...
                # lCells = list(filter(lambda x: int(x.get('rowSpan')) < 5, lCells))
                # less dirty
                maxrow = max(int(x.get('row')) for x in lCells)
                binding_rowspan = max(5, maxrow * 0.8) 
                traceln(" - max row = %d  => considering rowspan > %d as binding cells"
                        % (maxrow, binding_rowspan))
                lValidCell, lBindingCell = [], []
                for ndCell in lCells:
                    if int(ndCell.get('rowSpan')) < binding_rowspan:
                        lValidCell.append(ndCell)
                    else:
                        lBindingCell.append(ndCell)
                nDiscarded = len(lBindingCell)
                if nDiscarded > 1: traceln("****************   WARNING  ****************")
                traceln(" - %d cells discarded as binding cells" % nDiscarded)
                for ndCell in lBindingCell:
                    ndCell.set("type", "table-binding")
                lCells = lValidCell
                
            # FOR COLUMN HEADER: get max(cell[0,i].span)
    #         try:
    #             maxRowSpan = max(int(x.get('rowSpan')) for x in filter(lambda x: x.get('row') == "0", lCells))
    #         except ValueError:
    #             maxRowSpan = 1
            maxRowSpan = computeMaxRowSpan(lCells)
            tag_DU_row_col_header(lCells, maxRowSpan)
            
            try:
                addSeparator(page, lCells)
                doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
#                 print('annotation done for %s  --> %s' % (sFilename, sOutFilename))
            except TableAnnotationException:
                traceln("No Table region in file ", sFilename, "  IGNORED!!")
                sys.exit(1)
        print('annotation done for %s  --> %s' % (sFilename, sOutFilename))
        del doc





