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




import sys, os, math
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

from tasks.DU_Table_CellBorder import getCellsSeparators

lLabelsBIESO_R  = ['B', 'I', 'E', 'S', 'O']  #O?
lLabelsSM_C     = ['M', 'S', 'O']   # single cell, multicells
#lLabels_OI      = ['O','I']   # inside/outside a table           
#lLabels_SPAN    = ['rspan','cspan','nospan']
lLabels_HEADER  = ['D','CH', 'O']


sDURow    = "DU_row"
sDUCol    = 'DU_col'
sDUHeader = 'DU_header'

class TableAnnotationException(Exception):
    pass


def tag_DU_row_col_header(root, lCells, maxRowSpan):
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
            lText[0].set(sDURow,lLabelsBIESO_R[3])
        elif len(lText) > 1:
    #         lText.sort(key=lambda x:float(x.prop('y')))
            lText[0].set(sDURow,lLabelsBIESO_R[0])
            [x.set(sDURow,lLabelsBIESO_R[1]) for x in lText[1:-1]]
            lText[-1].set(sDURow,lLabelsBIESO_R[2])
    #         MultiPageXml.setCustomAttr(lText[0],"table","rtype",lLabelsBIESO_R[0])
    #         MultiPageXml.setCustomAttr(lText[-1],"table","rtype",lLabelsBIESO_R[2])
    #         [MultiPageXml.setCustomAttr(x,"table","rtype",lLabelsBIESO_R[1]) for x in lText[1:-1]]    
        
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
        [x.set(sDURow,lLabelsBIESO_R[-1]) for x in lText]
        [x.set(sDUCol,lLabelsSM_C[-1]) for x in lText]
        [x.set(sDUHeader,lLabels_HEADER[-1]) for x in lText]
        
    return

def removeSeparator(root):
    lnd = MultiPageXml.getChildByName(root, 'SeparatorRegion')
    n = len(lnd)
    for nd in lnd:
        nd.getparent().remove(nd)
    return n  
        
def addSeparator(root, lCells):
    """
    Add separator that correspond to cell boundaries
    modify the XML DOM
    """
    dRow, dCol = getCellsSeparators(lCells)

    try:
        ndTR = MultiPageXml.getChildByName(root,'TableRegion')[0]
    except IndexError:
        raise TableAnnotationException("No TableRegion!!! ")

    lRow = sorted(dRow.keys())
    lB = []
    for row in lRow:
        (x1, y1), (x2, y2) = dRow[row]
        b = math.degrees(math.atan((y2-y1) / (x2-x1)))
        lB.append(b)
        
        ndSep = MultiPageXml.createPageXmlNode("SeparatorRegion")
        ndSep.set("orient", "horizontal angle=%.2f" % b)
        ndSep.set("row", "%d" % row)
        ndTR.append(ndSep)
        ndCoord = MultiPageXml.createPageXmlNode("Coords")
        MultiPageXml.setPoints(ndCoord, [(x1, y1), (x2, y2)])
        ndSep.append(ndCoord)
        sStat = "\tHORIZONTAL: Average=%.1f°  stdev=%.2f°  min=%.1f° max=%.1f°" % (
        np.average(lB), np.std(lB), min(lB), max(lB)
        )
    ndTR.append(etree.Comment(sStat))
    traceln(sStat)
    
    lCol = sorted(dCol.keys())
    lB = []
    for col in lCol:
        (x1, y1), (x2, y2) = dCol[col]
        b = 90  -math.degrees(math.atan((x2-x1) / (y2 - y1)))
        lB.append(b)
        ndSep = MultiPageXml.createPageXmlNode("SeparatorRegion")
        ndSep.set("orient", "vertical %.2f" % b)
        ndSep.set("col", "%d" % col)
        ndTR.append(ndSep)
        ndCoord = MultiPageXml.createPageXmlNode("Coords")
        MultiPageXml.setPoints(ndCoord, [(x1, y1), (x2, y2)])
        ndSep.append(ndCoord)
    sStat = "\tVERTICAL  : Average=%.1f°  stdev=%.2f°  min=%.1f° max=%.1f°" % (
        np.average(lB), np.std(lB), min(lB), max(lB)
        )
    ndTR.append(etree.Comment(sStat))
    traceln(sStat)

    return


def computeMaxRowSpan(lCells):
    """
        compute maxRowSpan for Row 0
        ignore cells for which rowspan = #row
    """
    nbRows = max(int(x.get('row')) for x in lCells)
    try: 
        return max(int(x.get('rowSpan')) for x in filter(lambda x: x.get('row') == "0" and x.get('rowSpan') != str(nbRows+1), lCells))
    except ValueError :
        return 1
    
# ------------------------------------------------------------------
def main(lsFilename, lsOutFilename):
    #for the pretty printer to format better...
    parser = etree.XMLParser(remove_blank_text=True)
    for sFilename, sOutFilename in zip(lsFilename, lsOutFilename):
        doc = etree.parse(sFilename, parser)
        root = doc.getroot()

        lCells= MultiPageXml.getChildByName(root,'TableCell')
        if not lCells:
            traceln("ERROR: no TableCell - SKIPPING THIS FILE!!!")
            continue
        
        # default: O for all cells: all cells must have all tags!
        for cell in lCells:
            lText = MultiPageXml.getChildByName(cell,'TextLine')
            [x.set(sDURow,lLabelsBIESO_R[-1]) for x in lText]
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
        maxRowSpan = computeMaxRowSpan(lCells)
        
        tag_DU_row_col_header(root, lCells, maxRowSpan)
            
        try:
            removeSeparator(root)
            addSeparator(root, lCells)
            doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
            traceln('annotation done for %s  --> %s' % (sFilename, sOutFilename))
        except TableAnnotationException:
            traceln("No Table region in file ", sFilename, "  IGNORED!!")
        
        del doc


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
                lsFilename = [os.path.join(sInput, "col", s) for s in os.listdir(os.path.join(sInput, "col")) if s.endswith(".mpxml") ]
                if not lsFilename:
                    lsFilename = [os.path.join(sInput, "col", s) for s in os.listdir(os.path.join(sInput, "col")) if s.endswith(".pxml") ]
                lsFilename.sort()
                lsOutFilename = [ os.path.dirname(s) + os.sep + "c_" + os.path.basename(s) for s in lsFilename]
            else:
                traceln("%s is not a folder"%sys.argv[1])
                raise IndexError()
    except IndexError:
        traceln("Usage: %s ( input-file output-file | folder )" % sys.argv[0])
        exit(1)
            
    traceln(lsFilename)
    traceln("%d files to be processed" % len(lsFilename))
    traceln(lsOutFilename)

    main(lsFilename, lsOutFilename)


