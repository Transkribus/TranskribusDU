# -*- coding: utf-8 -*-

"""
    Annotate textlines  for Table understanding (finding rows and columns)
    
    Additionnally, it creates the horizontal graphical lines of the table.
    (by adding TableHLine elements in an element TableGraphicalLine of TableRegion)
    
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

lLabelsBIEOS_R  = ['B', 'I', 'E', 'S', 'O']  #O?
# lLabelsBIEOS_R  = ['RB', 'RI', 'RE', 'RS', 'RO']  #O?
lLabelsSM_C     = ['M', 'S', 'O']   # single cell, multicells
#lLabels_OI      = ['O','I']   # inside/outside a table           
#lLabels_SPAN    = ['rspan','cspan','nospan']
lLabels_HEADER  = ['D','CH', 'O']


sDURow= "DU_row"
sDUCol= 'DU_col'
sDUHeader = 'DU_header'

def tag_DU_row_col_header(lCells, maxRowSpan):
    """
    Tag the XML nodes corresponding to those cells
    Modify the XML DOM
    """
    for cell in lCells:
    
        lText = MultiPageXml.getChildByName(cell,'TextLine')
         
    #     # header wise
        
        if int(cell.get('row')) < maxRowSpan:
            [x.set(sDUHeader,lLabels_HEADER[1]) for x in lText]
        else:
            [x.set(sDUHeader,lLabels_HEADER[0]) for x in lText]
        
        # ROW WISE
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
        
        
        
        #COLUM WISE
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
                (sx,sy) = sPair.split(',')
                lXY.append( (int(sx), int(sy)) )
            ## HOW to define a CM element!!!!
            (x1,y1,x2,y2) = Polygon(lXY).getBoundingBox()
            if x2> cx2 and (x2 - cx2) > 0.75 * (cx2 - x1):
                txt.set(sDUCol,lLabelsSM_C[0])
            else:
                txt.set(sDUCol,lLabelsSM_C[1])
                
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
            #now the top segments contribute to row separator of index: row
            dRowSep_lSgmt[row].extend(lT)
            #now the bottom segments contribute to row separator of index: row+rowSpan
            dRowSep_lSgmt[row+rowSpan].extend(lB)
            
            dColSep_lSgmt[col].extend(lL)
            dColSep_lSgmt[col+colSpan].extend(lR)
        except ValueError: pass
        
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

    ndTR = MultiPageXml.getChildByName(root,'TableRegion')[0]

    lB = []
    for irow, lSegment in dRowSep_lSgmt.items():
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
    for icol, lSegment in dColSep_lSgmt.items():
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
# ------------------------------------------------------------------
#load mpxml 
sFilename = sys.argv[1]
sOutFilename = sys.argv[2]

#for the pretty printer to format better...
parser = etree.XMLParser(remove_blank_text=True)
doc = etree.parse(sFilename, parser)
root=doc.getroot()

lCells= MultiPageXml.getChildByName(root,'TableCell')

# default: O for all cells: all cells must have all tags!
for cell in lCells:
    lText = MultiPageXml.getChildByName(cell,'TextLine')
    [x.set(sDURow,lLabelsBIEOS_R[-1]) for x in lText]
    [x.set(sDUCol,lLabelsSM_C[-1]) for x in lText]
    [x.set(sDUHeader,lLabels_HEADER[-1]) for x in lText]
# ignore "binding" cells
# dirty...
lCells = list(filter(lambda x: int(x.get('rowSpan')) < 5, lCells))

# FOR COLUMN HEADER: get max(cell[0,i].span)
maxRowSpan = max(int(x.get('rowSpan')) for x in filter(lambda x: x.get('row') == "0", lCells))

tag_DU_row_col_header(lCells, maxRowSpan)

addSeparator(root, lCells)

doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
print('annotation done for %s'%sys.argv[1])





