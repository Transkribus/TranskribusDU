# -*- coding: utf-8 -*-

"""
    Annotate textlines  for Table understanding (finding rows and columns)
    
    Copyright Naver Labs Europe 2017 
    H. Déjean

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
from lxml import etree

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from xml_formats.PageXml import MultiPageXml 
from util.Polygon import Polygon

lLabelsBIEOS_R  = ['B', 'I', 'E', 'S', 'O']  #O?
lLabelsSM_C     = ['M', 'S', 'O']   # single cell, multicells
#lLabels_OI      = ['O','I']   # inside/outside a table           
#lLabels_SPAN    = ['rspan','cspan','nospan']
lLabels_HEADER  = ['D','CH', 'O']

#load mpxml 
sFilename = sys.argv[1]
doc = etree.parse(sFilename)
root=doc.getroot()

sDURow= "DU_row"
sDUCol= 'DU_col'
sDUHeader = 'DU_header'

# textline outside table
lRegions= MultiPageXml.getChildByName(root,'TextRegion')
for region in lRegions:
    lText =  MultiPageXml.getChildByName(region,'TextLine')
    [x.set(sDURow,lLabelsBIEOS_R[-1]) for x in lText]
    [x.set(sDUCol,lLabelsSM_C[-1]) for x in lText]
    [x.set(sDUHeader,lLabels_HEADER[-1]) for x in lText]

lCells= MultiPageXml.getChildByName(root,'TableCell')

# ignore "binding" cells
# dirty...
lCells = list(filter(lambda x: int(x.get('rowSpan')) < 5, lCells))
# FOR COLUMN HEADER: get max(cell[0,i].span)
maxRowSpan = max(int(x.get('rowSpan')) for x in filter(lambda x: x.get('row') == "0", lCells))

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
    lsPair = sPoints.split(' ')
    lXY = list()
    for sPair in lsPair:
        (sx,sy) = sPair.split(',')
        lXY.append( (int(sx), int(sy)) )
    (cx,cy,cx2,cy2) = Polygon(lXY).getBoundingBox()     
     
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
 
doc.write(sys.argv[2],encoding='utf-8',pretty_print=True,xml_declaration=True)
print('annotation done for %s'%sys.argv[1])





