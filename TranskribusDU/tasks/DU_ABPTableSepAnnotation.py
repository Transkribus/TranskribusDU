# -*- coding: utf-8 -*-

"""
    Annotate Separators for Table understanding (table separators or not)
    
    Copyright Naver Labs Europe 2018 
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

from shapely.geometry import Polygon as ShapePo
from shapely.geometry import LineString
from shapely.prepared import prep
from shapely.affinity import scale
import numpy as np

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from xml_formats.PageXml import MultiPageXml 
from util.Polygon import Polygon

lLabels_OI      = ['O','I']   # inside/outside a table           


sDUSep = "DU_sep"

def tagSeparatorRegion(lPages):
    """
        tag separatorRegion
    """
    for page in lPages:
        lSeparators = MultiPageXml.getChildByName(page,'SeparatorRegion')
        lTables = MultiPageXml.getChildByName(page,'TableRegion')
        if lTables == []: 
            print ("no table for %s"%sys.argv[1])
            sys.exit(0)
        # default O
        [x.set(sDUSep,lLabels_OI[0]) for x in lSeparators]
        
        for table in lTables:
            lPolygonTables = [ShapePo(MultiPageXml.getPointList(x)) for x in lTables ]
            lPolygonSep = [LineString(MultiPageXml.getPointList(x)) for x in lSeparators ]
        
            for table in lPolygonTables:
                table_prep = prep(table)
                [lSeparators[i].set(sDUSep,lLabels_OI[1]) for i,x in enumerate(lPolygonSep) if table_prep.intersects(x)]
        
        ## fix bindings
        for table in lTables:
            lCells = MultiPageXml.getChildByName(table,'TableCell')
            lCells = list(filter(lambda x: int(x.get('rowSpan')) > 6, lCells))
            lPolygonCells = [ShapePo(MultiPageXml.getPointList(x)) for x in lCells ]
            for cell in lPolygonCells:
                cell_prep = prep(scale(cell,xfact=0.5))
                for i,x in enumerate(lPolygonSep):
                    if cell_prep.intersects(x) and (x.bounds[3]-x.bounds[1]) > (x.bounds[2]-x.bounds[0]) :
                        lSeparators[i].set(sDUSep,lLabels_OI[0])
        
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

lPages = MultiPageXml.getChildByName(root,'Page')

tagSeparatorRegion(lPages)

doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
print('annotation done for %s'%sys.argv[1])





