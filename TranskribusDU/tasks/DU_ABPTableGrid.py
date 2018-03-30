# -*- coding: utf-8 -*-

"""
    Map a grid to the annotated table separators
    
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


class GridAnnotator:
    """
    mapping the table separators to a regular grid
    """
    iGRID_STEP = 33 # odd number is better
    
    def __init__(self, iGridHorizStep=iGRID_STEP, iGridVertiStep=iGRID_STEP):
        self.iGridHorizStep = iGridHorizStep
        self.iGridVertiStep = iGridVertiStep

    def iterGridHorizontalLines(self, iPageWidth, iPageHeight):
        """
        Coord of the horizontal lines of the grid, given the page size (pixels)
        Return iterator on the (x1,y1, x2,y2)
        """
        xmin, ymin, xmax, ymax = self.getGridBB(iPageWidth, iPageHeight)
        for y in range(ymin, ymax+1, self.iGridVertiStep):
            yield (xmin, y, xmax, y)
                
    def iterGridVerticalLines(self, iPageWidth, iPageHeight):
        """
        Coord of the vertical lines of the grid, given the page size (pixels)
        Return iterator on the (x1,y1, x2,y2)
        """
        xmin, ymin, xmax, ymax = self.getGridBB(iPageWidth, iPageHeight)
        for x in range(xmin, xmax+1, self.iGridHorizStep):
            yield (x, ymin, x, ymax)
                
    def getGridBB(self, iPageWidth, iPageHeight):
            xmin = 0
            xmax = (iPageWidth // self.iGridHorizStep) * self.iGridHorizStep
            ymin = 0
            ymax = (iPageHeight // self.iGridVertiStep) * self.iGridVertiStep
            return xmin, ymin, xmax, ymax
                    
    def get_grid_GT_index_from_DOM(self, root, fMinPageCoverage):
        """
        get the index in our grid of the table lines
        return lists of index, for horizontal and for vertical grid lines, per page
        return [(h_list, v_list), ...]
        """
        ltlHlV = []
        for ndPage in MultiPageXml.getChildByName(root, 'Page'):
            w, h = int(ndPage.get("imageWidth")), int(ndPage.get("imageHeight"))

            lHi, lVi = [], []
    
            ndTR = MultiPageXml.getChildByName(ndPage,'TableRegion')[0]
            
            #enumerate the table separators
            for ndSep in MultiPageXml.getChildByName(ndTR,'SeparatorRegion'):
                sPoints=MultiPageXml.getChildByName(ndSep,'Coords')[0].get('points')
                [(x1,y1),(x2,y2)] = Polygon.parsePoints(sPoints).lXY
                
                dx, dy = abs(x2-x1), abs(y2-y1)
                if dx > dy:
                    #horizontal table line
                    if dx > (fMinPageCoverage*w):
                        ym = (y1+y2)/2.0
                        i = int(round(ym / self.iGridVertiStep, 0)) 
                        lHi.append(i)
                else:
                    if dy > (fMinPageCoverage*h):
                        xm = (x1+x2)/2.0
                        i = int(round(xm / self.iGridHorizStep, 0)) 
                        lVi.append(i)
            ltlHlV.append( (lHi, lVi) )
        return ltlHlV

    def getLabel(self, i, lGTi):
        """
        given the minimum and maximum index of lines in GT grid lines
        produce the label of line of index i
        return the label
        """
        imin, imax = min(lGTi), max(lGTi) # could be optimized  
        if i < imin:
            return "O"              # Outside
        elif i == imin:
            return "B"              # Border
        elif imin < i and i < imax:
            if i in lGTi:
                return "S"          # Separator
            else:
                return "I"          # Ignore
        elif i == imax:
            return "B"              # Border
        else:
            return "O"              # Outside
        
    def add_grid_to_DOM(self, root, ltlHlV=None):
        """
        Add the grid lines to the DOM
        Tag them if ltlHlV is given
        Modify the XML DOM
        """
        for iPage, ndPage in enumerate(MultiPageXml.getChildByName(root, 'Page')):
            
            if ltlHlV is None:
                lHi, lVi = [], []
            else:
                lHi, lVi = ltlHlV[iPage]
    
            w, h = int(ndPage.get("imageWidth")), int(ndPage.get("imageHeight"))
            
            ndTR = MultiPageXml.getChildByName(root,'TableRegion')[0]
        
            def addPageXmlSeparator(nd, i, lGTi, x1, y1, x2, y2):
                ndSep = MultiPageXml.createPageXmlNode("GridSeparator")
                if lGTi:
                    # propagate the groundtruth info we have
                    sLabel = self.getLabel(i, lGTi)
                    ndSep.set("type", sLabel)
                if abs(x2-x1) > abs(y2-y1):
                    ndSep.set("orient", "0")
                else:
                    ndSep.set("orient", "90")
                    
                nd.append(ndSep)
                ndCoord = MultiPageXml.createPageXmlNode("Coords")
                MultiPageXml.setPoints(ndCoord, [(x1, y1), (x2, y2)])
                ndSep.append(ndCoord)
                return ndSep
            
            #Vertical grid lines 
            for i, (x1,y1,x2,y2) in enumerate(self.iterGridVerticalLines(w,h)):
                addPageXmlSeparator(ndTR, i, lVi, x1, y1, x2, y2)

            #horizontal grid lines 
            for i, (x1,y1,x2,y2) in enumerate(self.iterGridHorizontalLines(w,h)):
                addPageXmlSeparator(ndTR, i, lHi, x1, y1, x2, y2)
                
        return
    
                
# ------------------------------------------------------------------
if __name__ == "__main__":
    #load mpxml 
    sFilename = sys.argv[1]
    sOutFilename = sys.argv[2]
    
    iGridStep_H = 33  #odd number is better
    iGridStep_V = 33  #odd number is better
    
    fMinPageCoverage = 0.5  # minimum proportion of the page crossed by a grid line
                            # we want to ignore col- and row- spans
    
    #for the pretty printer to format better...
    parser = etree.XMLParser(remove_blank_text=True)
    doc = etree.parse(sFilename, parser)
    root=doc.getroot()
    
    doer = GridAnnotator(iGridStep_H, iGridStep_V)
        
    #map the groundtruth table separators to our grid
    ltlHlV = doer.get_grid_GT_index_from_DOM(root, fMinPageCoverage)
    
    #create DOM node reflecting the grid 
    # we add GridSeparator elements. Groundtruth ones have type="1"
    doer.add_grid_to_DOM(root, ltlHlV)
    
    #tag_grid(root, lSep)
    
    doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
    print('Annotated grid added to %s'%sys.argv[1])





