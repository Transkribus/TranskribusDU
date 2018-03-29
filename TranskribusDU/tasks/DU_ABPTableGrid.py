# -*- coding: utf-8 -*-

"""
    Map a grid to teh annotated table lines
    
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


# class GridAnnotator:
#     
#     def __init__(self, root, iGridHorizStep, iGridVertiStep):
#         self.root = root
#         self.iGridHorizStep = iGridHorizStep
#         self.iGridVertiStep = iGridVertiStep

def add_grid(root, iGridHorizStep, iGridVertiStep, ltlHlV=None):
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
        xmin = 0
        xmax = (w // iGridHorizStep) * iGridHorizStep
        ymin = 0
        ymax = (h // iGridVertiStep) * iGridVertiStep
        
        ndTR = MultiPageXml.getChildByName(root,'TableRegion')[0]
    
        #Vertical grid lines 
        i, x = 0, 0
        while x <= xmax:
            ndSep = MultiPageXml.createPageXmlNode("GridSeparator")
            if i in lVi: ndSep.set("type", "1")
            ndTR.append(ndSep)
            ndCoord = MultiPageXml.createPageXmlNode("Coords")
            MultiPageXml.setPoints(ndCoord, [(x, ymax), (x, ymin)])
            ndSep.append(ndCoord)
            i += 1
            x += iGridHorizStep
        assert x == (xmax + iGridHorizStep)
        #Horizontal grid lines 
        i, y = 0, 0
        while y <= ymax:
            ndSep = MultiPageXml.createPageXmlNode("GridSeparator")
            if i in lHi: ndSep.set("type", "1")
            ndTR.append(ndSep)
            ndCoord = MultiPageXml.createPageXmlNode("Coords")
            MultiPageXml.setPoints(ndCoord, [(xmin, y), (xmax, y)])
            ndSep.append(ndCoord)
            i += 1
            y += iGridVertiStep
        assert y == (ymax + iGridVertiStep)
    return

def get_grid_index(root, iGridHorizStep, iGridVertiStep):
    """
    get the index in our grid of the table lines
    return lists of index, for horizontal and for vertical grid lines, per page
    return [(h_list, v_list), ...]
    """
    ltlHlV = []
    for ndPage in MultiPageXml.getChildByName(root, 'Page'):
        lHi, lVi = [], []

        ndTR = MultiPageXml.getChildByName(ndPage,'TableRegion')[0]
        
        #enumerate the table separators
        for ndSep in MultiPageXml.getChildByName(ndTR,'SeparatorRegion'):
            sPoints=MultiPageXml.getChildByName(ndSep,'Coords')[0].get('points')
            [(x1,y1),(x2,y2)] = Polygon.parsePoints(sPoints).lXY
            if abs(x2-x1) > abs(y2-y1):
                #horizontal table line
                ym = (y1+y2)/2.0
                i = int(round(ym / iGridVertiStep, 0)) 
                lHi.append(i)
            else:
                xm = (x1+x2)/2.0
                i = int(round(xm / iGridHorizStep, 0)) 
                lVi.append(i)
        ltlHlV.append( (lHi, lVi) )
    return ltlHlV
                
# ------------------------------------------------------------------
#load mpxml 
sFilename = sys.argv[1]
sOutFilename = sys.argv[2]

iGridStep_H = 33  #odd number is better
iGridStep_V = 33  #odd number is better

#for the pretty printer to format better...
parser = etree.XMLParser(remove_blank_text=True)
doc = etree.parse(sFilename, parser)
root=doc.getroot()

lSep= MultiPageXml.getChildByName(root,'SeparatorRegion')

ltlHlV = get_grid_index(root, iGridStep_H, iGridStep_V)

print(ltlHlV)
add_grid(root, iGridStep_H, iGridStep_V, ltlHlV)

#tag_grid(root, lSep)

doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
print('annotation done for %s'%sys.argv[1])





