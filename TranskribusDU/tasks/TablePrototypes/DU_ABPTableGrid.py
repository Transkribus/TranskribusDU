# -*- coding: utf-8 -*-

"""
    Map a grid to the annotated table separators
    
    Copyright Naver Labs Europe 2018
    JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




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

    @classmethod
    def gridIndex(cls, x, step_grid, x_grid_start, x_grid_end = None):
        """
        closest grid line to the given value
        """
        assert x_grid_end is None or x <= x_grid_end, "value out of grid: %s" \
                    % ((step_grid, x_grid_start, x_grid_end),)
        return int(round( (x - x_grid_start) / float(step_grid), 0)) # py2...

    @classmethod
    def snapToGridIndex(cls, x, step_grid):
        """
        closest grid line to the given value
        return the grid line index
        """
        return int(round( x / float(step_grid), 0)) # py2...

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
    
            l = MultiPageXml.getChildByName(ndPage,'TableRegion')
            if l:
                assert len(l) == 1, "More than 1 TableRegion??"
                ndTR = l[0]
                
                #enumerate the table separators
                for ndSep in MultiPageXml.getChildByName(ndTR,'SeparatorRegion'):
                    sPoints=MultiPageXml.getChildByName(ndSep,'Coords')[0].get('points')
                    [(x1,y1),(x2,y2)] = Polygon.parsePoints(sPoints).lXY
                    
                    dx, dy = abs(x2-x1), abs(y2-y1)
                    if dx > dy:
                        #horizontal table line
                        if dx > (fMinPageCoverage*w):
                            ym = (y1+y2)/2.0   # 2.0 to support python2
                            #i = int(round(ym / self.iGridVertiStep, 0)) 
                            i = self.snapToGridIndex(ym, self.iGridVertiStep)
                            lHi.append(i)
                    else:
                        if dy > (fMinPageCoverage*h):
                            xm = (x1+x2)/2.0
                            #i = int(round(xm / self.iGridHorizStep, 0)) 
                            i = self.snapToGridIndex(xm, self.iGridHorizStep)
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
        return the number of grid lines created
        """
        domid = 0 #to add unique separator id and count them
        
        for iPage, ndPage in enumerate(MultiPageXml.getChildByName(root, 'Page')):
            try:
                lHi, lVi = ltlHlV[iPage]
            except IndexError:
                lHi, lVi = [], []
    
            w, h = int(ndPage.get("imageWidth")), int(ndPage.get("imageHeight"))
            
            ndTR = MultiPageXml.getChildByName(root,'TableRegion')[0]
        
            def addPageXmlSeparator(nd, i, lGTi, x1, y1, x2, y2, domid):
                ndSep = MultiPageXml.createPageXmlNode("GridSeparator")
                if lGTi:
                    # propagate the groundtruth info we have
                    sLabel = self.getLabel(i, lGTi)
                    ndSep.set("type", sLabel)
                if abs(x2-x1) > abs(y2-y1):
                    ndSep.set("orient", "0")
                else:
                    ndSep.set("orient", "90")
                ndSep.set("id", "s_%d"%domid)
                nd.append(ndSep)
                ndCoord = MultiPageXml.createPageXmlNode("Coords")
                MultiPageXml.setPoints(ndCoord, [(x1, y1), (x2, y2)])
                ndSep.append(ndCoord)
                return ndSep
            
            #Vertical grid lines 
            for i, (x1,y1,x2,y2) in enumerate(self.iterGridVerticalLines(w,h)):
                domid += 1
                addPageXmlSeparator(ndTR, i, lVi, x1, y1, x2, y2, domid)

            #horizontal grid lines 
            for i, (x1,y1,x2,y2) in enumerate(self.iterGridHorizontalLines(w,h)):
                domid += 1
                addPageXmlSeparator(ndTR, i, lHi, x1, y1, x2, y2, domid)
                
        return domid
    
    def remove_grid_from_dom(self, root):
        """
        clean the DOM from any existing grid (useful to choose at run-time the 
        grid increment (called step)
        return the number of removed grid lines
        """        
        for iPage, ndPage in enumerate(MultiPageXml.getChildByName(root, 'Page')):
    
            
            lnd = MultiPageXml.getChildByName(root,'GridSeparator')
            n = len(lnd)
            for nd in lnd:
                nd.getparent().remove(nd)
            #check...
            lnd = MultiPageXml.getChildByName(root,'GridSeparator')
            assert len(lnd) == 0
        return n
    
# ------------------------------------------------------------------
if __name__ == "__main__":
    #load mpxml 
    sFilename = sys.argv[1]
    sOutFilename = sys.argv[2]
    
    print("-adding grid: %s --> %s"%(sFilename, sOutFilename))
    iGridStep_H = 33  #odd number is better
    iGridStep_V = 33  #odd number is better
    
    # Some grid line will be O or I simply because they are too short.
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





