# -*- coding: utf-8 -*-

"""
    Find cuts of a page and annotate them based on the table separators
    
    Copyright Naver Labs Europe 2018
    JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os
from optparse import OptionParser
import operator
from collections import defaultdict

from lxml import etree
import numpy as np
import shapely.geometry as geom
import shapely.affinity

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln

from xml_formats.PageXml import MultiPageXml, PageXml

from util.Polygon import Polygon
from util.Shape import ShapeLoader, PolygonPartition

from tasks.DU_Table.DU_ABPTableSkewed_CutAnnotator import _isBaselineNotO, _isBaselineInTable,\
    computePRF
import tasks.DU_Table.DU_ABPTableRCAnnotation
from util.partitionEvaluation import evalPartitions
from util.jaccard import jaccard_distance

class CutAnnotator:
    """
    Cutting the page horizontally
    """
    fRATIO = 0.66
    
    def __init__(self):
        pass
    
    def get_separator_YX_from_DOM(self, root, fMinPageCoverage):
        """
        get the x and y of the GT table separators
        return lists of y, for horizontal and of x for vertical separators, per page
        return [(y_list, x_list), ...]
        """
        ltlYlX = []
        for ndPage in MultiPageXml.getChildByName(root, 'Page'):
            w, h = int(ndPage.get("imageWidth")), int(ndPage.get("imageHeight"))

            lYi, lXi = [], []
    
            l = MultiPageXml.getChildByName(ndPage,'TableRegion')
            if len(l) != 1:
                if l:
                    traceln("** warning ** %d TableRegion instead of expected 1" % len(l))
                else:
                    traceln("** warning ** no TableRegion, expected 1")
            if l:
                for ndTR in l:
                    #enumerate the table separators
                    for ndSep in MultiPageXml.getChildByName(ndTR,'SeparatorRegion'):
                        sPoints=MultiPageXml.getChildByName(ndSep,'Coords')[0].get('points')
                        [(x1,y1),(x2,y2)] = Polygon.parsePoints(sPoints).lXY
                        
                        dx, dy = abs(x2-x1), abs(y2-y1)
                        if dx > dy:
                            #horizontal table line
                            if dx > (fMinPageCoverage*w):
                                #ym = (y1+y2)/2.0   # 2.0 to support python2
                                lYi.append((y1,y2))
                        else:
                            if dy > (fMinPageCoverage*h):
                                #xm = (x1+x2)/2.0
                                lXi.append((x1,x2))
            ltlYlX.append( (lYi, lXi) )
                
        return ltlYlX

    def getHisto(self, lNd, w, _fMinHorizProjection, h, _fMinVertiProjection
                 , fRatio=1.0
                 , fMinHLen=None):
        """
        
        return two Numpy array reflecting the histogram of projections of objects
        first array along Y axis (horizontal projection), 2nd along X axis 
        (vertical projection)
        
        when fMinHLen is given , we do not scale horizontally text shorter than fMinHLen
        """
        
        hy = np.zeros((h,), np.float)
        hx = np.zeros((w,), np.float)
        
        for nd in lNd:
            sPoints=MultiPageXml.getChildByName(nd,'Coords')[0].get('points')
            try:
                x1,y1,x2,y2 = Polygon.parsePoints(sPoints).fitRectangle()
            
                if fMinHLen is None or abs(x2-x1) > fMinHLen: 
                    _x1, _x2 = self.scale(x1, x2, fRatio)
                else:
                    _x1, _x2 = x1, x2
                _y1, _y2 = self.scale(y1, y2, fRatio)
                hy[_y1:_y2+1] += float(x2 - x1) / w
                hx[_x1:_x2+1] += float(y2 - y1) / h
            except ZeroDivisionError:
                pass
            except ValueError:
                pass
        
        return hy, hx
    
    @classmethod
    def scale(cls, a, b, fRatio):
        """
        a,b are integers
        apply a scaling factor to the segment
        make sure its length remains non-zero
        return 2 integers
        """
        if fRatio == 1.0: return (a,b) # the code below does it, but no need...
        
        l = b - a                   # signed length
        ll = int(round(l * fRatio)) # new signed length
        
        dl2 = (l - ll) / 2.0
        ll2a = int(round(dl2))
        ll2b = (l - ll) - ll2a
        
        return a + ll2a, b - ll2b
        
    # labels...
    def _getLabel(self, i,j, liGT):
        """
        i,j are the index of teh start and end of interval of zeros
        liGT is a list of pair of pixel coordinates
        
        an interval of zeros is positive if it contains either end of the 
        separator or its middle.
        """
        for iGT, jGT in liGT:
            mGT = (iGT+jGT) // 2
            if i <= iGT and iGT <= j:
                return "S"
            elif i <= jGT and jGT <= j:
                return "S"
            elif i <= mGT and mGT <= j:
                return "S"            
        return "O"

    def getCentreOfZeroAreas(self, h, liGT=None):
        """
        liGT is the groundtruth indices
        return a list of center of areas contains consecutive 0s
        """
        lij = []    #list of area indices
        
        i0 = None   # index of start of a 0 area 
        imax = h.shape[0]
        i = 0
        while i < imax:
            if i0 is None:  # we were in a non-zero area
                if h[i] <= 0: i0 = i  # start of an area of 0s
            else:           # we were in a zero area
                if h[i] > 0:
                    # end of area of 0s
                    lij.append((i0, i-1))
                    i0 = None
            i += 1
        if not i0 is None: 
            lij.append((i0, imax-1))
            
        
        if liGT is None:
            liLbl = [None] * len(lij)
        else:
            liLbl = [self._getLabel(i,j,liGT) for (i,j) in lij]
        
        #take middle
        li = [ (j + i) // 2 for (i,j) in lij ]
        
        return li, liLbl

    def getLowestOfZeroAreas(self, h, liGT=None):
        """
        liGT is the groundtruth indices
        return a list of lowest points of areas contains consecutive 0s
        """
        lijm = []    #list of area indices
        
        i0 = None   # index of start of a 0 area 
        imax = h.shape[0]
        i = 0
        minV, minI = None, None
        while i < imax:
            if i0 is None:  # we were in a non-zero area
                if h[i] <= 0: 
                    i0 = i  # start of an area of 0s
                    minV, minI = h[i0], i0
            else:           # we were in a zero area
                if h[i] > 0:
                    # end of area of 0s
                    lijm.append((i0, i-1, minI))
                    i0 = None
                else:
                    if h[i] <= minV:    # take rightmost
                        minV, minI = h[i], i
            i += 1
        if not i0 is None: 
            minV, minI = h[i0], i0
            i = i0 + 1
            while i < imax:
                if h[i] < minV:     # tale leftmost
                    minV, minI = h[i], i
                i += 1
            lijm.append((i0, imax-1, minI))
            
        
        if liGT is None:
            liLbl = [None] * len(lijm)
        else:
            liLbl = [self._getLabel(i,j,liGT) for (i,j,_m) in lijm]
        
        #take middle
        li = [ m for (_i,_j, m) in lijm ]
        
        return li, liLbl


            
    def add_cut_to_DOM(self, root,
                       fMinHorizProjection=0.05,
                       fMinVertiProjection=0.05,
                       ltlYlX=[]
                       , fRatio = 1.0
                       , fMinHLen = None):
        """
        for each page, compute the histogram of projection of text on Y then X
        axis.
        From this histogram, find cuts. 
        fMinProjection determines the threholds as a percentage of width (resp 
        height) of page. Any bin lower than it is considered as zero.
        Map cuts to table separators to annotate them
        Dynamically tune the threshold for cutting so as to reflect most separators
        as a cut.
        Tag them if ltlYlX is given 
        
        ltlYlX is a list of (ltY1Y2, ltX1X2) per page. 
        ltY1Y2 is the list of (Y1, Y2) of horizontal separators, 
        ltX1X2 is the list of (X1, X2) of vertical separators.
         
        Modify the XML DOM by adding a separator cut, annotated if GT given
        """
        domid = 0 #to add unique separator id
        llX, llY = [], []
        for iPage, ndPage in enumerate(MultiPageXml.getChildByName(root, 'Page')):
            try:
                lYi, lXi = ltlYlX[iPage]
            #except TypeError:
            except:
                lYi, lXi = [], []
    
            w, h = int(ndPage.get("imageWidth")), int(ndPage.get("imageHeight"))
            
            #Histogram of projections
            lndTexLine = MultiPageXml.getChildByName(ndPage, 'TextLine')
            aYHisto, aXHisto = self.getHisto(lndTexLine, 
                                             w, fMinHorizProjection,
                                             h, fMinVertiProjection
                                             , fRatio
                                             , fMinHLen=fMinHLen)
            
            aYHisto = aYHisto - fMinHorizProjection
            aXHisto = aXHisto - fMinVertiProjection

            #find the centre of each area of 0s and its label
            lY, lYLbl = self.getCentreOfZeroAreas(aYHisto, lYi)
            # lX, lXLbl = self.getCentreOfZeroAreas(aXHisto, lXi)
            lX, lXLbl = self.getLowestOfZeroAreas(aXHisto, lXi)
            
            traceln(lY)
            traceln(lX)
            
            traceln(" - %d horizontal cuts" % len(lY))
            traceln(" - %d vertical cuts"   % len(lX))
            
            #ndTR = MultiPageXml.getChildByName(ndPage,'TableRegion')[0]
        
            # horizontal grid lines 
            for y, ylbl in zip(lY, lYLbl):
                domid += 1
                self.addPageXmlSeparator(ndPage, ylbl, 0, y, w, y, domid)

            # Vertical grid lines 
            for x, xlbl in zip(lX, lXLbl):
                domid += 1
                self.addPageXmlSeparator(ndPage, xlbl, x, 0, x, h, domid)     
            
            llX.append(lX)
            llY.append(lY)
                  
        return (llY, llX)
    
    @classmethod
    def addPageXmlSeparator(cls, nd, sLabel, x1, y1, x2, y2, domid):
        ndSep = MultiPageXml.createPageXmlNode("CutSeparator")
        if not sLabel is None:
            # propagate the groundtruth info we have
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
            
    def remove_cuts_from_dom(self, root):
        """
        clean the DOM from any existing cut 
        return the number of removed cut lines
        """        
        lnd = MultiPageXml.getChildByName(root,'CutSeparator')
        n = len(lnd)
        for nd in lnd:
            nd.getparent().remove(nd)
        #check...
        lnd = MultiPageXml.getChildByName(root,'CutSeparator')
        assert len(lnd) == 0
        return n        
    
    def loadPageCol(self, ndPage, fRatio
                    , shaper_fun=ShapeLoader.node_to_Point
                    , funIndex=lambda x: x._du_index):
        """
        load the page, looking for Baseline
        can filter by DU_row
        return a list of shapely objects
             , a dict of sorted list of objects, by column
             
        GT BUG: some Baseline are assigned to the wrong Cell
        => we also fix this here....
        
        """
        loBaseline        = []                # list of Baseline shapes
        i = 0
        
        dsetTableByCol = defaultdict(set) # sets of object ids, by col
        dsetTableDataByCol = defaultdict(set) # sets of object ids, by col
        dO = {}
        
        dNodeSeen = {}
        # first associate a unique id to each baseline and list them
        lshapeCell = []
        lOrphanBaselineShape = []
        
        lCells = MultiPageXml.getChildByName(ndPage, "TableCell")
        maxHeaderRowSpan = tasks.DU_Table.DU_ABPTableRCAnnotation.computeMaxRowSpan(lCells)
        traceln("   - maxHeaderRowSpan=", maxHeaderRowSpan)
        for ndCell in lCells:
            row, col = int(ndCell.get("row")), int(ndCell.get("col"))
            rowSpan = int(ndCell.get("rowSpan"))
            plg = ShapeLoader.node_to_Polygon(ndCell)
            #ymin, ymax of polygon
            lx = [_x for _x, _y in plg.exterior.coords]
            xmin, xmax = min(lx), max(lx)
            plg._row = row
            plg._col = col
            plg._xmin, plg._xmax = xmin, xmax
            lshapeCell.append(plg)
            
            for nd in MultiPageXml.getChildByName(ndCell, "Baseline"):
                nd.set("du_index", "%d" % i)
                ndParent = nd.getparent()
                dNodeSeen[ndParent.get('id')] = True
    
                # Baseline as a shapely object
                try:
                    o = shaper_fun(nd) #make a LineString
                except Exception as e:
                    traceln("ERROR: id=", nd.getparent().get("id"))
                    raise e
                # scale the objects, as done when cutting!!
                # useless currently since we make a Point...
                o = shapely.affinity.scale(o, xfact=fRatio, yfact=fRatio)
                
                o._du_index = i
                o._du_nd = nd
                o._dom_id = nd.getparent().get("id")
                loBaseline.append(o)
    
                # is this object in the correct cell???
                # We must use the centroid of the text box, otherwise a baseline
                # may be assigned to the next row
                # NOOO x = ShapeLoader.node_to_Polygon(ndParent).centroid.x
                # we must look for the leftest coordinate
                # NO CHECK FOR COLUMNS
                
                dsetTableByCol[col].add(funIndex(o))
                
                if (row+rowSpan) > maxHeaderRowSpan:
                    dsetTableDataByCol[col].add(funIndex(o))
                    
                i += 1

#         if lOrphanBaselineShape:
#             traceln("    *** error: %d Baseline in incorrect row - fixing this..." % len(lOrphanBaselineShape))
#             for o in lOrphanBaselineShape:
#                 bestrow, bestdeltacol = 0, 9999
#                 try:
#                     y = o.y
#                 except:
#                     y = o.centroid.y
#                 for plg in lshapeCell:
#                     if plg._ymin <= y and y <= plg._ymax:
#                         # sounds good
#                         deltacol = abs(o._bad_cell._col - plg._col)
#                         if deltacol == 0:
#                             # same column, ok it is that one
#                             bestrow = plg._row
#                             break
#                         else:
#                             if bestdeltacol > deltacol:
#                                 bestdeltacol = deltacol
#                                 bestrow = plg._row
#                 traceln("\t id=%s misplaced in row=%s instead of row=%s" %(
#                     o._du_nd.getparent().get("id")
#                     , o._bad_cell._row
#                     , bestrow))
#                 dsetTableByCol[bestrow].add(o._du_index)
#                 del o._bad_cell
            
        # and (UGLY) process all Baseline outside any TableCell...
        
        for nd in MultiPageXml.getChildByName(ndPage, "Baseline"):
            try:
                dNodeSeen[nd.getparent().get('id')]
            except:
                #OLD "GOOD" CODE HERE
                nd.set("du_index", "%d" % i)
    
                # Baseline as a shapely object
                o = shaper_fun(nd) #make a LineString

                # scale the objects, as done when cutting!!
                o = shapely.affinity.scale(o, xfact=fRatio)
                
                o._du_index = i
                o._du_nd = nd
                o._dom_id = nd.getparent().get("id")
                loBaseline.append(o)
    
                i += 1
        
        return loBaseline, dsetTableByCol, dsetTableDataByCol, maxHeaderRowSpan
    

class NoSeparatorException(Exception):
    pass
                
class BaselineCutAnnotator(CutAnnotator):
    """
    Much simpler approach: 
    - a block is defined by its baseline.
    - the baseline of each block defines a possible cut 
    - a parameter defines if the corresponding block is above or below the cut
    - so a cut defines a partition of the page block
    
    We use the table annotation to determine the baseline that is the on top
    or bottom of each table line (or column)
    """

    bSIO = False   # by default, we use SO as labels
    #iModulo = 1
    
    def __init__(self, bCutIsBeforeText=True):
        CutAnnotator.__init__(self)
        self.bCutIsBeforeText = bCutIsBeforeText
        
        #self._fModulo = float(self.iModulo)

    @classmethod
    def setLabelScheme_SIO(cls):
        cls.bSIO = True
        return True
    
#     def setModulo(self, iModulo):
#         self.iModulo = iModulo
#         self._fModulo = float(self.iModulo)
    
#     def moduloSnap(self, x, y):
#         """
#         return the same coordinate modulo the current modulo
#         """
#         return (int(round(x / self.fModulo)) * self.iModulo,
#                 int(round(y / self.fModulo)) * self.iModulo)

    @classmethod
    def getDomBaselineXY(cls, domNode):
        """
        find the baseline descendant node and return its "central" point
        """
        try:
            ndBaseline = MultiPageXml.getChildByName(domNode,'Baseline')[0]
        except IndexError as e:
            traceln("WARNING:  No Baseline child in ", domNode.get('id'))
            raise e
        x, y = cls.getPolylineAverageXY(ndBaseline)
        # modulo should be done only after the GT assigns labels.
        return (x, y)
        
    @classmethod
    def getPolylineAverageXY(cls, ndPolyline):
        """
        weighted average X and average Y of a polyline
        the weight indicate how long each segment at a given X, or Y, was.
        """
        sPoints=ndPolyline.get('points')
        lXY = Polygon.parsePoints(sPoints).lXY
        
        # list of X and Y values and respective weights
        lXYWxWy = [((x1+x2)/2.0, abs(y2-y1),    # for how long at this X?
                    (y1+y2)/2.0, abs(x2-x1)) \
                 for (x1,y1), (x2, y2) in zip(lXY, lXY[1:])] 
        fWeightedSumX = sum(x*wx for  x, wx, _,  _ in lXYWxWy)
        fWeightedSumY = sum(y*wy for  _,  _, y, wy in lXYWxWy)
        fSumWeightX   = sum(  wx for _, wx , _,  _ in lXYWxWy)
        fSumWeightY   = sum(  wy for _,  _ , _, wy in lXYWxWy)
        
        Xavg = int(round(fWeightedSumX/fSumWeightX)) if fSumWeightX > 0 else 0
        Yavg = int(round(fWeightedSumY/fSumWeightY)) if fSumWeightY > 0 else 0
        
#         Xavg, Yavg = self.moduloSnap(Xavg, Yavg)
        
        return (Xavg, Yavg)

    def _getLabelFromSeparator(self, ltXY, tlYlX, w, h):
        """
        ltXY is the list of (X, Y) of the "central" point of each baseline
        tlYlX are the coordinates of the GT separators
            ltY1Y2 is the list of (Y1, Y2) of horizontal separators, 
            ltX1X2 is the list of (X1, X2) of vertical separators.
        w, h are the page width and height
        
        if self.bCutIsBeforeText is True, we look for the highest baseline below
        or on each separator (which is possibly not horizontal)

        if self.bCutIsBeforeText is False, we look for the lowest baseline above
        or on each separator (which is possibly not horizontal)
        
        #TODO
        Same idea for vertical separators  ( ***** NOT DONE ***** )
        
        return lX, lY, lXLbl, lYLbl
        """
        ltY1Y2, ltX1X2 = tlYlX
        
        #rough  horizontal and vertical bounds
        try:
            ymin = operator.add(*min(ltY1Y2)) / 2.0  # ~~ (miny1+miny2)/2.0
            ymax = operator.add(*max(ltY1Y2)) / 2.0
            xmin = operator.add(*min(ltX1X2)) / 2.0
            xmax = operator.add(*max(ltX1X2)) / 2.0
        except ValueError:
            raise NoSeparatorException("No groundtruth")

        # find best baseline for each table separator
        setBestY = set()
        for (y1, y2) in ltY1Y2:
            bestY = 999999 if self.bCutIsBeforeText else -1
            bFound = False
            for x, y in ltXY:
                if x < xmin or xmax < x: # text outside table, ignore it 
                    continue
                #y of separator at x
                ysep = int(round(y1 + float(y2-y1) * x / w))
                if self.bCutIsBeforeText:
                    if ysep <= y and y < bestY and y < ymax:
                        #separator is above and baseline is above all others
                        bestY, bFound = y, True
                else:
                    if ysep >= y and y  > bestY and y > ymin:
                        bestY, bFound = y, True
            if bFound:
                setBestY.add(bestY)
        
        setBestX = set()
        for (x1, x2) in ltX1X2:
            bestX = 999999 if self.bCutIsBeforeText else -1
            bFound = False
            for x, y in ltXY:
                if y < ymin or ymax < y: # text outside table, ignore it 
                    continue
                #x of separator at Y
                xsep = int(round(x1 + float(x2-x1) * x / h))
                if self.bCutIsBeforeText:
                    if xsep <= x and x < bestX and x < xmax:
                        #separator is above and baseline is above all others
                        bestX, bFound = x, True
                else:
                    if xsep >= x and x  > bestX and x > xmin:
                        bestX, bFound = x, True
            if bFound:
                setBestX.add(bestX)

        # zero or one cut given a position
        lY = list(set(y for _, y in ltXY))  # zero or 1 cut per Y
        lY.sort()
        lX = list(set(x for x, _ in ltXY))  # zero or 1 cut per X
        lX.sort()
        
        if self.bSIO:
            # O*, S, (S|I)*, O*
            if setBestY:
                lYLbl = [ ("S" if y in setBestY \
                           else ("I" if ymin <= y and y <= ymax else "O")) \
                           for y in lY]
            else:
                lYLbl = ["O"] * len(lY)  # should never happen...
            if setBestX:
                lXLbl = [ ("S" if x in setBestX \
                           else ("I" if xmin <= x and x <= xmax else "O")) \
                           for x in lX]
            else:
                lXLbl = ["O"] * len(lX)  # should never happen...
        else:                
            # annotate the best baseline-based separator
            lYLbl = [ ("S" if y in setBestY else "O") for y in lY]
            lXLbl = [ ("S" if x in setBestX else "O") for x in lX]
        
        return lY, lYLbl, lX, lXLbl


#     def _getLabelFromCells(self, ltXY, lCells):
#         """
#         
#         NOT FINISHED
#         
#         SOME spans are ignored, some not
#         
#         This is done when making the straight separator, based on their length.
#         
#         ltXY is the list of (X, Y) of the "central" point of each baseline
#         lCells is the list of cells of the table
#         
#         For Y labels (horizontal cuts):
#         - if self.bCutIsBeforeText is True, we look for the highest baseline of 
#         each table line.
#         - if self.bCutIsBeforeText is False, we look for the lowest baseline of 
#         each table line.
# 
#         same idea for X labels (vertical cuts)
#         
#         returns the list of Y labels, the list of X labels
#         """
#                                             
#         lYLbl, lXLbl = [], []
#         
#         traceln("DIRTY: ignore rowspan above 5")
#         lCells = list(filter(lambda x: int(x.get('rowSpan')) < 5, lCells))   
#         dBestByRow = collections.defaultdict(lambda _: None)  # row->best_Y
#         dBestByCol = collections.defaultdict(lambda _: None)  # col->best_X
#         
#         dRowSep_lSgmt = collections.defaultdict(list)
#         dColSep_lSgmt = collections.defaultdict(list)
#         for cell in lCells:
#             row, col, rowSpan, colSpan = [int(cell.get(sProp)) for sProp \
#                                           in ["row", "col", "rowSpan", "colSpan"] ]
#             coord = cell.xpath("./a:%s" % ("Coords"),namespaces={"a":MultiPageXml.NS_PAGE_XML})[0]
#             sPoints = coord.get('points')
#             plgn = Polygon.parsePoints(sPoints)
#             lT, lR, lB, lL = plgn.partitionSegmentTopRightBottomLeft()
#             
#             #now the top segments contribute to row separator of index: row
#             dRowSep_lSgmt[row].extend(lT)
#             #now the bottom segments contribute to row separator of index: row+rowSpan
#             dRowSep_lSgmt[row+rowSpan].extend(lB)
#             
#             dColSep_lSgmt[col].extend(lL)
#             dColSep_lSgmt[col+colSpan].extend(lR)

    
    def add_cut_to_DOM(self, root, ltlYlX=[]):
        """
        for each page:
        - sort the block by their baseline average y
        - the sorted list of Ys defines the cuts.

        Tag them if ltlYlX is given
            ltlYlX is a list of (ltY1Y2, ltX1X2) per page. 
            ltY1Y2 is the list of (Y1, Y2) of horizontal separators, 
            ltX1X2 is the list of (X1, X2) of vertical separators.

        Modify the XML DOM by adding a separator cut, annotated if GT given
        """
        domid = 0 #to add unique separator id
        
        ltlYCutXCut = []
        for iPage, ndPage in enumerate(MultiPageXml.getChildByName(root, 'Page')):
            w, h = int(ndPage.get("imageWidth")), int(ndPage.get("imageHeight"))
    
            # list of Ys of baselines, and indexing of block by Y
            #list of (X,Y)
            ltXY = []
            lndTexLine = MultiPageXml.getChildByName(ndPage, 'TextLine')
            for ndBlock in lndTexLine:
                try:
                    ltXY.append(self.getDomBaselineXY(ndBlock))
                except:
                    pass

            # Groundtruth if any
            #lCells= MultiPageXml.getChildByName(ndPage, 'TableCell')

            # let's collect the segment forming the separators
            try:
                lY, lYLbl, lX, lXLbl = self._getLabelFromSeparator(ltXY,
                                                           ltlYlX[iPage], w, h)
            except NoSeparatorException:
                lX = list(set(x for x, _ in ltXY))  # zero or 1 cut per X
                lY = list(set(y for _, y in ltXY))  # zero or 1 cut per Y
                lX.sort()   # to have a nice XML
                lY.sort()
                lXLbl = [None] * len(lX)
                lYLbl = [None] * len(lY)
            
            ndTR = MultiPageXml.getChildByName(root,'TableRegion')[0]
        
            #Vertical grid lines 
            for y, ylbl in zip(lY, lYLbl):
                domid += 1
                self.addPageXmlSeparator(ndTR, ylbl, 0, y, w, y, domid)
            traceln(" - added %d horizontal cuts" % len(lX))

            #horizontal grid lines 
            for x, xlbl in zip(lX, lXLbl):
                domid += 1
                self.addPageXmlSeparator(ndTR, xlbl, x, 0, x, h, domid)     
            traceln(" - added %d vertical   cuts" % len(lY))
            
            ltlYCutXCut.append( ([y for _,y in ltXY],
                                 [x for x,_ in ltXY]))   
            
        return ltlYCutXCut


# ------------------------------------------------------------------
def main(sFilename, sOutFilename, fMinHorizProjection=0.05, fMinVertiProjection=0.05
         , bBaselineFirst=False
         , bBaselineLast=False
         , bSIO=False):
    
    print("- cutting: %s --> %s"%(sFilename, sOutFilename))
    
    # Some grid line will be O or I simply because they are too short.
    fMinPageCoverage = 0.5  # minimum proportion of the page crossed by a grid line
                            # we want to ignore col- and row- spans
    
    #for the pretty printer to format better...
    parser = etree.XMLParser(remove_blank_text=True)
    doc = etree.parse(sFilename, parser)
    root=doc.getroot()
    
    if bBaselineFirst:
        doer = BaselineCutAnnotator(bCutIsBeforeText=True)
        if bSIO: doer.setLabelScheme_SIO()
    elif bBaselineLast:
        doer = BaselineCutAnnotator(bCutIsBeforeText=False)
        if bSIO: doer.setLabelScheme_SIO()
    else:
        doer = CutAnnotator()
    
    print("doer=%s"%doer)
    
    #map the groundtruth table separators to our grid, per page (1 in tABP)
    ltlYlX = doer.get_separator_YX_from_DOM(root, fMinPageCoverage)
    
    # Find cuts and map them to GT
    # 
    if bBaselineFirst or bBaselineLast:
        doer.add_cut_to_DOM(root, ltlYlX=ltlYlX)
    else:        
        doer.add_cut_to_DOM(root, ltlYlX=ltlYlX,
                            fMinHorizProjection=fMinHorizProjection,
                            fMinVertiProjection=fMinVertiProjection,)
    
    #l_DU_row_Y, l_DU_row_GT = doer.predict(root)
    
    doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
    print('Annotated cut separators added into %s'%sOutFilename)

global_maxHeaderRowSpan = None
def _isBaselineInTableData(nd):
    """
    a Baseline in a TableRegion belongs to a TableCell element
    """
    global global_maxHeaderRowSpan
    v = nd.getparent().getparent().get("row")
    if v is None:
        return False
    else:
        return int(v) >= global_maxHeaderRowSpan


def get_col_partition(doer, sxpCut, dNS
                      , sFilename, lFilterFun
                      , fRatio
                      , bVerbose=False
                      , funIndex=lambda x: x._du_index
                      ):
    """
    return the GT partition in columns, as well as 1 partition per filter function
    """
    global global_maxHeaderRowSpan

    if bVerbose: traceln("- loading %s"%sFilename)
    parser = etree.XMLParser()
    doc = etree.parse(sFilename, parser)
    root=doc.getroot()

    llsetRun = []
    
    pnum = 0
    lndPage = MultiPageXml.getChildByName(root, 'Page')
    assert len(lndPage) == 1, "NOT SUPPORTED: file has many pages - soorry"
    for ndPage in  lndPage:
        pnum += 1
        if bVerbose: traceln("   - page %s - loading table GT" % pnum)
        
        loBaseline, dsetTableByCol, dsetTableDataByCol, global_maxHeaderRowSpan = doer.loadPageCol(ndPage, fRatio
                                                                                                   , funIndex=funIndex)
        
        if bVerbose: traceln("   - found %d objects on page" % (len(loBaseline)))
        
        # make a dictionary of cumulative sets, and the set of all objects
        lTableColK = sorted(dsetTableByCol.keys())
        lTableDataColK = sorted(dsetTableDataByCol.keys())
        if bVerbose: 
            traceln("   - found %d cols" % (len(lTableColK)))
            traceln("   - found %d objects in the table" % (sum(len(v) for v in dsetTableByCol.values())))
            traceln("   - found %d objects in the table data" % (sum(len(v) for v in dsetTableDataByCol.values())))
        lNdCut = ndPage.xpath(sxpCut, namespaces=dNS)
        if bVerbose: 
            traceln("   - found %d cuts" % (len(lNdCut)))
        else:
            traceln("- loaded %40s " % sFilename
                    , " %6d cols %6d 'S' cuts" % (  len(lTableColK)
                                                  , len(lNdCut))
                    , " %6d objects %6d table objects" % (
                                                        len(loBaseline)
                                                        , sum(len(v) for v in dsetTableByCol.values())
                                                        )
                    )
        loCut = []
        for ndCut in lNdCut:
            #now we need to infer the bounding box of that object
            (x1, y1), (x2, y2) = PageXml.getPointList(ndCut)  #the polygon
            # Create the shapely shape
            loCut.append(geom.LineString([(x1, y1), (x2, y2)]))   
        
        w,h = float(ndPage.get("imageWidth")), float(ndPage.get("imageHeight"))
#             # Add a fictive cut at top of page
#             loCut.append(geom.LineString([(0, 0), (w, 0)]))
#             # Add a fictive cut at end of page
#             loCut.append(geom.LineString([(0, h), (w, h)]))
        
        # order it by line centroid x
        loCut.sort(key=lambda o: o.centroid.x)
    
        # dcumset is the GT!!
        lsetGT     = [dsetTableByCol[k]     for k in lTableColK] # list of set of du_index
        lsetDataGT = [dsetTableDataByCol[k] for k in lTableDataColK]
        
        # NOW, look at predictions
        for filterFun in lFilterFun:
            loBaselineInTable = [o for o in loBaseline if filterFun(o._du_nd)]
            if bVerbose: traceln("   - %d objects on page predicted in table (%d out)" % (
                    len(loBaselineInTable)
                    , len(loBaseline) - len(loBaselineInTable)))
            
            # Now create the list of partitions created by the Cuts
            lsetRun = []
            partition    = PolygonPartition(loBaselineInTable)
            if True: # or bCutOnLeft:
                #cut if above the text that led to its creation
                setAllPrevIds = set([]) # cumulative set of what was already taken
                for oCut in loCut:
                    lo = partition.getObjectOnRightOfLine(oCut)
                    setIds = set(funIndex(o) for o in lo)
                    #print(oCut.centroid.x, setIds)
                    if setAllPrevIds:
                        prevColIds = setAllPrevIds.difference(setIds) # content of previous row
                        if prevColIds:
                            #an empty set is denoting alternative cuts leading to same partition 
                            lsetRun.append(prevColIds)
                    setAllPrevIds = setIds
            else:
                assert False, "look at this code..."
#                     #cut if below the text that led to its creation
#                     cumSetIds = set([]) # cumulative set
#                     for oCut in loCut:
#                         lo = partition.getObjectAboveLine(oCut)
#                         setIds = set(o._du_index for o in lo)
#                         rowIds = setIds.difference(cumSetIds) # only last row!
#                         if rowIds:
#                             #an empty set is denoting alternative cuts leading to same partition 
#                             lsetRun.append(rowIds)
#                         cumSetIds = setIds
#             _debugPartition("run", lsetRun)
#             _debugPartition("ref", lsetGT)
            llsetRun.append(lsetRun)
    return lsetGT, lsetDataGT, llsetRun


def op_eval_col(lsFilename, fSimil, fRatio, bVerbose=False):
    """
    We load the XML
    - get the CutSeparator elements
    - get the text objects (geometry=Baseline)
    - 
    """
    global global_maxHeaderRowSpan
    nOk, nErr, nMiss = 0, 0, 0

    if fSimil is None:
        #lfSimil = [ i / 100 for i in range(75, 101, 5)]
        lfSimil = [ i / 100 for i in range(70, 101, 10)]
    else:
        lfSimil = [fSimil]

    # we use only BIO + separators
    dOkErrMissOnlyCol  = { fSimil:(0,0,0) for fSimil in lfSimil }
    dOkErrMissOnlyCol.update({'name':'OnlyCol'
                            , 'FilterFun':_isBaselineNotO})
    # we use the TableRegion + separators
    dOkErrMissTableCol = { fSimil:(0,0,0) for fSimil in lfSimil }
    dOkErrMissTableCol.update({'name':'TableCol'
                             , 'FilterFun':_isBaselineInTable})

    # we use the TableRegion excluding the header + separators
    dOkErrMissTableDataCol = { fSimil:(0,0,0) for fSimil in lfSimil }
    dOkErrMissTableDataCol.update({'name':'TableDataCol'
                             , 'FilterFun':_isBaselineInTableData})
    
    ldOkErrMiss = [dOkErrMissOnlyCol, dOkErrMissTableCol, dOkErrMissTableDataCol]
    
    lFilterFun = [d['FilterFun'] for d in ldOkErrMiss]
            
    # sxpCut = './/pc:CutSeparator[@orient="0" and @DU_type="S"]' #how to find the cuts
    sxpCut = './/pc:CutSeparator[@orient="90"]' #how to find the cuts
    dNS = {"pc":PageXml.NS_PAGE_XML}
    
    doer = CutAnnotator()

    traceln(" - Cut selector = ", sxpCut)
    
    # load objects: Baseline and Cuts
    for n, sFilename in enumerate(lsFilename):
        lsetGT, lsetDataGT, llsetRun = get_col_partition(doer, sxpCut, dNS
                      , sFilename, lFilterFun
                      , fRatio
                      , bVerbose=False
                    , funIndex=lambda x: x._du_index  # simpler to view
#                       , funIndex=lambda x: x._dom_id  # more precise
                      )
        pnum = 1 # only support single-page file...
        for dOkErrMiss, lsetRun in zip(ldOkErrMiss, llsetRun):
            if dOkErrMiss['name'] == "TableDataCol":
                # we need to filter also the GT to discard the header from the column
                _lsetGT = lsetDataGT
            else:
                _lsetGT = lsetGT
            if bVerbose:
                traceln("----- RUN ----- ")
                for s in lsetRun: traceln("run ", sorted(s))
                traceln("----- REF ----- ")
                for s in _lsetGT: traceln("ref ", sorted(s))
            for fSimil in lfSimil:
                nOk, nErr, nMiss = dOkErrMiss[fSimil]
                _nOk, _nErr, _nMiss, _lFound, _lErr, _lMissed = evalPartitions(lsetRun, _lsetGT, fSimil, jaccard_distance)
                nOk   += _nOk
                nErr  += _nErr
                nMiss += _nMiss
                if bVerbose or fSimil == 1.0:
                    _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
                    traceln("%4d %8s simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f   ok=%6d  err=%6d  miss=%6d  %s page=%d" %(
                          n+1, dOkErrMiss['name'], fSimil
                        , _fP, _fR, _fF
                        , _nOk, _nErr, _nMiss
                        , os.path.basename(sFilename), pnum))
                dOkErrMiss[fSimil] = (nOk, nErr, nMiss)
    
    for dOkErrMiss in [dOkErrMissOnlyCol, dOkErrMissTableCol, dOkErrMissTableDataCol]:
        traceln()
        name = dOkErrMiss['name']
        for fSimil in lfSimil:
            nOk, nErr, nMiss = dOkErrMiss[fSimil]
            fP, fR, fF = computePRF(nOk, nErr, nMiss)
            traceln("ALL %8s  simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f " % (name, fSimil, fP, fR, fF )
                    , "        "
                    ,"ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss))
    return (nOk, nErr, nMiss)

def test_scale():
    
    assert (1,3) == CutAnnotator.scale(1, 3, 1.0)
    assert (3,1) == CutAnnotator.scale(3, 1, 1.0)

    def symcheck(a, b, r, aa, bb):
        assert (aa, bb) == CutAnnotator.scale(a, b, r), (a, b, r, aa, bb)
        assert (bb, aa) == CutAnnotator.scale(b, a, r), (b, a, r, bb, aa)
    symcheck(1, 2, 1.0, 1, 2)    
    symcheck(1, 1, 1.0, 1, 1)    
    symcheck(1, 10, 1.0, 1, 10)    
    
    assert (2,7) == CutAnnotator.scale(0 , 10, 0.5)
    assert (8,3) == CutAnnotator.scale(10, 0 , 0.5)
    
    assert (-2,-7) == CutAnnotator.scale(-0 , -10, 0.5)
    assert (-8,-3) == CutAnnotator.scale(-10, -0 , 0.5)

    assert (1,1) == CutAnnotator.scale(1, 1, 0.33)

# ------------------------------------------------------------------
if __name__ == "__main__":
    usage = ""
    parser = OptionParser(usage=usage, version="0.1")
    parser.add_option("--baseline_first", dest='bBaselineFirst',  action="store_true", help="Cut based on first baeline of row or column") 
    parser.add_option("--SIO"           , dest='bSIO'          ,  action="store_true", help="SIO labels") 

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    
    #load mpxml 
    sFilename = args[0]
    try:
        sOutFilename = args[1]
    except:
        sp, sf = os.path.split(sFilename)
        sOutFilename = os.path.join(sp, "cut-" + sf)        
    try:
        fMinH = float(args[2])
    except:
        fMinH = None
    if fMinH is None:
        main(sFilename, sOutFilename, bBaselineFirst=options.bBaselineFirst, bSIO=options.bSIO)
    else:
        fMinV = float(args[4])    # specify none or both
        main(sFilename, sOutFilename, fMinH, fMinV, bBaselineFirst=options.bBaselineFirst, bSIO=options.bSIO)
        


