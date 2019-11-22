# -*- coding: utf-8 -*-

"""
    Find cuts of a page along different slopes 
    and annotate them based on the table row content (which defines a partition)
    
    Copyright Naver Labs Europe 2018
    JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os
from optparse import OptionParser
import math
from collections import defaultdict, Counter

from lxml import etree
import shapely.geometry as geom
import shapely.ops

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln

from xml_formats.PageXml import MultiPageXml , PageXml
from util.Shape import ShapeLoader, PolygonPartition
from util.partitionEvaluation import evalPartitions
from util.jaccard import jaccard_distance
from util.Polygon import Polygon


class SkewedCutAnnotator:
    """
    Finding Skewed cuts and projecting GT to them
    
    Approach: 
    - a block is defined by its baseline.
    - we look for sloped separator crossing the page
    - each cut defines a partition
    - we build a dictionary part-> correspondign cuts
    - we select for eac key the cut with most frequent slope
    
    
    We use the table annotation to determine the GT tag of the cuts
    """
 
    #     slope    rad        deg
    #     1%    0.009999667    0.572938698
    #     2%    0.019997334    1.145762838
    #     3%    0.029991005    1.718358002
    #     4%    0.039978687    2.290610043
    #     5%    0.049958396    2.862405226

    # store angles as radians, so convert from degrees
    lfANGLE = [math.radians(x) for x in [-2, -1, 0, +1, +2]]
    lfANGLE = [math.radians(x) for x in [0]]
    # lfANGLE = [math.radians(x) for x in (_x/10 for _x in range(-20, +21, 5))]
        
    #lfANGLE = [math.radians(90+x) for x in [0]]

    gt_n     = 0   # how many valid Cut found? (valid = reflecting a row)
    gt_nOk   = 0   # how many GT table rows covered by a cut?
    gt_nMiss = 0   # how many GT table rows not reflected by a cut?
    nCut     = 0
    def __init__(self, bCutAbove, lAngle=lfANGLE):
        traceln("** SkewedCutAnnotator bCutAbove=%s  Angles (°): %s" %(bCutAbove, [math.degrees(v) for v in lAngle]))
        self.bCutAbove = bCutAbove # do we generate a cut line above or below each object?
        self.lAngle = lAngle
        
    # --- GT statistics
    @classmethod
    def gtStatReset(cls):
        cls.gt_n     = 0   # how many valid Cut found? (valid = reflecting a row)
        cls.gt_nOk   = 0   # how many GT table rows covered by a cut?
        cls.gt_nMiss = 0   # how many GT table rows not reflected by a cut?

    @classmethod
    def gtStatAdd(cls, n, nOk, nMiss, nCut):
        cls.gt_n += n
        cls.gt_nOk += nOk
        cls.gt_nMiss += nMiss
        cls.nCut += nCut

    @classmethod
    def gtStatReport(cls, t_n_nOk_nMiss=None):
        try:
            # to force displayign certain values
            n, nOk, nMiss = t_n_nOk_nMiss
            label = " >"
            nCut = None
        except:
            n, nOk, nMiss = cls.gt_n, cls.gt_nOk, cls.gt_nMiss
            nCut = cls.nCut
            label = "summary"
        nTotGT = nOk + nMiss + 0.00001
        traceln("GT: %s %7d cut reflecting a GT table row  (%.2f%%)" % (label, n, 100 * n / nTotGT))
        traceln("GT: %s %7d GT table row     reflected by a cut (%.2f%%)" % (label, nOk  , 100*nOk   / nTotGT))
        traceln("GT: %s %7d GT table row not reflected by a cut (%.2f%%)" % (label, nMiss, 100*nMiss / nTotGT))
        if not(nCut is None):
            traceln("GT: %s %7d cuts in total (%.2f%%)" % (label, nCut, nCut/nTotGT*100))
            
        
#     # def loadPage(self, ndPage, shaper_fun=ShapeLoader.node_to_LineString):
#     def loadPage_v1(self, ndPage, shaper_fun=ShapeLoader.node_to_Point):
#         """
#         load the page, looking for Baseline
#         can filter by DU_row
#         return a list of shapely objects
#              , a dict of sorted list of objects, by row
#              
#         GT BUG: some Baseline are assigned to the wrong Cell
#         => we also fix this here....
#         
#         """
#         loBaseline        = []                # list of Baseline shapes
# 
#         dsetTableByRow = defaultdict(set) # sets of object ids, by row
#         
#         # first associate a unique id to each baseline and list them
#         for i, nd in enumerate(MultiPageXml.getChildByName(ndPage, "Baseline")):
#             nd.set("du_index", "%d" % i)
#             # -> TextLine -> TableCell (possibly)
#             ndPrnt = nd.getparent()
#             row_lbl = ndPrnt.get("DU_row")
#             row = ndPrnt.getparent().get("row")
#             # row can be None
# 
#             # Baseline as a shapely object
#             o = shaper_fun(nd) #make a LineString
#             o._du_index = i
#             o._du_row    = row      # can be None
#             o._du_DU_row = row_lbl  # can be None
#             o._du_nd = nd
#             loBaseline.append(o)
# 
#             if not row is None: 
#                 dsetTableByRow[int(row)].add(i)
#         
#         return loBaseline, dsetTableByRow
    
    def loadPage(self, ndPage
                 , shaper_fun=ShapeLoader.node_to_Point
                 , funIndex=lambda x: x._du_index
                 , bIgnoreHeader=False
               ):
        """
        load the page, looking for Baseline
        can filter by DU_row
        return a list of shapely objects
             , a dict of sorted list of objects, by row
             
        GT BUG: some Baseline are assigned to the wrong Cell
        => we also fix this here....
        
        """
        loBaseline        = []                # list of Baseline shapes
        i = 0
        
        dsetTableByRow = defaultdict(set) # sets of object ids, by row
        
        dNodeSeen = {}
        # first associate a unique id to each baseline and list them
        lshapeCell = []
        lOrphanBaselineShape = []
        for ndCell in MultiPageXml.getChildByName(ndPage, "TableCell"):
            row, col = ndCell.get("row"), ndCell.get("col")
            plg = ShapeLoader.node_to_Polygon(ndCell)
            #ymin, ymax of polygon
            ly = [_y for _x, _y in plg.exterior.coords]
            ymin, ymax = min(ly), max(ly)
            plg._row = int(row)
            plg._col = int(col)
            plg._ymin, plg._ymax = ymin, ymax
            
            i0 = i
            for nd in MultiPageXml.getChildByName(ndCell, "Baseline"):
                nd.set("du_index", "%d" % i)
                ndParent = nd.getparent()
                dNodeSeen[ndParent.get('id')] = True
                if bIgnoreHeader and ndParent.get("DU_header") == "CH":
                    continue
                row_lbl = ndParent.get("DU_row")
    
                # Baseline as a shapely object
                try:
                    o = shaper_fun(nd) #make a LineString
                except Exception as e:
                    traceln("ERROR: id=", nd.getparent().get("id"))
                    raise e
                o._du_index = i
                o._du_DU_row = row_lbl  # can be None
                o._du_nd = nd
                o._dom_id = nd.getparent().get("id")
                loBaseline.append(o)
    
                # is this object in the correct cell???
                # We must use the centroid of the text box, otherwise a baseline
                # may be assigned to the next row
                #y = o.centroid.y # NOO!!
                y = ShapeLoader.node_to_Polygon(ndParent).centroid.y
                # if ymin <= y and y <= ymax:
                # we allow the content of a cell to overlap the cell lower border
                if ymin <= y:
                    dsetTableByRow[int(row)].add(funIndex(o))
                else:
                    # this is an orphan!
                    o._bad_cell = plg
                    lOrphanBaselineShape.append(o)
                
                i += 1
            
            if bIgnoreHeader and i0 == i:  
                continue # empty cells, certainly due to headers, ignore it.

            lshapeCell.append(plg)
            # end for
            
        if lOrphanBaselineShape:
            traceln("    *** error: %d Baseline in incorrect row - fixing this..." % len(lOrphanBaselineShape))
            for o in lOrphanBaselineShape:
                bestrow, bestdeltacol = 0, 9999
                try:
                    y = o.y
                except:
                    y = o.centroid.y
                for plg in lshapeCell:
                    if plg._ymin <= y and y <= plg._ymax:
                        # sounds good
                        deltacol = abs(o._bad_cell._col - plg._col)
                        if deltacol == 0:
                            # same column, ok it is that one
                            bestrow = plg._row
                            break
                        else:
                            if bestdeltacol > deltacol:
                                bestdeltacol = deltacol
                                bestrow = plg._row
                traceln("\t id=%s misplaced in row=%s instead of row=%s" %(
                    o._du_nd.getparent().get("id")
                    , o._bad_cell._row
                    , bestrow))
                dsetTableByRow[bestrow].add(funIndex(o))
                del o._bad_cell
            
        # and (UGLY) process all Baseline outside any TableCell...
        
        for nd in MultiPageXml.getChildByName(ndPage, "Baseline"):
            try:
                dNodeSeen[nd.getparent().get('id')]
            except:
                #OLD "GOOD" CODE HERE
                nd.set("du_index", "%d" % i)
                # -> TextLine -> TableCell (possibly)
                ndPrnt = nd.getparent()
                row_lbl = ndPrnt.get("DU_row")
    
                # Baseline as a shapely object
                o = shaper_fun(nd) #make a LineString
                o._du_index = i
                o._du_row    = None      # Must be None
                o._du_DU_row = row_lbl  # can be None
                o._du_nd = nd
                o._dom_id = nd.getparent().get("id")
                
                loBaseline.append(o)
    
                i += 1
        
        return loBaseline, dsetTableByRow

    @classmethod
    def makeCumulativeTableByRow(cls, dsetTableByRow, bDownward=False):
        """
        get a dictionary row-index -> set of row objects
        make a cumulative dictionary row-index -> frozenset of object from row 0 to row K 
        if bDonward is False, cumul is done from K to 0
        return (sorted list of keys, cumulative dictionary, set of all objects)
        """
        dcumset = defaultdict(set)
        cumset  = set()  # cumulative set of all index (of table objects)
        lTableRowK = sorted(dsetTableByRow.keys(), reverse=bDownward)
        for k in lTableRowK:
            cumset.update(dsetTableByRow[k])
            dcumset[k] = frozenset(cumset)
        if bDownward: lTableRowK.reverse()
        return lTableRowK, dcumset, cumset        
        
    def findHCut(self, ndPage, loBaseline, dsetTableByRow, fCutHeight=25, iVerbose=0):
        """
        find "horizontal" cuts that define a unique partition of the page text
        return a list of LineString
        """
        traceln("   - cut are made %s the text baseline centroid" % ("above" if self.bCutAbove else "below"))
        
        # GT: row -> set of object index
        bGT = len(dsetTableByRow) > 0

        traceln("   - Minimal cut height=", fCutHeight)
        w = int(ndPage.get("imageWidth"))
            
        dlCut_by_Partition = defaultdict(list) # dict partition -> list of cut lines
        partition    = PolygonPartition(loBaseline)
        
        _partitionFun = partition.getObjectBelowLineByIds if self.bCutAbove else partition.getObjectAboveLineByIds
        
        # Now consider in turn each baseline as the pivot for creating a separator
        # below it
        for oBaseline in loBaseline:
            
            # for each candidate "skewed" cuts
            for angle in self.lAngle:
                oCut = self.getTangentLineStringAtAngle(angle, oBaseline, w)
                
                if partition.isValidRibbonCut(oCut, -fCutHeight if self.bCutAbove else fCutHeight): 
                    #ok, store this candidate cut and associated partition!
                    tIds = _partitionFun(oCut)
                    dlCut_by_Partition[tIds].append(oCut)
                    oCut._du_support = oBaseline
                    oCut._du_angle = angle
                    oCut._du_label = "O" # just put "O" for wxvisu to show things
                    if bGT: oCut.__du_tIds = tIds   # temporarily
                    
        lloCutByPartition = list(dlCut_by_Partition.values())
        cntCut = sum(len(v) for v in lloCutByPartition)
        traceln("   - found %d \"horizontal\" cuts" % cntCut)

        # keep one cut per partition
        cntByAngle = Counter(o._du_angle for lo in lloCutByPartition for o in lo)
        if True:
            # preferring the closest to the average angle
            try:
                avgAngle = sum(k*v for k,v in cntByAngle.items()) / sum(cntByAngle.values())
            except ZeroDivisionError:
                avgAngle = 0
            lambdaScore = lambda o: - abs(o._du_angle - avgAngle)
        else:
            # preferring the most frequent angle
            lambdaScore = lambda o: cntByAngle[o._du_angle]
            cntCountByAngleDeg = {math.degrees(k):v for k,v in cntByAngle.items()}
            lDeg = sorted((cntCountByAngleDeg.keys())
                          , reverse=True, key=lambda o: cntCountByAngleDeg[o])
            traceln("   - Observed skew angles: " + "|".join([
                    " %.2f : %d "%(d, cntCountByAngleDeg[d]) for d in lDeg
                    ]))

        # Code below is correct but we need to do some more things for having better features
        # loCut = sorted((max(lo, key=lambda o: cntByAngle[o._du_angle]) 
        #                  for lo in lloCutByPartition)
        #                  , key=lambda o: o.centroid.y)
        loNewCut = []
        for _loCut in lloCutByPartition:
            # most frequent angle given the partition
            oCutBest = max(_loCut, key=lambdaScore)
            # create _du_set_support containing the set of node that lead to the same partition
            # set of nodes that generated this particular partition
            oCutBest._du_set_support = set(_o._du_support._du_index for _o in _loCut)
            # frequency of the cut's angle over the page
            oCutBest._du_angle_freq = cntByAngle[oCutBest._du_angle] / cntCut
            # cumulative frequency of all cuts that are represented by the chosen one
            oCutBest._du_angle_cumfreq = sum(cntByAngle[_o._du_angle] for _o in _loCut) / cntCut
            loNewCut.append(oCutBest)
        loCut = sorted(loNewCut, key=lambda o: o.centroid.y)

        traceln("   - kept %d \"horizontal\" unique cuts" % len(loCut))
        if loCut:
            traceln("   - average count of support nodes per cut : %.3f" %
                    (sum(len(_o._du_set_support) for _o in loCut) / len(loCut)))
            
        if bGT:
            traceln("  - loading GT Cell information")
            # make a dictionary of cumulative sets, and the set of all objects
            lTableRowK, dcumset, cumset = self.makeCumulativeTableByRow(dsetTableByRow, self.bCutAbove)
            # to tag at best the last cuts determining a valid partition...
            bestLastlSep = []
            bestLastLen = 99999999
            traceln("\tfound %d objects in table" % len(cumset))
            
            dGTCoverage = { k:0 for k in lTableRowK }
            # OK, let's tag with S I O based on the partition created by each cut
            for oCut in loCut:
                # build the set of index of text in the table,
                #  and above the cut
                setIdx = cumset.intersection(set(oCut.__du_tIds))
                
#                 if oCut.centroid.y == 1177:
#                     print(" oCut ", list(oCut.coords))
#                     print(sorted(list(setIdx)))
#                     print(setIdx.difference(dcumset[4]))
#                     print(dcumset[4].difference(setIdx))
#                     print(list(loBaseline[83].coords))
#                     print(loBaseline[83]._du_nd.getparent().get("id"))
#                     lkjljl

                # print(oCut._du_index, "nb table object above=", len(setIdx))
                #if setIdx in dcumset.values():  # a valid partition (compatible with the table)
                bNotFound = True
                for k in lTableRowK:
                    if setIdx == dcumset[k]:    # a valid partition (compatible with the table)
                        bNotFound = False
                        dGTCoverage[k] += 1
                        # ok, that partition was found
                        
                        if setIdx == cumset:
                            # is it the last separator of the table, or above the last??
                            # is it the last separator of the table, or below the last??
                            if len(tIds) <= bestLastLen:  # better end of table, because less O
                                if len(tIds) == bestLastLen: # same, in fact
                                    bestLastlSep.append(oCut)
                                else:
                                    bestLastlSep = [oCut]
                                    bestLastLen = len(setIdx) 
                            label = "O"  # we fix some of them at the end
                        else:
                            # ok this is a valid table partition
                            label = "S"
                if bNotFound:
                    if len(setIdx) > 0:
                        label = "I"  # some table elements above, but not all
                    else:
                        label = "O"
                oCut._du_label = label
                
                del oCut.__du_tIds
            for oCut in bestLastlSep: oCut._du_label = "S"
            
            c = Counter(oCut._du_label for oCut in loCut)
            lk = sorted(c.keys())
            traceln("GT:  > CUT Label count:  ", "  ".join("%s:%d"%(k, c[k]) for k in lk))            
            
            n       = sum(dGTCoverage.values())
            nOk     = len([k for k,v in dGTCoverage.items() if v > 0])
            nMiss   = len([k for k,v in dGTCoverage.items() if v == 0])
            if nMiss > 0:
                for k,v in dGTCoverage.items():
                    if v == 0: traceln("missed k=%d"%k)
            self.gtStatReport((n, nOk, nMiss))
            self.gtStatAdd(n, nOk, nMiss, len(loCut))
            self.gtStatReport()
        
        return loCut
        
    def getTangentLineStringAtAngle(self, a, o, w):
        """
        Find the line with given angle (less than pi/2 in absolute value) that is immediately below the object
        (angle in radians)
        return a Line
        """
        return geom.LineString( self.getTangentAtAngle(a, o, w) )
    
    def getTangentAtAngle(self, a, o, w):
        """
        Find the line with given angle (less than pi/2 in absolute value) that is immediately below the object
        (angle in radians)
        return a Line
        """
        EPSILON = 1

        minx, miny, maxx, maxy = o.bounds
        
        # first a line with this slope at some distance from object
        xo = (minx + maxx) // 2
        if self.bCutAbove:
            yo = miny - (maxx - minx) - 100
        else:
            yo = maxy + (maxx - minx) + 100
        y0, yw = self._getTangentAlongXCoord(a, xo, yo, w)
        oLine = geom.LineString([(0,y0), (w, yw)])
        
        #nearest points
        pt1, _pt2 = shapely.ops.nearest_points(o, oLine)
        x,y = pt1.x, pt1.y
        y0, yw = self._getTangentAlongXCoord(a, x, y, w)
        
        if self.bCutAbove:
            return (0, math.floor(y0-EPSILON)), (w, math.floor(yw-EPSILON))
        else:
            return (0, math.ceil(y0+EPSILON)), (w, math.ceil(yw+EPSILON))
        
    def _getTangentAlongXCoord(self, a, x, y, w):
        """
        intersection of the line, with angle a, at x=0 and x=w
        return y0, yw
        """
        if abs(a) <= 0.001:
            # math.radians(0.1) -> 0.0017453292519943296
            # this is horizontal!
            y0 = y
            yw = y
        else:
            t = math.tan(a)
            y0 = y - x * t
            yw = y + (w - x) * t
        return y0, yw
    
    def remove_cuts_from_dom(self, root):
        """
        clean the DOM from any existing cut 
        return the number of removed cut lines
        """        
        lnd = MultiPageXml.getChildByName(root,'CutSeparator')
        n = len(lnd)
        for nd in lnd:
            nd.getparent().remove(nd)
        return n        

    def add_Hcut_to_Page(self, ndPage, loCut, domid=0):
        """
        Add the cut to the page as a CutSeparator
        """
        for oCut in loCut:
            domid += 1
            self.addPageXmlSeparator(ndPage, oCut, domid)
        
        return domid

    @classmethod
    def addPageXmlSeparator(cls, ndPage, oCut, domid):
        ndSep = MultiPageXml.createPageXmlNode("CutSeparator")
        # propagate the groundtruth info we have
        ndSep.set("DU_type"    , oCut._du_label)
        ndSep.set("orient"  , "0")
        ndSep.set("DU_angle"  , "%.1f"%math.degrees(oCut._du_angle))
        ndSep.set("DU_angle_freq"           , "%.3f"%oCut._du_angle_freq)
        ndSep.set("DU_angle_cumul_freq"     , "%.3f"%oCut._du_angle_cumfreq)
        ndSep.set("DU_set_support"          , "%s"  %oCut._du_set_support)
        ndSep.set("id"      , "cs_%d" % domid)
        ndPage.append(ndSep)
        ndCoord = MultiPageXml.createPageXmlNode("Coords")
        MultiPageXml.setPoints(ndCoord, oCut.coords)
        ndSep.append(ndCoord)
        return ndSep


class NoSeparatorException(Exception):
    pass
                


# ------------------------------------------------------------------
def test__getTangentCoord(capsys):
    
    b1 = geom.Polygon([(1,2), (2,2), (2,3), (1,3)]) 
    
    doer = SkewedCutAnnotator(bCutAbove=True)

    def printAngle(a, oLine):
        [(xa,ya), (xb,yb)] = list(oLine.coords)
        aa = math.atan((yb-ya) / (xb-xa))
        print("asked %.2f°  got %.2f°   (diff=%.4f°)" % (math.degrees(a), math.degrees(aa), math.degrees(aa-a)))
        
    with capsys.disabled():
        p11 = geom.Point((1, 1))
        oLine = doer.getTangentLineStringAtAngle(0, p11, 100)
        assert list(oLine.coords) == [(0, 2.0), (100, 2.0)]
        
        oLine = doer.getTangentLineStringAtAngle(-0.1*math.pi/2, p11, 100)
        assert oLine.distance(p11) > 0
        
        oLine = doer.getTangentLineStringAtAngle(0.2*math.pi/2, p11, 100)
        assert oLine.distance(p11) > 0
    

        oLine = doer.getTangentLineStringAtAngle(0, b1, 10)
        assert list(oLine.coords) == [(0, 4), (10, 4)]
        
        a = -0.1*math.pi/2
        oLine = doer.getTangentLineStringAtAngle(a, b1, 5000) #typical page width
        print()
        printAngle(a, oLine)
        #print(oLine)
        assert oLine.distance(b1) > 0
        
        a = 0.2*math.pi/2
        oLine = doer.getTangentLineStringAtAngle(a, b1, 5000)
        printAngle(a, oLine)
        #print(oLine)
        assert oLine.distance(b1) > 0
    
def test__getTangentCoord_cut_above(capsys):
    
    b1 = geom.Polygon([(1,2), (2,2), (2,3), (1,3)]) 
    
    doer = SkewedCutAnnotator(bCutAbove=True)

    def printAngle(a, oLine):
        [(xa,ya), (xb,yb)] = list(oLine.coords)
        aa = math.atan((yb-ya) / (xb-xa))
        print("asked %.2f°  got %.2f°   (diff=%.4f°)" % (math.degrees(a), math.degrees(aa), math.degrees(aa-a)))
        
    with capsys.disabled():
        p11 = geom.Point((1, 1))
        oLine = doer.getTangentLineStringAtAngle(0, p11, 100)
        assert list(oLine.coords) == [(0, 0.0), (100, 0.0)]
        
        oLine = doer.getTangentLineStringAtAngle(-0.1*math.pi/2, p11, 100)
        assert oLine.distance(p11) > 0
        
        oLine = doer.getTangentLineStringAtAngle(0.2*math.pi/2, p11, 100)
        assert oLine.distance(p11) > 0
    

        oLine = doer.getTangentLineStringAtAngle(0, b1, 10)
        assert list(oLine.coords) == [(0, 1), (10, 1)]
        
        a = -0.1*math.pi/2
        oLine = doer.getTangentLineStringAtAngle(a, b1, 5000) #typical page width
        print()
        printAngle(a, oLine)
        #print(oLine)
        assert oLine.distance(b1) > 0
        
        a = 0.2*math.pi/2
        oLine = doer.getTangentLineStringAtAngle(a, b1, 5000)
        printAngle(a, oLine)
        #print(oLine)
        assert oLine.distance(b1) > 0
    
    
# ------------------------------------------------------------------
def op_cut(sFilename, sOutFilename, lDegAngle, bCutAbove, fMinHorizProjection=0.05, fCutHeight=25):
    #for the pretty printer to format better...
    parser = etree.XMLParser(remove_blank_text=True)
    doc = etree.parse(sFilename, parser)
    root=doc.getroot()
    
    doer = SkewedCutAnnotator(bCutAbove, lAngle = [math.radians(x) for x in lDegAngle])
        
    pnum = 0
    domid = 0
    for ndPage in  MultiPageXml.getChildByName(root, 'Page'):
        pnum += 1
        traceln(" --- page %s - constructing separator candidates" % pnum)
        
        #load the page objects and the GT partition (defined by the table) if any
        loBaseline, dsetTableByRow = doer.loadPage(ndPage)
        traceln(" - found %d objects on page" % (len(loBaseline)))

        # find almost-horizontal cuts and tag them if GT is available
        loHCut = doer.findHCut(ndPage, loBaseline, dsetTableByRow, fCutHeight)  

        #create DOM node reflecting the cuts 
        #first clean (just in case!)
        n = doer.remove_cuts_from_dom(ndPage) 
        if n > 0: 
            traceln(" - removed %d pre-existing cut lines" % n)
        
        # if GT, then we have labelled cut lines in DOM
        domid = doer.add_Hcut_to_Page(ndPage, loHCut, domid)

    doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
    print('Annotated cut separators added to %s'%sOutFilename)



def computePRF(nOk, nErr, nMiss):
    eps = 0.00001
    fP = 100 * nOk / (nOk + nErr + eps)
    fR = 100 * nOk / (nOk + nMiss + eps)
    fF = 2 * fP * fR / (fP + fR + eps)
    return fP, fR, fF

def _debugPartition(s, lset):
    traceln("---- ", s)
    for s in lset:
        traceln(s)
        
def _isBaselineNotO(nd):
    """
    filter Baseline tagged as 'O' or not tagged at all
    """
    v = nd.getparent().get("DU_row")
    return v is None or v not in ["O"]
    
def _isBaselineInTable(nd):
    """
    a Baseline in a TableRegion belongs to a TableCell element
    """
    v = nd.getparent().getparent().get("row")
    return not(v is None)

def get_row_partition(doer, sxpCut, dNS
                      , sFilename, lFilterFun
                      , bCutAbove=True
                      , bVerbose=False
                      , funIndex=lambda x: x._du_index
                      , bIgnoreHeader=False
                      ):
    """
    return the GT partition in rows, as well as 1 partition per filter fucntion
    """
    # load objects: Baseline and Cuts
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
        loBaseline, dsetTableByRow = doer.loadPage(ndPage, funIndex=funIndex
                                                   , bIgnoreHeader=bIgnoreHeader)
        if bVerbose: traceln("   - found %d objects on page" % (len(loBaseline)))
        
        # make a dictionary of cumulative sets, and the set of all objects
        lTableRowK = sorted(dsetTableByRow.keys())
        if bVerbose: 
            traceln("   - found %d rows" % (len(lTableRowK)))
            traceln("   - found %d objects in the table" % (sum(len(v) for v in dsetTableByRow.values())))
        lNdCut = ndPage.xpath(sxpCut, namespaces=dNS)
        if bVerbose: 
            traceln("   - found %d 'S' cut" % (len(lNdCut)))
        else:
            traceln("- loaded %40s " % sFilename
                    , " %6d rows %6d 'S' cuts" % (  len(lTableRowK)
                                                  , len(lNdCut))
                    , " %6d objects %6d table objects" % (
                                                        len(loBaseline)
                                                        , sum(len(v) for v in dsetTableByRow.values())
                                                        )
                    )
        loCut = []
        for ndCut in lNdCut:
            #now we need to infer the bounding box of that object
            (x1, y1), (x2, y2) = PageXml.getPointList(ndCut)  #the polygon
            # Create the shapely shape
            loCut.append(geom.LineString([(x1, y1), (x2, y2)]))   
        
        w,h = float(ndPage.get("imageWidth")), float(ndPage.get("imageHeight"))
        # Add a fictive cut at top of page
        loCut.append(geom.LineString([(0, 0), (w, 0)]))
        # Add a fictive cut at end of page
        loCut.append(geom.LineString([(0, h), (w, h)]))
        
        # order it by line centroid Y
        loCut.sort(key=lambda o: o.centroid.y)
    
        # dcumset is the GT!!
        lsetGT = [dsetTableByRow[k] for k in lTableRowK] # list of set of du_index

        # NOW, look at predictions
        for filterFun in lFilterFun:
            
            loBaselineInTable = [o for o in loBaseline if filterFun(o._du_nd)]
            if bVerbose: traceln("   - %d objects on page predicted in table (%d out)" % (
                    len(loBaselineInTable)
                    , len(loBaseline) - len(loBaselineInTable)))
            
            # Now create the list of partitions created by the Cuts
            lsetRun = []
            partition    = PolygonPartition(loBaselineInTable)
            if bCutAbove:
                #cut if above the text that led to its creation
                setAllPrevIds = set([]) # cumulative set of what was already taken
                for oCut in loCut:
                    lo = partition.getObjectBelowLine(oCut)
                    setIds = set(funIndex(o) for o in lo)
                    if setAllPrevIds:
                        prevRowIds = setAllPrevIds.difference(setIds) # content of previous row
                        if prevRowIds:
                            #an empty set is denoting alternative cuts leading to same partition 
                            lsetRun.append(prevRowIds)
                    setAllPrevIds = setIds
            else:
                #cut if below the text that led to its creation
                cumSetIds = set([]) # cumulative set
                for oCut in loCut:
                    lo = partition.getObjectAboveLine(oCut)
                    setIds = set(funIndex(o) for o in lo)
                    rowIds = setIds.difference(cumSetIds) # only last row!
                    if rowIds:
                        #an empty set is denoting alternative cuts leading to same partition 
                        lsetRun.append(rowIds)
                    cumSetIds = setIds
#             _debugPartition("run", lsetRun)
            llsetRun.append(lsetRun)
#             _debugPartition("ref", lsetGT)
    return lsetGT, llsetRun

def op_eval_row(lsFilename, fSimil, bCutAbove, bVerbose=False
                , bIgnoreHeader=False):
    """
    We load the XML
    - get the cut with @DU_type="S"
    - get the text objects (geometry=Baseline)
    - 
    """
    nOk, nErr, nMiss = 0, 0, 0

    if fSimil is None:
        #lfSimil = [ i / 100 for i in range(75, 101, 5)]
        lfSimil = [ i / 100 for i in range(70, 101, 10)]
    else:
        lfSimil = [fSimil]

    # we use only BIO+SIO
    dOkErrMissOnlyRow  = { fSimil:(0,0,0) for fSimil in lfSimil }
    dOkErrMissOnlyRow.update({'name':'OnlyRow'
                            , 'FilterFun':_isBaselineNotO})
    # we use the SIO and the TableRegion
    dOkErrMissTableRow = { fSimil:(0,0,0) for fSimil in lfSimil }
    dOkErrMissTableRow.update({'name':'TableRow'
                             , 'FilterFun':_isBaselineInTable})
    ldOkErrMiss = [dOkErrMissOnlyRow, dOkErrMissTableRow]
    
    sxpCut = './/pc:CutSeparator[@orient="0" and @DU_type="S"]' #how to find the cuts
    dNS = {"pc":PageXml.NS_PAGE_XML}
    
    doer = SkewedCutAnnotator(bCutAbove)

    traceln(" - Cut selector = ", sxpCut)
    
    # load objects: Baseline and Cuts
    for n, sFilename in enumerate(lsFilename):
        lsetGT, llsetRun = get_row_partition(doer, sxpCut, dNS
                                             , sFilename
                                             , [dOkErrMiss['FilterFun'] for dOkErrMiss in ldOkErrMiss]
                                             , bCutAbove=True, bVerbose=False
                                             , funIndex=lambda o: o._dom_id
                                             , bIgnoreHeader=bIgnoreHeader
                                             )
        pnum = 1 # only support single-page file...
        for dOkErrMiss, lsetRun in zip(ldOkErrMiss, llsetRun):
            for fSimil in lfSimil:
                nOk, nErr, nMiss = dOkErrMiss[fSimil]
                _nOk, _nErr, _nMiss, _lFound, _lErr, _lMissed = evalPartitions(lsetRun, lsetGT, fSimil, jaccard_distance)
                if bVerbose:
                    traceln(" - - - simil = %.2f" % fSimil)
                    traceln("----- RUN ----- ")
                    for s in lsetRun: traceln("  run ", sorted(s))
                    traceln("----- REF ----- ")
                    for s in lsetGT: traceln("  ref ", sorted(s))
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
    
    for dOkErrMiss in [dOkErrMissOnlyRow, dOkErrMissTableRow]:
        traceln()
        name = dOkErrMiss['name']
        for fSimil in lfSimil:
            nOk, nErr, nMiss = dOkErrMiss[fSimil]
            fP, fR, fF = computePRF(nOk, nErr, nMiss)
            traceln("ALL %8s  simil:%.2f  P %5.1f  R %5.1f  F1 %5.1f " % (name, fSimil, fP, fR, fF )
                    , "        "
                    ,"ok=%d  err=%d  miss=%d" %(nOk, nErr, nMiss))
    return (nOk, nErr, nMiss)


def op_eval_old(lsFilename, fSimil, bDetail=False):
    """
    We load the XML
    - get the cut with @type="S"
    - get the text objects (geometry=Baseline)
    - 
    """
    nOk, nErr, nMiss = 0, 0, 0

    # OLD STYLE (May'18)
    sxpCut = './/pc:CutSeparator[@orient="0" and @type="S"]' #how to find the cuts
        
    dNS = "./pc:TextEquiv"            

    doer = SkewedCutAnnotator(True)

    traceln(" - Cut selector = ", sxpCut)
    
    def getPolylineAverageXY(ndPolyline):
        """
        COPIED FROM  tasks.DU_ABPTableCutAnnotator.BaselineCutAnnotator
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

    def baseline_loader(nd):
        """
        load the baseline as done in DU_ABPTableCutAnnotator
        """
        x, y = getPolylineAverageXY(nd)
        # make a short horizontal line out of a point
        return geom.LineString([(x-10,y), (x+10, y)])

    # load objects: Baseline and Cuts
    for sFilename in lsFilename:
        traceln("- loading %s"%sFilename)
        parser = etree.XMLParser()
        doc = etree.parse(sFilename, parser)
        root=doc.getroot()
    
        pnum = 0
        for ndPage in  MultiPageXml.getChildByName(root, 'Page'):
            pnum += 1
            traceln("   - page %s - loading table GT" % pnum)
            loBaseline, dsetTableByRow = doer.loadPage(ndPage, shaper_fun=baseline_loader)
            traceln("   - found %d objects on page" % (len(loBaseline)))
            # make a dictionary of cumulative sets, and the set of all objects
            lTableRowK = sorted(dsetTableByRow.keys(), reverse=True) # bottom to top
            traceln("   - found %d objects in the table" % (sum(len(v) for v in dsetTableByRow.values())))

            lNdCut = ndPage.xpath(sxpCut, namespaces={"pc":PageXml.NS_PAGE_XML})
            traceln("   - found %d 'S' cut" % (len(lNdCut)))
            loCut = []
            for ndCut in lNdCut:
                #now we need to infer the bounding box of that object
                (x1, y1), (x2, y2) = PageXml.getPointList(ndCut)  #the polygon
                # make sure that the cut is above the baseline that created it
                y1 -= 1
                y2 -= 1
                assert y1 == y2  # in this version, the cuts were horizontal
                # Create the shapely shape
                loCut.append(geom.LineString([(x1, y1), (x2, y2)]))   
            # order it by line centroid Y
            loCut.sort(key=lambda o: o.centroid.y, reverse=True) # from bottom to top  
        
            # dcumset is the GT!!
            lsetGT = [dsetTableByRow[k] for k in lTableRowK] # list of set of du_index
    
            # Now create the list of partitions created by the Cuts, excluding the 'O'
            lsetRun = []
            partition    = PolygonPartition(loBaseline)
            cumSetIds = set([]) # cumulative set
            for oCut in loCut:
                lo = partition.getObjectBelowLine(oCut)
                setIds = set(o._du_index for o in lo if _isBaselineInTable(o._du_nd))
                rowIds = setIds.difference(cumSetIds) # only last row!
                if rowIds:
                    #an empty set is denoting alternative cuts leading to same partition 
                    lsetRun.append(rowIds)
                cumSetIds = setIds
#             _debugPartition("run", lsetRun)
#             _debugPartition("ref", lsetGT)
            _nOk, _nErr, _nMiss, _lFound, _lErr, _lMissed = evalPartitions(lsetRun, lsetGT, fSimil, jaccard_distance)
            nOk   += _nOk
            nErr  += _nErr
            nMiss += _nMiss
            if bDetail:
                _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
                traceln("ok=%d  err=%d  miss=%d  P=%.1f  R=%.1f  F1=%.1f %s page=%d" %(
                    _nOk, _nErr, _nMiss
                    , _fP, _fR, _fF
                    , sFilename, pnum))
    
    fP, fR, fF = computePRF(nOk, nErr, nMiss)
    
    traceln("SUMMARY == P=%.1f%%\tR=%.1f%%\tF1=%.1f" % (fP, fR, fF ))
    traceln("ok=%d  err=%d  miss=%d  P=%.1f  R=%.1f  F1=%.1f" %(
        nOk, nErr, nMiss
        , fP, fR, fF))
    return (nOk, nErr, nMiss)

# ------------------------------------------------------------------
def op_gt_recall(lsFilename, bCutAbove, lDegAngle, fMinHorizProjection=0.05, fCutHeight=25):
    cAll = Counter()
    for sFilename in lsFilename:
        traceln("- loading GT: %s"%sFilename)
        
        #for the pretty printer to format better...
        parser = etree.XMLParser(remove_blank_text=True)
        doc = etree.parse(sFilename, parser)
        root=doc.getroot()
        
        doer = SkewedCutAnnotator(bCutAbove, lAngle = [math.radians(x) for x in lDegAngle])
            
        pnum = 0
        for ndPage in  MultiPageXml.getChildByName(root, 'Page'):
            pnum += 1
            traceln(" --- page %s - constructing separator candidates" % pnum)
            
            #load the page objects and the GT partition (defined by the table) if any
            loBaseline, dsetTableByRow = doer.loadPage(ndPage)
            traceln(" - found %d objects on page" % (len(loBaseline)))
    
            # find almost-horizontal cuts and tag them if GT is available
            loHCut = doer.findHCut(ndPage, loBaseline, dsetTableByRow, fCutHeight)
            cAll.update(Counter(o._du_label for o in loHCut))

    lk = sorted(cAll.keys())
    traceln("GT: ALL CUT Label count:  ", "  ".join("%s:%d"%(k, cAll[k]) for k in lk))
            
        
# ------------------------------------------------------------------
if __name__ == "__main__":
    usage = ""
    parser = OptionParser(usage=usage, version="0.1")
    parser.add_option("--height", dest="fCutHeight", default=10
                      ,  action="store", type=float, help="Minimal height of a cut") 
    parser.add_option("--simil", dest="fSimil", default=None
                      ,  action="store", type=float, help="Minimal similarity for associating 2 partitions") 
    parser.add_option("--angle", dest='lsAngle'
                      ,  action="store", type="string", default="0"
                        ,help="Allowed cutting angles, in degree, comma-separated") 
    parser.add_option("--cut-below", dest='bCutBelow',  action="store_true", default=False
                        ,help="Each object defines one or several cuts above it (instead of above as by default)") 
#     parser.add_option("--cut-above", dest='bCutAbove',  action="store_true", default=None
#                         , help="Each object defines one or several cuts above it (instead of below as by default)") 
    parser.add_option("-v", "--verbose", dest='bVerbose',  action="store_true", default=False)

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    
    options.bCutAbove = not(options.bCutBelow)
    
    #load mpxml 
    op = args[0]
    # --------------------------------------
    if op == "cut":
        sFilename = args[1]
        sOutFilename = args[2]
        traceln("- cutting : %s  --> %s" % (sFilename, sOutFilename))
        lDegAngle = [float(s) for s in options.lsAngle.split(",")]
        traceln("- Allowed angles (°): %s" % lDegAngle)
        op_cut(sFilename, sOutFilename, lDegAngle, options.bCutAbove, fCutHeight=options.fCutHeight)
    # --------------------------------------
    elif op == "eval":
        lsFilename = args[1:]
        traceln("- evaluating cut-based partitions (fSimil=%s): " % options.fSimil, lsFilename)
        op_eval_row(lsFilename, options.fSimil, options.bCutAbove, options.bVerbose)
        # --------------------------------------
    elif op == "eval_bsln":
        lsFilename = args[1:]
        traceln("- evaluating baseline-based partitions : ", lsFilename)
        op_eval_old(lsFilename, options.fSimil, True)
        # --------------------------------------
    elif op == "gt_recall":
        lsFilename = args[1:]
        traceln("- GT recall : %s" % lsFilename)
        lDegAngle = [float(s) for s in options.lsAngle.split(",")]
        traceln("- Allowed angles (°): %s" % lDegAngle)
        op_gt_recall(lsFilename, options.bCutAbove, lDegAngle, fCutHeight=options.fCutHeight)
    else:
        print("Usage: %s [cut|eval|eval_bsln|gt_eval]")
        


