# -*- coding: utf-8 -*-

"""
We expect XML file with cluster defined by one algo.

For each Page:
    We tabulate the clusters (build a table where each cluster is a cell)
    We compute the row, col, row_span, col_span attributes of each cluster

Overwrite the input XML files, adding attributes to the cluster definitions

If the cluster do not have a defined shape, we compute a shape based on a minimum_rotated_rectangle

Created on 26/9/2019

Copyright NAVER LABS Europe 2019

@author: JL Meunier
"""

import sys, os
from optparse import OptionParser
from collections import defaultdict
from lxml import etree

import numpy as np
import shapely.ops
from shapely import affinity

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln, trace
from xml_formats.PageXml import PageXml

from tasks.intersect_cluster import Cluster
from graph.Block import Block
from util.Shape import ShapeLoader

# ----------------------------------------------------------------------------
xpCluster       = ".//pg:Cluster"
xpClusterEdge   = ".//pg:ClusterEdge"
xpEdge      = ".//pg:Edge"
# sFMT        = "(%s_∩_%s)"  pb with visu
sAlgoAttr   = "algo"
xpPage      = ".//pg:Page"
dNS = {"pg":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
# ----------------------------------------------------------------------------


class TableCluster(Cluster, Block):
    thTopAligned = 20      # a difference less than 20 pixel on y1 means top-aliogned
    # scale BB by these ratio (horizontally and vertically)
    scale_H   = 0.66    # better if same as in DU_Table_Col_Cut
    # scale_H   = 1.0     # to get hard cases
    scale_V   = 1       # do not shrink

    cnt = 0
    
    def __init__(self, name, setID, shape=None):
        Cluster.__init__(self, name, setID, shape=shape)
        # we do not __init__ Block - useless, we just need a few methods
        self.dsEdge = defaultdict(set)  # dic  edge_type -> neighbours set
        self.cnt = TableCluster.cnt
        TableCluster.cnt += 1
        
    @classmethod        
    def induceClusterEdge(cls, ndPage, lCluster):
        """
        compute inter- cluster edges from inter- cluster-item edges  
        
        no so good for horizontal edges... :-/
        """
        # revert dictionay itemID -Cluster
        dCluster_by_Item = { x:c for c in lCluster for x in c.getSetID() }
        for _nd in ndPage.xpath(xpEdge, namespaces=dNS):
            _A, _B = _nd.get("src"), _nd.get("tgt")
            _AC, _BC = dCluster_by_Item[_A], dCluster_by_Item[_B]
            if _AC != _BC:
                TableCluster.link(_AC, _BC, edge_type=_nd.get("type"))
        del dCluster_by_Item

    @classmethod        
    def computeClusterEdge(cls, _ndPage, lCluster):
        """
        compute edge using g2 method from class Block :-)
        A bit computationally heavy, but safe code...
        """
        lHEdge, lVEdge = Block.findPageNeighborEdges(lCluster, bShortOnly=False, iGraphMode=2)
        for edge in lHEdge:
            TableCluster.link(edge.A, edge.B, "HorizontalEdge")
        for edge in lVEdge:
            TableCluster.link(edge.A, edge.B, "VerticalEdge")
          
    @classmethod
    def addEdgesToXml(cls, ndPage, sAlgo, lCluster):
        cnt = 0
        ndPage.append(etree.Comment("\nInter-cluster edges by tabulate_cluster scale_H=%.2f sclae_V=%.2f\n" %(
            cls.scale_H, cls.scale_V)))
        
        setEdges = set()
        
        for A in lCluster:
            for edge_type, lLinked in A.dsEdge.items():
                for B in lLinked:
                    if A.cnt >= B.cnt: continue
                    if (A, B, edge_type) not in setEdges:
                        # ok, let's add the edge A <--> B
                        ndEdge = PageXml.createPageXmlNode("ClusterEdge")
                        ndEdge.set("src", A.name)
                        ndEdge.set("tgt", B.name)
                        ndEdge.set("type", edge_type)
                        ndEdge.set("algo", sAlgo)
                        if True:
                            ptA = A.shape.representative_point()
                            ptB = B.shape.representative_point()
                        
                        else:
                            ptA, ptB = shapely.ops.nearest_points(A.shape, B.shape)
                        PageXml.setPoints(ndEdge, list(ptA.coords) + list(ptB.coords)) 
                        ndEdge.tail = "\n"
                        ndPage.append(ndEdge)
                        
                        setEdges.add((A, B, edge_type))
                        cnt += 1
        del setEdges
        
        return cnt

    @classmethod
    def removeEdgesFromXml(cls, ndPage):
        """
        Given an algo, remove all its clusters from a page 
        """
        i = 0
        for nd in ndPage.xpath(xpClusterEdge, namespaces=dNS):
            ndPage.remove(nd)
            i += 1
        return i

    @classmethod
    def link(cls, A, B, edge_type=""):
        """
        record an edge between those 2 clusters
        """
        assert A != B
        A.dsEdge[edge_type].add(B)
        B.dsEdge[edge_type].add(A)

    @classmethod
    def computeClusterBoundingBox(cls, lCluster):
        for c in lCluster:
            c.setBB(c.shape.bounds)   
            assert c.x1 < c.x2        
            assert c.y1 < c.y2              
            if cls.scale_H != 1 or cls.scale_V != 1:
                c.scaled_shape = affinity.scale(c.shape, xfact=cls.scale_H, yfact=cls.scale_V)
            else:    
                c.scaled_shape = c.shape

    @classmethod
    def setTableAttribute(self, ndPage, setID, sAttr1, s1, sAttr2=None, s2=None):
        """
        set attributes such as "col" and "colSPan" of a set of objects given by their ID
        """
        lNode = [ndPage.xpath(".//*[@id='%s']"%_id, namespaces=dNS)[0] for _id in setID]
        for nd in lNode:
            nd.set(sAttr1, str(s1))
            if bool(sAttr2): 
                nd.set(sAttr2, str(s2))
                   
    @classmethod
    def tabulate(cls, ndPage, lCluster, bVerbose=False):
        """
        Top-down tabulation in the 4 directions
        """
        
        cls.tabulate_top_down(lCluster)
        for c in lCluster: 
            c.row1 = c.minrow
            c.node.set("row", str(c.row1))
        maxRow = max(c.row1 for c in lCluster)
            #c.node.set("col", str(c.mincol))
            #c.node.set("rowSpan", str(c.maxrow - c.minrow + 1))
            #c.node.set("colSpan", str(c.maxcol - c.mincol + 1))
        
        cls.rotateClockWise90deg(lCluster, bVerbose=bVerbose)
        cls.tabulate_top_down(lCluster)
        for c in lCluster: 
            c.col1 = c.minrow
            c.node.set("col", str(c.col1))
        maxCol = max(c.col1 for c in lCluster)
 
        cls.rotateClockWise90deg(lCluster, bVerbose=bVerbose)
        cls.tabulate_top_down(lCluster)
        for c in lCluster: 
            c.row2 = maxRow - c.minrow
            rowSpan = str(1 + c.row2 - c.row1)
            c.node.set("rowSpan", rowSpan)
            cls.setTableAttribute(ndPage, c.getSetID(), "row", c.row1, "rowSpan", rowSpan)

        cls.rotateClockWise90deg(lCluster, bVerbose=bVerbose)
        cls.tabulate_top_down(lCluster)
        for c in lCluster: 
            c.col2 = maxCol - c.minrow
            colSpan = str(1 + c.col2 - c.col1)
            c.node.set("colSpan", colSpan)
            cls.setTableAttribute(ndPage, c.getSetID(), "col", c.col1, "colSpan", colSpan)
 
    @classmethod
    def tabulate_rows(cls, ndPage, lCluster, bVerbose=False):
        """
        Top-down and bottom-up tabulations
        """
       
        cls.tabulate_top_down(lCluster)

        maxRow = max(c.minrow for c in lCluster)
        traceln("   maxRow=", maxRow)
        
#         if False:
#             for c in lCluster: 
#                 c.row1 = c.minrow
#                 c.node.set("row", str(c.row1))
#             cls.rotateClockWise180deg(lCluster, bVerbose=bVerbose)
#             cls.tabulate_top_down(lCluster)
#             for c in lCluster: 
#                 c.row2 = max(maxRow - c.minrow, c.row1)
#                 rowSpan = str(1 + c.row2 - c.row1)
#                 c.node.set("rowSpan", rowSpan)
#                 cls.setTableAttribute(ndPage, c.getSetID(), "row", c.row1, "rowSpan", rowSpan)
#         elif False:
#             for c in lCluster: 
#                 c.node.set("row", str(c.minrow))
#                 rowSpan = str(9)
#                 c.node.set("rowSpan", rowSpan)
#                 cls.setTableAttribute(ndPage, c.getSetID(), "row", c.minrow, "rowSpan", rowSpan)
        # tabulate top-down, then compute the separators and use them for
        # deciding the row and rowSpan
        # we get a list of linear separators, to be reflected as SeparatorRegion
        cls.map_to_rows(ndPage, maxRow, lCluster)
        
        for c in lCluster: 
            c.node.set("row", str(c.row1))
            rowSpan = str(1 + c.row2 - c.row1)
            c.node.set("rowSpan", rowSpan)
            cls.setTableAttribute(ndPage, c.getSetID(), "row", c.row1, "rowSpan", rowSpan)

    
    @classmethod
    def use_cut_columns(cls, ndPage):
        """
        use the name of the cut cluster to compute the col
        colSPan is always 1 in that case
        """
        #<Cluster name="0" algo="cut" content="" cut_X="330">
        for ndCluster in ndPage.xpath(xpCluster+"[@algo='cut']", namespaces=dNS):
            col = str(int(ndCluster.get("name")) - 1)
            setID = set(ndCluster.get("content").split())
            ndCluster.set("col", col)
            ndCluster.set("colSpan", "1")
            cls.setTableAttribute(ndPage, setID, "col", col, "colSpan", "1")
                
    @classmethod
    def tabulate_top_down(cls, lCluster):
        """
        compute minrow and maxrow values
        """        
        for c in lCluster:
            assert c.x1 <= c.x2
            assert c.y1 <= c.y2
        
        step = 1
        step_max = len(lCluster) + 1
        
        for c in lCluster: c.minrow = -1

        lTodoCluster = lCluster
        prevSetUpdated = None
        bNoLoop = True
        while lTodoCluster and bNoLoop:  
            setUpdated = set()           
            traceln("  - STEP %d"%step)
            # since we keep increasing the minrow, its maximum value cannot 
            # exceed len(lCluster), which is reached with at most step_max steps
            assert step <= step_max, "algorithm error"
            
            # visit all vertically from top cluster
            lTodoCluster.sort(key=lambda o: o.y1)
            # faster?? lCurCluster.sort(key=operator.attrgetter("x1"))
#             print([c.name for c in lTodoCluster])
#             for i in [0, 1]:
#                 print(lCluster[i].name, " y1=", lCluster[i].y1, " y2=", lCluster[i].y2)
            for c in lTodoCluster:
                setUpdated.update(c.visitStackDown(0))
            # visit all, horizontally from leftest clusters
            lTodoCluster.sort(key=lambda o: o.x1)
            for c in lTodoCluster:
                setUpdated.update(c.visitPeerRight())

            lTodoCluster.sort(key=lambda o: o.x2, reverse=True)
            for c in lTodoCluster:
                setUpdated.update(c.visitPeerLeft())
            
            if not prevSetUpdated is None and prevSetUpdated == setUpdated:
                traceln(" - loop detected - stopping now.")
                bNoLoop = False
            prevSetUpdated = setUpdated
            lTodoCluster = list(setUpdated)
            traceln("  ... %d updated" % len(lTodoCluster))
            step += 1
        
        if not bNoLoop:
            # need to fix the problem...
            # because of the loop, we have holes in the list of row numbers
            lMinrow = list(set(c.minrow for c in lCluster))
            lMinrow.sort()
            curRow = 0
            for iMinrow in range(len(lMinrow)):
                minrow = lMinrow[iMinrow]
                if minrow > curRow:
                    # missing row number...
                    delta = minrow - curRow
                    for c in lCluster:
                        if c.minrow >= curRow: 
                            c.minrow -= delta
                    for j in range(iMinrow, len(lMinrow)):
                        lMinrow[j] = lMinrow[j] - delta
                curRow += 1
        
    def visitStackDown(self, minrow, setVisited=set()):
        """
        e.g., stacking from top to bottom, we get a visit from upward, so we update our minrow accordingly
        return the set of updated items
        """
        #if self.name == "(6_I_agglo_345866)" and minrow > 17: print(self.name, minrow)
        setUpdated = set()
         
        if minrow > self.minrow:
            # the stack above us tells us about our minrow!
            self.minrow = minrow
            setUpdated.add(self)
            
        for c in self.dsEdge["VerticalEdge"]:
            # make sure we go downward
            # if c.y1 > self.y1:
            # and that the edge is a valid one
            # which implies the 1st condition!
            if self.y2 < c.y1:
                if self.minrow >= c.minrow:
                    # otherwise no need...
                    setUpdated.update(c.visitStackDown(self.minrow + 1, setVisited))
            elif self.y1 < c.y1:
                # c starts within self...
                # maybe there is skewing?
                if self.scaled_shape.intersects(c.scaled_shape):
                    # since we do not increase minrow, we need to make sure
                    # we do not infinite loop...
                    # (increasing minrow forces us to move downward the page and to end at some point)
                    if self.minrow > c.minrow or not self in setVisited:
                        setVisited.add(self)
                        setUpdated.update(c.visitStackDown(self.minrow, setVisited))
                else:
                    # I believe one is mostly above the other
                    if self.minrow >= c.minrow:
                        setUpdated.update(c.visitStackDown(self.minrow + 1, setVisited))
                 
        return setUpdated
 
    def visitPeerRight(self):
        """
        go from left to right, making sure the minrow is consistent with the geometric relationships 
        """
        setUpdated = set()
        a = self
        for b in self.dsEdge["HorizontalEdge"]:
            # make sure we go in good direction: rightward
            if a.x2 <= b.x1:
                minrow = max(a.minrow, b.minrow)
                bAB = TableCluster.isTopAligned(a, b)   # top justified
                bA = bAB or a.y1 > b.y1         # a below b 
                bB = bAB or a.y1 < b.y1         # a above b
                
                if bA and minrow > a.minrow:
                    a.minrow = minrow
                    setUpdated.add(a)
                    
                if bB and minrow > b.minrow:
                    b.minrow = minrow
                    setUpdated.add(b)
                setUpdated.update(b.visitPeerRight())
        return setUpdated
            
    def visitPeerLeft(self):
        """
        go from left to right, making sure the minrow is consistent with the geometric relationships 
        """
        setUpdated = set()
        a = self
        for b in self.dsEdge["HorizontalEdge"]:
            # make sure we go in good direction: leftward
            if b.x2 <= a.x1:
                minrow = max(a.minrow, b.minrow)
                bAB = TableCluster.isTopAligned(a, b)   # top justified
                bA = bAB or a.y1 > b.y1         # a below b 
                bB = bAB or a.y1 < b.y1         # a above b
                
                if bA and minrow > a.minrow:
                    a.minrow = minrow
                    setUpdated.add(a)
                    
                if bB and minrow > b.minrow:
                    b.minrow = minrow
                    setUpdated.add(b)
                setUpdated.update(b.visitPeerRight())
                
        return setUpdated

    @classmethod
    def isTopAligned(cls, a, b):
        return abs(a.y1 - b.y1) < cls.thTopAligned
    

    @classmethod
    def rotateClockWise90deg(cls, lCluster, bVerbose=True):
        if bVerbose: traceln(" -- rotation 90° clockwise")
        for c in lCluster:
            c.x1, c.y1, c.x2, c.y2 = -c.y2, c.x1, -c.y1, c.x2
            c.dsEdge["HorizontalEdge"], c.dsEdge["VerticalEdge"]  = c.dsEdge["VerticalEdge"], c.dsEdge["HorizontalEdge"]
        return
    
    @classmethod
    def rotateClockWise180deg(cls, lCluster, bVerbose=True):
        if bVerbose: traceln(" -- rotation 180° clockwise")
        for c in lCluster:
            c.x1, c.y1, c.x2, c.y2 = -c.x2, -c.y2, -c.x1, -c.y1
        return

    @classmethod
    def map_to_rows(cls, ndPage, maxRow, lCluster):
        """
        find lienar separators separating rows
        """
        # reflect each cluster by the highest point (highest ending points of baselines)
        dMinYByRow = defaultdict(lambda :9999999999)
        n = 2 * sum(len(c) for c in lCluster)
        X = np.zeros(shape=(n, 2))  # x,y coordinates
        i = 0
        for c in lCluster:
            c.maxY = -1
            c.minY = 9999999999
            for _id in c.getSetID():
                """
                <TextLine id="r1l5" custom="readingOrder {index:4;}" DU_cluster="0" row="0" rowSpan="1" col="0" colSpan="1">
                  <Coords points="217,688 245,685 273,685 301,688 329,690 358,689 358,646 329,647 301,645 273,642 245,642 217,645"/>
                  <Baseline points="217,688 245,685 273,685 301,688 329,690 358,689"/>
                  <TextEquiv><Unicode>ung.</Unicode></TextEquiv>
                </TextLine>
                 """
                nd = ndPage.xpath(".//*[@id='%s']/pg:Baseline"%_id, namespaces=dNS)[0]
                ls = ShapeLoader.node_to_LineString(nd)
                pA, pB = ls.boundary.geoms
                minY = min(pA.y, pB.y)
                c.minY = min(c.minY, minY)
                c.maxY = max(c.maxY, max((pA.y, pB.y)))
                dMinYByRow[c.minrow] = min(dMinYByRow[c.minrow], minY)
                # for the linear separators
                X[i,:] = (pA.x, pA.y)
                i = i + 1
                X[i,:] = (pB.x, pB.y)
                i = i + 1
                
        # check consistency
        for c in lCluster:
            for i in range(maxRow, c.minrow, -1):
                if c.minY > dMinYByRow[i]:
                    assert c.minrow < i
                    # how possible??? fix!!
                    c.minrow = i
                    break
 
        # compute row1 and row2
        for c in lCluster:
            c.row1 = c.minrow
            c.row2 = c.minrow
            for i in range(0, maxRow+1):
                if c.maxY > dMinYByRow[i]:
                    c.row2 = i
                else:
                    break
               
        # now compute maxRow - 1 separators!
        w = float(ndPage.get("imageWidth"))
        Y = np.zeros(shape=(n,))    # labels
#         lAB = [getLinearSeparator(X, np.clip(Y, row, row+1)) 
#                for row in range(maxRow-1)]
        
        for nd in ndPage.xpath(".//pg:SeparatorRegion[@algo]", namespaces=dNS):
            ndPage.remove(nd)
        
        for row in range(maxRow+1):
            Y0 = dMinYByRow[row] - 20
            Yw = Y0
            ndSep = PageXml.createPageXmlNode("SeparatorRegion")
            ndSep.set("algo", "tabulate_rows")
            ndCoords = PageXml.createPageXmlNode("Coords")
            ndCoords.set("points", "%d,%d %d,%d" %(0, Y0, w, Yw))
            ndSep.append(ndCoords)
            ndSep.tail = "\n"
            ndPage.append(ndSep)
        
        return 


def main(sInputDir, sAlgo, bCol=False, scale_H=None, scale_V=None, bVerbose=False):
    
    if not scale_H is None: TableCluster.scale_H = scale_H
    if not scale_V is None: TableCluster.scale_V = scale_V
    
    traceln("scale_H=", TableCluster.scale_H)
    traceln("scale_V=", TableCluster.scale_V)
    
    # filenames without the path
    lsFilename = [os.path.basename(name) for name in os.listdir(sInputDir) if name.endswith("_du.pxml") or name.endswith("_du.mpxml")]
    traceln(" - %d files to process, to tabulate clusters '%s'" % (
        len(lsFilename)
        , sAlgo))
    lsFilename.sort()
    for sFilename in lsFilename:
        sFullFilename = os.path.join(sInputDir, sFilename)
        traceln(" -------- FILE : ", sFullFilename)
        cnt = 0
        doc = etree.parse(sFullFilename)
        
        for iPage, ndPage in enumerate(doc.getroot().xpath(xpPage, namespaces=dNS)):
            lCluster = TableCluster.load(ndPage, sAlgo, bNode=True)  # True to keep a pointer to the DOM node
        
            if bVerbose:
                trace(" --- Page %d : %d cluster '%s' " %(iPage+1, len(lCluster), sAlgo))
            if len(lCluster) == 0:
                traceln("*** NO cluster '%s' *** we keep this page unchanged"%sAlgo)
                continue
            _nbRm = TableCluster.removeEdgesFromXml(ndPage)
            if bVerbose:
                traceln("\n  %d ClusterEdge removed"%_nbRm)
            
            TableCluster.computeClusterBoundingBox(lCluster)
            
            if True:
                # edges are better this way!
                lBB = []
                for c in lCluster: 
                    lBB.append(c.getBB())
                    c.scale(TableCluster.scale_H, TableCluster.scale_V)
                TableCluster.computeClusterEdge(ndPage, lCluster)
                for c, bb in zip(lCluster, lBB):
                    c.setBB(bb) 
                # for c in lCluster: c.scale(1.0/TableCluster.scale_H, 1.0/TableCluster.scale_V)
            else:
                # compute inter- cluster edges from inter- cluster-item edges  
                TableCluster.induceClusterEdge(ndPage, lCluster)

            # store inter-cluster edges
            cntPage = TableCluster.addEdgesToXml(ndPage, sAlgo, lCluster)
            if bVerbose:
                traceln("    %d inter-cluster edges   " %(cntPage))

            # compute min/max row/col for each cluster
            # WARNING - side effect on lCluster content and edges
            if bCol:
                TableCluster.tabulate(ndPage, lCluster, bVerbose=bVerbose)
            else:
                TableCluster.tabulate_rows(ndPage, lCluster, bVerbose=bVerbose)
                TableCluster.use_cut_columns(ndPage)
                            
            cnt += cntPage
        traceln("%d inter-cluster edges" %(cnt))
            
        
        doc.write(sFullFilename,
          xml_declaration=True,
          encoding="utf-8",
          pretty_print=True
          #compression=0,  #0 to 9
          )        
        
        del doc
        
    traceln(" done   (%d files)" % len(lsFilename))



# ----------------------------------------------------------------------------
if __name__ == "__main__":
    
    version = "v.01"
    sUsage="""
Tabulate the clusters from given @algo and compute the row, col, row_span, col_span attributes of each cluster

Usage: %s <sInputDir> <algo>
   
""" % (sys.argv[0])

    parser = OptionParser(usage=sUsage)
    parser.add_option("--scale_h", dest='fScaleH',  action="store", type="float"
                      , help="objects are horizontally scaled by this factor")   
    parser.add_option("--scale_v", dest='fScaleV',  action="store", type="float"
                      , help="objects are vertically scaled by this factor")   
    parser.add_option("--col", dest='bCol',  action="store_true"
                      , help="Columns also tabulated instead of derived from 'cut' clusters")   
    parser.add_option("-v", "--verbose", dest='bVerbose',  action="store_true"
                      , help="Verbose mode")   
    (options, args) = parser.parse_args()
    
    try:
        sInputDir, sA = args
    except ValueError:
        sys.stderr.write(sUsage)
        sys.exit(1)
    
    # ... checking folders
    if not os.path.normpath(sInputDir).endswith("col")  : sInputDir = os.path.join(sInputDir, "col")

    if not os.path.isdir(sInputDir): 
        sys.stderr.write("Not a directory: %s\n"%sInputDir)
        sys.exit(2)
    
    # ok, go!
    traceln("Input  is : ", os.path.abspath(sInputDir))
    traceln("algo is : ", sA)
    if options.bCol:
        traceln("columns also tabulated")
    else:
        traceln("columns are those of projection profile")
        
    main(sInputDir, sA, bCol=options.bCol
         , scale_H=options.fScaleH, scale_V=options.fScaleV
         , bVerbose=options.bVerbose)
    
    traceln("Done.")