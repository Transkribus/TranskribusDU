# -*- coding: utf-8 -*-

"""
We expect XML file with cluster defined by several algo.
For each Page:
    We intersect the cluster of one algo with cluster of the other and 
    We generate new clusters named after the algo names, e.g. (A_I_B)

Overwrite the input XML files, adding new cluster definitions

Created on 9/9/2019

Copyright NAVER LABS Europe 2019

@author: JL Meunier
"""

import sys, os
from optparse import OptionParser

from lxml import etree

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln, trace
from util.Shape import ShapeLoader
from xml_formats.PageXml import PageXml

# ----------------------------------------------------------------------------
xpCluster   = ".//pg:Cluster"
# sFMT        = "(%s_∩_%s)"  pb with visu
sFMT        = "(%s_I_%s)"
sAlgoAttr   = "algo"
xpPage      = ".//pg:Page"
dNS = {"pg":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
# ----------------------------------------------------------------------------

class Cluster:
    cnt = 0
    
    def __init__(self, name, setID, shape=None):
        self.name   = name
        self.setID  = setID
        self.shape = shape
        # self.node = ...  the load method can set a .node attribute pointing to the DOM node
         
    def getSetID(self): return self.setID
    
    def __len__(self): return len(self.setID)
    
    @classmethod
    def remove(cls, ndPage, sAlgo):
        """
        Given an algo, remove all its clusters from a page 
        """
        i = 0
        for nd in ndPage.xpath(xpCluster+"[@%s='%s']"%(sAlgoAttr, sAlgo)
                           , namespaces=dNS):
            ndPage.remove(nd)
            i += 1
        return i
            
    @classmethod
    def load(cls, ndPage, sAlgo, bNode=False, sLvl=None):
        """
        Given an algo, load all its cluster from the page.
        Compute their shape, if not provided in the XML, as a minimum rotated rectangle
        """
        l = []
        if sLvl is None:
            xpath = xpCluster+"[@%s='%s']"%(sAlgoAttr, sAlgo)
        else:
            xpath = xpCluster+"[@%s='%s' and @level='%s']"%(sAlgoAttr, sAlgo,sLvl) 
        for nd in ndPage.xpath(xpath
                                                    , namespaces=dNS):
            c = cls.loadClusterNode(ndPage, nd, sAlgo)
            if not c is None: 
                if bNode: c.node = nd
                l.append(c)
        return l
    
    @classmethod
    def loadClusterNode(cls, ndPage, nd, sAlgo, bComputeShape=True):
        """
        Load a cluster from its XML node
        Compute its shape, if not provided in the XML, as a minimum rotated rectangle       
        """
        name  = nd.get("name")
        if name is None: 
            name = "%s_%d"%(sAlgo, cls.cnt)
            cls.cnt += 1
            nd.set("name", name)
        setID = set(nd.get("content").split())
        if bool(setID):
            try:
                shape = ShapeLoader.node_to_Polygon(nd)
            except IndexError:
                if bComputeShape: 
                    shape = cls.computeShape(ndPage, setID)
                else:
                    shape = None
            return cls(name, setID, shape)
        else:
            return None

    @classmethod
    def store(cls, ndPage, lCluster, sAlgo):
        """
        Store those "qlgo" clusters in the page node
        """
        ndPage.append(etree.Comment("\nClusters created by cluster intersection\n"))

        for c in lCluster:
            ndPage.append(c.makeClusterNode(sAlgo))
            
    def makeClusterNode(self, sAlgo):
        """
        Create an XML node reflecting the cluster
        """
        ndCluster = PageXml.createPageXmlNode('Cluster')  
        ndCluster.set("name", self.name)   
        ndCluster.set("algo", sAlgo)   
        # add the space separated list of node ids
        ndCluster.set("content", " ".join(self.setID))   
        ndCoords = PageXml.createPageXmlNode('Coords')        
        ndCluster.append(ndCoords)
        if self.shape is None:
            ndCoords.set('points', "")
        else:                     
            ndCoords.set('points', ShapeLoader.getCoordsString(self.shape)) 
        ndCluster.tail = "\n"                    
        return ndCluster

    @classmethod
    def intersect(cls, one, other):
        """
        return None or a cluster made by intersecting two cluster
        the shape of the intersection if the intersection of shapes, or None if not applicable
        """
        setID = one.setID.intersection(other.setID)
        if bool(setID):
            try:
                shapeInter = one.shape.intersection(other.shape)
            except ValueError:
                shapeInter = None
            return cls(sFMT % (one.name, other.name), setID, shapeInter)
        else:
            return None
    
    @classmethod
    def computeShape(cls, ndPage, setID, bConvexHull=False):
        """ 
        compute a shape for this cluster, as the minimum rotated rectangle of its content
        or optionally as the convex hull
        """
        # let's find the nodes and compute the shape
        lNode = [ndPage.xpath(".//*[@id='%s']"%_id, namespaces=dNS)[0] for _id in setID]
        return       ShapeLoader.convex_hull(lNode, bShapelyObject=True)   \
                if bConvexHull                                             \
                else ShapeLoader.minimum_rotated_rectangle(lNode, bShapelyObject=True)
                

def main(sInputDir, sAlgoA, sAlgoB, bShape=False, bConvexHull=False, bVerbose=False):
    sAlgoC = sFMT % (sAlgoA, sAlgoB)
       
    # filenames without the path
    lsFilename = [os.path.basename(name) for name in os.listdir(sInputDir) if name.endswith("_du.pxml") or name.endswith("_du.mpxml")]
    traceln(" - %d files to process, to produce clusters '%s'" % (
        len(lsFilename)
        , sAlgoC))
        
    for sFilename in lsFilename:
        sFullFilename = os.path.join(sInputDir, sFilename)
        traceln(" - FILE : ", sFullFilename)
        cntCluster, cntPage = 0, 0
        doc = etree.parse(sFullFilename)
        
        for iPage, ndPage in enumerate(doc.getroot().xpath(xpPage, namespaces=dNS)):
            nRemoved = Cluster.remove(ndPage, sAlgoC)
            
            lClusterA = Cluster.load(ndPage, sAlgoA)
            lClusterB = Cluster.load(ndPage, sAlgoB)
        
            if bVerbose:
                trace("Page %d : (%d clusters REMOVED),   %d cluster '%s'   %d clusters '%s'" %(iPage+1
                  , nRemoved
                  , len(lClusterA), sAlgoA
                  , len(lClusterB), sAlgoB))
            
            lClusterC = []
            for A in lClusterA:
                for B in lClusterB:
                    C = Cluster.intersect(A, B)
                    if not C is None:
                        lClusterC.append(C)
             
            if bVerbose: traceln( "    -> %d clusters" % (len(lClusterC)))
            if bShape or bConvexHull:
                for c in lClusterC: 
                    c.shape = Cluster.computeShape(ndPage, c.setID, bConvexHull=bConvexHull)
                
            cntCluster += len(lClusterC)
            cntPage += 1
            
            Cluster.store(ndPage, lClusterC, sAlgoC)
        
        doc.write(sFullFilename,
          xml_declaration=True,
          encoding="utf-8",
          pretty_print=True
          #compression=0,  #0 to 9
          )        
        
        del doc
        traceln(" %d clusters over %d pages" % (cntCluster, cntPage))
        
    traceln(" done   (%d files)" % len(lsFilename))



# ----------------------------------------------------------------------------
if __name__ == "__main__":
    
    version = "v.01"
    sUsage="""
Produce the intersection of two types of clusters, selected by their @algo attrbute.

Usage: %s <sInputDir> <algoA> <algoB> 
   
""" % (sys.argv[0])

    parser = OptionParser(usage=sUsage)
    parser.add_option("-v", "--verbose", dest='bVerbose',  action="store_true"
                      , help="Verbose mode")   
    parser.add_option("-s", "--shape", dest='bShape',  action="store_true"
                      , help="Compute the shape of the intersection content as minimum rotated rectangle, instead of intersection of shapes")   
    parser.add_option("--hull", dest='bConvexHull',  action="store_true"
                      , help="Compute the shape of the intersection content as convex hull, instead of intersection of shapes")   
    (options, args) = parser.parse_args()
    
    try:
        sInputDir, sA, sB = args
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
    traceln("algo A is : ", sA)
    traceln("algo B is : ", sB)
    if options.bShape or options.bConvexHull:
        traceln("Shape of intersections based on content!")
    else:
        traceln("Shape of intersections is the intersection of shapes!")

    main(sInputDir, sA, sB, options.bShape, options.bConvexHull, options.bVerbose)
    
    traceln("Input  was : ", os.path.abspath(sInputDir))
    traceln("algo A was : ", sA)
    traceln("algo B was : ", sB)
    if options.bShape or options.bConvexHull:
        trace("Shape of intersections based on content: ")
        if options.bConvexHull:
            traceln(" as a convex hull")
        else:
            traceln(" as a minimum rotated rectangle")
    else:
        traceln("Shape of intersections is the intersection of shapes!")

    traceln("Done.")