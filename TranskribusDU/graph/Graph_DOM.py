# -*- coding: utf-8 -*-

"""
    Computing the graph for a XML document
    
    Copyright Xerox(C) 2016, 2019 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import math

from lxml import etree
import shapely.geometry as geom

from common.trace import traceln
from util import XYcut
from .Graph import Graph
from . import Edge
from xml_formats.PageXml import PageXml, MultiPageXml


class Graph_DOM(Graph):
    """
    Graph for DOM input
    """
    # --- NODE TYPES and LABELS
    _lNodeType       = []       #the list of node types for this class of graph
    _bMultitype      = False    # equivalent to len(_lNodeType) > 1
    _dLabelByCls     = None     #dictionary across node types
    _dClsByLabel     = None     #dictionary across node types
    _nbLabelTot      = 0        #total number of labels

    sIN_FORMAT  = "XML"   # tell here which input format is expected
    sOUTPUT_EXT = ".mpxml"

    def __init__(self, lNode = [], lEdge = []):
        Graph.__init__(self, lNode, lEdge)
    
    def parseDocFile(self, sFilename, iVerbose=0):
        """
        Load that document as a CRF Graph.
        Also set the self.doc variable!
        
        Return a CRF Graph object
        """
    
        self.doc = etree.parse(sFilename)
        self.lNode, self.lEdge = list(), list()
        #load the block of each page, keeping the list of blocks of previous page
        lPrevPageNode = None

        for pnum, page, domNdPage in self._iter_Page_DocNode(self.doc):
            #now that we have the page, let's create the node for each type!
            lPageNode = list()
            setPageNdDomId = set() #the set of DOM id
            # because the node types are supposed to have an empty intersection
                            
            lPageNode = [nd for nodeType in self.getNodeTypeList() for nd in nodeType._iter_GraphNode(self.doc, domNdPage, page) ]
            
            #check that each node appears once
            setPageNdDomId = set([nd.domid for nd in lPageNode])
            assert len(setPageNdDomId) == len(lPageNode), "ERROR: some nodes fit with multiple NodeTypes"
            
        
            self.lNode.extend(lPageNode)
            
            lPageEdge = Edge.Edge.computeEdges(lPrevPageNode, lPageNode, self.iGraphMode)
            
            self.lEdge.extend(lPageEdge)
            if iVerbose>=2: traceln("\tPage %5d    %6d nodes    %7d edges"%(pnum, len(lPageNode), len(lPageEdge)))
            
            lPrevPageNode = lPageNode
        if iVerbose: traceln("\t\t (%d nodes,  %d edges)"%(len(self.lNode), len(self.lEdge)) )
        
        return self

    def addEdgeToDoc(self, Y=None, ndPage=None):
        """
        To display the graph conveniently we add new Edge elements
        """
        assert Y == None
        if self.lNode:
            ndPage = self.lNode[0].page.node if ndPage is None else ndPage    
            ndPage.append(etree.Comment("Edges added to the XML for convenience"))
            for edge in self.lEdge:
                A, B = edge.A , edge.B   #shape.centroid, edge.B.shape.centroid
                ndEdge = PageXml.createPageXmlNode("Edge")
                ndEdge.set("src", edge.A.node.get("id"))
                ndEdge.set("tgt", edge.B.node.get("id"))
                ndEdge.set("type", edge.__class__.__name__)
                ndEdge.tail = "\n"
                ndPage.append(ndEdge)
                PageXml.setPoints(ndEdge, [(A.x1, A.y1), (B.x1, B.y1)]) 
        
        return         

    @classmethod
    def saveDoc(cls, sFilename, doc, _lg, sCreator, sComment):
        """
        _lg is not used since we have enriched the DOC (in doc parameter)
        """
        # build a decent output filename
        sDUFilename = cls.getOutputFilename(sFilename)
        
        MultiPageXml.setMetadata(doc, None, sCreator, sComment)
  
        doc.write(sDUFilename,
                  xml_declaration=True,
                  encoding="utf-8",
                  pretty_print=True
                  #compression=0,  #0 to 9
                  )
        return sDUFilename
    
    def detachFromDoc(self):
        """
        Detach the graph from the DOM node, which can then be freed
        """
        if self.doc != None:
            for nd in self.lNode: nd.detachFromDOM()
            self.doc = None
 
    @classmethod
    def exportToDom(cls, lg, bGraph=False):
        """
            export a set of graph as (Multi)PageXml
        """
        #create document
        pageDoc,pageNode = PageXml.createPageXmlDocument('graph2DOM', filename="",imgW=0, imgH=0)
        pageW, pageH = 0,0
        for iG,g in enumerate(lg):
            if iG > 0:
                pageNode = PageXml.createPageXmlNode("Page")
                pageDoc.getroot().append(pageNode)
                
#             lRegions = g.lCluster
#             for region in lRegions:
#                 lNodes = [g.lNode[idx] for idx in region]
                
            if g.lCluster:
                lRegionNodes = [ [g.lNode[idx] for idx in region] for region in g.lCluster ]
            else:
                lRegionNodes = [ g.lNode ]
            for iR, lNodes in enumerate(lRegionNodes):
                #regionType = region.type
                regionNode = PageXml.createPageXmlNode("TextRegion")
                regionNode.set("id", "R%d"%iR)
                pageNode.append(regionNode)
                
                #regionNode.set("type",str(regionType))
                # Region geometry
                mp = geom.MultiPolygon(b.getShape() for b in lNodes)
                xmin, ymin, xmax, ymax = geom.MultiPolygon(b.getShape() for b in lNodes).bounds
                coordsNode = PageXml.createPageXmlNode('Coords')
                regionNode.append(coordsNode)
                PageXml.setPoints(coordsNode,[(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)])
                
                # update page dimension
                pageW = max(pageW, xmax)
                pageH = max(pageH, ymax)

                lSegment = [(o.y1, o.y2, o) for o in lNodes]
                lLines,_,_ = XYcut.mergeSegments(lSegment, 2)
                for iL, (ymin,ymax,lw) in enumerate(lLines):
                    xmin, ymin, xmax, ymax = geom.MultiPolygon(b.getShape() for b in lw).bounds
                    textLineNode = PageXml.createPageXmlNode('TextLine')
                    textLineNode.set("id", "R%d_L%d" % (iR, iL))

                    coordsNode = PageXml.createPageXmlNode('Coords')
                    textLineNode.append(coordsNode)
                    PageXml.setPoints(coordsNode,[(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)])
                    regionNode.append(textLineNode)
                    
                    for iW, w in enumerate(lw):
                        wordNode = PageXml.createPageXmlNode('Word')
                        # standard Block attributes for DOM objects, so that addEdgeTODoc wan work
                        w.node = wordNode
                        w.domid = "R%d_L%d_W%d" % (iR, iL, iW)
                        wordNode.set("id", w.domid)
                        textLineNode.append(wordNode)
                        coordsNode = PageXml.createPageXmlNode('Coords')
                        wordNode.append(coordsNode)
                        PageXml.setPoints(coordsNode,w.shape.exterior.coords)
                        
                        textEquiv=  PageXml.createPageXmlNode('TextEquiv')
                        wordNode.append(textEquiv)
                        unicode =   PageXml.createPageXmlNode('Unicode')
                        textEquiv.append(unicode)
                        unicode.text=w.text
            # end of page
            pageNode.set('imageWidth' , str(math.ceil(pageW)))
            pageNode.set('imageHeight', str(math.ceil(pageH)))
            pageW, pageH = 0,0
            
            if bGraph:
                g.addEdgeToDoc(g, ndPage=pageNode)
                    
        return pageDoc   