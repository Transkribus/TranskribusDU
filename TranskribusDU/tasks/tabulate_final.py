# -*- coding: utf-8 -*-

"""
We expect XML file with TextLine having the row, col, rowSpan, colSpan attributes

For each Page:
    We delete any empty table (or complain if not empty)
    We select TextLine with rowSPan=1 and colSpan=1
    We create one cell for each pair of row and col number
    We inject the TexLine into its cell
    We create a TableRegion to contain the cells
    We delete empty regions
    We resize non-empty regions
    
We compute the cell and table geometries and store them.

Created on 21/10/2019

Copyright NAVER LABS Europe 2019

@author: JL Meunier
"""

import sys, os
from optparse import OptionParser
from collections import defaultdict
from lxml import etree

from shapely.ops import cascaded_union        
from shapely.geometry import Polygon
from util.XYcut import mergeSegments
try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln
from xml_formats.PageXml import PageXml

from util.Shape import ShapeLoader

# ----------------------------------------------------------------------------
xpPage      = ".//pg:Page"
dNS = {"pg":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
# ----------------------------------------------------------------------------


def processRegions(ndPage,bVerbose=False):
    """
        Delete empty regions
        resize no empty regions
        delete TextEquiv
    """
    lDel=[]
    lndRegions =  ndPage.xpath(".//pg:TextRegion", namespaces=dNS)
    for ndRegion in lndRegions:

        [ndRegion.remove(te) for te in ndRegion.xpath("./pg:TextEquiv", namespaces=dNS) ]

        lTL=  ndRegion.xpath(".//pg:TextLine", namespaces=dNS)
        if lTL == []:
            # to be deleted
            lDel.append(ndRegion)
        else:
            #resize it
            oHull = ShapeLoader.convex_hull(lTL, bShapelyObject=True)
            PageXml.getChildByName(ndRegion,'Coords')[0].set("points", ShapeLoader.getCoordsString(oHull, bFailSafe=True))
#             contour = cascaded_union([p if p.is_valid else p.convex_hull for p in lTL ]) 
#             o = contour.minimum_rotated_rectangle
#             ndRegion.getChildByName('Coords').set("points", ShapeLoader.getCoordsString(o, bFailSafe=True))

    # delete empty regions
    [ ndRegion.getparent().remove(ndRegion) for ndRegion in lDel]
    
    if bVerbose:
        traceln(" - %d regions deleted"%(len(lDel)))
        traceln(" - %d regions updated"%(len(lndRegions) - len(lDel)))

class TableRegion:
    
    def __init__(self, pagenum, tablenum):
        self.pagenum  = pagenum
        self.tablenum = tablenum
        # (row, col) -> list of nodes
        self._dCellNd  = defaultdict(list)
        
    def addToCell(self, row, col, nd):
        self._dCellNd[(row, col)].append(nd)

    def makeTableNode(self):
        """
        Make a DOM tree for this table
        """    
        lK = self._dCellNd.keys()            
        lRow = list(set(_row for _row, _col in lK))
        lRow.sort()
        lCol = list(set(_col for _row, _col in lK))
        lCol.sort()
        if lCol == [] or lRow == []:  return None
        
        ndTable = PageXml.createPageXmlNode("TableRegion")
        ndTable.set("id", "p%s_%s" % (self.pagenum, self.tablenum))
        ndTable.tail = "\n"
        lCellShape = []
        lNdCell = []
        
        self.nbCol = max(lCol)+1
        self.nbRow = max(lRow)+1
        self.lRows=[[] for i in range(max(lRow)+1)]
        self.lCols=[[] for i in range(max(lCol)+1)]
        #print (lRow,lCol)
        for row in lRow:
            for col in lCol:
                lNdText = self._dCellNd[(row, col)]
                #     <TableCell row="0" col="1" rowSpan="1" colSpan="1" id="TableCell_1500971530732_2485">
                #        <Coords points="480,42 485,323 878,323 874,38"/>

                if lNdText:
                    ndCell = PageXml.createPageXmlNode("TableCell") 
                    ndCell.set("id", "p%s_t%s_r%s_c%s"%(self.pagenum, self.tablenum, row, col))

                    # shape of the cell
                    oHull = ShapeLoader.convex_hull(lNdText, bShapelyObject=True)
                    #oHull = ShapeLoader.contourObject(lNdText)
                    lCellShape.append(oHull)  # keep those to compute table contour

                    # Coords sub-element                    
                    ndCoords = PageXml.createPageXmlNode("Coords")
                    ndCoords.set("points", ShapeLoader.getCoordsString(oHull, bFailSafe=True))
                    ndCoords.tail = "\n"
                    ndCell.append(ndCoords)
                    
                    # row="0" col="0" rowSpan="1" colSpan="1" leftBorderVisible="false" rightBorderVisible="false" topBorderVisible="false" bottomBorderVisible="false"
                    ndCell.set("row"    , str(row))
                    ndCell.set("rowSpan", "1")
                    ndCell.set("col"    , str(col))
                    ndCell.set("colSpan", "1")
                    ndCell.tail = "\n"

                    #add corner 
                    cornerNode = PageXml.createPageXmlNode("CornerPts")
                    cornerNode.text = "0 1 2 3"                    
                    ndCell.append(cornerNode)
                    
                    for nd in lNdText: ndCell.append(nd)
                    
                    lNdCell.append(ndCell)
                    self.lRows[row].append(ndCell)
                    self.lCols[col].append(ndCell)
        
        # Table geometry
        ndCoords = PageXml.createPageXmlNode("Coords")
        contour = cascaded_union([p if p.is_valid else p.convex_hull for p in lCellShape ]) 
        o = contour.minimum_rotated_rectangle
        ndCoords.set("points", ShapeLoader.getCoordsString(o, bFailSafe=True))
        ndCoords.tail = "\n"
        ndTable.append(ndCoords)
        
        for nd in lNdCell:
            ndTable.append(nd)
            
        return ndTable

    def addEmptyCell(self,ndTable,i,j):
        """
            the web tool requires all cells (even empty) for a better display
            
            get all cells of row i 
            get all cells of col j
            get ymin,ymax of row i and xmin,xmax of col j: intersection  -> cell zone
            (do it for all cells?)  
        """
        rowPol= ShapeLoader.contourObject(self.lRows[i])
        colPol= ShapeLoader.contourObject(self.lCols[j])
        if colPol.is_empty  or rowPol.is_empty: return
#         print (colPol)
        r= Polygon( [ [0,rowPol.bounds[1]],[10000,rowPol.bounds[1]],[10000,rowPol.bounds[3]],[0,rowPol.bounds[3]]] )
        c= Polygon( [ [colPol.bounds[0],0],[colPol.bounds[0],10000],[colPol.bounds[2],10000],[colPol.bounds[2],0]] )
        oHull=r.intersection(c)
#         print ("hull",oHull)
        ndCell = PageXml.createPageXmlNode("TableCell") 
        ndCell.set("id", "p%s_t%s_r%s_c%s"%(self.pagenum, self.tablenum, i, j))

        # Coords sub-element                    
        ndCoords = PageXml.createPageXmlNode("Coords")
        ndCoords.set("points", ShapeLoader.getCoordsString(oHull, bFailSafe=True))
        ndCoords.tail = "\n"
        ndCell.append(ndCoords)
        
        # row="0" col="0" rowSpan="1" colSpan="1" leftBorderVisible="false" rightBorderVisible="false" topBorderVisible="false" bottomBorderVisible="false"
        ndCell.set("row"    , str(i))
        ndCell.set("rowSpan", "1")
        ndCell.set("col"    , str(j))
        ndCell.set("colSpan", "1")
        ndCell.tail = "\n"
#         print (i,j,ndCell)
        ndTable.append(ndCell)
        
        
                        
        
        
    def  orderCell(self,ndTable):
        """
            Ycut on the textlines of each cell
            
            detach TL from cell
            order them
            add them to the cell  
        """
        for row in range(self.nbRow):
            for col in range(self.nbCol):
                lNdText= self._dCellNd[(row,col)]
                if lNdText == []:
                    #self.addEmptyCell(ndTable,row, col) ##add empty cell
                    continue
                cellNd= lNdText[0].getparent()
                for nd in  lNdText:cellNd.remove(nd)
                lmapping = [ ShapeLoader.node_to_Polygon(x) for i,x in enumerate(lNdText)]
                lSegment = [(o.bounds[1], o.bounds[3], lNdText[i]) for i,o in enumerate(lmapping)]
                lLines,_,_ = mergeSegments(lSegment, 2)
#                 [ cellNd.append(tl)  for _,_,ltl in lLines for tl in ltl]
                for _,_,ltl in lLines:
                    ltl = list(ltl)
                    ltl.sort(key=lambda x:ShapeLoader.node_to_Polygon(x).bounds[0])
                    [cellNd.append(tl) for tl in ltl]
                    
        
def main(sInputDir, bForce=False, bVerbose=False):
    
    # filenames without the path
    lsFilename = [os.path.basename(name) for name in os.listdir(sInputDir) if name.endswith("_du.pxml") or name.endswith("_du.mpxml")]
    traceln(" - %d files to process, to tabulate clusters" % (
        len(lsFilename)))
    lsFilename.sort()
    for sFilename in lsFilename:
        sFullFilename = os.path.join(sInputDir, sFilename)
        traceln(" -------- FILE : ", sFullFilename)
        doc = etree.parse(sFullFilename)
        
        for iPage, ndPage in enumerate(doc.getroot().xpath(xpPage, namespaces=dNS)):
            
            # find and delete any pre-existing table
            # if bForce, then move any TextLMine under Page before tabe deletion
            lNdTable = ndPage.xpath(".//pg:TableRegion", namespaces=dNS)
            if bVerbose:
                if bForce:
                    traceln(" - %d pre-existing table to be deleted, preserving its contents by moving it under Page node" % len(lNdTable))
                else:
                    traceln(" - %d pre-existing table to be deleted IF EMPTY" % len(lNdTable))
            for ndTable in lNdTable:
                lNd =  ndTable.xpath(".//pg:TextLine", namespaces=dNS)
                if lNd:
                    if bForce:
                        for nd in lNd:
                            nd.getparent().remove(nd)
                            ndPage.append(nd)
                    else:
                        raise ValueError("Pre-existing Table not empty")
                ndTable.getparent().remove(ndTable)
            
            # enumerate text, and add to cell
            # ignore any text in col|row-spanning cells
            table = TableRegion(iPage+1, 1) # only one table for now!
            lNdText = ndPage.xpath('.//pg:TextLine[@rowSpan="1" and @colSpan="1"]', namespaces=dNS)
            for ndText in lNdText:
                ndText.getparent().remove(ndText)
                table.addToCell(  int(ndText.get("row"))
                                , int(ndText.get("col"))
                                , ndText)
            # make the <TableRegion> !
            ndTable = table.makeTableNode()
            if ndTable is not None:
                # reorder cell content
                table.orderCell(ndTable)
                # add it to the page
                ndPage.append(ndTable)
            
            processRegions(ndPage,bVerbose)
        
        doc.write(sFullFilename+'.f',
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
Create a TableRegion for non-spanning cells.
Rely on row, col, rowSpan, colSpan attributes of the TextLine

Usage: %s <sInputDir>
   
""" % (sys.argv[0])

    parser = OptionParser(usage=sUsage)
    parser.add_option("-v", "--verbose", dest='bVerbose',  action="store_true"
                      , help="Verbose mode")  
    parser.add_option("-f", "--force", dest='bForce',  action="store_true"
                      , help="Force deletion of pre-existing tables, if not empty keeps its contents")  
    (options, args) = parser.parse_args()
    
    try:
        [sInputDir] = args
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
        
    main(sInputDir, bForce=options.bForce, bVerbose=options.bVerbose)
    
    traceln("Done.")
