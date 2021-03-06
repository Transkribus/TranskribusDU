# -*- coding: utf-8 -*-

"""
Transform clusters into TableRegion/TableCells and populate them with TextLines

Created on August 2019

Copyright NAVER LABS Europe 2019
@author: Hervé Déjean
"""

import sys, os
from optparse import OptionParser
from collections import defaultdict

from lxml import etree
import numpy as np
import statistics 
from statistics import mode 
from shapely.geometry import MultiPolygon,GeometryCollection

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) ))
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln
#from xml_formats.PageXml import PageXml
from util.Shape import ShapeLoader
from shapely.affinity import scale
#dNS = {"pg":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
# ----------------------------------------------------------------------------

    
def deleteRegionsinDOM(page,lRegionsNd):
    [page.remove(c) for c in lRegionsNd]
  
def main(sInputDir 
         , bVerbose=False):
    
    # filenames without the path
    lsFilename = [os.path.basename(name) for name in os.listdir(sInputDir) if name.endswith("179.ds_xml")]
    print(" - %d .ds_xml files to process" % len(lsFilename))
    for sMPXml in lsFilename:
        print(" - .ds_xml FILE : ", sMPXml)
        
        # 0 - load input file
        doc = etree.parse(os.path.join(sInputDir,sMPXml))
        clusterZone2TableCell(doc)

        # export to csv as well? 
        doc.write(os.path.join(sInputDir,sMPXml+'.test'),
                  xml_declaration = True,
                  encoding="utf-8",
                  pretty_print=True
                  #compression=0,  #0 to 9
                  )        
            
    
def getTEXTs(ndPage):
    lTL= ndPage.xpath(".//TEXT")
    dIds= {}
    for tl in lTL: dIds[tl.get('id')]=tl
    return dIds     



def computeProfile(dCells,dTextInCells,dCellCndPerRow,dCellCndPerCol):
    """
        for each row: its position as defined by the cells in column
        for each col: its position as defined by the cells in row
        
        if sparse?: cellshape is_empty == true
        
        ONLY FOR LONG ENOUGH ROWS?
    """

    for colid,colCells in dCellCndPerCol.items():
        colCells.sort(key = lambda cell:cell.centroid.y)
    
            
#     print ('- rows')  
    dUpdateCells=dict()    
    modeMax  = 0
    for rowid,rowCells in dCellCndPerRow.items():
        lPosition=[]
        rowCells.sort(key = lambda cell:cell.centroid.x)
        [lPosition.append(dCellCndPerCol[dCells[cell.wkt][1]].index(cell)) for cell in rowCells]
#         print (rowid,lPosition)     
        # take most frequent and update cell
        if lPosition != []:
            try:modeMax =mode(lPosition)
            except statistics.StatisticsError:
#                 print (lPosition)
                modeMax=modeMax+1
                print ("WARNING ROW MODE",rowid,lPosition,modeMax)    
#             print (rowid,lPosition,modeMax)   
            if not modeMax in lPosition:
                modeMax=lPosition[0]
                print('WARNING ROW first',modeMax,lPosition)
            for cell in rowCells:
                    dUpdateCells[cell.wkt]=[modeMax, dCells[cell.wkt][1]]
                       
    # dCellCndPerRow deprecated now 
#     print ('- columns')
    modeMax=0
    for colid,colCells in dCellCndPerCol.items():
        lPosition = []
#         print ([dCells[cell.wkt][0] for cell in colCells] )
        try:[lPosition.append(dCellCndPerRow[dCells[cell.wkt][0]].index(cell)) for cell in colCells]
        #try:[lPosition.append(dCells[cell.wkt][0]) for cell in colCells if dCells[cell.wkt][1] ==colid]
        except:pass
        # take most frequent   
    #         print (colid,lPosition)
#         modeMax=0
        if lPosition != []:
            #modeMax  = 0
            try:modeMax =mode(lPosition)
            except statistics.StatisticsError:
#                 print (lPosition)
                modeMax=modeMax+1
#             print (colid,lPosition,modeMax)
            if not modeMax in lPosition:
                modeMax=lPosition[0]
                print('WARNING COLUMN',modeMax,lPosition)
            for cell in colCells:dUpdateCells[cell.wkt]=[dUpdateCells[cell.wkt][0],modeMax]
    
    
    dNewCells=dict()
    for c in dUpdateCells:
        try:
            if c in dTextInCells.keys():
                #dNewCells[c].extend([t.text for t in dTextInCells[c] if t.text is not None])
                dNewCells[c].extend( dTextInCells[c] ) #[t.text for t in dTextInCells[c] if t.text is not None])

            else:dNewCells[c]=[]
        except KeyError:
            dNewCells[c]=[]
            if c in dTextInCells.keys():
                #dNewCells[c].extend([t.text for t in dTextInCells[c] if t.text is not None])
                dNewCells[c].extend( dTextInCells[c] ) #[t.text for t in dTextInCells[c] if t.text is not None])

            else: dNewCells[c]=[]
#             print ([t.text for t in dTextInCells[c]],dCells[c])
#         else:
#             print ("?",dCells[c]) 
    
    # merge cells with the same i,j  (check if compatible?)             

#     for c in dNewCells:
#         print (dUpdateCells[c],[t.text for t in dNewCells[c] if t.text is not None])
    
    return dUpdateCells,dNewCells
    #done?
    # 
    #span -> see next cell number  span= difference of position
    
def storeEdges(ndPage):
    """
        create a dictionary to store edges
        dEdges[continue|break][(id1,ied2)]=weight
        
        <EDGE src="r1l2" tgt="r2l42" type="HorizontalEdge" w="0.948" label="break" points="57.12,607.68 105.12,607.44"/>

        return dEdges
    """
    xEdgeNd   = ".//EDGE"
    lEdgeNd = ndPage.xpath(xEdgeNd)
    
    dEdges={'continue':{},'break':{} }
    nbEdges=0
    for nd in lEdgeNd: 
        dEdges[nd.get('label')][(nd.get('src'),nd.get('tgt'))] = float(nd.get('w'))
        dEdges[nd.get('label')][(nd.get('tgt'),nd.get('src'))] = float(nd.get('w'))
        nbEdges+= 1
#     print('%d edges stored.'%nbEdges)
    
    return dEdges

  

def storeSeparators(pageNd):
    """
        <SeparatorRegion points="482.4,218.64 485.52,625.68" x="482.4" y="218.64" height="407.04" width="3.12"/>
    """
    lSepNd=pageNd.xpath(".//SeparatorRegion")

def connected_components(dEdges,lIds,TH=1):
    lcluster=[]  
    visited = np.zeros(len(lIds),dtype='int64')
    for ii in range(len(lIds)):
        visited_index = []
        def DFS(i):
            if visited[i] == 1:
                return
            visited[i]=1
            visited_index.append(i)
            for j in range(len(lIds)):
                rel = False
                if (lIds[i],lIds[j]) in dEdges['continue']:
                    if dEdges['continue'][(lIds[i],lIds[j])] >= TH:
                        rel=True
                
                if visited[j]!=1 and rel:
                    visited_index.append(j)
                    DFS(j)
            return visited_index

        dfs_i = DFS(ii) 
        if dfs_i is not None:
            lcluster.append(list(map(lambda x:lIds[x],set(dfs_i))))

    return lcluster

def bestRegionsAssignment(txt,lRegions):
    """
         find the best (max overlap for self) region  for each txt
         lTxts: list of polygons
         lRegions: list of polygons
    """
    from rtree import index
    
    assert txt.convex_hull.is_valid
     
    txtidx = index.Index()
    lP = []
    [lP.append(e) for e in lRegions if e.is_valid]
    for i,elt in enumerate(lRegions):
        txtidx.insert(i, lP[i].bounds)
    lSet = txtidx.intersection(txt.bounds)
    lOverlap = []
    for ei in lSet:
        if lP[ei].is_valid:
            intersec= txt.intersection(lP[ei]).area
            if intersec >0:
                lOverlap.append((ei,lP[ei],intersec))
    if lOverlap != []:        
        lOverlap.sort(key=lambda xyz:xyz[-1])
#             print ("??",self,lRegions[lOverlap[-1][0]])
        return lRegions[lOverlap[-1][0]]
    
    return None



def checkClusteringPerRow(pageNd,dCellCndPerRow,dTextInCells,dClusterWithTL, dEdges):
    """
        recompute clusters for the rows
    """
    lNewPoly=[]
    lNewClusterWithTL = []
    lNewClusterWithCC = []
    lRowtodel=[]
    lTLtoBeAdded=[]
    for rowid,rowCells in dCellCndPerRow.items():
        lNbClusterPerCol=[]
        lClusterPerCol=[]
        for cell in rowCells:
            if cell.wkt in dTextInCells.keys():
                lIds = [tl.get('id') for tl in dTextInCells[cell.wkt] ] 
                lClusterCells = connected_components(dEdges,lIds,TH=0.5)
                lNbClusterPerCol.append(len(lClusterCells))
                lClusterPerCol.append(lClusterCells)
                #print (len(rowCells),lIds,lClusterCells,lNbClusterPerCol)
            #else empty cell
        if len(lNbClusterPerCol) > 2:
#             print (lNbClusterPerCol)
            try:nbRows = mode(lNbClusterPerCol)
            except statistics.StatisticsError:nbRows=2
            if nbRows !=1:
                print ('WARNING CUT ',rowid,"mode",nbRows)
                lRowtodel.append(rowid)
                dClusterCells=defaultdict(list)
                for colcluster in lClusterPerCol:
                    if len(colcluster) == nbRows:
                        # get tl instead of ids
                        # take the first element only for comparing position
                        colclustertxt = []
                        for cc in colcluster:
                            colclustertxt.append([ tl for tl in dClusterWithTL[rowid] if tl.get('id') in cc ])
                        sortedC = sorted(colclustertxt,key=lambda x:ShapeLoader.node_to_Polygon(x[0]).centroid.y)
                        for i,cc in enumerate(sortedC):
                            dClusterCells[i].append(cc)
                    else:
                        for cc in colcluster:
                            lTLtoBeAdded.extend([ tl for tl in dClusterWithTL[rowid] if tl.get('id') in cc ])
                for i,lcc in dClusterCells.items():
#                     print (i,lcc)
                    rect = ShapeLoader.contourObject([x for cc in lcc for x in cc]).envelope  # minimal_rectangle?
                    rect=scale(rect,xfact = 2)
                    lNewPoly.append(rect)
                    lNewClusterWithCC.append(lcc)
                    lNewClusterWithTL.append([x for cc in lcc for x in cc])
#                         lNewCluster.Append()
                    # need also to create a cluster with list of ids!!
                    rNd=etree.Element('LINE')
                    rNd.set('points',ShapeLoader.getCoordsString(rect))
                    pageNd.append(rNd)
                    # final check: if overlap between rectangle: top rectangle is the cut 
    
    #return updated list of lCLusters
    return  lRowtodel,lNewPoly, lNewClusterWithTL,lNewClusterWithCC,lTLtoBeAdded
    
def getClusterTL(ndPage,lClusters,dIds):
    """
        get tl nodes per cluster
    """
    dTLCluster={}
    for i,c in enumerate(lClusters):
        dTLCluster[i]=[]
        lIds=c.get('content').split()
        for id in lIds:
            try:dTLCluster[i].append(dIds[id])
            except KeyError:pass
#         [dTLCluster[i].append(dIds[id]) for id in lIds]
#         try: [dTLCluster[i].append(dIds[id]) for id in lIds]
#         except KeyError:
#             # TableCell_1575304898029_1251l1 in cluster???? file EDIN:Sanktpoeltendom_01-14_1888_1894_3374_02-Taufe_0035.ds_xml 
#             pass 
    
    return dTLCluster

def assignConCompToCells(lCC,lCells):
    """
        get textlines from ids and best-assign them to the list of dCells (shape)
        
        # do it at the cc level?
    """
    dCellCC={}
    for i,cc in enumerate(ShapeLoader.contourObject(lCC)):
        if not cc.is_empty:
            cellshape = bestRegionsAssignment(cc,lCells)
            if cellshape:
                try:dCellCC[cellshape.wkt].append(lCC[i])
                except KeyError: dCellCC[cellshape.wkt]= [lCC[i]]
    
    return dCellCC

def assignTextLinesToCells(lTL,lCells):
    """
        get textlines from ids and best-assign them to the list of dCells (shape)
        
        # do it at the cc level?
    """
#     print (lCells)
#     print ([ShapeLoader.node_to_Polygon(nd) for nd in lTL])
    dCellTL={}
    for i,txt in enumerate([ShapeLoader.node_to_Polygon(nd) for nd in lTL]):
        if not txt.is_empty:
            cellshape = bestRegionsAssignment(txt,lCells)
            if cellshape:
                try:dCellTL[cellshape.wkt].append(lTL[i])
                except KeyError: dCellTL[cellshape.wkt]= [lTL[i]]
    return dCellTL


def transformMinima2envelop(pageNd,dClusterWithTL):
    """
    """
    lClustersPoly=[]
    for row in  dClusterWithTL:
#         bb = ShapeLoader.contourObject(dClusterWithTL[row]).envelope
#         if not bb.is_empty:
#             cellNd = etree.Element('COL')
#             pageNd.append(cellNd)
#             cellNd.set('points',ShapeLoader.getCoordsString(bb))    
        
        ch = ShapeLoader.contourObject(dClusterWithTL[row]).convex_hull
        lClustersPoly.append(ch)
        if not ch.is_empty:
            cellNd = etree.Element('LINE')
            pageNd.append(cellNd)
            cellNd.set('points',ShapeLoader.getCoordsString(ch))
    
    return lClustersPoly
    
def shakeRectangleShapes(pageNd,lClusterPoly,dCellCndPerRow,dTextInCells):
    """
        delete one 'col' and see if the rectangle is stable (area? comparison; rather angle)
        
        angle see baseline object or 
        take the longest line in the polygon and compute its angle
            radian = math.atan((shape.lastpoint.x - shape.firstpoint.x)/(shape.lastpoint.y - shape.firstpoint.y))  
            degrees = radian * 180 / math.pi  
        return degrees  
    """
    #sort by size (reverse)
#     lClusterPoly = [ShapeLoader.node_to_Polygon(nd) for nd in lClusters]
    
    ## can be restricted to ones with high angle?
    lNewPoly=[]
    for i in sorted(dCellCndPerRow.keys()):
        lcellRows=dCellCndPerRow[i]
        # skip first
        ltl=[]
        for c in lcellRows[1:]:
            if c.wkt in dTextInCells:
                for tl in dTextInCells[c.wkt]:
                    assert ShapeLoader.node_to_Polygon(tl).is_valid
                    ltl.append(tl)
        rectangle = ShapeLoader.minimum_rotated_rectangle(ltl,bShapelyObject=True)
#         print (i,lClusterPoly[i].intersection(rectangle).area / lClusterPoly[i].area)
        if not rectangle.is_empty:
            if abs(rectangle.bounds[1]-rectangle.bounds[3]) < abs(lClusterPoly[i].bounds[1]-lClusterPoly[i].bounds[3]):
#                 rectangle=scale(rectangle,xfact=1.5)
                cellNd = etree.Element('COL')
                pageNd.append(cellNd)
                rectangle=scale(rectangle,xfact=2)
                lNewPoly.append(rectangle)
                cellNd.set('points',ShapeLoader.getCoordsString(rectangle))    
                
            else: 
                lClusterPoly[i]=scale(lClusterPoly[i],xfact=1.5)
                lNewPoly.append(lClusterPoly[i])
        else:                
            lClusterPoly[i]=scale(lClusterPoly[i],xfact=1.5)
            lNewPoly.append(lClusterPoly[i])
        ## get length and compute angle !!!  take the one with more horizontal angle::
#         cellNd = etree.Element('COL')
#         pageNd.append(cellNd)
#         if not rectangle.is_empty:
#             rectangle=scale(rectangle,xfact=2)
#             cellNd.set('points',ShapeLoader.getCoordsString(rectangle))    
#     print (len(lClusterPoly),len(lNewPoly))    
    return lNewPoly

def ajustRectangleShapes(pageNd,lClusterPoly):
    """
        reduce a cluster rectangle with just area which does not overlap other area
    """
    #lClusterPoly = [ShapeLoader.node_to_Polygon(nd) for nd in lClusters]

    lNewPoly=[]
    for i,pol1 in enumerate(lClusterPoly):
#         print (i,'init:',pol1.area,pol1.bounds)
        init=pol1.wkt
        # take all other and createa multipo
#         multi=MultiPolygon([p for p in lClusterPoly if p != pol1]).buffer(0)
        for j,pol2 in enumerate(lClusterPoly[i+1:]):
            if pol1.intersection(pol2).area > pol2.area*0.01:
                pol1 = pol1.symmetric_difference(pol2).difference(pol2)
#                 pol1=pol1.difference(pol2)
                if not pol1.is_empty and type(pol1) in [MultiPolygon,GeometryCollection]:
                    pol1=max(pol1.geoms,key=lambda p:p.area)

        cellNd = etree.Element('COLX')
        pageNd.append(cellNd)
        if not pol1.is_empty:
            cellNd.set('points',ShapeLoader.getCoordsString(pol1))        
        lNewPoly.append(pol1)
#     lNewPoly.extend(lClusterPoly[1:])
#     print(len(lNewPoly) ,len(lClusterPoly))
    return lNewPoly            

    
def mergeContainedShapes():
    """
        if a shape is contained in another shape: merge both
    """
def createCellZoneFromIntersection(lClusterPoly,lZonePoly):        
    """
    """
    #dictionary of 'cells ' per row
    # key = rowid  
    dCellCndPerRow = dict()

    #dictionary of 'cells ' per column
    # key = colid  
    dCellCndPerCol = dict()
    dCells=dict()
    #CREATE CELLS FROM MINIMUN_ROTATED_RECTANGLE
    dShapeCells={}  #key = cell.wkt  item: shape
    for i,r in enumerate(lClusterPoly):
        dCellCndPerRow[i]=[]
        for j,z in enumerate(lZonePoly):
            cellpoly = z.buffer(0).intersection(r.buffer(0))
            if not cellpoly.is_empty:
                dCellCndPerRow[i].append(cellpoly)
                try:dCellCndPerCol[j].append(cellpoly)
                except KeyError:dCellCndPerCol[j]=[cellpoly]
                dCells[cellpoly.wkt]=[i,j]
                dShapeCells[cellpoly.wkt]=cellpoly
    return dCells,dShapeCells,dCellCndPerRow,dCellCndPerCol

def createCellsFromZoneClusters(pageNd,dIds,lZones,lClusters,dEdges):
    """
    
        TOP DOWN: CREATE A GRID TABLE AND REFINE IT
    
        lZones: list of zones for columns (ordered left right by construction)
        lClusters: list of cluster NODE!! 
        
        create cells by zone intersections
    
    
        idea! compute 'invariance' of the angle of  the rectangle: if one column less: should be the same angle
           variance for detecting merge: del on column and see if several clusters?
        
        
        1 test if rectangle OK
            if all ok and no overlap: create rows and col
        
        # for overlapping:   -> connected componants for each column
            create alternate rectangles by skipping one cell 
        
        3 merge if same  #order (at cell level)
        
        4 split if n connected cells per column
            # how: create alternate rectangles?

    """ 
    lClusters = sorted(lClusters,key=lambda x:ShapeLoader.node_to_Polygon(x).centroid.y)
    lZones = sorted(lZones,key=lambda x:ShapeLoader.node_to_Polygon(x).centroid.x)

    dClusterWithTL = getClusterTL(pageNd,lClusters,dIds)
    
    #lClusterPoly = [ShapeLoader.node_to_Polygon(nd) for nd in lClusters]

    lZonePoly = [ShapeLoader.node_to_Polygon(nd) for nd in lZones]
    
    ## if one clusterpoly is contained in another: merge

    lClusterPoly = transformMinima2envelop(pageNd,dClusterWithTL)
#     return 0,0,0

    dCells,dShapeCells,dCellCndPerRow,dCellCndPerCol = createCellZoneFromIntersection(lClusterPoly, lZonePoly)
    dTextInCells={} #key = cell.wkt
    for i  in range(len(lClusters)):dTextInCells.update( assignTextLinesToCells(dClusterWithTL[i],dCellCndPerRow[i])) 
    lClusterPoly= shakeRectangleShapes(pageNd,lClusterPoly,dCellCndPerRow,dTextInCells)

    lClusterPoly= ajustRectangleShapes(pageNd,lClusterPoly)

    # again recompute cell zones    
    dCells,dShapeCells,dCellCndPerRow,dCellCndPerCol = createCellZoneFromIntersection(lClusterPoly, lZonePoly)
    dTextInCells={} #key = cell.wkt
    for i  in range(len(lClusters)):dTextInCells.update( assignTextLinesToCells(dClusterWithTL[i],dCellCndPerRow[i])) 
    
    # do it first ??
    #check if row is a merge
    lRowtobeDel,lNewRowCluster,lNewClusterTL,lNewClusterCC,lTLtoBeAdded = checkClusteringPerRow(pageNd,dCellCndPerRow,dTextInCells,dClusterWithTL, dEdges)
    if lNewRowCluster:
        lClusterPoly = [ShapeLoader.node_to_Polygon(nd) for i,nd in enumerate(lClusters) if i not in lRowtobeDel]
        dClusterWithTL = getClusterTL(pageNd,[c for i,c in enumerate(lClusters) if i not in lRowtobeDel],dIds)
        for i,ltl in enumerate(lNewClusterTL):
            dClusterWithTL[len(lClusterPoly)+i]=ltl
        lClusterPoly.extend(lNewRowCluster)   
#         lClusterPoly= ajustRectangleShapes(pageNd,lClusterPoly)

        dCells,dShapeCells,dCellCndPerRow,dCellCndPerCol = createCellZoneFromIntersection(lClusterPoly, lZonePoly)
        dTextInCells={} #key = cell.wkt
        for i  in range(len(lClusterPoly)):
            dTextInCells.update(assignTextLinesToCells(dClusterWithTL[i],dCellCndPerRow[i]))
                 
        
        ## populate elements which are not ij the grid/cells : at cc level or at tl level ?
        for i  in range(len(lClusterPoly)):
            dTextInCells.update(assignTextLinesToCells(lTLtoBeAdded,dCellCndPerRow[i]))
    ## oversegmentation : test links at cell level if compatible!see 0068
    
    #COMPUTE PROFILE
    dUpdateCells,dNewCells = computeProfile(dCells,dTextInCells,dCellCndPerRow,dCellCndPerCol)


    
    # TABLE CREATION
    
    return    dUpdateCells,dNewCells,dShapeCells


    
def tableCreation(ndPage,dCellsIJ,dCellTxt,dCoordCell):
    """
        find TABLE tag
        add CELLS 
    
    """
    # get TABLE node
    xTable   = ".//TABLE"
    lTableNds = ndPage.xpath(xTable)
    # discard fake table!!
    tableNd= lTableNds[-1]
    
    for cellwkt in dCellsIJ.keys():
        cellNd = etree.Element('CELL')
        i,j = dCellsIJ[cellwkt]
        x1,y1,x2,y2=dCoordCell[cellwkt].bounds
        cellNd.set('row',f'{i}')
        cellNd.set('col',f'{j}')
        cellNd.set('x',str(x1))
        cellNd.set('y',str(y1))
        cellNd.set('height',str(abs(y2-y1)))
        cellNd.set('width',str(abs(x2-x1)))

            # empty        
#         cellNd.set('points',ShapeLoader.getCoordsString(dCoordCell[cellwkt]))
        try:cellNd.set('points',ShapeLoader.getCoordsString(dCoordCell[cellwkt]))
        except:
            # take largest 
            pol=max(dCoordCell[cellwkt].geoms,key=lambda p:p.area)
            cellNd.set('points',ShapeLoader.getCoordsString(pol))
            
        # populate with TL!:
        # sort by Y increasing!!
        dCellTxt[cellwkt].sort(key=lambda x:ShapeLoader.node_to_Point(x).bounds[1])
        [cellNd.append(t) for t in dCellTxt[cellwkt]]
        #cellNd.text= " ".join(dCellTxt[cellwkt])
        
        cellNd.set('colSpan',"1")
        cellNd.set('rowSpan',"1")
        tableNd.append(cellNd)                
    
    return 
    
            
            
# def getCellCluster(ndPage,xpCluster):
#     """
#         <Cluster name="(7_I_agglo_101575)" algo="(cut_I_agglo)" content="p1_r7l22 p1_r7l23"><Coords points="1635,2937 1539,2930 1546,2812 1642,2819 1635,2937"/></Cluster>
# 
#     """
#     from statistics import mean
#     dColdict=defaultdict(list)
#     dRowdict=defaultdict(list)
#     dTmpRowIDPos=defaultdict(list)
#     dRowIDPos=defaultdict(list)
#     lClusters= ndPage.xpath(xpCluster, namespaces=dNS)
#     prevcolid=-1
#     rowposition=0
# 
#     dcol2=defaultdict(list)
#     # assume order by columns!!!    
#     for c  in lClusters:
#         name= c.get('name')
#         colid,rowid= [ int(i) for i in  name.strip('()').split('_I_agglo_') if isinstance(int(i),int)]    
#         dcol2[colid].append(c)
#     
#     #for c  in lClusters:
#     for colid in dcol2:
#         c= dcol2[colid].sort(key=lambda cell:mean([ShapeLoader.node_to_Point(x).y for x in cell]))
#         name= c.get('name')
#         colid,rowid= [ int(i) for i in  name.strip('()').split('_I_agglo_') if isinstance(int(i),int)]
#         # why? assume ordered by column??
#         if colid != prevcolid:rowposition=-1
#         rowposition += 1
#         dColdict[colid].extend(c.get('content').split())
#         dRowdict[rowid].extend(c.get('content').split())
#         prevcolid = colid
#       
#       
#     for key, values in dTmpRowIDPos.items():
#         print (key, max(set(values), key = values.count), values)
#         dRowIDPos[max(set(values), key = values.count)].append(key) #max(set(values), key = values.count)
#     
#     lF=defaultdict(list)
#     for i,pos in enumerate(sorted(dRowIDPos.keys())):
#         print (i,pos,dRowIDPos[pos], [dTmpRowIDPos[x] for x in dRowIDPos[pos] ])
#         lF[i]= dRowIDPos[pos]
#     
#     #return dColdict,dRowdict,dRowIDPos #lCells
#     return dColdict,dRowdict,lF #lCells
   
   
    
def loadSeparatorZones(ndPage,algo,name):
    """
     algo="sep" name="V"
    """
    xpZones   = f".//Zone[ @name='{name}']"
    lZones = ndPage.xpath(xpZones)
    
    return lZones
        
def loadRowCluster(ndPage,algo):
    """
        load cluster algo = aglo
    """    
    xpCluster   = f".//Cluster[@algo='{algo}']"
    lClusters= ndPage.xpath(xpCluster)
    return lClusters
        
    
def clusterZone2TableCell(doc):    
    """
        row as clusters
        column as zones
    """
    root = doc.getroot()
    xpPages= "./PAGE"
    for _, ndPage in enumerate(doc.xpath(xpPages)):
        dIds = getTEXTs(ndPage)
        lColZones = loadSeparatorZones(ndPage,'sep','V')
        lRowClusters = loadRowCluster(ndPage,'agglo')
        if len(lRowClusters) ==0: 
            print("NO ROW CLUSTER !!")
            return        
        if len(lColZones) ==0: 
            print("NO COLUMN !!")
            return
        dEdges=storeEdges(ndPage)
        print (f'nb cols: {len(lColZones)}, nbRows: {len(lRowClusters)}')
        
        lCellsIJ, dCellsText,dShapeCell = createCellsFromZoneClusters(ndPage,dIds,lColZones,lRowClusters,dEdges)
        
        # sanity check ofr cells shape: if one cell overlaps with others: minimal rectnagle to be fixed
        # as input TABLE/COL
        tableCreation(ndPage,lCellsIJ,dCellsText,dShapeCell)

        #clean up:  delete empty regions



# ----------------------------------------------------------------------------
if __name__ == "__main__":
    
    version = "v.01"
    sUsage="""
Usage: %s <sInputDir>   
    
""" % (sys.argv[0])

    parser = OptionParser(usage=sUsage)
    parser.add_option("-v", "--verbose", dest='bVerbose',  action="store_true"
                      , help="Verbose mode")     
    (options, args) = parser.parse_args()
    
    try:
        sInputDir = args[0]
    except ValueError:
        sys.stderr.write(sUsage)
        sys.exit(1)
    
    # ... checking folders
    #if not os.path.normpath(sInputDir).endswith("col")  : sInputDir = os.path.join(sInputDir, "col")
    # all must be ok by now
    lsDir = [sInputDir]
    if not all(os.path.isdir(s) for s in lsDir):
        for s in lsDir:
            if not os.path.isdir(s): sys.stderr.write("Not a directory: %s\n"%s)
        sys.exit(2)
    bVerbose=options.bVerbose
    main(sInputDir, bVerbose=options.bVerbose)
    
    traceln("Done.")