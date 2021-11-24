# -*- coding: utf-8 -*-

"""
Transform clusters into TableRegion/TableCells and populate them with TextLines

Created on August 2019

Copyright NAVER LABS Europe 2019
@author: Hervé Déjean
"""

import sys, os, glob
from optparse import OptionParser
from copy import deepcopy
from collections import Counter
from collections import defaultdict

from lxml import etree
import numpy as np
from shapely.ops import cascaded_union        


try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) ))
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln, trace
from xml_formats.PageXml import PageXml
from util.Shape import ShapeLoader
dNS = {"pg":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
# ----------------------------------------------------------------------------

def getClusterCoords(lElts):
    
        lp = []
        for e in lElts:
            try:
                lp.append(ShapeLoader.node_to_Polygon(e))
            except ValueError:
                pass
        contour = cascaded_union([p if p.is_valid else p.convex_hull for p in lp ])     
        # print(contour.wkt)
        try:spoints = ' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in contour.convex_hull.exterior.coords)
        except:
            try: spoints = ' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in contour.convex_hull.coords)
            # JL got once a: NotImplementedError: Multi-part geometries do not provide a coordinate sequence
            except: spoints = ""    
        return spoints
    
def deleteRegionsinDOM(page,lRegionsNd):
    [page.remove(c) for c in lRegionsNd]
  
def main(sInputDir 
         , bVerbose=False):
    
    lSkippedFile = []
    
    # filenames without the path
    lsFilename = [os.path.basename(name) for name in os.listdir(sInputDir) if name.endswith("_du.mpxml")]
    traceln(" - %d .mpxml files to process" % len(lsFilename))
    for sMPXml in lsFilename:
        traceln(" - .mpxml FILE : ", sMPXml)
        
        # 0 - load input file
        doc = etree.parse(os.path.join(sInputDir,sMPXml))
        cluster2TableCell(doc,bVerbose)
        
#         doc.write(os.path.join(sInputDir,sMPXml),
#                   xml_declaration = True,
#                   encoding="utf-8",
#                   pretty_print=True
#                   #compression=0,  #0 to 9
#                   )        
            
    
def propagateTypeToRegion(ndRegion):
    """
        compute the most frequent type in the Textlines and assigns it to the new region
    """
    dType=Counter()
    for t in ndRegion:
        dType[t.get('type')]+=1
    mc = dType.most_common(1)
    if mc :
        if mc[0][0]:ndRegion.set('type',mc[0][0])
        #  structure {type:page-number;}
        # custom="structure {type:page-number;}"
        if mc[0][0]:ndRegion.set('custom',"structure {type:%s;}"%mc[0][0])
        
    
def addTableCellsToDom(page,ipage,lc,bVerbose):
    """
        create a dom node for each cluster
        update DU_cluster for each Textline
    """
    # create TableRegion first !
    
    for ic,dC in enumerate(lc):
        ndRegion = PageXml.createPageXmlNode('TableCell')     
        
        #update elements
        lTL = lc[dC] 
        print (lTL)
#         for id in c.get('content').split():
#             elt = page.xpath('.//*[@id="%s"]'%id)[0]
#             elt.getparent().remove(elt)
#             ndRegion.append(elt)
#             lTL.append((elt))
        ndRegion.set('id',"p%d_r%d"%(ipage,ic))
        coords = PageXml.createPageXmlNode('Coords')        
        ndRegion.append(coords)
        coords.set('points',getClusterCoords(lTL))   
        propagateTypeToRegion(ndRegion)

        page.append(ndRegion)
     
def getTextLines(ndPage):
    lTL= ndPage.xpath(".//pg:TextLine", namespaces=dNS)
    dIds= {}
    for tl in lTL: dIds[tl.get('id')]=tl
    return dIds
            
            
def getCellCluster(ndPage,xpCluster):
    """
        <Cluster name="(7_I_agglo_101575)" algo="(cut_I_agglo)" content="p1_r7l22 p1_r7l23"><Coords points="1635,2937 1539,2930 1546,2812 1642,2819 1635,2937"/></Cluster>

    """
    from statistics import mean
    dColdict=defaultdict(list)
    dRowdict=defaultdict(list)
    dTmpRowIDPos=defaultdict(list)
    dRowIDPos=defaultdict(list)
    lClusters= ndPage.xpath(xpCluster, namespaces=dNS)
    prevcolid=-1
    rowposition=0

    dcol2=defaultdict(list)
    # assume order by columns!!!    
    for c  in lClusters:
        name= c.get('name')
        colid,rowid= [ int(i) for i in  name.strip('()').split('_I_agglo_') if isinstance(int(i),int)]    
        dcol2[colid].append(c)
    
    #for c  in lClusters:
    for colid in dcol2:
        c= dcol2[colid].sort(key=lambda cell:mean([ShapeLoader.node_to_Point(x).y for x in cell]))
        name= c.get('name')
        colid,rowid= [ int(i) for i in  name.strip('()').split('_I_agglo_') if isinstance(int(i),int)]
        # why? assume ordered by column??
        if colid != prevcolid:rowposition=-1
        rowposition += 1
        dColdict[colid].extend(c.get('content').split())
        dRowdict[rowid].extend(c.get('content').split())
        prevcolid = colid
      
      
    for key, values in dTmpRowIDPos.items():
        print (key, max(set(values), key = values.count), values)
        dRowIDPos[max(set(values), key = values.count)].append(key) #max(set(values), key = values.count)
    
    lF=defaultdict(list)
    for i,pos in enumerate(sorted(dRowIDPos.keys())):
        print (i,pos,dRowIDPos[pos], [dTmpRowIDPos[x] for x in dRowIDPos[pos] ])
        lF[i]= dRowIDPos[pos]
    ss
    #return dColdict,dRowdict,dRowIDPos #lCells
    return dColdict,dRowdict,lF #lCells
   
   
def createTable(dColdict,dRowdict,dRowIDPos,lIds):
    """
    
    sort rows by avg(y)
    """
    from statistics import mean 
    
    #get table dimensions
    for x in dRowIDPos.items(): print (x)
    nbCols= len(dColdict.keys())+1
    nbRows= len(dRowIDPos.keys())+1
    table = [[ [] for i in range(nbCols)] for j in range(nbRows)]
    print (nbRows,nbCols,len(table),len(table[0]))
    for irow in sorted(dRowIDPos.keys()):
        for jcol in sorted(dColdict.keys()):
            if jcol > 0:
#                 print (irow,jcol)
                # compute intersection??
                cellij=  list([value for row in dRowIDPos[irow]  for value in dRowdict[row] if value in dColdict[jcol] ])
    #             print(dRowdict[dRowIDPos[irow]] )
    #             print(dColdict[jcol])
                #print (irow,jcol-1,[''.join(lIds[id].itertext()).strip() for id in cellij])
    #             print (irow,jcol-1,[lIds[id] for id in cellij])
                table[irow][jcol-1]=[lIds[id] for id in cellij]
    
    # ignore empty row
#     table.sort(key=lambda row:mean([ShapeLoader.node_to_Point(x).y for cell in row for x in cell])   )
    for row in table:
        print ([len(x) for cell in row for x in cell])
#         for col in row:
#             print ([''.join(x.itertext()).strip() for x in col],end='')
#         print() 
    table.sort(key=lambda row:mean([ShapeLoader.node_to_Point(x).y for cell in row for x in cell])   )

#     for irow,row in enumerate(table):
#         lY=[]
#         for cell in row:
#             if cell != []:
#                 #print( [ShapeLoader.node_to_Point(x) for x in cell])
#                 mean([lY.extend(ShapeLoader.node_to_Point(x).y for x in cell)])
#                 
#         print (irow,mean(lY))
                
        
def cluster2TableCell(doc, fTH=0.5,bVerbose=True):
    """
    
    """
    root = doc.getroot()
    
    
    xpCluster   = ".//pg:Cluster[@algo='(cut_I_agglo)']"
    xpTextRegions     = ".//pg:TextRegion"
    
    # get pages
    for iPage, ndPage in enumerate(PageXml.xpath(root, "//pc:Page")[24:]): 
        # get cluster    
        dColdict,dRowdict,dRowIDPos = getCellCluster(ndPage,xpCluster) #ndPage.xpath(xpCluster, namespaces=dNS)
        lIds = getTextLines(ndPage)
#         lRegionsNd  =  ndPage.xpath(xpTextRegions, namespaces=dNS)
#         if bVerbose:traceln("\n%d clusters and %d regions found" %(len(dClusters),len(lRegionsNd)))
        try:
            lCells = createTable(dColdict,dRowdict,dRowIDPos,lIds)
        except KeyError:
            print(iPage)
            return
#         addTableCellsToDom(ndPage,iPage+1,lCells,bVerbose)
#         if bVerbose:traceln("%d regions created" %(len(dClusters)))            
#         deleteRegionsinDOM(ndPage, lRegionsNd)
    return doc



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
    if not os.path.normpath(sInputDir).endswith("col")  : sInputDir = os.path.join(sInputDir, "col")
    # all must be ok by now
    lsDir = [sInputDir]
    if not all(os.path.isdir(s) for s in lsDir):
        for s in lsDir:
            if not os.path.isdir(s): sys.stderr.write("Not a directory: %s\n"%s)
        sys.exit(2)
    bVerbose=options.bVerbose
    main(sInputDir, bVerbose=options.bVerbose)
    
    traceln("Done.")