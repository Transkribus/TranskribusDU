# -*- coding: utf-8 -*-

"""
Transform clusters into TextRegions and populate them with TextLines

Created on August 2019

Copyright NAVER LABS Europe 2019
@author: Hervé Déjean
"""

import sys, os
from optparse import OptionParser
from collections import Counter
from collections import defaultdict

from lxml import etree
from shapely.ops import cascaded_union        


try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
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
         , options):
    
    
    # filenames without the path
    lsFilename = [os.path.basename(name) for name in os.listdir(sInputDir) if name.endswith("_du.mpxml")]
    traceln("- %d .mpxml files to process" % len(lsFilename))
    for sMPXml in lsFilename:
        trace(".du_mpxml FILE : ", sMPXml)
        if options.bVerbose: traceln()
        
        # 0 - load input file
        doc = etree.parse(os.path.join(sInputDir,sMPXml))
        cluster2Region(doc,options.bVerbose)
        if options.bOverwrite:
            outfilename = sMPXml
        else: 
            outfilename = sMPXml[:-len('_du.mpxml')]+"." + options.outext
        traceln(" written in %s"%(os.path.join(sInputDir,outfilename)))
        doc.write(os.path.join(sInputDir,outfilename),
                  xml_declaration = True,
                  encoding="utf-8",
                  pretty_print=True
                  #compression=0,  #0 to 9
                  )        
            
    
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
        
    
def addRegionToDom(page,ipage,lc,bVerbose):
    """
        create a dom node for each cluster
        update DU_cluster for each Textline
    """
    for ic,dC in enumerate(lc):
        ndRegion = PageXml.createPageXmlNode('TextRegion')     
        
        #update elements
#         for id in c.get('content').split():
#             elt = page.xpath('.//*[@id="%s"]'%id)[0]
#             elt.getparent().remove(elt)
#             ndRegion.append(elt)
#             lTL.append((elt))
        lTL = lc[dC] 
        ndRegion.set('id',"p%d_r%d"%(ipage,ic))
        coords = PageXml.createPageXmlNode('Coords')        
        ndRegion.append(coords)
        coords.set('points',getClusterCoords(lTL))   
#         propagateTypeToRegion(ndRegion)
        for tl in lTL:
            tl.getparent().remove(tl)
            ndRegion.append(tl)
            #print (f"{tl.get('id')} added to {ndRegion.get('id')}")
        page.append(ndRegion)
             
def getCLusters(ndPage):
    dCluster=defaultdict(list)
    lTL= ndPage.xpath(".//*[@DU_cluster]", namespaces=dNS)
    for x in lTL:dCluster[x.get('DU_cluster')].append(x)
    return dCluster
                 
def cluster2Region(doc, fTH=0.5,bVerbose=True):
    """
    
    """
    root = doc.getroot()
    
    xpTextRegions     = ".//pg:TextRegion"
    
    # get pages
    for iPage, ndPage in enumerate(PageXml.xpath(root, "//pc:Page")): 
        # get cluster    
        dClusters= getCLusters(ndPage) #ndPage.xpath(xpCluster, namespaces=dNS)
        lRegionsNd  =  ndPage.xpath(xpTextRegions, namespaces=dNS)
        if bVerbose:traceln(" %d clusters and %d regions found" %(len(dClusters),len(lRegionsNd)))
        
        addRegionToDom(ndPage,iPage+1,dClusters,bVerbose)
        if bVerbose:traceln(" %d regions created" %(len(dClusters)))            
        deleteRegionsinDOM(ndPage, lRegionsNd)
        
        #
        lEdgesNd  =  ndPage.xpath(".//pg:Edge", namespaces=dNS)
        deleteRegionsinDOM(ndPage, lEdgesNd)

        lClustersNd  =  ndPage.xpath(".//pg:Cluster", namespaces=dNS)
        deleteRegionsinDOM(ndPage, lClustersNd)

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
    parser.add_option("--outputextension", dest='outext',  action="store", type='string'
                      , help="add extention to the input file name")  
    parser.add_option("--overwite", dest='bOverwrite',  action="store_true", default=False
                      , help="overwrite input file") 
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
    
    main(sInputDir, options)
    
    traceln("Done.")
    