# -*- coding: utf-8 -*-

"""
    Create column clusters by projection profile
    
    Copyright Naver Labs Europe(C) 2019 JL Meunier
"""
import sys, os
from optparse import OptionParser

from lxml import etree

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) ))) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln

from util.Polygon import Polygon
from xml_formats.PageXml import MultiPageXml, PageXml
from graph.pkg_GraphBinaryConjugateSegmenter.GraphBinaryConjugateSegmenter import GraphBinaryConjugateSegmenter
from tasks.DU_Table.DU_ABPTableCutAnnotator import CutAnnotator


def main(lsFilename
         , fRatio, fMinHLen
         , fMinHorizProjection
         , fMinVertiProjection=0.05
         ):
    
    for sFilename in lsFilename:
        iDot = sFilename.rindex('.')
        if sFilename[:iDot].endswith("_du"): continue
        if True:
            sOutFilename = sFilename[:iDot] + "_du.mpxml"  # + "_du" + sFilename[iDot:]
        else: 
            # to mimic the bug of DU_Task.predict until today...
            sOutFilename = sFilename[:iDot-1] + "_du.mpxml" 
        traceln("- cutting: %s --> %s"%(sFilename, sOutFilename))
        
        #for the pretty printer to format better...
        parser = etree.XMLParser(remove_blank_text=True)
        doc = etree.parse(sFilename, parser)
        root=doc.getroot()
        
        doer = CutAnnotator()
        
        #     # Some grid line will be O or I simply because they are too short.
        #     fMinPageCoverage = 0.5  # minimum proportion of the page crossed by a grid line
        #                             # we want to ignore col- and row- spans
        #map the groundtruth table separators to our grid, per page (1 in tABP)
        # ltlYlX = doer.get_separator_YX_from_DOM(root, fMinPageCoverage)
        
        # clean any previous cuts:
        doer.remove_cuts_from_dom(root)
        
        # Find cuts and map them to GT
        llY, llX = doer.add_cut_to_DOM(root
                            #, ltlYlX=ltlYlX
                            , fMinHorizProjection=fMinHorizProjection
                            , fMinVertiProjection=fMinVertiProjection
                            , fRatio=fRatio
                            , fMinHLen=fMinHLen)

        add_cluster_to_dom(root, llX)
                
        doc.write(sOutFilename, encoding='utf-8', pretty_print=True, xml_declaration=True)
        traceln('Clusters and cut separators added into %s'%sOutFilename)
        
        del doc
    

def add_cluster_to_dom(root, llX):
    """
    Cluster the Textline based on the vertical cuts
    """
    
    for lX, (_iPage, ndPage) in zip(llX, enumerate(MultiPageXml.getChildByName(root, 'Page'))):
        w, _h = int(ndPage.get("imageWidth")), int(ndPage.get("imageHeight"))

        lX.append(w)
        lX.sort()
        # cluster of objects on
        imax = len(lX)
        dCluster = { i:list() for i in range(imax) }
        
        #Histogram of projections
        lndTextline = MultiPageXml.getChildByName(ndPage, 'TextLine')
        
        # hack to use addClusterToDom
        class MyBlock:
            def __init__(self, nd):
                self.node = nd
                
        o = GraphBinaryConjugateSegmenter()
        o.lNode = []
        for nd in lndTextline:
            o.lNode.append(MyBlock(nd))
        
        for iNd, ndTextline in enumerate(lndTextline):
            sPoints=MultiPageXml.getChildByName(ndTextline,'Coords')[0].get('points')
            try:
                x1,_y1,x2,_y2 = Polygon.parsePoints(sPoints).fitRectangle()
                xm = (x1 + x2) / 2.0
                bLastColumn = True
                for i, xi in enumerate(lX):
                    if xm <= xi: 
                        dCluster[i].append(iNd)
                        ndTextline.set("DU_cluster", str(i))
                        bLastColumn = False
                        break
                if bLastColumn:
                    i = imax
                    dCluster[i].append(iNd)
                    ndTextline.set("DU_cluster", str(i))
            except ZeroDivisionError:
                pass
            except ValueError:
                pass    
    
        # add clusters
        lNdCluster = o.addClusterToDom(dCluster, bMoveContent=False, sAlgo="cut", pageNode=ndPage)
        
        # add a cut_X attribute to the clusters
        for ndCluster in lNdCluster:
            i = int(ndCluster.get('name'))
            ndCluster.set("cut_X", str(lX[i]))
        
    
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    usage = """<file>+|<dir>
Generate _du.mpxml files.
"""
    version = "v.01"
    parser = OptionParser(usage=usage, version="0.1")
    parser.add_option("--ratio", dest='fRatio', action="store"
                      , type=float
                      , help="Apply this ratio to the bounding box"
                      , default=0.66) 
    parser.add_option("--fMinHLen", dest='fMinHLen', action="store"
                      , type=float
                      , help="Do not scale horizontally a bounding box with width lower than this" 
                      , default=75) 
     
    parser.add_option("--fHorizRatio", dest='fMinHorizProjection', action="store"
                      , type=float
                      , help="On the horizontal projection profile, it ignores profile lower than this ratio of the page width"
                      , default=0.05) 
            
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    traceln(options)
    
    if args and all(map(os.path.isfile, args)):
        lsFile = args
        traceln("Working on files: ", lsFile)
    elif args and os.path.isdir(args[0]):
        sDir = args[0]
        traceln("Working on folder: ", sDir)
        lsFile = [os.path.join(sDir, s) for s in os.listdir(sDir) if s.lower().endswith("pxml")]
    else:
        traceln("Usage : %s " % sys.argv[0], usage)
        sys.exit(1)
    main(lsFile
         , options.fRatio, fMinHLen=options.fMinHLen
         , fMinHorizProjection=options.fMinHorizProjection
         )    

