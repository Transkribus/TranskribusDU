# -*- coding: utf-8 -*-

"""
    Create column segmenters
    
    Copyright Naver Labs Europe(C) 2018 JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os
from optparse import OptionParser

from lxml import etree


try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _exit
from tasks.DU_ABPTableCutAnnotator import CutAnnotator


def main(sFilename, sOutFilename
         , fRatio, fMinHLen
         , fMinHorizProjection, fMinVertiProjection
         ):
    
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
    
    # Find cuts and map them to GT
    # 
    doer.add_cut_to_DOM(root
                        #, ltlYlX=ltlYlX
                        , fMinHorizProjection=fMinHorizProjection
                        , fMinVertiProjection=fMinVertiProjection
                        , fRatio=fRatio
                        , fMinHLen=fMinHLen)
    
    #l_DU_row_Y, l_DU_row_GT = doer.predict(root)
    
    doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
    traceln('Annotated cut separators added into %s'%sOutFilename)
    
    del doc
    
    
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    usage = """<file>+|<from-dir> <to-dir>"""
    version = "v.01"
    parser = OptionParser(usage=usage, version="0.1")
    parser.add_option("--ratio", dest='fRatio', action="store"
                      , type=float
                      , help="Apply this ratio to the bounding box"
                      , default=CutAnnotator.fRATIO) 
    parser.add_option("--fMinHLen", dest='fMinHLen', action="store"
                      , type=float
                      , help="Do not scale horizontally a bounding box with width lower than this" 
                      , default=75) 
     
    parser.add_option("--fHorizRatio", dest='fMinHorizProjection', action="store"
                      , type=float
                      , help="On the horizontal projection profile, it ignores profile lower than this ratio of the page width"
                      , default=0.05) 
    parser.add_option("--fVertRatio", dest='fMinVertiProjection', action="store"
                      , type=float
                      , help="On the vertical projection profile, it ignores profile lower than this ratio of the page height" 
                      , default=0.05) 
#     parser.add_option("--SIO"           , dest='bSIO'          ,  action="store_true", help="SIO labels") 
#     parser.add_option("--annotate", dest='bAnnotate',  action="store_true",default=False,  help="Annotate the textlines with BIES labels")    
    
#     parser.add_option("--detail", dest='bDetailedReport',  action="store_true", default=False,help="Display detailed reporting (score per document)") 
#     parser.add_option("--baseline", dest='bBaseline',  action="store_true", default=False, help="report baseline method") 
#     parser.add_option("--line_see_line", dest='iLineVisibility',  action="store",
#                       type=int, default=GraphSkewedCut.iLineVisibility,
#                       help="seeline2line: how far in pixel can a line see another cut line?") 
#     parser.add_option("--block_see_line", dest='iBlockVisibility',  action="store",
#                       type=int, default=GraphSkewedCut.iBlockVisibility,
#                       help="seeblock2line: how far in pixel can a block see a cut line?") 
#     parser.add_option("--height", dest="fCutHeight", default=GraphSkewedCut.fCutHeight
#                       , action="store", type=float, help="Minimal height of a cut") 
#     #     parser.add_option("--cut-above", dest='bCutAbove',  action="store_true", default=False
#     #                         ,help="Each object defines one or several cuts above it (instead of below as by default)") 
#     parser.add_option("--angle", dest='lsAngle'
#                       ,  action="store", type="string", default="-1,0,+1"
#                         ,help="Allowed cutting angles, in degree, comma-separated") 
# 
#     parser.add_option("--graph", dest='bGraph',  action="store_true", help="Store the graph in the XML for displaying it") 
#     parser.add_option("--bioh", "--BIOH", dest='bBIOH',  action="store_true", help="Text are categorised along BIOH instead of BIO") 
            
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    traceln(options)
    
    if len(args) == 2 and os.path.isdir(args[0]) and os.path.isdir(args[1]):
        # ok, let's work differently...
        sFromDir,sToDir = args
        for s in os.listdir(sFromDir):
            if not s.lower().endswith("pxml"): pass
            sFilename = sFromDir + "/" + s
            sp, sf = os.path.split(s)
            sOutFilename = sToDir + "/" + "cut-" + sf
            traceln(sFilename,"  -->  ", sOutFilename)
            main(sFilename, sOutFilename
             , options.fRatio, fMinHLen=options.fMinHLen
             , fMinHorizProjection=options.fMinHorizProjection
             , fMinVertiProjection=options.fMinVertiProjection 
             )    
    else:
        for sFilename in args:
            sp, sf = os.path.split(sFilename)
            sOutFilename = os.path.join(sp, "cut-" + sf)    
            traceln(sFilename,"  -->  ", sOutFilename)
            main(sFilename, sOutFilename
                 , options.fRatio, fMinHLen=options.fMinHLen
                 , fMinHorizProjection=options.fMinHorizProjection
                 , fMinVertiProjection=options.fMinVertiProjection
                 )    

