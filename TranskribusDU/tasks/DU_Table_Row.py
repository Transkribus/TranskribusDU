# -*- coding: utf-8 -*-

"""
    *** Same as its parent apart that text baselines are reflected as a LineString (instead of its centroid)
    
    DU task for ABP Table: 
        doing jointly row BIO and near horizontal cuts SIO
    
    block2line edges do not cross another block.
    
    The cut are based on baselines of text blocks, with some positive or negative inclination.

    - the labels of cuts are SIO 
    
    Copyright Naver Labs Europe(C) 2018 JL Meunier

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os

import math

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _exit
from tasks.DU_CRF_Task import DU_CRF_Task
from tasks.DU_ABPTableSkewed import GraphSkewedCut, main
from tasks.DU_ABPTableSkewed_CutAnnotator import SkewedCutAnnotator
from tasks.DU_ABPTableSkewed_txtBIO_sepSIO_line import DU_ABPTableSkewedRowCutLine
from tasks.DU_ABPTableSkewed_txtBIOH_sepSIO_line import DU_ABPTableSkewedRowCutLine_BIOH

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    
    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)
#     parser.add_option("--annotate", dest='bAnnotate',  action="store_true",default=False,  help="Annotate the textlines with BIES labels")    
    
    #FOR GCN
    # parser.add_option("--revertEdges", dest='bRevertEdges',  action="store_true", help="Revert the direction of the edges") 
    parser.add_option("--detail", dest='bDetailedReport',  action="store_true", default=False,help="Display detailed reporting (score per document)") 
    parser.add_option("--baseline", dest='bBaseline',  action="store_true", default=False, help="report baseline method") 
    parser.add_option("--line_see_line", dest='iLineVisibility',  action="store",
                      type=int, default=GraphSkewedCut.iLineVisibility,
                      help="seeline2line: how far in pixel can a line see another cut line?") 
    parser.add_option("--block_see_line", dest='iBlockVisibility',  action="store",
                      type=int, default=GraphSkewedCut.iBlockVisibility,
                      help="seeblock2line: how far in pixel can a block see a cut line?") 
    parser.add_option("--height", dest="fCutHeight", default=GraphSkewedCut.fCutHeight
                      , action="store", type=float, help="Minimal height of a cut") 
    #     parser.add_option("--cut-above", dest='bCutAbove',  action="store_true", default=False
    #                         ,help="Each object defines one or several cuts above it (instead of below as by default)") 
    parser.add_option("--angle", dest='lsAngle'
                      ,  action="store", type="string", default="-1,0,+1"
                        ,help="Allowed cutting angles, in degree, comma-separated") 

    parser.add_option("--graph", dest='bGraph',  action="store_true", help="Store the graph in the XML for displaying it") 
    parser.add_option("--bioh", "--BIOH", dest='bBIOH',  action="store_true", help="Text are categorised along BIOH instead of BIO") 
    parser.add_option("--text", "--txt", dest='bTxt',  action="store_true", help="Use textual features.") 
            
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    options.bCutAbove = True    # Forcing this!

    if options.bBIOH:
        DU_CLASS = DU_ABPTableSkewedRowCutLine_BIOH
    else:
        DU_CLASS = DU_ABPTableSkewedRowCutLine
    
    if options.bGraph:
        import os.path
        # hack
        DU_CLASS.bCutAbove = options.bCutAbove
        traceln("\t%s.bCutAbove=" % DU_CLASS.__name__, DU_CLASS.bCutAbove)
        DU_CLASS.lRadAngle = [math.radians(v) for v in [float(s) for s in options.lsAngle.split(",")]]
        traceln("\t%s.lRadAngle=" % DU_CLASS.__name__, DU_CLASS.lRadAngle)
        for sInputFilename in args:
            sp, sf = os.path.split(sInputFilename)
            sOutFilename = os.path.join(sp, "graph-" + sf)
            doer = DU_CLASS("debug", "."
                           , iBlockVisibility=options.iBlockVisibility
                           , iLineVisibility=options.iLineVisibility
                           , fCutHeight=options.fCutHeight
                           , bCutAbove=options.bCutAbove
                           , lRadAngle=[math.radians(float(s)) for s in options.lsAngle.split(",")]
                           , bTxt=options.bTxt)
            o = doer.cGraphClass()
            o.parseXmlFile(sInputFilename, 9)
            o.addEdgeToDOM()
            print('Graph edges added to %s'%sOutFilename)
            o.doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
        SkewedCutAnnotator.gtStatReport()
        exit(0)
    
    # --- 
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        _exit(usage, 1, e)
    
    main(DU_CLASS, sModelDir, sModelName, options)
