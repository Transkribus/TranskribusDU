# -*- coding: utf-8 -*-

"""
    Example DU task for Dodge, using the logit textual feature extractor
    
    Copyright Xerox(C) 2017 JL. Meunier

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
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os
from crf import FeatureDefinition_PageXml_GTBooks

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _checkFindColDir, _exit

from crf.Graph_MultiPageXml import Graph_MultiPageXml
from crf.NodeType_PageXml   import NodeType_PageXml_type_GTBooks

from DU_CRF_Task import DU_CRF_Task
from DU_BL_Task import DU_Baseline
from crf.FeatureDefinition_PageXml_GTBooks import FeatureDefinition_GTBook

# ===============================================================================================================

lLabels = ['TOC-entry', 'caption', 'catch-word'
                         , 'footer', 'footnote', 'footnote-continued'
                         , 'header', 'heading', 'marginalia', 'page-number'
                         , 'paragraph', 'signature-mark']   #EXACTLY as in GT data!!!!
lIgnoredLabels = None

nbClass = len(lLabels)

"""
if you play with a toy collection, which does not have all expected classes, you can reduce those.
"""
lActuallySeen = [0, 4, 7, 9, 10]
# lActuallySeen = None
if lActuallySeen:
    print "REDUCING THE CLASSES TO THOSE SEEN IN TRAINING"
    lIgnoredLabels  = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
    lLabels         = [lLabels[i] for i in lActuallySeen ]
    print len(lLabels)          , lLabels
    print len(lIgnoredLabels)   , lIgnoredLabels
    nbClass = len(lLabels) + 1  #because the ignored labels will become OTHER

#DEFINING THE CLASS OF GRAPH WE USE
DU_GRAPH = Graph_MultiPageXml
nt = NodeType_PageXml_type_GTBooks("gtb"                   #some short prefix because labels below are prefixed with it
                      , lLabels
                      , lIgnoredLabels
                      , False    #no label means OTHER
                      )
nt.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes
                  , "./pc:TextEquiv")       #how to get their text
               )
DU_GRAPH.addNodeType(nt)

"""
The constraints must be a list of tuples like ( <operator>, <NodeType>, <states>, <negated> )
where:
- operator is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
- states is a list of unary state names, 1 per involved unary. If the states are all the same, you can pass it directly as a single string.
- negated is a list of boolean indicated if the unary must be negated. Again, if all values are the same, pass a single boolean value instead of a list 
"""
if False:
    DU_GRAPH.setPageConstraint( [    ('ATMOSTONE', nt, 'pnum' , False)    #0 or 1 catch_word per page
                                   , ('ATMOSTONE', nt, 'title'    , False)    #0 or 1 heading pare page
                                 ] )

# ===============================================================================================================


class DU_BL_V1(DU_Baseline):
    def __init__(self, sModelName, sModelDir,logitID,sComment=None):
        DU_Baseline.__init__(self, sModelName, sModelDir,DU_GRAPH,logitID)



if __name__ == "__main__":

    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)

    # ---
    #parse the command line
    (options, args) = parser.parse_args()
    # ---
    try:
        sModelDir, sModelName = args
    except Exception as e:
        _exit(usage, 1, e)

    doer = DU_BL_V1(sModelName, sModelDir,'logit_5')

    if options.rm:
        doer.rm()
        sys.exit(0)

    traceln("- classes: ", DU_GRAPH.getLabelNameList())

    if hasattr(options,'l_train_files') and hasattr(options,'l_test_files'):
        f=open(options.l_train_files)
        lTrn=[]
        for l in f:
            fname=l.rstrip()
            lTrn.append(fname)
        f.close()

        g=open(options.l_test_files)
        lTst=[]
        for l in g:
            fname=l.rstrip()
            lTst.append(fname)

        tstReport=doer.train_save_test(lTrn, lTst, options.warm,filterFilesRegexp=False)
        traceln(tstReport)


    else:

        lTrn, lTst, lRun = [_checkFindColDir(lsDir) for lsDir in [options.lTrn, options.lTst, options.lRun]]

        if lTrn:
            doer.train_save_test(lTrn, lTst, options.warm)
        elif lTst:
            doer.load()
            tstReport = doer.test(lTst)
            traceln(tstReport)

        if lRun:
            doer.load()
            lsOutputFilename = doer.predict(lRun)
            traceln("Done, see in:\n  %s"%lsOutputFilename)

