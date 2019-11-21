# -*- coding: utf-8 -*-

"""
    Example DU task for Dodge, using the logit textual feature extractor
    
    Copyright Xerox(C) 2017 JL. Meunier


    
    
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
from crf.NodeType_PageXml   import NodeType_PageXml_type_NestedText
from DU_CRF_Task import DU_CRF_Task
from DU_BL_Task import DU_Baseline
from crf.FeatureDefinition_PageXml_GTBooks import FeatureDefinition_GTBook

# ===============================================================================================================

lLabels =  ['TOC-entry'         #0
            , 'caption'
            , 'catch-word'
                         , 'footer'
                         , 'footnote'                #4
                         , 'footnote-continued'
                         , 'header'             #6
						 , 'heading'          #7
                         , 'marginalia'
                         , 'page-number'    #9
                         , 'paragraph'    #10
                         , 'signature-mark']
lIgnoredLabels = None

nbClass = len(lLabels)

"""
if you play with a toy collection, which does not have all expected classes, you can reduce those.
"""
lActuallySeen = [4, 6, 7, 9, 10]
#lActuallySeen = [4, 6]
"""
                0-            TOC-entry    5940 occurences       (   2%)  (   2%)
                1-              caption     707 occurences       (   0%)  (   0%)
                2-           catch-word     201 occurences       (   0%)  (   0%)
                3-               footer      11 occurences       (   0%)  (   0%)
                4-             footnote   36942 occurences       (  11%)  (  11%)
                5-   footnote-continued    1890 occurences       (   1%)  (   1%)
                6-               header   15910 occurences       (   5%)  (   5%)
                7-              heading   18032 occurences       (   6%)  (   6%)
                8-           marginalia    4292 occurences       (   1%)  (   1%)
                9-          page-number   40236 occurences       (  12%)  (  12%)
               10-            paragraph  194927 occurences       (  60%)  (  60%)
               11-       signature-mark    4894 occurences       (   2%)  (   2%)
"""
lActuallySeen = None
if lActuallySeen:
    traceln("REDUCING THE CLASSES TO THOSE SEEN IN TRAINING")
    lIgnoredLabels  = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
    lLabels         = [lLabels[i] for i in lActuallySeen ]
    traceln(len(lLabels)          , lLabels)
    traceln(len(lIgnoredLabels)   , lIgnoredLabels)
    nbClass = len(lLabels) + 1  #because the ignored labels will become OTHER

    #DEFINING THE CLASS OF GRAPH WE USE
    DU_GRAPH = Graph_MultiPageXml
    nt = NodeType_PageXml_type_NestedText("gtb"                   #some short prefix because labels below are prefixed with it
                          , lLabels
                          , lIgnoredLabels
                              , True    #no label means OTHER
                              )
else:
    #DEFINING THE CLASS OF GRAPH WE USE
    DU_GRAPH = Graph_MultiPageXml
    nt = NodeType_PageXml_type_NestedText("gtb"                   #some short prefix because labels below are prefixed with it
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

