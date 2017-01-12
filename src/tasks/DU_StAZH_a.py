# -*- coding: utf-8 -*-

"""
    First DU task for StAZH
    
    Copyright Xerox(C) 2016 JL. Meunier

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

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _checkFindColDir, _exit

from crf.Graph_MultiPageXml import Graph_MultiPageXml
from crf.NodeType_PageXml   import NodeType_PageXml
from DU_CRF_Task import DU_CRF_Task


# ===============================================================================================================
#DEFINING THE CLASS OF GRAPH WE USE
DU_GRAPH = Graph_MultiPageXml
nt = NodeType_PageXml("TR"                   #some short prefix because labels below are prefixed with it
                      , ['catch-word', 'header', 'heading', 'marginalia', 'page-number']   #EXACTLY as in GT data!!!!
                      , []      #no ignored label/ One of those above or nothing, otherwise Exception!!
                      , True    #no label means OTHER
                      )
nt.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes
                  , "./pc:TextEquiv")       #how to get their text
               )
DU_GRAPH.addNodeType(nt)
# ===============================================================================================================

 
class DU_StAZH_a(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , working on a MultiPageXMl document at TextRegion level
    , with the below labels 
    """

    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None): 
        
        DU_CRF_Task.__init__(self
                             , sModelName, sModelDir
                             , DU_GRAPH
                             , dFeatureConfig = {
                                    'n_tfidf_node'    : 500
                                  , 't_ngrams_node'   : (2,4)
                                  , 'b_tfidf_node_lc' : False    
                                  , 'n_tfidf_edge'    : 250
                                  , 't_ngrams_edge'   : (2,4)
                                  , 'b_tfidf_edge_lc' : False    
                              }
                             , dLearnerConfig = {
                                    'C'                : .1 
#                                    'C'                : 1.0 
                                 , 'njobs'            : 4
                                 , 'inference_cache'  : 50
                                , 'tol'              : .1
#                                  , 'tol'              : 0.05
                                , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 250
                                 }
                             , sComment=sComment
                             )
        
        self.addBaseline_LogisticRegression()    #use a LR model as baseline
    #=== END OF CONFIGURATION =============================================================


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
        
    doer = DU_StAZH_a(sModelName, sModelDir)
    
    if options.rm:
        doer.rm()
        sys.exit(0)
    
    traceln("- classes: ", DU_GRAPH.getLabelNameList())
    
    
    #Add the "col" subdir if needed
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
    
