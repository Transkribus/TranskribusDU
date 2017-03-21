# -*- coding: utf-8 -*-

"""
    Example DU task for Dodge
    
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

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _checkFindColDir, _exit

from crf.Graph_DSXml import Graph_DSXml
from crf.NodeType_DSXml   import NodeType_DS
from DU_CRF_Task import DU_CRF_Task,DU_CRF_FS_Task


import dodge_graph

'''
# ===============================================================================================================
#DEFINING THE CLASS OF GRAPH WE USE
DU_DODGE_GRAPH = Graph_DSXml
nt = NodeType_DS("Ddg"                   #some short prefix because labels below are prefixed with it
                      , ['title', 'pnum']   #EXACTLY as in GT data!!!!
                      , []      #no ignored label/ One of those above or nothing, otherwise Exception!!
                      , True    #no label means OTHER
                      )
nt.setXpathExpr( ".//BLOCK"        #how to find the nodes
               )
DU_DODGE_GRAPH.addNodeType(nt)

"""
The constraints must be a list of tuples like ( <operator>, <NodeType>, <states>, <negated> )
where:
- operator is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
- states is a list of unary state names, 1 per involved unary. If the states are all the same, you can pass it directly as a single string.
- negated is a list of boolean indicated if the unary must be negated. Again, if all values are the same, pass a single boolean value instead of a list 
"""


#DU_DODGE_GRAPH.setPageConstraint([('ATMOSTONE', nt, 'pnum' , False)  #0 or 1 catch_word per page
#                               , ('ATMOSTONE', nt, 'title'    , False)  #0 or 1 heading pare page
#                                  ])

# ===============================================================================================================
'''

 
class DU_Dodge(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , working on a DS XML document at BLOCK level
    , with the below labels 
    """
    sXmlFilenamePattern = "*_ds.xml"

    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None): 
        
        DU_CRF_Task.__init__(self
                             , sModelName, sModelDir
                             , dodge_graph.DU_GRAPH
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
                                 , 'njobs'            : 4
                                 , 'inference_cache'  : 50
                                 #, 'tol'              : .1
                                 , 'tol'              : .05
                                 , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 250
                                 }
                             , sComment=sComment
                             , cFeatureDefinition=None  #SO THAT WE USE THE SAME FEATURES AS FOR PageXml (because it is the features by default)
                             )
        
        self.addBaseline_LogisticRegression()    #use a LR model as baseline
    #=== END OF CONFIGURATION =============================================================


class DU_Dodge_CRF_FS(DU_CRF_FS_Task):
    """
    We will do a CRF model for a DU task
    , working on a DS XML document at BLOCK level
    , with the below labels
    """
    #sXmlFilenamePattern = "*_ds.xml"
    #sXmlFilenamePattern ="GraphML_R33*_ds.xml"
    sXmlFilenamePattern = "*_ds.xml"

    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None):

        DU_CRF_FS_Task.__init__(self
                             , sModelName, sModelDir
                             , DU_DODGE_GRAPH
                             , dFeatureConfig = {
                                    'n_tfidf_node'    : 500
                                  , 't_ngrams_node'   : (2,4)
                                  , 'b_tfidf_node_lc' : False
                                  , 'n_tfidf_edge'    : 250
                                  , 't_ngrams_edge'   : (2,4)
                                  , 'b_tfidf_edge_lc' : False
                                  , 'feat_select':'chi2'
                              }
                             , dLearnerConfig = {
                                   'C'                : .1
                                 , 'njobs'            : 1
                                 , 'inference_cache'  : 50
                                 #, 'tol'              : .1
                                 , 'tol'              : .05
                                 , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 250
                                 }
                             , sComment=sComment
                             , cFeatureDefinition=None  #SO THAT WE USE THE SAME FEATURES AS FOR PageXml (because it is the features by default)
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
        
    doer = DU_Dodge(sModelName, sModelDir)
    
    if options.rm:
        doer.rm()
        sys.exit(0)
    
    traceln("- classes: ", DU_DODGE_GRAPH.getLabelNameList())
    
    
    #Add the "out" subdir if needed

    lTrn, lTst, lRun = [_checkFindColDir(lsDir, "out") for lsDir in [options.lTrn, options.lTst, options.lRun]] 
    #print lTrn
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
