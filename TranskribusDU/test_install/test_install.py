# -*- coding: utf-8 -*-

"""
    Testing the Python numpy, pystruct, AD3, cvxopt installation - takes about 30 seconds
    
    Copyright Xerox(C) 2016 JL. Meunier


    
    
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
    TranskribusDU_version.version
from common.trace import traceln
from tasks import  _checkFindColDir

from crf.Graph import Graph
from crf.Graph_MultiPageXml import Graph_MultiPageXml
from crf.NodeType_PageXml   import NodeType_PageXml

from tasks.DU_CRF_Task import DU_CRF_Task

 
class DU_Test(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , working on a MultiPageXMl document at TextRegion level
    , with the below labels 
    """
    @classmethod
    def getConfiguredGraphClass(cls):
        Graph.resetNodeTypes()
        
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
        return DU_GRAPH
    
    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None): 
        
        DU_CRF_Task.__init__(self
                             , sModelName, sModelDir
                             , dFeatureConfig = {
                                    'n_tfidf_node'    : 10
                                  , 't_ngrams_node'   : (2,2)
                                  , 'b_tfidf_node_lc' : False    
                                  , 'n_tfidf_edge'    : 10
                                  , 't_ngrams_edge'   : (2,2)
                                  , 'b_tfidf_edge_lc' : False    
                              }
                             , dLearnerConfig = {
                                   'C'                : .1 
                                 , 'njobs'            : 2
                                 , 'inference_cache'  : 10
                                 , 'tol'              : .1
                                 , 'save_every'       : 5     #save every 50 iterations,for warm start
                                 #, 'max_iter'         : 1000
                                 , 'max_iter'         : 2
                                 }
                             , sComment=sComment
                             )
        
        self.addBaseline_LogisticRegression()    #use a LR model as baseline
        

def test_main():
    """
    Usage :
    pytest test_install.py

    If no exception raised, then OK!!
    """
    import tempfile
    sModelDir = tempfile.mkdtemp()
    sModelName = "toto"
    traceln("Working in temp dir: %s"%sModelDir)

    sTranskribusTestDir = os.path.join(os.path.dirname(__file__), "trnskrbs_3820")

    #We train, test, predict on the same document(s)
    lDir = _checkFindColDir( [sTranskribusTestDir])

    traceln("- training and testing a model")
    doer = DU_Test(sModelName, sModelDir, "Installation test, with almost fake training and testing.")
    oTestReport = doer.train_save_test(lDir, lDir)
    traceln(oTestReport)
    del doer
    traceln("DONE")
    
    traceln("- loading and testing a model")
    doer2 = DU_Test(sModelName, sModelDir, "Testing the load of a model")
    doer2.load()
    oTestReport = doer2.test(lDir)
    traceln(oTestReport)
    del doer2
    traceln("DONE")
    
    traceln("- loading and predicting")
    doer3 = DU_Test(sModelName, sModelDir, "Predicting")
    doer3.load()
    l = doer3.predict(lDir)
    del doer3
    traceln(l)
    traceln("DONE")
    
    #cleaning!
    mdl = DU_Test(sModelName, sModelDir) 
    mdl.rm()
        
if __name__ == "__main__":
    """
    Usage:
    python test_install.py
    
    If no exception raised, then OK!!
    """
    test_main()
    print("OK, test completed successfully.")
