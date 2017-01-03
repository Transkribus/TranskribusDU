# -*- coding: utf-8 -*-

"""
    Testing the Python numpy, pystruct, AD3, libxml2, cvxopt installation - takes about 30 seconds
    
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
from tasks import  _checkFindColDir

from crf.Graph_MultiPageXml import Graph_MultiPageXml
from crf.NodeType_PageXml   import NodeType_PageXml

from tasks.DU_CRF_Task import DU_CRF_Task

 
class DU_Test(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , working on a MultiPageXMl document at TextRegion level
    , with the below labels 
    """

    #=== CONFIGURATION ====================================================================
    Metadata_Creator = "XRCE Document Understanding CRF-based + constraints - v0.1"
    Metadata_Comments = None
    
    featureExtractorConfig = { 
                        'n_tfidf_node'    : 10
                      , 't_ngrams_node'   : (2,2)
                      , 'b_tfidf_node_lc' : False    
                      , 'n_tfidf_edge'    : 10
                      , 't_ngrams_edge'   : (2,2)
                      , 'b_tfidf_edge_lc' : False    
                      }

    learnerConfig = { 'C'                 : .1 
                     , 'njobs'            : 2
                     , 'inference_cache'  : 10
                     , 'tol'              : .1
                     , 'save_every'       : 5     #save every 50 iterations,for warm start
                     #, 'max_iter'         : 1000
                     , 'max_iter'         : 8
                     }

    def getGraphClass(self):
        #we will load data from MultiPageXml XML files
        DU_Graph = Graph_MultiPageXml
        
        nt = NodeType_PageXml("TR"                   #some short prefix because labels below are prefixed with it
                              , ['catch-word', 'header', 'heading', 'marginalia', 'page-number']   #EXACTLY as in GT data!!!!
                              , []      #no ignored label/ One of those above or nothing, otherwise Exception!!
                              , True    #no label means OTHER
                              )
        nt.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_Graph.addNodeType(nt)
        
        traceln("- classes: ", DU_Graph.getLabelNameList())
                
        return DU_Graph


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

    doer = DU_Test(dFeatureConfig=DU_Test.featureExtractorConfig
                  , dLearnerConfig=DU_Test.learnerConfig)
    
    #We train, test, predict on the same document(s)
    lDir = _checkFindColDir( [sTranskribusTestDir])

    doer.train_test(sModelName, sModelDir, lDir, lDir)
    if False:
        #you can also test the prediction, but it will generate a file "7749_du.mpxml" in trnskrbs_3820/col
        doer.predict(sModelName, sModelDir, lDir)
    
    #Hack to clean
    mdl = DU_CRF_Task.ModelClass(sModelName, sModelDir) 
    os.unlink(mdl.getModelFilename())
    os.unlink(mdl.getTransformerFilename())
    os.unlink(mdl.getConfigurationFilename())
    os.rmdir(sModelDir)
        
if __name__ == "__main__":
    """
    Usage:
    python test_install.py
    
    If no exception raised, then OK!!
    """
    test_main()
    print "OK, test completed successfully."
