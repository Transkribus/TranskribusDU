# -*- coding: utf-8 -*-

"""
    DU task for BAR - see https://read02.uibk.ac.at/wiki/index.php/Document_Understanding_BAR
    
    Copyright Xerox(C) 2017 JL Meunier

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

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln

from crf.Graph_MultiPageXml import FactorialGraph_MultiContinuousPageXml
from crf.NodeType_PageXml   import NodeType_PageXml_type_woText
from DU_CRF_Task import DU_FactorialCRF_Task
from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_T_PageXml_StandardOnes_noText
from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText

from DU_BAR import main
 
class DU_BAR_sem_sgm(DU_FactorialCRF_Task):
    """
    We will do a Factorial CRF model using the Multitype CRF 
    , with the below labels 
    """
    sLabeledXmlFilenamePattern = "*.du_mpxml"

    # ===============================================================================================================
    #DEFINING THE CLASS OF GRAPH WE USE
    DU_GRAPH = FactorialGraph_MultiContinuousPageXml
    
    #---------------------------------------------
    lLabels1 = ['heading', 'header', 'page-number', 'resolution-number', 'resolution-marginalia', 'resolution-paragraph', 'other']
    
    nt1 = NodeType_PageXml_type_woText("sem"                   #some short prefix because labels below are prefixed with it
                          , lLabels1
                          , None                                #keep this to None, unless you know very well what you do. (FactorialCRF!)
                          , False    #no label means OTHER
                          , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                          )
    nt1.setLabelAttribute("DU_sem")
    nt1.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes, MUST be same as for other node type!! (FactorialCRF!)
                      , "./pc:TextEquiv")       #how to get their text
                   )
    DU_GRAPH.addNodeType(nt1)

    #---------------------------------------------
    #lLabels2 = ['heigh', 'ho', 'other']
    #lLabels2 = ['heigh', 'ho']
    lLabels2 = ['B', 'I', 'E']  #we never see any S...  , 'S']
    
    nt2 = NodeType_PageXml_type_woText("sgm"                   #some short prefix because labels below are prefixed with it
                          , lLabels2
                          , None                                #keep this to None, unless you know very well what you do. (FactorialCRF!)
                          , False    #no label means OTHER
                          , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                          )
    nt2.setLabelAttribute("DU_sgm")
    nt2.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes, MUST be same as for other node type!! (FactorialCRF!)
                      , "./pc:TextEquiv")       #how to get their text
                   )
    DU_GRAPH.addNodeType(nt2)
    
    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None, C=None, tol=None, njobs=None, max_iter=None, inference_cache=None): 
        
#         #edge feature extractor config is a bit teddious...
#         dFeatureConfig = { lbl:None for lbl in self.lLabels1+self.lLabels2 }
#         for lbl1 in self.lLabels1:
#             for lbl2 in self.lLabels2:
#                 dFeatureConfig["%s_%s"%(lbl1, lbl2)] = None
        
        DU_FactorialCRF_Task.__init__(self
                     , sModelName, sModelDir
                     , self.DU_GRAPH
                     , dLearnerConfig = {
                                   'C'                : .1   if C               is None else C
                                 , 'njobs'            : 8    if njobs           is None else njobs
                                 , 'inference_cache'  : 50   if inference_cache is None else inference_cache
                                 #, 'tol'              : .1
                                 , 'tol'              : .05  if tol             is None else tol
                                 , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 1000 if max_iter        is None else max_iter
                         }
                     , sComment=sComment
                     , cFeatureDefinition=FeatureDefinition_PageXml_StandardOnes_noText
#                     , cFeatureDefinition=FeatureDefinition_T_PageXml_StandardOnes_noText
#                     {
#                         #config for the extractor of nodes of each type
#                         "text": None,    
#                         "sprtr": None,
#                         #config for the extractor of edges of each type
#                         "text_text": None,    
#                         "text_sprtr": None,    
#                         "sprtr_text": None,    
#                         "sprtr_sprtr": None    
#                         }
                     )
        
        traceln("- classes: ", self.DU_GRAPH.getLabelNameList())

        self.bsln_mdl = self.addBaseline_LogisticRegression()    #use a LR model trained by GridSearch as baseline
        
    #=== END OF CONFIGURATION =============================================================

  
    def predict(self, lsColDir,sDocId):
        """
        Return the list of produced files
        """
#         self.sXmlFilenamePattern = "*.a_mpxml"
        return DU_CRF_Task.predict(self, lsColDir,sDocId)


if __name__ == "__main__":
    main(DU_BAR_sem_sgm)