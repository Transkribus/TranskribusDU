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

from crf.Graph_MultiPageXml import Graph_MultiPageXml
from crf.Graph_Multi_SinglePageXml import Graph_MultiSinglePageXml
from crf.NodeType_PageXml   import NodeType_PageXml_type_woText, NodeType_PageXml_type
from tasks.DU_CRF_Task import DU_CRF_Task
from crf.FeatureDefinition_PageXml_std import FeatureDefinition_PageXml_StandardOnes
from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_T_PageXml_StandardOnes_noText
from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText

from tasks.DU_BAR import main
 
class DU_BAR_sem(DU_CRF_Task):
    """
    We will do a typed CRF model for a DU task
    , with the below labels 
    """
    sLabeledXmlFilenamePattern = "*.bar_mpxml"

    bHTR     = True  # do we have text from an HTR?
    bPerPage = True # do we work per document or per page?
    
    #=== CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        #DEFINING THE CLASS OF GRAPH WE USE
        if cls.bPerPage:
            DU_GRAPH = Graph_MultiSinglePageXml  # consider each age as if indep from each other
        else:
            DU_GRAPH = Graph_MultiPageXml

        lLabels1 = ['heading', 'header', 'page-number', 'resolution-number', 'resolution-marginalia', 'resolution-paragraph', 'other']
        
        #the converter changed to other unlabelled TextRegions or 'marginalia' TRs
        lIgnoredLabels1 = None
        
        """
        if you play with a toy collection, which does not have all expected classes, you can reduce those.
        """
        
#         lActuallySeen = None
#         if lActuallySeen:
#             print( "REDUCING THE CLASSES TO THOSE SEEN IN TRAINING")
#             lIgnoredLabels  = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
#             lLabels         = [lLabels[i] for i in lActuallySeen ]
#             print( len(lLabels)          , lLabels)
#             print( len(lIgnoredLabels)   , lIgnoredLabels)
        if cls.bHTR:
            ntClass = NodeType_PageXml_type
        else:
            #ignore text
            ntClass = NodeType_PageXml_type_woText
                         
        nt1 = ntClass("sem"                   #some short prefix because labels below are prefixed with it
                              , lLabels1
                              , lIgnoredLabels1
                              , False    #no label means OTHER
                              , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                              )
        nt1.setLabelAttribute("DU_sem")
        nt1.setXpathExpr( (".//pc:TextRegion"        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_GRAPH.addNodeType(nt1)
            
        return DU_GRAPH

    
    # ===============================================================================================================
    

    
    # """
    # if you play with a toy collection, which does not have all expected classes, you can reduce those.
    # """
    # 
    # lActuallySeen = None
    # if lActuallySeen:
    #     print "REDUCING THE CLASSES TO THOSE SEEN IN TRAINING"
    #     lIgnoredLabels  = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
    #     lLabels         = [lLabels[i] for i in lActuallySeen ]
    #     print len(lLabels)          , lLabels
    #     print len(lIgnoredLabels)   , lIgnoredLabels
    #     nbClass = len(lLabels) + 1  #because the ignored labels will become OTHER
    

    
    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None, C=None, tol=None, njobs=None, max_iter=None, inference_cache=None): 
        
        if self.bHTR:
            cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes
            dFeatureConfig = { 'bMultiPage':False, 'bMirrorPage':False  
                              , 'n_tfidf_node':500, 't_ngrams_node':(2,4), 'b_tfidf_node_lc':False
                              , 'n_tfidf_edge':250, 't_ngrams_edge':(2,4), 'b_tfidf_edge_lc':False }
        else:
            cFeatureDefinition = FeatureDefinition_PageXml_StandardOnes_noText
            dFeatureConfig = { 'bMultiPage':False, 'bMirrorPage':False  
                              , 'n_tfidf_node':None, 't_ngrams_node':None, 'b_tfidf_node_lc':None
                              , 'n_tfidf_edge':None, 't_ngrams_edge':None, 'b_tfidf_edge_lc':None }
        
        DU_CRF_Task.__init__(self
                     , sModelName, sModelDir
                     , dFeatureConfig = dFeatureConfig
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
                     , cFeatureDefinition=cFeatureDefinition
#                     , cFeatureDefinition=FeatureDefinition_T_PageXml_StandardOnes_noText
#                      , dFeatureConfig = {
#                          #config for the extractor of nodes of each type
#                          "text": None,    
#                          "sprtr": None,
#                          #config for the extractor of edges of each type
#                          "text_text": None,    
#                          "text_sprtr": None,    
#                          "sprtr_text": None,    
#                          "sprtr_sprtr": None    
#                          }
                     )
        
        traceln("- classes: ", self.getGraphClass().getLabelNameList())

        self.bsln_mdl = self.addBaseline_LogisticRegression()    #use a LR model trained by GridSearch as baseline
        
    #=== END OF CONFIGURATION =============================================================

  
    def predict(self, lsColDir,sDocId):
        """
        Return the list of produced files
        """
#         self.sXmlFilenamePattern = "*.a_mpxml"
        return DU_CRF_Task.predict(self, lsColDir,sDocId)


if __name__ == "__main__":
    main(DU_BAR_sem)