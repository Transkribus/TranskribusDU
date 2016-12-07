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

from crf.Graph_MultiPageXml_TextRegion import Graph_MultiPageXml_TextRegion
from crf.FeatureExtractors_PageXml_std import FeatureExtractors_PageXml_StandardOnes

from crf.Model_SSVM_AD3 import Model_SSVM_AD3

from DU_CRF_Task import DU_CRF_Task

from common.trace import traceln
 
class DU_StAZH_Graph(Graph_MultiPageXml_TextRegion):
    """
    Specializing the graph for this particular task
    """
    #            1            2            3        4                5
    TASK_LABELS = ['catch-word', 'header', 'heading', 'marginalia', 'page-number']
    
    def __init__(self): 
        Graph_MultiPageXml_TextRegion.__init__(self)
        self.setLabelList(self.TASK_LABELS, True)  #True means non-annotated node are of class 0 = OTHER
    
class DU_StAZH_a(DU_CRF_Task, FeatureExtractors_PageXml_StandardOnes):
    """
    We will do a CRF model for a DU task
    , working on a MultiPageXMl document at TextRegion level
    , with the labels defined in StAZH_Label class
    """

    n_tfidf_node    = 300
    t_ngrams_node   = (2,4)
    b_tfidf_node_lc = True    
    n_tfidf_edge    = 300
    t_ngrams_edge    = (2,4)
    b_tfidf_edge_lc = True    
    
    def __init__(self): 
        DU_CRF_Task.__init__(self)
        FeatureExtractors_PageXml_StandardOnes.__init__(self
                                                        , self.n_tfidf_node, self.t_ngrams_node, self.b_tfidf_node_lc
                                                        , self.n_tfidf_edge, self.t_ngrams_edge, self.b_tfidf_edge_lc)
    
    def run(self, sModelName, sModelDir, lsTrnColDir, lsTstColDir):
        """
        Train a model on the tTRN collections and test it using the TST collections
        """
        traceln("-"*50)
        traceln("Training model '%s' in folder '%s'"%(sModelName, sModelDir))
        traceln("Train collection(s):", lsTrnColDir)
        traceln("Test  collection(s):", lsTstColDir)
        traceln("-"*50)
        
        #list the train and test files
        ts_trn, lTSFilename_trn = self.listMaxTimestampFile(lsTrnColDir, "*.mpxml")
        _     , lTSFilename_tst = self.listMaxTimestampFile(lsTstColDir, "*.mpxml")
        
        du = DU_StAZH_Graph()
        
        traceln("- loading training graphs")
        lGraph_trn = du.loadDetachedGraphs(lTSFilename_trn, True, 1)  #True because we load the labels as well
        traceln(" %d graphs loaded"%len(lGraph_trn))
            
        traceln("- creating a %s model"%Model_SSVM_AD3)
        mdl = Model_SSVM_AD3(sModelName, sModelDir)

        mdl.setClassNames(du.getLabelList())
        traceln("- classes: ", du.getLabelList())

        traceln("- retrieving or creating feature extractors...")
        try:
            mdl.loadFittedTransformers(ts_trn)
        except:
            mdl.setTranformers(self.getTransformers())
            mdl.fitTranformers(lGraph_trn)
            self.clean_transformers(mdl.getTransformers())
            mdl.saveTransformers()
        traceln(" done")
        
        traceln("- training model...")
        mdl.train(lGraph_trn, True, ts_trn)
        traceln(" done")
        
        traceln("- loading test graphs")
        lGraph_tst = self.loadDetachedLabelledGraphs(lTSFilename_tst, True, 1)
        traceln(" %d graphs loaded"%len(lGraph_trn))

        mdl.test(lGraph_tst)


if __name__ == "__main__":
    
    sTopDir = "C:\\Local\\meunier\\git\\TranskribusDU\\usecases\\StAZH\\"
    doer = DU_StAZH_a()
    doer.run("DU_StAZH_a", "c:\\tmp_READ"
              , [sTopDir+"trnskrbs_3820\\col"]
              , [sTopDir+"trnskrbs_3832\\col"])
    
