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
from crf.Model import ModelException
from crf.Model_SSVM_AD3 import Model_SSVM_AD3

from DU_CRF_Task import DU_CRF_Task

from common.trace import traceln
 
class DU_StAZH_a(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , working on a MultiPageXMl document at TextRegion level
    , with the below labels 
    """

    #  0=OTHER        1            2            3        4                5
    TASK_LABELS = ['catch-word', 'header', 'heading', 'marginalia', 'page-number']

    n_tfidf_node    = 500
    t_ngrams_node   = (2,4)
    b_tfidf_node_lc = False    
    n_tfidf_edge    = 250
    t_ngrams_edge    = (2,4)
    b_tfidf_edge_lc = False    

    
    def __init__(self): 
        DU_CRF_Task.__init__(self)
    
    def getGraphClass(self):
        DU_StAZH_Graph = Graph_MultiPageXml_TextRegion
        DU_StAZH_Graph.setLabelList(self.TASK_LABELS, True)  #True means non-annotated node are of class 0 = OTHER
        traceln("- classes: ", DU_StAZH_Graph.getLabelList())
        return DU_StAZH_Graph

    def train_test(self, sModelName, sModelDir, lsTrnColDir, lsTstColDir):
        """
        Train a model on the tTRN collections and test it using the TST collections
        """
        traceln("-"*50)
        traceln("Training model '%s' in folder '%s'"%(sModelName, sModelDir))
        traceln("Train collection(s):", lsTrnColDir)
        traceln("Test  collection(s):", lsTstColDir)
        traceln("-"*50)
        
        #list the train and test files
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, "*.mpxml")
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, "*.mpxml")
        
        DU_StAZH_Graph = self.getGraphClass()
        
        traceln("- loading training graphs")
        lGraph_trn = DU_StAZH_Graph.loadGraphs(lFilename_trn, bDetach=True, bLabelled=True, iVerbose=1)
        traceln(" %d graphs loaded"%len(lGraph_trn))
            
        traceln("- creating a %s model"%Model_SSVM_AD3)
        mdl = Model_SSVM_AD3(sModelName, sModelDir)

        traceln("- retrieving or creating feature extractors...")
        try:
            mdl.loadTransformers(ts_trn)
        except ModelException:
            fe = FeatureExtractors_PageXml_StandardOnes(self.n_tfidf_node, self.t_ngrams_node, self.b_tfidf_node_lc
                                                        , self.n_tfidf_edge, self.t_ngrams_edge, self.b_tfidf_edge_lc)         
            fe.fitTranformers(lGraph_trn)
            fe.clean_transformers()
            mdl.setTranformers(fe.getTransformers())
            mdl.saveTransformers()
        traceln(" done")
        
        traceln("- training model...")
        mdl.train(lGraph_trn, True, ts_trn)
        traceln(" done")
        
        traceln("- loading test graphs")
        lGraph_tst = DU_StAZH_Graph.loadGraphs(lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
        traceln(" %d graphs loaded"%len(lGraph_tst))

        fScore, sReport = mdl.test(lGraph_tst, DU_StAZH_Graph.getLabelList())
        
        return

    def test(self, sModelName, sModelDir, lsTstColDir):
        traceln("-"*50)
        traceln("Trained model '%s' in folder '%s'"%(sModelName, sModelDir))
        traceln("Test  collection(s):", lsTstColDir)
        traceln("-"*50)
        
        #list the train and test files
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, "*.mpxml")
        
        traceln("- loading a %s model"%Model_SSVM_AD3)
        mdl = Model_SSVM_AD3(sModelName, sModelDir)
        mdl.load()
        traceln(" done")
        
        DU_StAZH_Graph = self.getGraphClass()
        
        traceln("- loading test graphs")
        lGraph_tst = DU_StAZH_Graph.loadGraphs(lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
        traceln(" %d graphs loaded"%len(lGraph_tst))

        fScore, sReport = mdl.test(lGraph_tst, DU_StAZH_Graph.getLabelList())
        
        return

    def predict(self, sModelName, sModelDir, lsColDir):
        traceln("-"*50)
        traceln("Trained model '%s' in folder '%s'"%(sModelName, sModelDir))
        traceln("Collection(s):", lsColDir)
        traceln("-"*50)
        
        #list the train and test files
        sMPXMLExtension = ".mpxml"
        _     , lFilename = self.listMaxTimestampFile(lsColDir, "*"+sMPXMLExtension)
        
        traceln("- loading a %s model"%Model_SSVM_AD3)
        mdl = Model_SSVM_AD3(sModelName, sModelDir)
        mdl.load()
        traceln(" done")
        
        DU_StAZH_Graph = self.getGraphClass()
        
        traceln("- loading collection as graphs")
        for sFilename in lFilename:
            if sFilename.endswith("_du"+sMPXMLExtension): continue #:)
            [g] = DU_StAZH_Graph.loadGraphs([sFilename], bDetach=False, bLabelled=False, iVerbose=1)
            Y = mdl.predict(g)
            doc = g.setDomLabels(Y)
            sDUFilename = sFilename[:-len(sMPXMLExtension)]+"_du"+sMPXMLExtension
            doc.saveFormatFileEnc(sDUFilename, "utf-8", True)  #True to indent the XML
            doc.freeDoc()
            del Y, g
            traceln("\t done")
        traceln(" done")

        return


if __name__ == "__main__":
    
    sTopDir = "C:\\Local\\meunier\\git\\TranskribusDU\\usecases\\StAZH\\"
    if False:
        doer = DU_StAZH_a()
        doer.train_test("DU_StAZH_a", "c:\\tmp_READ"
                        , [sTopDir+"trnskrbs_3820\\col"]
                        , [sTopDir+"trnskrbs_3832\\col"])
    
    if True:
        doer = DU_StAZH_a()
        doer.test("DU_StAZH_a", "c:\\tmp_READ"
                        , [sTopDir+"trnskrbs_3832\\col"])
    
    if True:
        doer = DU_StAZH_a()
        doer.predict("DU_StAZH_a", "c:\\tmp_READ"
#                         , [sTopDir+"trnskrbs_3829\\col"])
                        , [sTopDir+"trnskrbs_3989\\col"])
                                
                                