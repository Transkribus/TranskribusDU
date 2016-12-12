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
import glob
from optparse import OptionParser

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from tasks import sCOL, _checkFindColDir, _exit

from crf.Graph_MultiPageXml_TextRegion import Graph_MultiPageXml_TextRegion
from crf.FeatureExtractors_PageXml_std import FeatureExtractors_PageXml_StandardOnes
from crf.Model import ModelException
from crf.Model_SSVM_AD3 import Model_SSVM_AD3
from xml_formats.PageXml import MultiPageXml

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



if __name__ == "__main__":

    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    # --- 
#     try:    sDir = args.pop(0)
#     except: _exit(usage, 1)
#     sColDir = _checkFindColDir(sDir)
#     
#     sTopDir = "C:\\Local\\meunier\\git\\TranskribusDU\\usecases\\StAZH\\"
#     if False:
#         doer = DU_StAZH_a()
#         doer.train_test("DU_StAZH_a", "c:\\tmp_READ"
#                         , [sTopDir+"trnskrbs_3820\\col"]
#                         , [sTopDir+"trnskrbs_3832\\col"])
#     
#     if True:
#         doer = DU_StAZH_a()
#         doer.test("DU_StAZH_a", "c:\\tmp_READ"
#                         , [sTopDir+"trnskrbs_3832\\col"])
#     
#     if True:
#         doer = DU_StAZH_a()
#         doer.predict("DU_StAZH_a", "c:\\tmp_READ"
# #                         , [sTopDir+"trnskrbs_3829\\col"])
#                         , [sTopDir+"trnskrbs_3989\\col"])
#                                 
    try:
        sModelName, sModelDir = args
    except Exception as e:
        _exit(usage, 1, e)
        
    doer = DU_StAZH_a()
    
    #Add the "col" subdir if needed
    lTrn, lTst, lRun = [_checkFindColDir(lsDir) for lsDir in [options.lTrn, options.lTst, options.lRun]]

    if lTrn:
        doer.train_test(sModelName, sModelDir, lTrn, lTst)
    elif lTst:
        doer.test(sModelName, sModelDir, lTst)
    
    if lRun:
        doer.predict(sModelName, sModelDir, lRun)
    
