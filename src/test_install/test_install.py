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

from crf.Graph_MultiPageXml_TextRegion import Graph_MultiPageXml_TextRegion

from tasks.DU_CRF_Task import DU_CRF_Task

 
class DU_Test(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , working on a MultiPageXMl document at TextRegion level
    , with the below labels 
    """

    #  0=OTHER        1            2            3        4                5
    TASK_LABELS = ['catch-word', 'header', 'heading', 'marginalia', 'page-number']

    n_tfidf_node    = 10
    t_ngrams_node   = (2,2)
    b_tfidf_node_lc = False    
    n_tfidf_edge    = 10
    t_ngrams_edge    = (2,2)
    b_tfidf_edge_lc = False    

    
    def __init__(self): 
        DU_CRF_Task.__init__(self)
    
    def getGraphClass(self):
        DU_StAZH_Graph = Graph_MultiPageXml_TextRegion
        DU_StAZH_Graph.setLabelList(self.TASK_LABELS, True)  #True means non-annotated node are of class 0 = OTHER

        return DU_StAZH_Graph


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
    try:
        sTranskribusTestDir = os.path.join(os.path.dirname(__file__), "trnskrbs_3820")

        doer = DU_Test()
        
        doer.configureLearner(njobs=2, save_every=5, max_iter=8)
        
        #We train, test, predict on teh same document(s)
        lDir = _checkFindColDir( [sTranskribusTestDir])

        doer.train_test(sModelName, sModelDir, lDir, lDir)
        if False:
            #you can also test the prediction, but it will generate a file "7749_du.mpxml" in trnskrbs_3820/col
            doer.predict(sModelName, sModelDir, lDir)
    finally:
        #Hack to clean
        mdl = DU_CRF_Task.ModelClass(sModelName, sModelDir) 
        os.unlink(mdl.getModelFilename())
        os.unlink(mdl.getTransformerFilename())
        os.rmdir(sModelDir)
        
if __name__ == "__main__":
    """
    Usage:
    python test_install.py
    
    If no exception raised, then OK!!
    """
    test_main()
