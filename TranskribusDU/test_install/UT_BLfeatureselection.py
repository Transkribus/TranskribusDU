import sys, os
import unittest

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


class UT_BLfeatureselection(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(UT_BLfeatureselection, self).__init__(*args, **kwargs)
        self.name='testing the feature selection for a Baseline Model'
        #load the data once for the test

    def test_01(self):
        #Train the Baseline Model






if __name__ == '__main__':
    unittest.main()
