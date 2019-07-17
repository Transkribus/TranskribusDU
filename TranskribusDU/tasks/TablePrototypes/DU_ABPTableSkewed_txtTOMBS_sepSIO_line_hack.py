# -*- coding: utf-8 -*-

"""
    *** 
    
    Labelling is T O M B S
     It depends on the distance between the baseline and its above and below valid (S) cut
    
    Cuts are SIO

    Copyright Naver Labs Europe(C) 2018 JL Meunier

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

import numpy as np
from lxml import etree
import shapely.affinity


try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks.DU_CRF_Task import DU_CRF_Task
from tasks.DU_ABPTableSkewed import GraphSkewedCut_H, My_FeatureDefinition_v3, NodeType_PageXml_Cut_Shape, main_command_line
from tasks.DU_ABPTableSkewed import Edge_BL
from tasks.DU_ABPTableSkewed_txtBIO_sepSIO import NodeType_BIESO_to_BIO_Shape

from xml_formats.PageXml import MultiPageXml

from util.Shape import ShapeLoader

from tasks.DU_ABPTableSkewed_txtTOMBS_sepSIO_line import *

#------------------------------------------------------------------------------------------------------

# WE add one feature for _ishort

from crf.Transformer import Transformer
import tasks.DU_ABPTableSkewed

class Node1HotFeatures_noText(Transformer):
    """
    we will get a list of block and return a one-hot encoding, directly
    noText = without _any_ text-related feature
    """
    def transform(self, lNode):
        #We allocate TWO more columns to store in it the tfidf and idf computed at document level.
        #a = np.zeros( ( len(lNode), 10 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        a = np.zeros( ( len(lNode), 7 + 5 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
        for i, blk in enumerate(lNode): 
            a[i,0] = int(blk.pnum%2 == 0)
            #new in READ
            #are we in page 1 or 2 or next ones?
            a[i, max(1 , min(3, blk.pnum))]      = 1.0  #  a[i, 1-2-3 ]
            #are we in page -2 or -1 or previous ones?
            a[i, 6+max(-2, blk.pnum-blk.page.pagecnt)]  = 1.0  #  a[i, 4-5-6 ]
            #a[i,blk.orientation] = 1.0   
            a[i, 7 + blk.cls] = 1.0
             
        return a
tasks.DU_ABPTableSkewed.Node1HotFeatures_noText = Node1HotFeatures_noText


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main_command_line(DU_ABPTableSkewedRowCutLine)
