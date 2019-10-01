# -*- coding: utf-8 -*-

"""
    *** Same as its parent apart that text baselines are reflected as a LineString (instead of its centroid)
    
    DU task for ABP Table: 
        doing jointly row BISO and near horizontal cuts SIO
    
    block2line edges do not cross another block.
    
    The cut are based on baselines of text blocks, with some positive or negative inclination.

    - the labels of cuts are SIO 
    
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

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from xml_formats.PageXml import MultiPageXml
from util.Shape import ShapeLoader

from tasks.DU_CRF_Task import DU_CRF_Task
from tasks.DU_ABPTableSkewed import GraphSkewedCut_H, My_FeatureDefinition_v3, NodeType_PageXml_Cut_Shape, main_command_line

from crf.Transformer import Transformer
from crf.NodeType_PageXml   import NodeType_PageXml_type_woText


import tasks.DU_ABPTableSkewed
class Node1HotFeatures_noText(Transformer):
    """
    we will get a list of block and return a one-hot encoding, directly
    noText = without _any_ text-related feature
    """
    def transform(self, lNode):
        #We allocate TWO more columns to store in it the tfidf and idf computed at document level.
        #a = np.zeros( ( len(lNode), 10 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        a = np.zeros( ( len(lNode), 7 + 4 ) , dtype=np.float64)  # 4 possible orientations: 0, 1, 2, 3
        
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


class NodeType_BISO_Shape(NodeType_PageXml_type_woText):
    """
    """
    
    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = self.sDefaultLabel
        
        sXmlLabel = domnode.get(self.sLabelAttr)
        
        sXmlLabel = {'B':'B',
                     'I':'I',
                     'E':'I',
                     'S':'S',
                     'O':'O'}[sXmlLabel]
        try:
            sLabel = self.dXmlLabel2Label[sXmlLabel]
        except KeyError:
            #not a label of interest
            try:
                self.checkIsIgnored(sXmlLabel)
                #if self.lsXmlIgnoredLabel and sXmlLabel not in self.lsXmlIgnoredLabel: 
            except:
                raise ValueError("Invalid label '%s'"
                                 " (from @%s or @%s) in node %s"%(sXmlLabel,
                                                           self.sLabelAttr,
                                                           self.sDefaultLabel,
                                                           etree.tostring(domnode)))

        return sLabel

    def _iter_GraphNode(self, doc, domNdPage, page):
        """
        to add the shape object reflecting the baseline
        """
        for blk in super()._iter_GraphNode(doc, domNdPage, page):
            try:
                ndBaseline = blk.node.xpath(".//pc:Baseline", namespaces=self.dNS)[0]
                try:
                    o = ShapeLoader.node_to_LineString(ndBaseline)
                except ValueError:
                    traceln("SKIPPING INVALID Baseline: ", etree.tostring(ndBaseline))
                    continue
                blk.shape = o
                blk.du_index = int(ndBaseline.get("du_index"))
                yield blk
            except:
                pass
        return

class GraphSkewedCut_H_lines(GraphSkewedCut_H):
    
    # reflecting text baseline as a LineString
    shaper_fun = ShapeLoader.node_to_SingleLine


    def addEdgeToDOM(self):
        """
        To display the grpah conveniently we add new Edge elements
        Since we change the BAseline representation, we show the new one
        """
        super().addEdgeToDOM()
        
        for blk in self.lNode:
            assert blk.type.name in ["row", "sepH"], blk.type.name
            
            if blk.type.name == "row":
                ndBaseline = blk.node.xpath(".//pc:Baseline", namespaces=self.dNS)[0]
                o = self.shaper_fun(ndBaseline)
                MultiPageXml.setPoints(ndBaseline, list(o.coords))
            
        return
   

class DU_ABPTableSkewedRowCutLine(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , with the below labels 
    """
    sXmlFilenamePattern = "*.mpxml"
    #sXmlFilenamePattern = "*.pxml"
    
    iBlockVisibility    = None
    iLineVisibility     = None
    fCutHeight          = None
    bCutAbove           = None
    lRadAngle           = None

    #=== CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        
        # Textline labels
        #  Begin Inside End Single Other
        lLabels_BISO_row  = ['B', 'I', 'S', 'O'] 

        # Cut lines: 
        #  Border Ignore Separator Outside
        lLabels_SIO_Cut  = ['S', 'I', 'O']
       
        #DEFINING THE CLASS OF GRAPH WE USE
        DU_GRAPH = GraphSkewedCut_H_lines
        
        DU_GRAPH.iBlockVisibility   = cls.iBlockVisibility
        DU_GRAPH.iLineVisibility    = cls.iLineVisibility
        DU_GRAPH.fCutHeight         = cls.fCutHeight
        DU_GRAPH.bCutAbove          = cls.bCutAbove
        DU_GRAPH.lRadAngle          = cls.lRadAngle
        
        # ROW
        ntR = NodeType_BISO_Shape("row"
                              , lLabels_BISO_row
                              , None
                              , False
                              , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))
                              )
        ntR.setLabelAttribute("DU_row")
        ntR.setXpathExpr( (".//pc:TextLine"        #how to find the nodes
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_GRAPH.addNodeType(ntR)
        
        # CUT
        ntCutH = NodeType_PageXml_Cut_Shape("sepH"
                              , lLabels_SIO_Cut
                              , None
                              , False
                              , None        # equiv. to: BBoxDeltaFun=lambda _: 0
                              )
        ntCutH.setLabelAttribute("DU_type")
        ntCutH.setXpathExpr( ('.//pc:CutSeparator[@orient="0"]' #how to find the nodes
                        # the angle attribute give the true orientation (which is near 0)
                          , "./pc:TextEquiv")       #how to get their text
                       )
        DU_GRAPH.addNodeType(ntCutH)        
        
        DU_GRAPH.setClassicNodeTypeList( [ntR ])
        
        return DU_GRAPH
        
    def __init__(self, sModelName, sModelDir, 
                 iBlockVisibility = None,
                 iLineVisibility = None,
                 fCutHeight = None,
                 bCutAbove = None,
                 lRadAngle = None,
                 sComment = None,
                 C=None, tol=None, njobs=None, max_iter=None,
                 inference_cache=None): 
        
        DU_ABPTableSkewedRowCutLine.iBlockVisibility = iBlockVisibility
        DU_ABPTableSkewedRowCutLine.iLineVisibility  = iLineVisibility
        DU_ABPTableSkewedRowCutLine.fCutHeight       = fCutHeight
        DU_ABPTableSkewedRowCutLine.bCutAbove        = bCutAbove
        DU_ABPTableSkewedRowCutLine.lRadAngle        = lRadAngle

        DU_CRF_Task.__init__(self
                     , sModelName, sModelDir
                     , dFeatureConfig = {'row_row':{}, 'row_sepH':{},
                                         'sepH_row':{}, 'sepH_sepH':{},
                                         'sepH':{}, 'row':{}}
                     , dLearnerConfig = {
                                   'C'                : .1   if C               is None else C
                                 , 'njobs'            : 4    if njobs           is None else njobs
                                 , 'inference_cache'  : 50   if inference_cache is None else inference_cache
                                 #, 'tol'              : .1
                                 , 'tol'              : .05  if tol             is None else tol
                                 , 'save_every'       : 50     #save every 50 iterations,for warm start
                                 , 'max_iter'         : 10   if max_iter        is None else max_iter
                         }
                     , sComment=sComment
                     #,cFeatureDefinition=FeatureDefinition_PageXml_StandardOnes_noText
                     ,cFeatureDefinition=My_FeatureDefinition_v3
                     )


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main_command_line(DU_ABPTableSkewedRowCutLine)
