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

#------------------------------------------------------------------------------------------------------

# WE add one feature for _ishort

from crf.Transformer import Transformer
import tasks.DU_ABPTableSkewed
class Block2CutLine_EdgeTransformer_qtty(Transformer):
    def transform(self, lEdge):
        N = 5
        a = np.zeros( ( len(lEdge), 2 * N) , dtype=np.float64)  
        for i, edge in enumerate(lEdge):
#             z = 0 if edge._type < 0 else N  # _type is -1 or 1 
            if edge._type < 0:
                z = 0
                ishort = 1 if edge.len < GraphSkewedCut_H_TOMBS_lines.iCutCloseDistanceTop else 0
            else:
                z = N
                ishort = 1 if edge.len < GraphSkewedCut_H_TOMBS_lines.iCutCloseDistanceBot else 0
                
            a[i, z:z+N] = (1
                           , len(edge.B.set_support)
                           , edge.A._in_edge_up
                           , edge.A._in_edge_down
                           , ishort
                           )
#             print(a[i,:].tolist())
        # traceln("Block2CutLine_EdgeTransformer", a[:min(100, len(lEdge)),])
        return a
tasks.DU_ABPTableSkewed.Block2CutLine_EdgeTransformer_qtty = Block2CutLine_EdgeTransformer_qtty

class Block2CutLine_FakeEdgeTransformer(Transformer):
    """
    a fake transformer that return as many features as the union of real ones above
    """
    def transform(self, lEdge):
        assert not(lEdge)
        return np.zeros( ( len(lEdge), 2*8 + 2*5) , dtype=np.float64)
tasks.DU_ABPTableSkewed.Block2CutLine_FakeEdgeTransformer = Block2CutLine_FakeEdgeTransformer 

#------------------------------------------------------------------------------------------------------
class GraphSkewedCut_H_TOMBS_lines(GraphSkewedCut_H):

    # reflecting text baseline as a LineString
    shaper_fun = ShapeLoader.node_to_SingleLine
    
    iCutCloseDistanceTop = 45  # any block close enough become T or S
    iCutCloseDistanceBot = 45  # any block close enough become  B or S

    @classmethod
    def showClassParam(cls):
        bShown = super().showClassParam()
        if bShown:
            #also show ours!
            traceln("  - iCutCloseDistanceTop : "     , cls.iCutCloseDistanceTop)
            traceln("  - iCutCloseDistanceBot : "     , cls.iCutCloseDistanceBot)
        
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

    """
    To compute TOMBS labels, it is better to use the built graph...
    """
    def parseDomLabels(self):
        """
        Parse the label of the graph from the dataset, and set the node label
        return the set of observed class (set of integers in N+)
        """
        # WE expect I or O for text blocks!!
        setSeensLabels = super().parseDomLabels()
        
        # now look at edges to compute T M B S
        # REMEMBER, we did: edge.len = dist / self.iBlockVisibility
        maxLenTop = self.iCutCloseDistanceTop / self.iBlockVisibility
        maxLenBot = self.iCutCloseDistanceBot / self.iBlockVisibility
        
        # --- ASSUMPTION !!! ---
        T, _O, M, B, S = 0, 1, 2, 3, 4
        sepS, _sepI, _sepO = 5, 6, 7
        
        for edge in self.lEdge:
            if type(edge) == Edge_BL and edge.B.cls == sepS:
                cls = edge.A.cls
                if edge._type < 0:   # this short edge goes up
                    if edge.len <= maxLenTop:
                        # Ok, this will be a T or B or S!
                        # which means the text block is teh 1st CRF node type
                        # REMEMBER, we did: edge._type = -1 if blk.y_bslne >= y else +1
                        if cls == M:
                            newcls = T
                        elif cls == B:
                            newcls = S
                        else:
                            continue
                        edge.A.cls = newcls
                        setSeensLabels.add(newcls)
                else:           # sthis hort edge goes down
                    if edge.len <= maxLenBot:
                        if cls == M:
                            newcls = B
                        elif cls == T:
                            newcls = S
                        else:
                            continue
                        edge.A.cls = newcls
                        setSeensLabels.add(newcls)
                
        # traceln(self._dClsByLabel)
        return setSeensLabels       
    

class NodeType_BIESO_to_TOMBS_Shape(NodeType_BIESO_to_BIO_Shape):
    """
    Convert BIESO labeling to SIOStSmSb
    """
    bColumnHeader = False     # ignore headers for now
    
    dConverter = {   'B':'M',
                     'I':'M',
                     'E':'M',
                     'S':'M',  #  St Sm Sb => specific processing to get it
                     'O':'O',
                     'CH':'CH'}
    
    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sXmlLabel = domnode.get(self.sLabelAttr)
        
        # in case we also deal with column headers
        if self.bColumnHeader and 'CH' == domnode.get("DU_header"):
            sXmlLabel = 'CH'
        
        sXmlLabel = self.dConverter[sXmlLabel]
        try:
            sLabel = self.dXmlLabel2Label[sXmlLabel]
        except KeyError:
            raise ValueError("Invalid label '%s'"
                             " (from @%s or @%s) in node %s"%(sXmlLabel,
                                                       self.sLabelAttr,
                                                       self.sDefaultLabel,
                                                       etree.tostring(domnode)))
        # traceln(etree.tostring(domnode), sLabel)
        return sLabel


class DU_ABPTableSkewedRowCutLine(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , with the below labels 
    """
    sXmlFilenamePattern = "*.mpxml"  # *_du.* files are now ignored by DU_CRF_Task
    
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
        lLabels_TOMBS_blk  = ['T', 'O', 'M', 'B', 'S'] 

        # Cut lines: 
        #  Border Ignore Separator Outside
        lLabels_SIO_Cut  = ['S', 'I', 'O']
       
        #DEFINING THE CLASS OF GRAPH WE USE
        DU_GRAPH = GraphSkewedCut_H_TOMBS_lines
        
        DU_GRAPH.iBlockVisibility   = cls.iBlockVisibility
        DU_GRAPH.iLineVisibility    = cls.iLineVisibility
        DU_GRAPH.fCutHeight         = cls.fCutHeight
        DU_GRAPH.bCutAbove          = cls.bCutAbove
        DU_GRAPH.lRadAngle          = cls.lRadAngle
        
        # ROW
        ntR = NodeType_BIESO_to_TOMBS_Shape("row"
                              , lLabels_TOMBS_blk
                              , None
                              , False
                              , None
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
        DU_ABPTableSkewedRowCutLine.bCutAbove        = True
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
