# -*- coding: utf-8 -*-

"""
    DU task for ABP Table: 
        doing jointly row BIESO and near horizontal cuts SIO
    
    block2line edges do not cross another block.
    
    The cut are based on baselines of text blocks, with some positive or negative inclination.

    - the labels of cuts are SIO 
    
    Copyright Naver Labs Europe(C) 2018 JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os
from lxml import etree

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _exit
from tasks.DU_CRF_Task import DU_CRF_Task
from tasks.DU_ABPTableSkewed_CutAnnotator import SkewedCutAnnotator

from util.Shape import ShapeLoader

from tasks.DU_ABPTableSkewed import GraphSkewedCut_H, My_FeatureDefinition_v3, NodeType_PageXml_Cut_Shape, main_command_line
from crf.NodeType_PageXml   import NodeType_PageXml_type_woText


class NodeType_BIESO_Shape(NodeType_PageXml_type_woText):
    """
    """
    
    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = self.sDefaultLabel
        domnode = graph_node.node
        sXmlLabel = domnode.get(self.sLabelAttr)
        
        sXmlLabel = {'B':'B',
                     'I':'I',
                     'E':'E',
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



class DU_ABPTableSkewedRowCut(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , with the below labels 
    """
    sXmlFilenamePattern = "*[0-9].mpxml"
    
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
        lLabels_BIESO_row  = ['B', 'I', 'E', 'S', 'O'] 

        # Cut lines: 
        #  Border Ignore Separator Outside
        lLabels_SIO_Cut  = ['S', 'I', 'O']
       
        #DEFINING THE CLASS OF GRAPH WE USE
        DU_GRAPH = GraphSkewedCut_H
        
        DU_GRAPH.iBlockVisibility   = cls.iBlockVisibility
        DU_GRAPH.iLineVisibility    = cls.iLineVisibility
        DU_GRAPH.fCutHeight         = cls.fCutHeight
        DU_GRAPH.bCutAbove          = cls.bCutAbove
        DU_GRAPH.lRadAngle          = cls.lRadAngle
        
        # ROW
        ntR = NodeType_BIESO_Shape("row"
                              , lLabels_BIESO_row
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
        
        DU_ABPTableSkewedRowCut.iBlockVisibility = iBlockVisibility
        DU_ABPTableSkewedRowCut.iLineVisibility  = iLineVisibility
        DU_ABPTableSkewedRowCut.fCutHeight       = fCutHeight
        DU_ABPTableSkewedRowCut.bCutAbove        = bCutAbove
        DU_ABPTableSkewedRowCut.lRadAngle        = lRadAngle

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
    main_command_line(DU_ABPTableSkewedRowCut)

