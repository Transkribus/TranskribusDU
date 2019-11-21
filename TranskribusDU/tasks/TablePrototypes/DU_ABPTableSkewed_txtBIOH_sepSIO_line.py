# -*- coding: utf-8 -*-

"""
    *** Same as DU_ABPTableSkewed_txtBIO_sepSIO_line, except that text have BIOH as labels
    
    DU task for ABP Table: 
        doing jointly row BIOH and near horizontal cuts SIO
    
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
from tasks.DU_CRF_Task import DU_CRF_Task
from tasks.DU_ABPTableSkewed import My_FeatureDefinition_v3, NodeType_PageXml_Cut_Shape, main_command_line
from tasks.DU_ABPTableSkewed_txtBIO_sepSIO import NodeType_BIESO_to_BIO_Shape_txt
from tasks.DU_ABPTableSkewed_txtBIO_sepSIO_line import GraphSkewedCut_H_lines, DU_ABPTableSkewedRowCutLine


class NodeType_BIESO_to_BIOH_Shape_txt(NodeType_BIESO_to_BIO_Shape_txt):
    """
    Convert BIESO labeling to SIO
    """
    
    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = self.sDefaultLabel
        domnode = graph_node.node
        
        sXmlLabel = domnode.get("DU_header")
        if sXmlLabel != 'CH':
            sXmlLabel = domnode.get(self.sLabelAttr)

        sXmlLabel = {'B':'B',
                     'I':'I',
                     'E':'I',
                     'S':'B',
                     'O':'O',
                     'CH':'CH'}[sXmlLabel]
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


class NodeType_BIESO_to_BIOH_Shape(NodeType_BIESO_to_BIOH_Shape_txt):
    """
    without text
    """
    def _get_GraphNodeText(self, doc, domNdPage, ndBlock, ctxt=None):
        return u""


class DU_ABPTableSkewedRowCutLine_BIOH(DU_ABPTableSkewedRowCutLine):
    """
    We will do a CRF model for a DU task
    , with the below labels 
    """

    #=== CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        
        # Textline labels
        #  Begin Inside End Single Other
        lLabels_BIOH_row  = ['B', 'I', 'O', 'CH'] 

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
        DU_GRAPH.bTxt               = cls.bTxt
        
        # ROW
        ntR = (     NodeType_BIESO_to_BIOH_Shape_txt if cls.bTxt \
               else NodeType_BIESO_to_BIOH_Shape \
               )("row"
                              , lLabels_BIOH_row
                              , None
                              , False
                              , None
                              )
        ntR.setLabelAttribute("DU_row")
        ntR.setXpathExpr( (".//pc:TextLine"        #how to find the nodes
                          , "./pc:TextEquiv/pc:Unicode")       #how to get their text
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
        

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main_command_line(DU_ABPTableSkewedRowCutLine_BIOH)
