# -*- coding: utf-8 -*-

"""
    *** Same as its parent apart that text baselines are reflected as a LineString (instead of its centroid)
    
    DU task for ABP Table: 
        doing jointly row BIO and near horizontal cuts SIO
    
    block2line edges do not cross another block.
    
    The cut are based on baselines of text blocks, with some positive or negative inclination.

    - the labels of cuts are SIO 
    
    Copyright Naver Labs Europe(C) 2018 JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from xml_formats.PageXml import MultiPageXml
from util.Shape import ShapeLoader

from tasks.DU_CRF_Task import DU_CRF_Task
from tasks.DU_ABPTableSkewed import GraphSkewedCut_H, My_FeatureDefinition_v3, NodeType_PageXml_Cut_Shape, main_command_line,\
    My_FeatureDefinition_v3_txt
from tasks.DU_ABPTableSkewed_txtBIO_sepSIO import NodeType_BIESO_to_BIO_Shape, NodeType_BIESO_to_BIO_Shape_txt


class GraphSkewedCut_H_lines(GraphSkewedCut_H):
    
    # reflecting text baseline as a LineString
    shaper_fun = ShapeLoader.node_to_SingleLine


    def addEdgeToDoc(self, Y=None):
        """
        To display the grpah conveniently we add new Edge elements
        Since we change the BAseline representation, we show the new one
        """
        super().addEdgeToDoc()
        
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
    bTxt                = None  # use textual features?

    #=== CONFIGURATION ====================================================================
    @classmethod
    def getConfiguredGraphClass(cls):
        """
        In this class method, we must return a configured graph class
        """
        
        # Textline labels
        #  Begin Inside End Single Other
        lLabels_BIO_row  = ['B', 'I', 'O'] 

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
        ntR = (     NodeType_BIESO_to_BIO_Shape_txt if cls.bTxt \
               else NodeType_BIESO_to_BIO_Shape \
               )("row"
                  , lLabels_BIO_row
                  , None
                  , False
                  , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))
                  )
        ntR.setLabelAttribute("DU_row")
        ntR.setXpathExpr( (".//pc:TextLine"                 #how to find the nodes
                          , "./pc:TextEquiv/pc:Unicode")    #how to get their text
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
        
    def __init__(self, sModelName, sModelDir
                 , iBlockVisibility = None
                 , iLineVisibility = None
                 , fCutHeight = None
                 , bCutAbove = None
                 , lRadAngle = None
                 , bTxt      = None
                 , sComment = None
                 , cFeatureDefinition   = None
                 , dFeatureConfig       = {}
                 , C=None, tol=None, njobs=None, max_iter=None
                 , inference_cache=None): 
        
        DU_ABPTableSkewedRowCutLine.iBlockVisibility = iBlockVisibility
        DU_ABPTableSkewedRowCutLine.iLineVisibility  = iLineVisibility
        DU_ABPTableSkewedRowCutLine.fCutHeight       = fCutHeight
        DU_ABPTableSkewedRowCutLine.bCutAbove        = bCutAbove
        DU_ABPTableSkewedRowCutLine.lRadAngle        = lRadAngle
        DU_ABPTableSkewedRowCutLine.bTxt             = bTxt

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
                     , cFeatureDefinition= My_FeatureDefinition_v3_txt if self.bTxt else My_FeatureDefinition_v3
                     )


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main_command_line(DU_ABPTableSkewedRowCutLine)
