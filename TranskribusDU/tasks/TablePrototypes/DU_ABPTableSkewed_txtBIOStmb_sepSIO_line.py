# -*- coding: utf-8 -*-

"""
    *** 
    
    Labelling is B I O St Sm Sb
    
    Singletons are split into:
    - St : a singleton on "top"    of its cell, vertically
    - Sm : a singleton in "middle" of its cell, vertically
    - Sb : a singleton in "bottom" of its cell, vertically
    
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

from lxml import etree
import shapely.affinity

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks.DU_CRF_Task import DU_CRF_Task
from tasks.DU_ABPTableSkewed import My_FeatureDefinition_v3, NodeType_PageXml_Cut_Shape, main_command_line
from tasks.DU_ABPTableSkewed_txtBIO_sepSIO import NodeType_BIESO_to_BIO_Shape
from tasks.DU_ABPTableSkewed_txtBIO_sepSIO_line import GraphSkewedCut_H_lines

from util.Shape import ShapeLoader

class NodeType_BIESO_to_SIOStSmSb_Shape(NodeType_BIESO_to_BIO_Shape):
    """
    Convert BIESO labeling to SIOStSmSb
    """
    bColumnHeader = False     # ignore headers for now
    
    dConverter = {   'B':'B',
                     'I':'I',
                     'E':'I',
                     'S':None,  #  St Sm Sb => specific processing to get it
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
        if sXmlLabel is None:
            # special processing for singletons TODO: make it more efficient?
            ptTxt = ShapeLoader.node_to_Polygon(domnode).centroid
            plgCell = ShapeLoader.node_to_Polygon(domnode.getparent())
            plgMiddle = shapely.affinity.scale(plgCell, 1, 0.333, 1, 'centroid')
            if plgMiddle.contains(ptTxt):
                sXmlLabel = "Sm"
            else:
                if ptTxt.y < plgCell.centroid.y:
                    sXmlLabel = "St"
                else:
                    sXmlLabel = "Sb"
        try:
            sLabel = self.dXmlLabel2Label[sXmlLabel]
        except KeyError:
#             #not a label of interest, can we ignore it?
#             try:
#                 self.checkIsIgnored(sXmlLabel)
#                 sLabel = self.sDefaultLabel
#                 #if self.lsXmlIgnoredLabel and sXmlLabel not in self.lsXmlIgnoredLabel: 
#             except:
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
        lLabels_SIOStSmSb_row  = ['B', 'I', 'O', 'St', 'Sm', 'Sb'] 

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
        ntR = NodeType_BIESO_to_SIOStSmSb_Shape("row"
                              , lLabels_SIOStSmSb_row
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
