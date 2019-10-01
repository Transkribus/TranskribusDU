# -*- coding: utf-8 -*-

"""
    DU task for segmenting text in rows using the conjugate graph after the SW
    re-engineering by JLM
    
    As of June 5th, 2015, this is the exemplary code
    
    Copyright NAVER(C)  2019  Jean-Luc Meunier
    
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

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln

from graph.GraphBinaryConjugateSegmenter    import GraphBinaryConjugateSegmenter
from graph.Graph_Multi_SinglePageXml        import Graph_MultiSinglePageXml
from graph.NodeType_PageXml                 import defaultBBoxDeltaFun
from graph.NodeType_PageXml                 import NodeType_PageXml_type
from tasks.DU_Task_Factory                  import DU_Task_Factory
from tasks.DU_Task_Features                 import Features_June19_Simple


class ConjugateSegmenterGraph_MultiSinglePageXml(GraphBinaryConjugateSegmenter, Graph_MultiSinglePageXml):
    """
    standard graph but used in conjugate mode
    """
    def __init__(self):
        super(ConjugateSegmenterGraph_MultiSinglePageXml, self).__init__()
#         GraphBinaryConjugateSegmenter.__init__(self)
#         Graph_MultiSinglePageXml.__init__(self)


# class My_ConjugateNodeType(NodeType_PageXml_type_woText):
class My_ConjugateNodeType(NodeType_PageXml_type):
    """
    We need this to extract properly the label from the label attribute of the (parent) TableCell element.
    """
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun):
        super(My_ConjugateNodeType, self).__init__(sNodeTypeName, lsLabel, lsIgnoredLabel, bOther, BBoxDeltaFun)

    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = domnode.getparent().get(self.sLabelAttr)
        
        return sLabel if not sLabel is None else "__none__"

    def setDomNodeLabel(self, domnode, sLabel):
        raise Exception("This shoud not occur in conjugate mode")    
    
    
def getConfiguredGraphClass(doer):
    """
    In this class method, we must return a configured graph class
    """
    # each graph reflects 1 page
    DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml  # consider each age as if independent from each other

    # ntClass = NodeType_PageXml_type
    ntClass = My_ConjugateNodeType

    nt = ntClass("row"                   #some short prefix because labels below are prefixed with it
                  , []                   # in conjugate, we accept all labels, andNone becomes "none"
                  , []
                  , False                # unused
                  , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  )    
    nt.setLabelAttribute("row")
    nt.setXpathExpr( (".//pc:TextLine"        #how to find the nodes            
                      #, "./pc:TextEquiv")       #how to get their text
                      , ".//pc:Unicode")       #how to get their text
                   )
    DU_GRAPH.addNodeType(nt)
    
    return DU_GRAPH


    
if __name__ == "__main__":
    #     import better_exceptions
    #     better_exceptions.MAX_LENGTH = None
    
    # standard command line options for CRF- ECN- GAT-based methods
    usage, parser = DU_Task_Factory.getStandardOptionsParser(sys.argv[0])

    traceln("VERSION: %s" % DU_Task_Factory.getVersion())

    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        DU_Task_Factory.exit(usage, 1, e)

    # === SETTING the graph type (and its node type) a,d the feature extraction pipe
    doer = DU_Task_Factory.getDoer(sModelDir, sModelName
                                   , options                    = options
                                   , fun_getConfiguredGraphClass= getConfiguredGraphClass
                                   , cFeatureDefinition         = Features_June19_Simple
                                   # OBSOLETE:  preferred way is to choose them in the feature class
                                   # OBSOLETE: , dFeatureConfig             = {'bMultiPage':False, 'bMirrorPage':False}                                           
                                   )
    
    # == LEARNER CONFIGURATION ===
    # setting the learner configuration, in a standard way 
    # (from command line options, or from a JSON configuration file)
    dLearnerConfig = doer.getStandardLearnerConfig(options)
    
    
#     # force a balanced weighting
#     print("Forcing balanced weights")
#     dLearnerConfig['balanced'] = True
    
    # of course, you can put yours here instead.
    doer.setLearnerConfiguration(dLearnerConfig)

    # === CONJUGATE MODE ===
    doer.setConjugateMode()
    
    # === GO!! ===
    # act as per specified in the command line (--trn , --fold-run, ...)
    doer.standardDo(options)
    
    del doer

