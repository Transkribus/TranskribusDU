# -*- coding: utf-8 -*-

"""
    DU task for segmenting text in rows using a BIO scheme
    
    Example of code after April SW re-engineering by JLM
    
    Copyright NAVER(C) 2019  Jean-Luc Meunier
    
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
import lxml.etree as etree

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln
from graph.Graph_Multi_SinglePageXml import Graph_MultiSinglePageXml
from graph.NodeType_PageXml   import NodeType_PageXml_type_woText
from graph.FeatureDefinition_PageXml_std_noText_v4 import FeatureDefinition_PageXml_StandardOnes_noText_v4
from tasks.DU_Task_Factory import DU_Task_Factory



# to convert from BIESO to BIO we create our own NodeType by inheritance
# class NodeType_BIESO_to_BIO_Shape(NodeType_PageXml_type_woText):
class NodeType_PageXml_type_woText_BIESO_to_BIO(NodeType_PageXml_type_woText):
    """
    Convert BIESO labeling to BIO
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
                     'S':'B',
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


def getConfiguredGraphClass(doer):
    """
    In this function, we return a configured graph.Graph subclass
    
    doer is a tasks.DU_task object created by tasks.DU_Task_Factory
    """
    #DEFINING THE CLASS OF GRAPH WE USE
    DU_GRAPH = Graph_MultiSinglePageXml

    lLabels = ['B', 'I', 'O']
    
    lIgnoredLabels = []
    
    """
    if you play with a toy collection, which does not have all expected classes, you can reduce those.
    """
    
    lActuallySeen = None
    if lActuallySeen:
        print( "REDUCING THE CLASSES TO THOSE SEEN IN TRAINING")
        lIgnoredLabels  = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
        lLabels         = [lLabels[i] for i in lActuallySeen ]
        print( len(lLabels)          , lLabels)
        print( len(lIgnoredLabels)   , lIgnoredLabels)
    
    nt = NodeType_PageXml_type_woText_BIESO_to_BIO(
                            "abp"                   #some short prefix because labels below are prefixed with it
                          , lLabels
                          , lIgnoredLabels
                          , False    #no label means OTHER
                          , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                          )
    nt.setLabelAttribute("DU_row")
    
    nt.setXpathExpr( (".//pc:TextLine"        #how to find the nodes
                      , "./pc:TextEquiv")       #how to get their text
                   )
    
    # ntA.setXpathExpr( (".//pc:TextLine | .//pc:TextRegion"        #how to find the nodes
    #                   , "./pc:TextEquiv")       #how to get their text
    #                 )
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

    doer = DU_Task_Factory.getDoer(sModelDir, sModelName
                                   , options                    = options
                                   , fun_getConfiguredGraphClass= getConfiguredGraphClass
                                   , cFeatureDefinition         = FeatureDefinition_PageXml_StandardOnes_noText_v4
                                   , dFeatureConfig             = {}                                           
                                   )
    
    # setting the learner configuration, in a standard way 
    # (from command line options, or from a JSON configuration file)
    dLearnerConfig = doer.getStandardLearnerConfig(options)
    # of course, you can put yours here instead.
    doer.setLearnerConfiguration(dLearnerConfig)


    # act as per specified in the command line (--trn , --fold-run, ...)
    doer.standardDo(options)
    
    del doer

