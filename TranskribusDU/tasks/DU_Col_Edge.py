# -*- coding: utf-8 -*-

"""
    DU task for segmenting text in cols using the conjugate graph after the SW
    re-engineering by JLM
    
    Copyright Xerox(C)  2019  Jean-Luc Meunier
    
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

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln

from tasks.DU_Task_Factory                          import DU_Task_Factory
from tasks.DU_Row_Edge                              import ConjugateSegmenterGraph_MultiSinglePageXml
from tasks.DU_Row_Edge                              import My_ConjugateNodeType
from graph.FeatureDefinition_PageXml_std_noText_v4  import FeatureDefinition_PageXml_StandardOnes_noText_v4


def getConfiguredGraphClass(doer):
    """
    In this class method, we must return a configured graph class
    """
    DU_GRAPH = ConjugateSegmenterGraph_MultiSinglePageXml  # consider each age as if indep from each other

    # ntClass = NodeType_PageXml_type
    ntClass = My_ConjugateNodeType

    nt = ntClass("col"                   #some short prefix because labels below are prefixed with it
                  , []
                  , []
                  , True                #unused
                  , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  )    
    nt.setLabelAttribute("col")
    nt.setXpathExpr( (".//pc:TextLine"        #how to find the nodes            
                      #nt1.setXpathExpr( (".//pc:TableCell//pc:TextLine"        #how to find the nodes
                      , "./pc:TextEquiv")       #how to get their text
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

    doer = DU_Task_Factory.getDoer(sModelDir, sModelName
                                   , options                    = options
                                   , fun_getConfiguredGraphClass= getConfiguredGraphClass
                                   , cFeatureDefinition         = FeatureDefinition_PageXml_StandardOnes_noText_v4
                                   , dFeatureConfig             = {}                                           
                                   )
    
    # setting the learner configuration, in a standard way 
    # (from command line options, or from a JSON configuration file)
    dLearnerConfig = doer.getStandardLearnerConfig(options)
    doer.setLearnerConfiguration(dLearnerConfig)

    doer.setConjugateMode()
    
    # act as per specified in the command line (--trn , --fold-run, ...)
    doer.standardDo(options)
    
    del doer

