# -*- coding: utf-8 -*-

"""
    DU task: predicting directly the row or col number
    
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
from shutil import copyfile
from collections import defaultdict
from lxml import etree

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln, trace
from xml_formats.PageXml import PageXml
from tasks.DU_Task_Factory                          import DU_Task_Factory
from graph.Graph_Multi_SinglePageXml                import Graph_MultiSinglePageXml
from graph.NodeType_PageXml                         import defaultBBoxDeltaFun
from graph.NodeType_PageXml                         import NodeType_PageXml_type_woText
from graph.FeatureDefinition_PageXml_std_noText_v4  import FeatureDefinition_PageXml_StandardOnes_noText_v4

# ----------------------------------------------------------------------------


sATTRIBUTE  = "row"
iMIN, iMAX  = 0, 15
iMAXMAX     = 99
sXPATH      = ".//pc:TextLine"

bOTHER      = True  # do we need OTHER as label?

# sXPATH      = ".//pc:TextLine[../@%s]" % sATTRIBUTE

"""
bad approach: the is forced to set the last loaded cell to row_9 
iMIN, iMAX  = "0", "9"
sXPATH      = ".//pc:TextLine[%s <= ../@%s and ../@%s <= %s]" % (iMIN, sATTRIBUTE, sATTRIBUTE, iMAX)
"""

# =======  UTILITY  =========
def split_by_max(crit, sDir):
    """
    here, we create sub-folders, where the files have the same number of row or col
    """
    assert crit in ["row", "col"]
    
    sColDir= os.path.join(sDir, "col")
    traceln("- looking at ", sColDir)
    lsFile = []
    for _fn in os.listdir(sColDir):
        _fnl = _fn.lower()
        if _fnl.endswith("_du.mpxml") or _fnl.endswith("_du.pxml"):
            continue
        if not(_fnl.endswith(".mpxml") or _fnl.endswith(".pxml")):
            continue
        lsFile.append(_fn)
    traceln(" %d files" % len(lsFile))
    
    dCnt = defaultdict(int)

    for sFilename in lsFile:
        trace("- %s" % sFilename)
        sInFile = os.path.join(sColDir, sFilename)
        doc = etree.parse(sInFile)
        rootNd = doc.getroot()
        vmax = -999
        xp = "//@%s" % crit
        try:
            vmax = max(int(_nd) for _nd in PageXml.xpath(rootNd, xp))
            assert vmax >= 0
            sToDir = "%s_%s_%d"%(sDir, crit, vmax)
        except ValueError:
            trace("  ERROR on file %s" % sInFile)
            vmax = None
            sToDir = "%s_%s_%s"%(sDir, crit, vmax)
        del doc
        sToColDir = os.path.join(sToDir, "col")
        try:
            os.mkdir(sToDir)
            os.mkdir(sToColDir)
        except FileExistsError: pass
        copyfile(sInFile, os.path.join(sToColDir, sFilename))
        traceln("   -> ", sToColDir)
        dCnt[vmax] += 1
    traceln("WARNING: %d invalid files"%dCnt[None]) 
    del dCnt[None]
    traceln(sorted(dCnt.items()))
    
    
# =======  DOER  =========

class My_NodeType_Exception(Exception):
    pass


class My_NodeType(NodeType_PageXml_type_woText):
    """
    We need this to extract properly the label from the label attribute of the (parent) TableCell element.
    """
    sLabelAttr = sATTRIBUTE
    
    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True
                 , BBoxDeltaFun=defaultBBoxDeltaFun):
        super(My_NodeType, self).__init__(sNodeTypeName, lsLabel
                                          , lsIgnoredLabel=lsIgnoredLabel
                                          , bOther=bOther
                                          , BBoxDeltaFun=BBoxDeltaFun)

    def parseDomNodeLabel(self, domnode, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        sLabel = domnode.getparent().get(self.sLabelAttr)
        if sLabel is None: 
            if self.bOther: 
                return self.getDefaultLabel()
            else:
                raise My_NodeType_Exception("Missing attribute @%s for node id=%s" % (self.sLabelAttr, domnode.get("id")))
        try: 
            i = int(sLabel) 
            if i > iMAX: 
                if self.bOther:
                    return self.getDefaultLabel()
                else:
                    raise My_NodeType_Exception("Too large label integer value: @%s='%s' for node id=%s" % (self.sLabelAttr, sLabel, domnode.get("id")))
            else:
                return sATTRIBUTE + "_" + sLabel
        except: 
            raise My_NodeType_Exception("Invalid label value: @%s='%s' for node id=%s" % (self.sLabelAttr, sLabel, domnode.get("id")))

    def setDomNodeLabel(self, domnode, sLabel):
        domnode.set("DU_%s"%self.sLabelAttr, sLabel)

    
def getConfiguredGraphClass(doer):
    """
    In this class method, we must return a configured graph class
    """
    DU_GRAPH = Graph_MultiSinglePageXml  # consider each age as if indep from each other

    # ntClass = NodeType_PageXml_type
    ntClass = My_NodeType

    nt = ntClass(sATTRIBUTE              # some short prefix because labels below are prefixed with it
                  , [str(n) for n in range(int(iMIN), int(iMAX)+1)]
                  # in --noother, we have a strict list of label values!
                  , ["%s_%d"%(sATTRIBUTE, n) for n in range(int(iMAX), int(iMAXMAX)+1)] if bOTHER else []
                  , bOTHER                # unused
                  , BBoxDeltaFun=lambda v: max(v * 0.066, min(5, v/3))  #we reduce overlap in this way
                  )    
    #nt.setLabelAttribute("row")
    nt.setXpathExpr( (sXPATH       #how to find the nodes            
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

    parser.add_option("--what", dest='sWhat'       ,  action="store", type="string", default="row"
                      , help='what to predict, e.g. "row", "col")')    
    parser.add_option("--max", dest='iMax'       ,  action="store", type="int"
                      , help="Maximum number to be found (starts at 0)")    
    parser.add_option("--intable", dest='bInTable'  ,  action="store_true"
                      , help="Ignore TextLine ouside the table region, using the GT")    
    parser.add_option("--noother", dest='bNoOther'  ,  action="store_true"
                      , help="No 'OTHER' label")    
 
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    
    if args and args[0] == "split-by-max":
        assert len(args) == 3, 'expected: split-by-max row|col <FOLDER>'
        split_by_max(args[1], args[2])
        exit(0)
        
        
    # standard arguments
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        DU_Task_Factory.exit(usage, 1, e)

    # specific options
    if options.iMax:            iMAX        = options.iMax
    if options.sWhat:           sATTRIBUTE  = options.sWhat 
    if bool(options.bInTable):  sXPATH      = sXPATH + "[../@%s]" % sATTRIBUTE
    if options.bNoOther:        bOTHER      = False
                                
    # some verbosity                                
    traceln('Prediction "%s" from %d to %d using selector %s  (%s)' % (
        sATTRIBUTE, 0, iMAX, sXPATH
        , "With label 'OTHER'" if bOTHER else "Without label 'OTHER'" ))
    
    # standard options        
    doer = DU_Task_Factory.getDoer(sModelDir, sModelName
                                   , options                    = options
                                   , fun_getConfiguredGraphClass= getConfiguredGraphClass
                                   , cFeatureDefinition         = FeatureDefinition_PageXml_StandardOnes_noText_v4
                                   , dFeatureConfig             = {}                                           
                                   )
    
    # setting the learner configuration, in a standard way 
    # (from command line options, or from a JSON configuration file)
    # dLearnerConfig = doer.getStandardLearnerConfig(options)
    if options.bECN:
        dLearnerConfig = {
                "name"                  :"default_8Lay1Conv",
                "dropout_rate_edge"     : 0.2,
                "dropout_rate_edge_feat": 0.2,
                "dropout_rate_node"     : 0.2,
                "lr"                    : 0.0001,
                "mu"                    : 0.0001,
                "nb_iter"               : 3000,
                "nconv_edge"            : 1,
                "node_indim"            : -1,
                "num_layers"            : 8,
                "ratio_train_val"       : 0.1,
                "patience"              : 100,
                "activation_name"       :"relu",
                "stack_convolutions"    : False
                }
    elif options.bCRF:
        dLearnerConfig = doer.getStandardLearnerConfig(options)
    else:
        raise "Unsupported method"
    
    if options.max_iter:
        traceln(" - max_iter=%d" % options.max_iter)
        dLearnerConfig["nb_iter"] = options.max_iter
            
    if False:
        # force a balanced weighting
        print("Forcing balanced weights")
        dLearnerConfig['balanced'] = True
    
    doer.setLearnerConfiguration(dLearnerConfig)

    # act as per specified in the command line (--trn , --fold-run, ...)
    doer.standardDo(options)
    
    del doer

