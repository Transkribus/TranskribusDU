# -*- coding: utf-8 -*-

"""
    DU task factory class.
    
    Copyright Xerox(C) 2019 JL. Meunier

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
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os
import traceback
from optparse import OptionParser
import random

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from graph.Graph import Graph
from tasks.DU_CRF_Task import DU_CRF_Task
from tasks.DU_ECN_Task import DU_ECN_Task
from tasks.DU_GAT_Task import DU_GAT_Task

class DU_Task_Factory:
    VERSION = "Factory_19"

    version          = None     # dynamically computed
    
    l_CHILDREN_CLASS = [DU_CRF_Task, DU_ECN_Task, DU_GAT_Task]
    # faster load for debug... l_CHILDREN_CLASS = [DU_CRF_Task]

    @classmethod
    def getStandardOptionsParser(cls, sys_argv0=None):
        usage = """"%s <model-folder> <model-name> [--rm] [--trn <col-dir> [--warm] [--vld <col-dir>]+]+ [--tst <col-dir>]+ [--run <col-dir> [--graph]]+
or for a cross-validation [--fold-init <N>] [--fold-run <n> [-w]] [--fold-finish] [--fold <col-dir>]+
[--pkl]
[--g1|--g2]

        For the named MODEL using the given FOLDER for storage:
        --rm  : remove all model data from the folder
        --trn : train a model using the given data (multiple --trn are possible)
                --warm/-w: warm-start the training if applicable
                --vld    : use this/these validation set(s) for training - can be a ratio to extract the validation set from the training set
        --tst : test the model using the given test collection (multiple --tst are possible)
        --run : predict using the model for the given collection (multiple --run are possible)
        
        --fold        : enlist one collection as data source for cross-validation
        --fold-init   : generate the content of the N folds 
        --fold-run    : run the given fold, if --warm/-w, then warm-start if applicable 
        --fold-finish : collect and aggregate the results of all folds that were run.

        --max_iter    : Maximum number of learning iterations allowed.
        --seed        : Seed for the randomizer (to reproduce results)
        --detail      : Display detailed reporting (score per document)
        
        --pkl         : store the data as a pickle file containing PyStruct data structure (lX, lY) and exit
        --graph       : store the graph in the output XML
        --g1          : default mode (historical): edges created only to closest overlapping block (downward and rightward)
        --g2          : implements the line-of-sight edges (when in line of sight, then link by an edge)
        """%sys_argv0

        #prepare for the parsing of the command line
        parser = OptionParser(usage=usage, version=cls.getVersion())
        
        parser.add_option("--trn", dest='lTrn',  action="append", type="string"
                          , help="Train or continue previous training session using the given annotated collection.")    
        parser.add_option("--vld", dest='lVld',  action="append", type="string"
                          , help="Use this validation data while training.")    
        parser.add_option("--tst", dest='lTst',  action="append", type="string"
                          , help="Test a model using the given annotated collection.")    
        parser.add_option("--run", dest='lRun',  action="append", type="string"
                          , help="Run a model on the given non-annotated collection.")    
        parser.add_option("--fold", dest='lFold',  action="append", type="string"
                          , help="Evaluate by cross-validation a model on the given annotated collection.")    
        parser.add_option("--fold-init", dest='iFoldInitNum',  action="store", type="int"
                          , help="Initialize the file lists for parallel cross-validating a model on the given annotated collection. Indicate the number of folds.")    
        parser.add_option("--fold-run", dest='iFoldRunNum',  action="store", type="int"
                          , help="Run one fold, prepared by --fold-init options. Indicate the fold by its number.")    
        parser.add_option("--fold-finish", dest='bFoldFinish',  action="store_true"
                          , help="Evaluate by cross-validation a model on the given annotated collection.")    
        parser.add_option("-w", "--warm", dest='warm',  action="store_true"
                          , help="To make warm-startable model and warm-start if a model exist already.")   
        parser.add_option("--pkl", dest='bPkl', action="store_true"
                          , help="GZip and pickle PyStruct data as (lX, lY) on disk.")    
        parser.add_option("--graph", dest='bGraph',  action="store_true"
                          , help="Store the graph in the XML for displaying it") 
        parser.add_option("--rm", dest='rm',  action="store_true"
                          , help="Remove all model files")   
        parser.add_option("--detail"     , dest='bDetailedReport'   , action="store_true"
                          , default=False,help="Display detailed reporting (score per document)") 
        parser.add_option("--max_iter"          , dest='max_iter'       ,  action="store", type="int"        # "append" would allow doing a gridsearch on max_iter...
                          , help="Maximum number of iterations allowed")    
        parser.add_option("--seed"          , dest='seed'       ,  action="store", type="int"        # "append" would allow doing a gridsearch on max_iter...
                          , help="Randomizer seed")  
        parser.add_option("--g1", dest='bG1',  action="store_true"
                          , help="default mode (historical): edges created only to closest overlapping block (downward and rightward)")   
        parser.add_option("--g2", dest='bG2',  action="store_true"
                          , help="implements the line-of-sight edges (when in line of sight, then link the nodes by an edge)")   

            
        # consolidate...
        for c in cls.l_CHILDREN_CLASS:
            c_usage, parser = c.updateStandardOptionsParser(parser)
            usage = usage + "\n" + c_usage 
        
        return usage, parser

    @classmethod
    def exit(cls, usage, status, exc=None):
        if usage:        traceln("\nUSAGE : %s\n"%usage)
        if exc != None:  traceback.print_exc()
        sys.exit(status)    

    @classmethod
    def getVersion(cls):
        cls.version = str(cls.VERSION) + ": " + " | ".join([c.getVersion() for c in cls.l_CHILDREN_CLASS])
        return cls.version 
        
    @classmethod
    def getDoer(cls, sModelDir, sModelName
                , options = None
                , bCRF    = None
                , bECN    = None
                , bGAT    = None
                , fun_getConfiguredGraphClass   = None
                , sComment                      = None
                , cFeatureDefinition            = None
                , dFeatureConfig                = {}                
                ):
        """
        Create the requested doer object 
        """
        assert not fun_getConfiguredGraphClass  is None, "You must provide a getConfiguredGraphClass method"
        assert not cFeatureDefinition           is None, "You must provide a cFeatureDefinition class"

        # Graph mode for computing edges
        assert not(options.bG1 and options.bG2), "Specify graph mode either 1 or 2 not both"
        iGraphMode = 1
        if options.bG1: iGraphMode = 1
        if options.bG2: iGraphMode = 2
        Graph.setGraphMode(iGraphMode)

        bCRF = bCRF or (not(options is None) and options.bCRF)
        bECN = bECN or (not(options is None) and options.bECN)
        bGAT = bGAT or (not(options is None) and options.bGAT)
        
        assert (bCRF or bECN or bGAT)       , "You must specify one learning method."
        assert [bCRF, bECN, bGAT].count(True) == 1  , "You must specify only one learning method."
        
        if bECN: 
            c = DU_ECN_Task
        elif bCRF: 
            c = DU_CRF_Task
        elif bGAT: 
            c = DU_GAT_Task
            
        c.getConfiguredGraphClass = fun_getConfiguredGraphClass

        doer = c(sModelName, sModelDir
                 , sComment             = sComment
                 , cFeatureDefinition   = cFeatureDefinition
                 , dFeatureConfig       = dFeatureConfig)

        if options.seed is None:
            random.seed()
            traceln("SETUP: Randomizer initialized automatically")
        else:
            random.seed(options.seed)
            traceln("SETUP: Randomizer initialized by your seed (%d)"%options.seed)
            
        traceln("SETUP: doer : class=%s  version=%s" % (doer.__class__.__qualname__, doer.getVersion()))
        
        return doer
