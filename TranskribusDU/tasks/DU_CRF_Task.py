# -*- coding: utf-8 -*-

"""
    CRF DU task core. Supports classical CRF and Typed CRF
    
    Copyright Xerox(C) 2016, 2017 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import trace, traceln
from common.chrono import chronoOn, chronoOff

from tasks.DU_Task import DU_Task
import graph.GraphModel

# dynamically imported
Model_SSVM_AD3              = None
Model_SSVM_AD3_Multitype    = None


class DU_CRF_Task(DU_Task):
    """
    DU learner based on graph CRF
    """
    VERSION = "CRF_v19"

    version          = None     # dynamically computed
    
    sMetadata_Creator = "NLE Document Understanding Typed CRF-based - v0.4"
    
    # sXmlFilenamePattern = "*[0-9]"+MultiPageXml.sEXT    #how to find the Xml files

    def __init__(self, sModelName, sModelDir
                 , sComment             = None
                 , cFeatureDefinition   = None
                 , dFeatureConfig       = {}
                 ): 
         
        super().__init__(sModelName, sModelDir
                         , sComment             = sComment
                         , cFeatureDefinition   = cFeatureDefinition
                         , dFeatureConfig       = dFeatureConfig)

        # Dynamic import
        global Model_SSVM_AD3, Model_SSVM_AD3_Multitype
        Model_SSVM_AD3           = DU_Task.DYNAMIC_IMPORT('.Model_SSVM_AD3', 'crf').Model_SSVM_AD3
        Model_SSVM_AD3_Multitype = DU_Task.DYNAMIC_IMPORT('.Model_SSVM_AD3_Multitype', 'crf').Model_SSVM_AD3_Multitype
        
        if self.iNbNodeType > 1:
            #check the configuration of a MULTITYPE graph
            setKeyGiven = set(dFeatureConfig.keys())
            lNT = self.cGraphClass.getNodeTypeList()
            setKeyExpected = {nt.name for nt in lNT}.union( {"%s_%s"%(nt1.name,nt2.name) for nt1 in lNT for nt2 in lNT} )
            
            setMissing = setKeyExpected.difference(setKeyGiven)
            setExtra   = setKeyGiven.difference(setKeyExpected)
            if setMissing: traceln("ERROR: missing feature extractor config for : ", ", ".join(setMissing))
            if setExtra:   traceln("ERROR: feature extractor config for unknown : ", ", ".join(setExtra))
            if setMissing or setExtra: raise ValueError("Bad feature extractor configuration for a multi-type CRF graph")
           
        self.cModelClass = Model_SSVM_AD3 if self.iNbNodeType == 1 else Model_SSVM_AD3_Multitype
        assert issubclass(self.cModelClass, graph.GraphModel.GraphModel), "Your model class must inherit from graph.GraphModel.GraphModel"
        
    #---  CONFIGURATION setters --------------------------------------------------------------------
    def isTypedCRF(self): 
        """
        if this a classical CRF or a Typed CRF?
        """
        return bool(self.iNbNodeType > 1)
    
    
    #---  COMMAND LINE PARSZER --------------------------------------------------------------------
    @classmethod
    def getVersion(cls):
        cls.version = "-".join([DU_Task.getVersion(), str(cls.VERSION)])
        return cls.version 

    @classmethod
    def updateStandardOptionsParser(cls, parser):
        usage = """
        --crf          : use graph-CRF learner
        --crf-njobs    : number of parallel training jobs
        
        --crf-XXX      : set the XXX trainer parameter. XXX can be max_iter, C, tol, inference-cache
                            If several values are given, a grid search is done by cross-validation. 
                            The best set of parameters is then stored and can be used thanks to the --best-params option.
        --best-params  : uses the parameters obtained by the previously done grid-search. 
                            If it was done on a model fold, the name takes the form: <model-name>_fold_<fold-number>, e.g. foo_fold_2

        """        
        # for CRF
        parser.add_option("--crf"        , dest='bCRF'           , action="store_true"
                          , default=False, help="use a graph-CRF Model")
        
        parser.add_option("--crf-njobs", dest='crf_njobs',  action="store", type="int"
                          , help="CRF training parameter njobs")
        parser.add_option("--crf-C"                 , dest='crf_C'              ,  action="append", type="float"
                          , help="CRF training parameter C")    
        parser.add_option("--crf-tol"               , dest='crf_tol'            ,  action="append", type="float"
                          , help="CRF training parameter tol")    
        parser.add_option("--crf-inference_cache"   , dest='crf_inference_cache',  action="append", type="int"
                          , help="CRF training parameter inference_cache")    
        parser.add_option("--best-params", dest='best_params',  action="store", type="string"
                          , help="Use the best  parameters from the grid search previously done on the given model or model fold") 

        parser.add_option("--storeX" , dest='storeX' ,  action="store", type="string", help="Dev: to be use with --run: load the data and store [X] under given filename, and exit")
        parser.add_option("--applyY" , dest='applyY' ,  action="store", type="string", help="Dev: to be use with --run: load the data, label it using [Y] from given file name, and store the annotated data, and exit")

        # REDUNDANT with --max_iter but for backward compatibility
        parser.add_option("--crf-max_iter"          , dest='max_iter'       ,  action="store", type="int"        # "append" would allow doing a gridsearch on max_iter...
                          , help="Maximum number of iterations allowed (same as max_iter)")    
                
#         # ECN and GAT disabled for now....
#         parser.add_option("--ecn"        , dest='bECN'           , action="store_true", default=False)
#         parser.add_option("--gat"        , dest='bGAT'           , action="store_true", default=False)

        return usage, parser
           
    #---  UTILITIES  ------------------------------------------------------------------------------------------
    def getStandardLearnerConfig(self, options):
        """
        Once the command line has been parsed, you can get the standard learner
        configuration dictionary from here.
        """
        o = options
        return  {  
              'njobs'           : 16   if o.crf_njobs           is None else o.crf_njobs
            , 'max_iter'        : 1000 if o.max_iter            is None else o.max_iter
            , 'C'               : .1   if o.crf_C               is None else o.crf_C
            , 'inference_cache' : 50   if o.crf_inference_cache is None else o.crf_inference_cache
            , 'tol'             : .05  if o.crf_tol             is None else o.crf_tol
            , 'save_every'      : 10
            , 'balanced'        : False     # balanced instead of uniform class weights
                }
# ------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    version = "v.01"
    usage, description, parser = DU_CRF_Task.getStandardOptionsParser(sys.argv[0], version)

    parser.print_help()
    
    traceln("\nThis module should not be run as command line. It does nothing. (And did nothing!)")
