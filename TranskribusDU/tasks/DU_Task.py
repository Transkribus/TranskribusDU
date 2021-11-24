# -*- coding: utf-8 -*-

"""
    DU task core.
    
    Copyright Xerox(C) 2016, 2017 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union?s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import sys, os, glob, datetime
import gc
import json
from importlib import import_module
from io import StringIO
import traceback
import tempfile
import lxml.etree as etree

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV  #0.18.1 REQUIRES NUMPY 1.12.1 or more recent
    
try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import trace, traceln
from common.chrono import chronoOn, chronoOff, pretty_time_delta
from common.TestReport import TestReportConfusion
import util.Tracking as Tracking
import util.metrics as metrics
from util.gzip_pkl import gzip_pickle_dump, gzip_pickle_load
from xml_formats.PageXml import MultiPageXml

from graph.GraphModel import GraphModel, GraphModelException, GraphModelNoEdgeException
from graph.GraphConjugate import GraphConjugateException
from graph.Graph import GraphException
from graph.Graph_JsonOCR import Graph_JsonOCR
from graph.Graph_DOM import Graph_DOM
from graph.Edge import CelticEdge
import graph.FeatureDefinition
from tasks import _checkFindColDir

from .DU_Table.DU_Table_Evaluator import eval_cluster_of_files, computePRF

import tasks.DU_Task_Features   as DU_TaskFeatures


# to activate the use of an oracle for predicting continue or break
bORACLE_EDGE_BREAK_CONTINUE = False


def getDataToPickle(doer, mdl, lGraph):
    """
    Generic pickler, widely used
    
    data that is specific to this task, which we want to pickle when --pkl is used
    for each node of each graph, we want to store the node text + geometry + label
    
    ( (text, (x1, y1, x2, y2), label )
    
    
    return [ 
            [ 
                (text, (x1, y1, x2, y2), label )  
                ...  (per block)
             ]
           ... (per graph)
           ]
    """
    lDataByGraph = []
    for g in lGraph:
        lNodeData = []
        for nd in g.lNode:
            data = (nd.text
                      , (nd.x1, nd.y1, nd.x2, nd.y2)
                      , nd.cls
                      )
            lNodeData.append(data)
        lDataByGraph.append(lNodeData)
    return lDataByGraph


class DU_Task:
    """
Document Understanding class

USAGE:
- get a CRF- or ECN-based instance by the getDoer class method
- choose a model name and a folder where it will be stored
- define the features and their configuration, or a list of (feature definition and its configuration)
- choose to learn using CRF or ECN (via command line options or method)
- define the learner configuration
- run the do method

    doer = DU_Task.getDoer(sModelDir, sModelName, getConfiguredGraphClass, options=.., bCRF=.., bECN=.., )
    # here, you can configure it further if needed ...
    # use the standard command line options to do something
    doer.standardDo(options)
    del doer
    
METHODs:
- training: train_save_test
- loading a trained model: load
- testing a trained model: test
- removing a model from the disk: rm
- if you want to use specific features: setFeatureDefinition
- if you want to also train/test against some baseline model: setBaselineList, addBaseline_LogisticRegression 

See DU_StAZH_b.py

    """
    VERSION          = "DU_Task_v19" 
    
    version          = None     # dynamically computed
    
    cModelClass      = None     #depends on the number of node types!
    cGraphClass      = None     #class of graph in use 
    
    cFeatureDefinition   = None # FeatureDefinition_PageXml_StandardOnes   #I keep this for backward compa
    
    sMetadata_Creator = "NLE Document Understanding: DU_Task"
    sMetadata_Comments = ""

    #dGridSearch_LR_conf = {'C':[0.1, 0.5, 1.0, 2.0] }  #Grid search parameters for LR baseline method training
    dGridSearch_LR_conf   = {'C':[0.01, 0.1, 1.0, 10.0] }  #Grid search parameters for LR baseline method training
    dGridSearch_LR_n_jobs = 4                              #Grid search: number of jobs
    
    # why this old-stuff??  sXmlFilenamePattern = "*[0-9]"+MultiPageXml.sEXT    #how to find the Xml files
    sXmlFilenamePattern = "*"+MultiPageXml.sEXT    #how to find the Xml files

    iNbNodeType = 1     # as of today, only CRF can do multitype
    
    bConjugate = False
    
    def configureGraphClass(self, configuredClass=None):
        """
        class method to set the graph class ONCE (subsequent calls are ignored)
        """
        if self.cGraphClass is None: #OK, let's set the class attribute!
            
            #if nothing in parameter, or we call the class method
            if configuredClass is None:
                configuredClass = self.getConfiguredGraphClass()
                    
                assert configuredClass is not None, "getConfiguredGraphClass returned None"
                
            self.cGraphClass = configuredClass
            self.bConjugate  = configuredClass.bConjugate

        assert self.cGraphClass is not None
        traceln("SETUP: Graph class is %s  (graph mode %d)" % (self.cGraphClass, self.cGraphClass.getGraphMode()))
        traceln("SETUP: Input format is '%s'" % (self.cGraphClass.getDocInputFormat()))
        
        return self.cGraphClass

    @staticmethod
    def DYNAMIC_IMPORT(name, package=None):
        chronoOn("import")
        trace("SETUP: Dynamic import of '%s' from '%s'" % (name, package))
        m = import_module(name, package)
        traceln("  done [%.1fs]" % chronoOff("import"))
        return m

    
    def __init__(self, sModelName, sModelDir
                 , sComment             = None
                 , cFeatureDefinition   = None
                 , dFeatureConfig       = {}
                 ): 
        self.sModelName    = sModelName
        self.sModelDir     = sModelDir
        if sComment: 
            self.sMetadata_Comments    = sComment

        self.configureGraphClass()

        
        self._mdl = None
        
        self._lBaselineModel = []
        self.bVerbose = True
        
        #for single- or multi-type CRF, the same applies!
        self.lNbClass = [len(nt.getLabelNameList()) for nt in self.cGraphClass.getNodeTypeList()]
        self.nbClass = sum(self.lNbClass)
        self.iNbNodeType = len(self.cGraphClass.getNodeTypeList())

        #--- feature definition and configuration per type
        #Feature definition and their config
        if cFeatureDefinition: self.cFeatureDefinition  = cFeatureDefinition
        assert issubclass(self.cFeatureDefinition, graph.FeatureDefinition.FeatureDefinition), "Your feature definition class must inherit from graph.FeatureDefinition.FeatureDefinition"
        
        self.config_extractor_kwargs = dFeatureConfig

        self.config_learner_kwargs = None   # to be set
        self.cModelClass = None             # to be specialized

        self._additional_data_fun = None

        self.fClusterTH = 0.5

    #---  DOER --------------------------------------------------------------------
    @classmethod
    def getVersion(cls):
        return str(cls.VERSION) 

    def standardDo(self, options, experiment_name="DU"):
        """
        do whatever is requested by an option from the parsed command line
        
        return None
        """
        global bORACLE_EDGE_BREAK_CONTINUE
        
        Tracking.set_no_tracking()
        
        # use as TH for clustering
        self.fClusterTH = options.iClusterTH / 100
            
        if options.bEdgeOracle: 
            traceln("*** EDGE ORACLE: predict 'break' or 'continue' label from the groundtruth ***")
            bORACLE_EDGE_BREAK_CONTINUE = True
            
        if bool(options.iServer):
            assert not bORACLE_EDGE_BREAK_CONTINUE
            self.load()
            # run in server mode!
            self.serve_forever(options.iServer, options.bServerDebug, options=options)
            return
        
        if options.rm:
            assert not bORACLE_EDGE_BREAK_CONTINUE
            self.rm()
            return

        if options.iCelticEdge > 0:
            traceln(" - celtic edges included (max per sector = %d)"%options.iCelticEdge)
            CelticEdge.setViewRange  (9e99)
            CelticEdge.setMaxBySector(options.iCelticEdge)
            # we must account for CelticEdge type
            DU_TaskFeatures.addCelticEdgeType()
        else:
            traceln(" - no celtic edge included")
            
    
        lTrn, lTst, lRun, lFold = [_checkFindColDir(lsDir) for lsDir in [options.lTrn, options.lTst, options.lRun, options.lFold]]
        
        # Validation set if any
        try:
            ratio_train_val = float(options.lVld[0])
            lVld            = []
            if ratio_train_val <= 0 or 1.0 <= ratio_train_val: raise Exception("Bad ratio, not in ]0, 1[")
        except:
            ratio_train_val = None
            lVld            = _checkFindColDir(options.lVld)
        #traceln("- classes: ", doer.getGraphClass().getLabelNameList())
    
        ## use. a_mpxml files
        #doer.sXmlFilenamePattern = doer.sLabeledXmlFilenamePattern
    
        if options.iFoldInitNum or options.iFoldRunNum or options.bFoldFinish:
            assert not bORACLE_EDGE_BREAK_CONTINUE
            if options.iFoldInitNum:
                """
                initialization of a cross-validation
                """
                splitter, ts_trn, lFilename_trn = self._nfold_Init(lFold, options.iFoldInitNum, test_size=0.25, random_state=None, bStoreOnDisk=True)
            elif options.iFoldRunNum:
                """
                Run one fold
                """
                if options.blPkl: raise ValueError("option lpkl not compatible with fold run")
                oReport = self._nfold_RunFoldFromDisk(options.iFoldRunNum, options.warm, options.bPkl)
                traceln(oReport)
            elif options.bFoldFinish:
                tstReport = self._nfold_Finish()
                traceln(tstReport)
            else:
                assert False, "Internal error"

            return
    
    
        if lFold:
            assert not bORACLE_EDGE_BREAK_CONTINUE
            if options.blPkl: raise ValueError("option lpkl not compatible with folds")
            loTstRpt = self.nfold_Eval(lFold, 3, .25, None, options.bPkl)
            sReportPickleFilename = os.path.join(self.sModelDir, self.sModelName + "__report.txt")
            traceln("Results are in %s"%sReportPickleFilename)
            GraphModel.gzip_cPickle_dump(sReportPickleFilename, loTstRpt)
        elif lTrn or lTst or lRun:
                
            if lTrn or lTst or (lRun and (options.bEvalRow or options.bEvalCol or options.bEvalCell or options.sEvalCluster or options.bEvalClusterLevel)):
                options.bMLFlow = options.bMLFlow or options.sMLFlowExp # force it!
                # ----------        Tracking stuff
                if options.sMLFlowURI:
                    if options.sMLFlowURI == "-" or options.sMLFlowURI.startswith("file"):
                        # tracking in local files
                        Tracking.set_tracking()
                    else:
                        Tracking.set_tracking_uri(options.sMLFlowURI)
                elif options.bMLFlow:
                    Tracking.set_tracking_uri()
                else:
                    Tracking.set_no_tracking()
                    
                # MLFLow Experiment name
                _s = options.sMLFlowExp if options.sMLFlowExp else experiment_name
                Tracking.set_experiment(_s)
                traceln("Tracking experiment = ", _s)
                # MLFLow Run name
                _s = options.sMLFlowRun if options.sMLFlowRun else self.sModelName
                Tracking.start_run(_s)
                traceln("Tracking run = ", _s)
                if os.environ.get("SLURM_JOB_ID"): Tracking.log_param("SLURM_JOB_ID", os.environ.get("SLURM_JOB_ID"))
                
                Tracking.log_artifact_string("General", json.dumps({
                      "main"              : str(os.path.abspath(sys.argv[0]))
                    , "main.args"         : str(sys.argv[1:])
                    , "main.graph_class"  : self.getGraphClass().__name__
                    , "main.graph_mode"   : self.getGraphClass().getGraphMode()
                    , "main.ModelDir"     : os.path.abspath(self.sModelDir) 
                    , "main.ModelName"    : self.sModelName
                    , "main.model_class"  : self.getModelClass().__name__
                    , "main.seed"         : options.seed
                    , "main.ext"          : options.sExt
                    , 'main.bWarm'        : options.warm
                    }, indent=True)
                                            )
                Tracking.log_artifact_string("Options", str(options))
                Tracking.log_artifact_string("Options.True", str({k:v for k,v in options.__dict__.items() if bool(v)}))
                
                if lTrn or lTst:
                    _dCfg = self.getStandardLearnerConfig(options)
                    # Tracking.log_params(_dCfg)
                    Tracking.log_artifact_string("LearningParam" 
                                                 , json.dumps(_dCfg, indent=True))
                    Tracking.log_artifact_string("Data", json.dumps({'lTrn':lTrn, 'lVld':lVld, 'lTst':lTst, 'ratio_train_val':ratio_train_val}
                                                                    , indent=True))
                
            if lTrn:
                assert not bORACLE_EDGE_BREAK_CONTINUE
                tstReport = self.train_save_test(lTrn, lTst, lVld, options.warm, options.bPkl
                                                 , ratio_train_val=ratio_train_val
                                                 , blPickleXY=options.blPkl)
                try:    traceln("Baseline best estimator: %s"%self.bsln_mdl.best_params_)   #for GridSearch
                except: pass
                traceln(self.getModel().getModelInfo())
                Tracking.log_artifact_string("Model", self.getModel().getModelInfo())
                if lTst:
                    traceln(tstReport)
                    Tracking.log_artifact_string("test_report", tstReport)
                    # Return global micro- Precision/Recall/F1, accuracy, support
                    _p,_r,_f,_a,_s = metrics.confusion_PRFAS(tstReport.getConfusionMatrix())
                    Tracking.log_metrics({'avgP':_p, 'avgR':_r, 'F1':_f, 'Accuracy':_a}, ndigits=3)
                    if options.bDetailedReport:
                        _sTstRpt = tstReport.getDetailledReport()
                        traceln(_sTstRpt)
                        Tracking.log_artifact_string("test_report_detailed", _sTstRpt)
            elif lTst:
                assert not bORACLE_EDGE_BREAK_CONTINUE
                self.load()
                tstReport = self.test(lTst)
                traceln(tstReport)
                Tracking.log_artifact_string("test_report", tstReport)
                # Return global micro- Precision/Recall/F1, accuracy, support
                _p,_r,_f,_a,_s = metrics.confusion_PRFAS(tstReport.getConfusionMatrix())
                Tracking.log_metrics({'avgP':_p, 'avgR':_r, 'avgF1':_f, 'Accuracy':_a}, ndigits=3)
                # details ...
                if options.bDetailedReport:
                    _sTstRpt = tstReport.getDetailledReport()
                    traceln(_sTstRpt)
                    Tracking.log_artifact_string("test_report_detailed", _sTstRpt)
#                     for test in lTst:
#                         sReportPickleFilename = os.path.join('..',test, self.sModelName + "__report.pkl")
#                         traceln('Report dumped into %s'%sReportPickleFilename)
#                         GraphModel.gzip_cPickle_dump(sReportPickleFilename, tstReport)
        
            if lRun:
#                 if options.storeX or options.applyY:
#                     try: self.load()
#                     except: pass    #we only need the transformer
#                     lsOutputFilename = self.runForExternalMLMethod(lRun, options.storeX, options.applyY, options.bRevertEdges)
#                 else:
                if options.blPkl: raise ValueError("option lpkl not compatible with run")
                if not bORACLE_EDGE_BREAK_CONTINUE: self.load(bPickle=options.bPkl)
                lsOutputFilename = self.predict(lRun, bGraph=options.bGraph,bOutXML=options.bOutXML
                                                , bPickle= options.bPkl)
                if options.bEvalRow or options.bEvalCol or options.bEvalCell or options.bEvalRegion or options.sEvalCluster:
                    if  options.sEvalCluster:
                        sLevel = options.sEvalCluster # only used for display n this case
                        l = self.getGraphClass().getNodeTypeList()
                        assert len(l) == 1, "Cannot compute cluster quality with multiple node types"
                        xpSelector = l[0].getXpathExpr()[0]  # node selector, text selector
                        nOk, nErr, nMiss, sRpt, _ = eval_cluster_of_files(lsOutputFilename
                                              , "cluster"
                                              , bIgnoreHeader=False
                                              , bIgnoreOutOfTable=True
                                              , xpSelector=xpSelector
                                              , sClusterGTAttr=options.sEvalCluster
                                             )
                    else:
                        if options.bEvalRow:        sLevel = "row"
                        elif options.bEvalCol:      sLevel = "col"
                        elif options.bEvalCell:     sLevel = "cell"
                        elif options.bEvalRegion:     sLevel = "region"
                        else: raise ValueError()
                        l = self.getGraphClass().getNodeTypeList()
                        assert len(l) == 1, "Cannot compute cluster quality with multiple node types"
                        xpSelector = l[0].getXpathExpr()[0]  # node selector, text selector                        
                        nOk, nErr, nMiss, sRpt, _ = eval_cluster_of_files(lsOutputFilename
                                              , sLevel
                                              , bIgnoreHeader=False
                                              , bIgnoreOutOfTable=True
                                              , xpSelector=xpSelector
                                             )
                    fP, fR, fF = computePRF(nOk, nErr, nMiss)
                    traceln(sRpt)
                    Tracking.log_artifact_string("Cluster_eval", sRpt)
                    Tracking.log_metrics({  ('P_%s'%sLevel) :fP
                                          , ('R_%s'%sLevel) :fR
                                          , ('F1_%s'%sLevel):fF}, ndigits=2)
                elif options.bEvalClusterLevel:
                    l = self.getGraphClass().getNodeTypeList()
                    assert len(l) == 1, "Cannot compute cluster quality with multiple node types"
                    nt = l[0]  #unique node type
                    xpSelector = nt.getXpathExpr()[0]  # node selector, text selector
                    for lvl in range(self.getGraphClass().getHierarchyDepth()):
                        nOk, nErr, nMiss, sRpt, _ = eval_cluster_of_files(lsOutputFilename
                                              , "cluster_lvl%d"%lvl
                                              , bIgnoreHeader=False
                                              , bIgnoreOutOfTable=False
                                              , xpSelector=xpSelector
                                              , sClusterGTAttr=nt.getLabelAttribute()[lvl]
                                             )
                        fP, fR, fF = computePRF(nOk, nErr, nMiss)
                        traceln(sRpt)
                        Tracking.log_artifact_string("Cluster_eval_lvl%d"%lvl, sRpt)
                        Tracking.log_metrics({  ('lvl%d_P' %lvl):fP
                                              , ('lvl%d_R' %lvl):fR
                                              , ('lvl%d_F1'%lvl):fF}, ndigits=2)
                        
                        
                traceln("Done, see in:\n  %s"%lsOutputFilename)
        else:
            traceln("No action specified in command line. Doing nothing... :)")
        
        Tracking.end_run("FINISHED")
        return

    def __del__(self):
        """
        trying to clean big objects
        """
        try:
            del self._mdl
            del self._lBaselineModel
            del self.cFeatureDefinition
            del self.cModelClass         
        except:
            pass
        self._mdl = None
        self._lBaselineModel = None
        self.cFeatureDefinition = None
        self.cModelClass = None
        
    #---  SERVER MODE ---------------------------------------------------------
    def serve_forever(self, iPort, bDebug=False, options={}):
        self.sTime_start = datetime.datetime.now().isoformat()
        self.sTime_load = self.sTime_start
        self.save_self = self
        
        import socket
        sURI = "http://%s:%d" % (socket.gethostbyaddr(socket.gethostname())[0], iPort)
        sDescr = """
- home page for humans: %s
- POST or GET on %s/predict with argument xml=... 
""" % ( sURI, sURI)
        traceln("SERVER MODE")
        traceln(sDescr)
        
        from flask import Flask
        from flask import request, abort
        from flask import render_template_string #, render_template
        from flask import redirect, url_for      #, send_from_directory, send_file
        from flask import Response
        
        # Create Flask app load app.config
        app = Flask(self.__class__.__name__)

        @app.route('/')
        def home_page():
            # String-based templates
            return render_template_string("""<!doctype html>
<title>DU_Task server</title>
<ul>
<li> <pre>Server start time : {{ start_time }}</pre>
<li> <pre>Model load time   : {{ load_time }}</pre>
<li> Model : ({{ model_type }}) {{ model_spec }}
</ul>
<p>
<a href='/reload'>reload the model</a>
<p>Provide some {{ input_format }} data and get PageXml output:
<form action="/predict" method="post">
  <textarea name="data" id="data" required rows="25" cols="80"></textarea>
  <input type="text" name="name" id="name" placeholder="form_cluster" size="30"/>
  <input type="text" name="nhyp" id="nhyp" placeholder="2"/>
  <input type="submit" value="predict"/>
</form>
<p>
If applicable: the name of the clustering method is one of: 
form_cluster   (the default one)
        
        HypothesisClusterer
        LocalHypothesisClusterer
        HierarchicalHypothesisClusterer
        LocalHierarchicalHypothesisClusterer

        HypothesisClustererBB
        LocalHypothesisClustererBB
        HierarchicalHypothesisClustererBB
        LocalHierarchicalHypothesisClustererBB
<p>Apart from "form_cluster", the number of hypothesis can be given. (Default to 2)
<p>
This server runs with those options: {{ sOptions }}
"""
                    , model_type=self.__class__.__name__
                    , model_spec=os.path.abspath(self.getModel().getModelFilename())
                    , input_format=self.getGraphClass().getDocInputFormat()
                    , start_time=self.sTime_start
                    , load_time=self.sTime_load
                    , sOptions=str(options))
            traceln("SERVER ENDING. BYE")
            
        @app.route('/predict', methods = ['POST'])
        def predict():
            try:    
                if hasattr(self, "http_init__predict_file"):
                    # to allow for customization of the service
                    self.http_init__predict_file(request)
                    
                sData = request.form['data']
                if sData.startswith("<?xml"):
                    # horrible hack....
                    # lxml rejects the XML declaration, want bytes, but reject bytes...
                    # did not find a good way to get raw value of the posted data
                    # sData = sData[sData.index("?>")+2:]
                    fd, sFullPath = tempfile.mkstemp(suffix=".xml", prefix="DU_Task_")
                    os.write(fd, sData.encode('utf-8'))
                    os.close(fd)
                    doc, lg = self._predict_file(self.getGraphClass(), [], sFullPath, bGraph=options.bGraph)
                    os.unlink(sFullPath)
                else:
                    doc, lg = self._predict_file(self.getGraphClass(), [], StringIO(sData), bGraph=options.bGraph)
                
                # if nothing to do, the method returns None...
                if doc is None:
                    # doc = etree.parse(StringIO(sXml))
                    return sData
                else:
                    if not(isinstance(doc, etree._ElementTree)):
                        traceln(" converting to PageXml...")
                        doc = Graph_DOM.exportToDom(lg)
                    sResp = etree.tostring(doc.getroot(), encoding='UTF-8', xml_declaration=False)
                    return Response(sResp, mimetype="application/xml")

            except Exception as e:
                traceln("-----  predict exception -------------------------")
                traceln(traceback.format_exc())
                traceln("--------------------------------------------------")
                abort(418, repr(e))

        @app.route('/reload')
        def reload():
            """
            Force to reload the model
            """
            traceln("Reloading the model")
            self.save_self.load(bForce=True)
            self.sTime_load = datetime.datetime.now().isoformat()
            return redirect(url_for('home_page'))
               
        @app.route('/stop')
        def stop():
            """
            Force to exit
            """
            traceln("Trying to stop the server...")
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                traceln("Exiting!  (but this may stop only one process..)")
                sys.exit(0)
            else:
                traceln("Shutting down")
                func()
            return redirect(url_for('home_page'))
            
        # RUN THE SERVER !!
        # CAUTION: TensorFlow incompatible with debug=True  (double load => GPU issue)
        app.run(host='0.0.0.0', port=iPort, debug=bDebug)
        

        return

       
    #---  CONFIGURATION setters --------------------------------------------------------------------
    def getGraphClass(self):    
        return self.cGraphClass
    
    def setModelClass(self, cModelClass): 
        self.cModelClass = cModelClass
        traceln("SETUP: Model class changed to ", self.cModelClass.__name__)
        assert issubclass(self.cModelClass, GraphModel), "Your model class must inherit from graph.GraphModel.GraphModel"
    
    def getModelClass(self):    
        return self.cModelClass
    
    def getModel(self):         
        return self._mdl
    
    def setLearnerConfiguration(self, dLearnerConfig, bCommandLineInput=False):
        if bCommandLineInput:
            # Because of the way of dealing with the command line, we may get singleton instead of scalar. We fix this here
            self.config_learner_kwargs      = {k:v[0] if type(v) is list and len(v)==1 else v for k,v in dLearnerConfig.items()}
        else:
            self.config_learner_kwargs = dLearnerConfig
        
        traceln("SETUP:", json.dumps(self.config_learner_kwargs
                                     , sort_keys=True, indent=4))
                
                

    def getStandardLearnerConfig(self, options):
        """
        Once the command line has been parsed, you can get the standard learner
        configuration dictionary from here.
        """
        traceln("WARNING: no learner configuration by default!!!")
        return {}
    
    """
    When some class is not represented on some graph, you must specify the number of class (per type if multiple types)
    Otherwise pystruct will complain about the number of states differeing from the number of weights
    """
    def setNbClass(self, useless_stuff):    #DEPRECATED - DO NOT CALL!! Number of class computed automatically
        traceln("SETUP: *** setNbClass is deprecated - update your code (but it should work fine!)")
        
    def getNbClass(self): #OK
        """
        return the total number of classes
        """
        return self.nbClass
    
    def setXmlFilenamePattern(self, sExt):
        """
        Set the expected file extension of the input data
        """    
        assert sExt, "Empty extension not allowed"
        if not sExt.startswith("."): sExt = "." + sExt
        self.sXmlFilenamePattern = "*" + sExt

        
    #----------------------------------------------------------------------------------------------------------    
    def setBaselineList(self, lMdl):
        """
        Add one or several baseline methods.
        set one or a list of sklearn model(s):
        - they MUST be initialized, so that the fit method can be called at train time
        - they MUST accept the sklearn usual predict method
        - they SHOULD support a concise __str__ method
        They will be trained with the node features, from all nodes of all training graphs
        """
        self._lBaselineModel = lMdl
        
    def getBaselineList(self):
        return self._lBaselineModel
    
    def addBaseline_LogisticRegression(self):
        """
        add as Baseline a Logistic Regression model, trained via a grid search
        """
        #we always have one LR model per type  (yes, even for single type ! ;-) )
        mdl = GridSearchCV(LogisticRegression(class_weight='balanced') 
                        , self.dGridSearch_LR_conf
                        , n_jobs=self.dGridSearch_LR_n_jobs)  
            
        return mdl

    #----------------------------------------------------------------------------------------------------------   
    # in case you want no output at all on stderr
    def setVerbose(self, bVerbose)  : self.bVerbose = bVerbose 
    def getVerbose(self)            : return self.bVerbose 
    
    def traceln(self, *kwargs)     : 
        if self.bVerbose: traceln(*kwargs)
        
    #----------------------------------------------------------------------------------------------------------    
    def load(self, bForce=False, bPickle=False):
        """
        Load the model from the disk
        if bForce == True, force the load, even if the model is already loaded in memory
        """
        if bForce or not self._mdl:
            traceln("- loading a %s model"%self.cModelClass)
            self._mdl = self.cModelClass(self.sModelName, self.sModelDir)
            if bPickle:
                # loading the model a minima, only to get the feature extractors
                GraphModel.load(self._mdl)
            else:
                self._mdl.load()
            traceln(" done")
        else:
            traceln("- %s model already loaded"%self.cModelClass)
            
        return
    
    def rm(self):
        """
        Remove from the disk any file for this model!!!
        """
        mdl = self.cModelClass(self.sModelName, self.sModelDir)
        
        for s in [  mdl.getModelFilename()
                  , mdl.getTransformerFilename()
                  , mdl.getConfigurationFilename()
                  , mdl.getBaselineFilename()
                  , mdl._getParamsFilename(self.sModelName, self.sModelDir)       ]:
            if os.path.exists(s):
                self.traceln("\t - rm %s"%s) 
                os.unlink(s)
        if os.path.exists(self.sModelDir) and not os.listdir(self.sModelDir):
            self.traceln("\t - rmdir %s"%self.sModelDir) 
            os.rmdir(self.sModelDir)
        return 
    
    def train_save_test(self, lsTrnColDir, lsTstColDir, lsVldColDir, bWarm=False, bPickleXY=False
                        , ratio_train_val=None
                        , blPickleXY=False):
        """
        - Train a model on the tTRN collections, if not empty.
        - Test the trained model using the lTST collections, if not empty.
        - Also train/test any baseline model associated to the main model.
        - Trained models are saved on disk, for testing, redicting or further training (by warm-start)
        - if bWarm==True: warm-start the training from any data stored on disk. Otherwise, a non-empty model folder raises a GraphModelException
        return a test report object
        """
        self.traceln("-"*50)
        self.traceln("Model files of '%s' in folder '%s'"%(self.sModelName, os.path.abspath(self.sModelDir)))
        self.traceln("Training with collection(s):", lsTrnColDir)
        self.traceln("Testing with  collection(s):", lsTstColDir)
        if lsVldColDir: self.traceln("Validating with  collection(s):", lsVldColDir)
        self.traceln("  (File pattern is %s)" % self.sXmlFilenamePattern)
        self.traceln("-"*50)
        
        #list the train and test files
        #NOTE: we check the presence of a digit before the '.' to eclude the *_du.xml files
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, self.sXmlFilenamePattern)
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, self.sXmlFilenamePattern)
        _     , lFilename_vld = self.listMaxTimestampFile(lsVldColDir, self.sXmlFilenamePattern)
        
        self.traceln("- creating a %s model"%self.cModelClass)
        oReport = self._train_save_test(self.sModelName, bWarm, lFilename_trn, ts_trn, lFilename_tst, lFilename_vld, bPickleXY
                                        , ratio_train_val=ratio_train_val
                                        , blPickleXY=blPickleXY)


        return oReport

    def test(self, lsTstColDir):
        """
        test the model
        return a TestReport object
        """
        self.traceln("-"*50)
        self.traceln("Trained model '%s' in folder '%s'"%(self.sModelName, self.sModelDir))
        self.traceln("Testing  collection(s):", lsTstColDir)
        self.traceln("-"*50)
        
        if not self._mdl: raise Exception("The model must be loaded beforehand!")
        
        #list the train and test files
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, self.sXmlFilenamePattern)
        
        DU_GraphClass = self.cGraphClass
        
        lPageConstraint = DU_GraphClass.getPageConstraint()
        if lPageConstraint: 
            for dat in lPageConstraint: self.traceln("\t\t%s"%str(dat))
        
        oReport = self._mdl.testFiles(lFilename_tst, lambda fn: DU_GraphClass.loadGraphs(self.cGraphClass, [fn], bDetach=True, bLabelled=True, iVerbose=1)
                                      , self.getBaselineList() != [])

        return oReport


    def predict(self, lsColDir, docid=None, bGraph=False, bOutXML=True, bPickle=False):
        """
        Return the list of produced files
        """
        if not self._mdl and not bORACLE_EDGE_BREAK_CONTINUE: raise Exception("The model must be loaded beforehand!")

        #list files
        if docid is None:
            self.traceln("-"*50)
            self.traceln("Predicting for collection(s):", lsColDir, "  (%s)" % self.sXmlFilenamePattern)
            self.traceln("-"*50)
            _     , lFilename = self.listMaxTimestampFile(lsColDir, self.sXmlFilenamePattern)
        # predict for this file only
        else:
            try:
                lFilename = [os.path.abspath(os.path.join(lsColDir[0], docid+MultiPageXml.sEXT  ))]
            except IndexError:
                raise Exception("a collection directory must be provided!")


        DU_GraphClass = self.getGraphClass()

        lPageConstraint = DU_GraphClass.getPageConstraint()
        if lPageConstraint:
            for dat in lPageConstraint: self.traceln("\t\t%s"%str(dat))

        chronoOn("predict")
        self.traceln("- loading collection as graphs, and processing each in turn. (%d files)"%len(lFilename))
        lsOutputFilename = []
        for sFilename in lFilename:
            if DU_GraphClass.isOutputFilename(sFilename):
                traceln(" - ignoring '%s' because of its extension" % sFilename) 
                continue 
            
            #try:
            doc, lg = self._predict_file(DU_GraphClass, lPageConstraint, sFilename, bGraph=bGraph
                                         , bPickle=bPickle)
            #except GraphException as e:
            #    doc = None
            #    traceln(str(e))
            #    chronoOff("predict_1") # not nice, I know....
            
            if doc is None:
                self.traceln("\t- no prediction to do for: %s"%sFilename)
            else:
                sCreator = self.sMetadata_Creator + " " + self.getVersion()
                sComment = self.sMetadata_Comments          \
                           if bool(self.sMetadata_Comments) \
                           else "Model: %s  %s  (%s)" % (
                               self.sModelName
                               , self._mdl.__class__.__name__
                               , os.path.abspath(self.sModelDir))
                # which output format
                if bOutXML:
                    if DU_GraphClass == Graph_DOM:
                        traceln(" ignoring export-to-DOM (already DOM output)")
                        pass
                    else:
                        doc = Graph_DOM.exportToDom(lg)
                        sDUFilename = Graph_DOM.saveDoc(sFilename, doc, lg, sCreator, sComment)
                        traceln(" - exported as XML to ", sDUFilename)
                else:
                    sDUFilename = DU_GraphClass.saveDoc(sFilename, doc, lg
                                                    , sCreator=sCreator
                                                    , sComment=sComment)

                del doc
                del lg
                lsOutputFilename.append(sDUFilename)                
        self.traceln(" done [%.2fs]"%chronoOff("predict"))
        return lsOutputFilename

    def _predict_file(self, DU_GraphClass, lPageConstraint, sFilename, bGraph=False
                      , bPickle=False):
        """
        Return the doc (a DOM?, a JSON?, another ?), the list of graphs
            Note: the doc can be None is no graph
        """
        chronoOn("predict_1")
        doc = None
        lg = DU_GraphClass.loadGraphs(self.cGraphClass, [sFilename]
                                      , bDetach=False
                                      , bLabelled=bORACLE_EDGE_BREAK_CONTINUE
                                      , iVerbose=1)

        if bPickle:
            self._pickleRunData(self._mdl, lg, sFilename)
        else:
            #normally, we get one graph per file, but in case we load one graph per page, for instance, we have a list
            for i,g in enumerate(lg):
                if not g.lNode: continue  # no node...
                    
                doc = g.doc
                if lPageConstraint:
                    #self.traceln("\t- prediction with logical constraints: %s"%sFilename)
                    self.traceln("\t- page constraints IGNORED!!")
                self.traceln("%d\t- prediction : %s"%(i,sFilename))
    
                try: self._predict_graph(i, g, lPageConstraint=lPageConstraint, bGraph=bGraph)
                except GraphConjugateException:pass
        self.traceln("\t done [%.2fs]"%chronoOff("predict_1"))
        return doc, lg

    def _predict_graph(self, ig, g, lPageConstraint=None, bGraph=False, bMetric=False):
        """
        predict for a graph
        side effect on the graph g
        return the graph
        """
        try:
            if bMetric: 
                try:    [Ygt] = self._mdl.get_lY([g])
                except:  
                    Ygt  = None
            if bORACLE_EDGE_BREAK_CONTINUE:
                Y = np.zeros((len(g.lEdge), 2), dtype=np.float)
                for i, e in enumerate(g.lEdge):
                    Y[i, 1-int(e.A.cls == e.B.cls)] = 1.0
            else:
                Y = self._mdl.predict(g, bProba=True)
            if self.bConjugate:
                g.setDocLabels(Y,self.fClusterTH)
            else:  
                g.setDocLabels(Y)

            if bMetric: 
                g.computeMetric(ig, g, np.argmax(Y, axis=1) if Y.ndim == 2 else Y, Ygt)
            
            if bGraph: g.addEdgeToDoc()
            del Y
        except GraphModelNoEdgeException:
            traceln("*** ERROR *** cannot predict due to absence of edge in graph")
        return g

    def checkLabelCoverage(self, lY):
        #check that all classes are represented in the dataset
        #it is done again in train but we do that here also because the extractor can take along time, 
        #   and we may discover afterwards it was a useless dataset.
        aLabelCount, _ = np.histogram( np.hstack(lY) , range(self.nbClass+1))
        traceln("   Labels count: ", aLabelCount, " (%d graphs)"%len(lY))
        traceln("   Labels      : ", self.getGraphClass().getLabelNameList())
        if np.min(aLabelCount) == 0:
            sMsg = "*** ERROR *** Label(s) not observed in data."
            #traceln( sMsg+" Label(s): %s"% np.where(aLabelCount[:] == 0)[0] )
            lMissingLabels = [self.getGraphClass().getLabelNameList()[i] for i in np.where(aLabelCount[:] == 0)[0]]
            traceln( sMsg+" Label(s): %s"% lMissingLabels )
            raise ValueError(sMsg)
        return True

    #-----  NFOLD STUFF
    def _nfold_Init(self, lsTrnColDir, n_splits=3, test_size=None, random_state=None, bStoreOnDisk=False):
        """
        initialize a cross-validation
        if bStoreOnDisk is true, store the details of each fold on disk, for paralell execution of each
        return a splitter object, training file timestamp and list 
        """
        self.traceln("-"*50)
        traceln("---------- INITIALIZING CROSS-VALIDATION ----------")
        self.traceln("Model files '%s' in folder '%s'"%(self.sModelName, self.sModelDir))
        #sConfigFile = os.path.join(self.sModelDir, self.sModelName+".py")
        #self.traceln("  Configuration file: %s"%sConfigFile)
        self.traceln("Evaluating with collection(s):", lsTrnColDir)
        self.traceln("-"*50)

        fnCrossValidDetails = os.path.join(self.sModelDir, "fold_def.pkl")
        if os.path.exists(fnCrossValidDetails):
            self.traceln("ERROR: I refuse to overwrite an existing CV setup. Remove manually the CV data! (files %s%s%s_fold* )"%(self.sModelDir, os.sep, self.sModelName))
            exit(1)
        
        #list the train files
        traceln(" - looking for %s files in %s"%(self.sXmlFilenamePattern, lsTrnColDir))
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, self.sXmlFilenamePattern)
        self.traceln("       %d train documents" % len(lFilename_trn))
        
        if test_size is None:
            test_size = 1.0 / n_splits
        splitter = ShuffleSplit(n_splits, test_size, random_state)
        
        if bStoreOnDisk:
            
            GraphModel.gzip_cPickle_dump(fnCrossValidDetails
                                              , (lsTrnColDir, n_splits, test_size, random_state))
            
            for i, (train_index, test_index) in enumerate(splitter.split(lFilename_trn)):
                iFold = i + 1
                traceln("---------- FOLD %d ----------"%iFold)
                lFoldFilename_trn = [lFilename_trn[i] for i in train_index]
                lFoldFilename_tst = [lFilename_trn[i] for i in test_index]
                traceln("--- Train with: %s files"%len(lFoldFilename_trn))
                traceln("--- Test  with: %s files"%len(lFoldFilename_tst))
                assert not( set(lFoldFilename_trn).intersection(lFoldFilename_tst)), set(lFoldFilename_trn).intersection(lFoldFilename_tst)
                
                #fnFoldDetails = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_def.pkl"%iFold)
                fnFoldDetails = os.path.join(self.sModelDir, "fold_%d_def.pkl" % iFold)
                oFoldDetails  = (iFold, ts_trn, lFilename_trn, train_index, test_index)
                GraphModel.gzip_cPickle_dump(fnFoldDetails, oFoldDetails)
                #store the list for TRN and TST in a human readable form
                for name, lFN in [('trn', lFoldFilename_trn), ('tst', lFoldFilename_tst)]:
                    #with open(os.path.join(self.sModelDir, self.sModelName+"_fold_%d_def_%s.txt"%(iFold, name)), "w") as fd:
                    with open(os.path.join(self.sModelDir, "fold_%d_def_%s.txt" % (iFold, name)),
                              "w") as fd:
                        fd.write("\n".join(lFN))
                        fd.write("\n")
                traceln("--- Fold info stored in : %s"%fnFoldDetails)
                
        return splitter, ts_trn, lFilename_trn

    def _nfold_RunFoldFromDisk(self, iFold, bWarm=False, bPickleXY=False):
        """
        Run the fold iFold
        Store results on disk
        """
        fnFoldDetails = os.path.join(self.sModelDir, "fold_%d_def.pkl"%abs(iFold))

        if os.path.exists(fnFoldDetails) is False:
            try:
                import fnmatch
                #Try to take an existing fold definition
                modelsFiles = os.listdir(self.sModelDir)
                found_files = fnmatch.filter(modelsFiles, '*'+"_fold_%d_def.pkl"%abs(iFold))
                if len(found_files)==1:
                    traceln('Found an existing Fold definition:',found_files[0])
                    fnFoldDetails=os.path.join(self.sModelDir,found_files[0])
                else:
                    raise Exception('Could not find a fold definition')
            except ImportError:
                traceln('Could not load Python 3 fnmatch module ')

        traceln("--- Loading fold info from : %s"% fnFoldDetails)
        oFoldDetails = GraphModel.gzip_cPickle_load(fnFoldDetails)
        (iFold_stored, ts_trn, lFilename_trn, train_index, test_index) = oFoldDetails
        assert iFold_stored == abs(iFold), "Internal error. Inconsistent fold details on disk."
        
        if iFold > 0: #normal case
            oReport = self._nfold_RunFold(iFold, ts_trn, lFilename_trn, train_index, test_index, bWarm=bWarm, bPickleXY=bPickleXY)
        else:
            traceln("Switching train and test data for fold %d"%abs(iFold))
            oReport = self._nfold_RunFold(iFold, ts_trn, lFilename_trn, test_index, train_index, bWarm=bWarm, bPickleXY=bPickleXY)
        
        fnFoldResults = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_TestReport.pkl"%iFold)
        GraphModel.gzip_cPickle_dump(fnFoldResults, oReport)
        traceln(" - Done (fold %d)"%iFold)
        
        return oReport

    def _nfold_Finish(self):
        traceln("---------- SHOWING RESULTS OF CROSS-VALIDATION ----------")
        
        fnCrossValidDetails = os.path.join(self.sModelDir, "fold_def.pkl")
        (lsTrnColDir, n_splits, test_size, random_state) = GraphModel.gzip_cPickle_load(fnCrossValidDetails)
        
        loReport = []
        for i in range(n_splits):
            iFold = i + 1
            fnFoldResults = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_TestReport.pkl"%iFold)
            traceln("\t-loading ", fnFoldResults)
            try:
                oReport = GraphModel.gzip_cPickle_load(fnFoldResults)
                
                loReport.append(oReport)
            except:
                traceln("\tWARNING: fold %d has NOT FINISHED or FAILED"%iFold)

        oNFoldReport = TestReportConfusion.newFromReportList(self.sModelName+" (ALL %d FOLDS)"%n_splits, loReport) #a test report based on the confusion matrix

        fnCrossValidDetails = os.path.join(self.sModelDir, self.sModelName+"_folds_STATS.txt")
        with open(fnCrossValidDetails, "a") as fd:
            #BIG banner
            fd.write("\n\n")
            fd.write("#"*80+"\n")
            fd.write("# AGGREGATING FOLDS RESULTS  " + "%s\n"%datetime.datetime.now().isoformat())
            fd.write("#"*80+"\n\n")
            
            for oReport in loReport: 
                fd.write(str(oReport))
                fd.write("%s\n"%(" _"*30))

            fd.write(str(oNFoldReport))
            
        return oNFoldReport

    def _nfold_RunFold(self, iFold, ts_trn, lFilename_trn, train_index, test_index, bWarm=False, bPickleXY=False):
        """
        Run this fold
        Return a TestReport object
        """
        traceln("---------- RUNNING FOLD %d ----------"%iFold)
        lFoldFilename_trn = [lFilename_trn[i] for i in train_index]
        lFoldFilename_tst = [lFilename_trn[i] for i in test_index]
        traceln("--- Train with: %s"%lFoldFilename_trn)
        traceln("--- Test  with: %s"%lFoldFilename_tst)
        
        self.traceln("- creating a %s model"%self.cModelClass)
        sFoldModelName = self.sModelName+"_fold_%d"%iFold
        
        oReport = self._train_save_test(sFoldModelName, bWarm, lFoldFilename_trn, ts_trn, lFoldFilename_tst, [], bPickleXY)

        fnFoldReport = os.path.join(self.sModelDir, self.sModelName+"_fold_%d_STATS.txt"%iFold)
        with open(fnFoldReport, "w") as fd:
            fd.write(str(oReport))
        
        return oReport
    
    def nfold_Eval(self, lsTrnColDir, n_splits=3, test_size=None, random_state=None, bPickleXY=False):
        """
        n-fold evaluation on the training data
        
        - list all files
        - generate a user defined number of independent train / test dataset splits. Samples are first shuffled and then split into a pair of train and test sets
        - for each split: 
            - train a CRF and all baseline model
            - test and make a TestReport
            - save the model
        - return a list of TestReports
        """
        
        splitter, ts_trn, lFilename_trn = self._nfold_Init(lsTrnColDir, n_splits, test_size, random_state)
        
        loTstRpt = []
        
        for i, (train_index, test_index) in enumerate(splitter.split(lFilename_trn)):
            oReport = self._nfold_RunFold(i+1, ts_trn, lFilename_trn, train_index, test_index, bPickleXY=False)
            traceln(oReport)
            loTstRpt.append(oReport)
        
        return loTstRpt

    #----------------------------------------------------------------------------------------------------------    
    def _pickleData(self, mdl, lGraph, name
                    , blPickleXY=False):
        """
        Pickling train or valid or test data
        if blPickleXY is True, we proceed graph by graph (to reduce memory footprint!)
        """
        #for GCN
        bGCN_revert = False
        if bGCN_revert:
            for g in lGraph: g.revertEdges()
        if blPickleXY:
            sDir = mdl.getTrainDataDirname(name) + ("_tXrY" if bGCN_revert else "_tXY")
            if not os.path.isdir(sDir): os.mkdir(sDir)
            # data
            nGraph, nf = 0, None
            for n, g in enumerate(lGraph):
                try:
                    [X], [Y] = mdl.get_lX_lY([g])
                except:
                    traceln("-----  SKIPPED graph index = %d   get_lX_lY exception -------------------------" % n)
                    traceln(traceback.format_exc())
                    continue
                if nf is None: nf, ef = X[0].shape[1], X[2].shape[1] 
                mdl.gzip_cPickle_dump( os.path.join(sDir, "%06d.pkl" % (n+1))
                                        , (X, Y))
                del X, Y
                if (n % 10) == 9: gc.collect()
                nGraph += 1

            if lGraph:
                # model characteristics
                mdl.gzip_cPickle_dump(os.path.join(sDir, "model_def.pkl")
                                      , (  mdl.getNbClass()
                                         , nf    # node #feat
                                         , ef    # edge #feat
                                         , nGraph
                                         , self.config_extractor_kwargs
                                         ))
                traceln("%d classes  |  %d node features  | %d edge features  |  %d graphs" % (  mdl.getNbClass()
                                                                                               , nf, ef
                                                                                               , nGraph))
            if nGraph < len(lGraph): traceln("SKIPPED %d graphs" % (len(lGraph)-nGraph))
            traceln(" --> gzipped pickle files in %s  (%d tXY files)" % ( os.path.abspath(sDir)
                                                                        , nGraph))
        else:
            lX, lY = mdl.get_lX_lY(lGraph)
            sFilename = mdl.getTrainDataFilename(name)
            if bGCN_revert:
                sFilename = sFilename.replace("_tlXlY_", "_tlXrlY_")
            mdl.gzip_cPickle_dump(sFilename, (lX, lY))
            traceln(" --> gzipped pickle file: ", os.path.abspath(sFilename))
        
        if self._additional_data_fun:
            # application specific data to be pickled
            lData = (self._additional_data_fun)(self, mdl, lGraph)
            sFilename = mdl.getAdditionalTrainDataFilename(name)
            mdl.gzip_cPickle_dump(sFilename, lData)
            traceln(" -->            see also: ", os.path.abspath(sFilename))
        return

    def _pickleRunData(self, mdl, lGraph, sInputFilename):
        """
        Pickling run data  (so only the X)
        """
        #for GCN
        bGCN_revert = False
        if bGCN_revert:
            for g in lGraph: g.revertEdges()
        lX = mdl.get_lX(lGraph)
        sFilename = mdl.getRunDataFilename(sInputFilename)
        if bGCN_revert:
            sFilename = sFilename.replace("_tlX_", "_tlXr_")
        mdl.gzip_cPickle_dump(sFilename, lX)
        traceln(" --> gzipped pickle file: ", os.path.abspath(sFilename))
        
        if self._additional_data_fun:
            # application specific data to be pickled
            lData = (self._additional_data_fun)(self, mdl, lGraph)
            sFilename = mdl.getAdditionalRunDataFilename(sInputFilename, "node")
            mdl.gzip_cPickle_dump(sFilename, lData)
            traceln(" -->            see also: ", os.path.abspath(sFilename))
        return
    
    @classmethod
    def getPickledDef(cls, sDir):
        """
        return nbClass, nbNodeFeat, nbEdgeFeat, nbGraph
        """
        return gzip_pickle_load(os.path.join(sDir, "model_def.pkl"))

    @classmethod
    def getPickledXY(cls, sFilepath, blPickleXY=False):
        """
        return an iterator over the XYs
        """
        if blPickleXY:
            traceln(" ...loading gzipped pickle files from %s" % os.path.abspath(sFilepath))
            # arbitrary order!
            return (gzip_pickle_load(fn) for fn in glob.iglob(os.path.join(sFilepath, "[0-9]*[0-9].pkl")))           
        else:
            traceln(" ...loading from zipped pickle file: ", os.path.abspath(sFilepath))
            (lX, lY) = gzip_pickle_load(sFilepath)
            return zip(lX, lY)
        
    def setAdditionalDataProvider(self, fun):
        """
        register a callback for generating specific additional data to be pickled
        the fun must have this prototype:
            fun(doer, mdl, lGraph) -> [data, ...]
            
            it returns one data per graph... which is pickled in a .data gzipped file
        """
        self._additional_data_fun = fun
        traceln("SETUP: Registering an additional data provider for pickling: ", fun)
        
    #----------------------------------------------------------------------------------------------------------    
    def _train_save_test(self, sModelName, bWarm, lFilename_trn, ts_trn, lFilename_tst, lFilename_vld, bPickleXY
                         , ratio_train_val=None
                         , blPickleXY=False):
        """
        used both by train_save_test and _nfold_runFold
        if provided, try using lFilename_vld as validation set to make best model.
        """
        mdl = self.cModelClass(sModelName, self.sModelDir)
        if blPickleXY:
            # let's do it immediately, to spot any use error ASAP
            self.traceln("- retrieving feature extractors...")
            mdl.loadTransformers() # note: we do not check if fresh, because some data can be fresher (e.g. TWY experiment in MENUs)
        else:
            if os.path.exists(mdl.getModelFilename()) and not bWarm: 
                raise GraphModelException("Model exists on disk already (%s), either remove it first or warm-start the training."%mdl.getModelFilename())
            
        mdl.configureLearner(**self.config_learner_kwargs)
        mdl.setBaselineModelList(self._lBaselineModel)
        if not blPickleXY: mdl.saveConfiguration( (self.config_extractor_kwargs, self.config_learner_kwargs) )
        self.traceln("\t - configuration: ", self.config_learner_kwargs )

        self.traceln("- loading training graphs")
        lGraph_trn = self.cGraphClass.loadGraphs(self.cGraphClass, lFilename_trn, bDetach=True, bLabelled=True, iVerbose=1)
        self.traceln(" --- %d training graphs loaded"%len(lGraph_trn))
        if len(lGraph_trn) == 0 and not (bPickleXY or blPickleXY): 
            raise ValueError("No training graph loaded!")

        if lFilename_vld:
            self.traceln("- loading validation graphs")
            lGraph_vld = self.cGraphClass.loadGraphs(self.cGraphClass, lFilename_vld, bDetach=True, bLabelled=True, iVerbose=1)
            self.traceln(" --- %d validation graphs loaded"%len(lGraph_vld))
        else:
            lGraph_vld = []
            if not (bPickleXY or blPickleXY):
                if ratio_train_val is None:
                    self.traceln("- no validation graphs")
                else:
                    lG = [g for g in lGraph_trn]
                    split_idx = int(ratio_train_val * len(lG))
                    lGraph_vld = lG[:split_idx ]
                    lGraph_trn = lG[ split_idx:]
                    del lG
                    self.traceln("- extracted %d validation graphs, got %d training graphs (ratio=%.3f)" 
                                 % (len(lGraph_vld), len(lGraph_trn), ratio_train_val))

        #for this check, we load the Y once...
        if self.bConjugate:
            mdl.setNbClass(len(self.cGraphClass.getEdgeLabelNameList()))
            for _g in lGraph_trn: _g.computeEdgeLabels()
            for _g in lGraph_vld: _g.computeEdgeLabels()
        else:
            assert self.nbClass and self.lNbClass, "internal error: I expected the number of class to be automatically computed at that stage"
            if self.iNbNodeType == 1:
                mdl.setNbClass(self.nbClass)
            else:
                mdl.setNbClass(self.lNbClass)
            
            if (bPickleXY or blPickleXY) and not(lGraph_trn):
                # probably only pickling a tst folder...
                pass
            else:
                self.checkLabelCoverage(mdl.get_lY(lGraph_trn)) #NOTE that Y are in bad order if multiptypes. Not a pb here
            
        if not blPickleXY:
            chronoOn("FeatExtract")
            self.traceln("- retrieving or creating feature extractors...")
            try:
                mdl.loadTransformers(ts_trn)
            except GraphModelException:
                try:
                    fe = self.cFeatureDefinition(**self.config_extractor_kwargs)
                except Exception as e:
                    traceln("ERROR: could not instantiate feature definition class: ", str(self.cFeatureDefinition))
                    raise e        
                fe.fitTranformers(lGraph_trn)
                fe.cleanTransformers()
                mdl.setTranformers(fe.getTransformers())
                mdl.saveTransformers()
            self.traceln(" done [%.1fs]"%chronoOff("FeatExtract"))
        
        # pretty print of features extractors
        sPrettyFeat = """--- Features ---
--- NODES : %s

--- EDGES : %s
--- -------- ---
""" % mdl.getTransformers()
        self.traceln(sPrettyFeat)
        Tracking.log_artifact_string("Features", sPrettyFeat)
        
        if bPickleXY or blPickleXY:
            if lGraph_trn: self._pickleData(mdl, lGraph_trn, "trn", blPickleXY=blPickleXY)
            if lGraph_vld: self._pickleData(mdl, lGraph_vld, "vld", blPickleXY=blPickleXY)
        else:
            self.traceln("- training model...")
            chronoOn("MdlTrn")
            mdl.train(lGraph_trn, lGraph_vld, True, ts_trn, verbose=1 if self.bVerbose else 0)
            mdl.save()
            tTrn = chronoOff("MdlTrn")
            self.traceln(" training done [%.1f s]  (%s)" % (tTrn, pretty_time_delta(tTrn)))
        
            # OK!!
            self._mdl = mdl
        
        if lFilename_tst:
            self.traceln("- loading test graphs")
            lGraph_tst = self.cGraphClass.loadGraphs(self.cGraphClass, lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
            if self.bConjugate:
                for _g in lGraph_tst: _g.computeEdgeLabels()
            self.traceln(" --- %d graphs loaded"%len(lGraph_tst))
            if bPickleXY or blPickleXY:
                self._pickleData(mdl, lGraph_tst, "tst", blPickleXY=blPickleXY)
                oReport = None
            else:
                oReport = mdl.test(lGraph_tst) # bug due to MPXML , lsDocName=lFilename_tst)
        else:
            oReport = None

        if bPickleXY or blPickleXY:
            self.traceln("- pickle done, exiting")
            exit(0)
        
        return oReport
    
    #----------------------------------------------------------------------------------------------------------    
    def listMaxTimestampFile(cls, lsDir, sPattern, bIgnoreDUFiles=True):
        """
        List the file following the given pattern in the given folders
        return the timestamp of the most recent one
        Return a timestamp and a list of filename
        """
        lFn, ts = [], None 
        for sDir in lsDir:
            lsFilename = sorted(glob.iglob(os.path.join(sDir, sPattern)))  
            if bIgnoreDUFiles:
                lsFilename = [s for s in lsFilename if not(os.path.splitext(s)[0].endswith("_du"))]
            lFn.extend([s.replace("\\", "/") for s in lsFilename]) #Unix-style is universal
            if lsFilename:
                ts_max =  max([os.path.getmtime(sFilename) for sFilename in lsFilename])
                ts = ts_max if ts is None else max(ts, ts_max)
                ts = max(ts, ts_max)
        return ts, lFn
    listMaxTimestampFile = classmethod(listMaxTimestampFile)


# -----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    usage, parser = DU_Task.getStandardOptionsParser(sys.argv[0])

    parser.print_help()
    
    traceln("\nThis module should not be run as command line. It does nothing. (And did nothing!)")
