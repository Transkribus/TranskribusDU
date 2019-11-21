# -*- coding: utf-8 -*-

"""
    DU task for BAR - see https://read02.uibk.ac.at/wiki/index.php/Document_Understanding_BAR
    
    Copyright Xerox(C) 2017 JL Meunier


    
    
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
from tasks import _checkFindColDir, _exit

from tasks.DU_CRF_Task import DU_CRF_Task


def main(DU_BAR):
    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)
    parser.add_option("--docid", dest='docid',  action="store",default=None,  help="only process docid")    
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    
    # --- 
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        _exit(usage, 1, e)
        
    doer = DU_BAR(sModelName, sModelDir,
                      C                 = options.crf_C,
                      tol               = options.crf_tol,
                      njobs             = options.crf_njobs,
                      max_iter          = options.max_iter,
                      inference_cache   = options.crf_inference_cache)
    
    
    if options.docid:
        sDocId=options.docid
    else:
        sDocId=None
    if options.rm:
        doer.rm()
        sys.exit(0)

    lTrn, lTst, lRun, lFold = [_checkFindColDir(lsDir) for lsDir in [options.lTrn, options.lTst, options.lRun, options.lFold]] 
#     if options.bAnnotate:
#         doer.annotateDocument(lTrn)
#         traceln('annotation done')    
#         sys.exit(0)
    
    ## use. a_mpxml files
    doer.sXmlFilenamePattern = doer.sLabeledXmlFilenamePattern


    if options.iFoldInitNum or options.iFoldRunNum or options.bFoldFinish:
        if options.iFoldInitNum:
            """
            initialization of a cross-validation
            """
            splitter, ts_trn, lFilename_trn = doer._nfold_Init(lFold, options.iFoldInitNum, bStoreOnDisk=True)
        elif options.iFoldRunNum:
            """
            Run one fold
            """
            oReport = doer._nfold_RunFoldFromDisk(options.iFoldRunNum, options.warm)
            traceln(oReport)
        elif options.bFoldFinish:
            tstReport = doer._nfold_Finish()
            traceln(tstReport)
        else:
            assert False, "Internal error"    
        #no more processing!!
        exit(0)
        #-------------------
        
    if lFold:
        loTstRpt = doer.nfold_Eval(lFold, 3, .25, None, options.pkl)
        import graph.GraphModel
        sReportPickleFilename = os.path.join(sModelDir, sModelName + "__report.txt")
        traceln("Results are in %s"%sReportPickleFilename)
        graph.GraphModel.GraphModel.gzip_cPickle_dump(sReportPickleFilename, loTstRpt)
    elif lTrn:
        doer.train_save_test(lTrn, lTst, options.warm, options.pkl)
        try:    traceln("Baseline best estimator: %s"%doer.bsln_mdl.best_params_)   #for GridSearch
        except: pass
        traceln(" --- CRF Model ---")
        traceln(doer.getModel().getModelInfo())
    elif lTst:
        doer.load()
        tstReport = doer.test(lTst)
        traceln(tstReport)
    
    if lRun:
        if options.storeX or options.applyY:
            try: doer.load() 
            except: pass    #we only need the transformer
            lsOutputFilename = doer.runForExternalMLMethod(lRun, options.storeX, options.applyY)
        else:
            doer.load()
            lsOutputFilename = doer.predict(lRun)
        traceln("Done, see in:\n  %s"%lsOutputFilename)

if __name__ == "__main__":
    raise Exception("This is an abstract module.")