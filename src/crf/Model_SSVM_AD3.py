# -*- coding: utf-8 -*-

"""
    Train, test, predict steps for a CRF model
    - CRF model is EdgeFeatureGraphCRF  (unary and pairwise potentials)
    - Train using SSM
    - Predict using AD3

    Copyright Xerox(C) 2016 JL. Meunier

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
import gc

from pystruct.utils import SaveLogger
from pystruct.learners import OneSlackSSVM
from pystruct.models import EdgeFeatureGraphCRF

from common.trace import traceln
from common.chrono import chronoOn, chronoOff
from Model import Model
from Graph import Graph
from TestReport import TestReport

class Model_SSVM_AD3(Model):
    #default values for the solver
    C                = .1 
    njobs            = 4
    inference_cache  = 50
    tol              = .1
    save_every       = 50     #save every 50 iterations,for warm start
    max_iter         = 1000
    
    def __init__(self, sName, sModelDir):
        """
        a CRF model, that uses SSVM and AD3, with a name and a folder where it will be stored or retrieved from
        """
        Model.__init__(self, sName, sModelDir)
        self.ssvm = None
        
    def configureLearner(self, inference_cache=None, C=None, tol=None, njobs=None, save_every=None, max_iter=None):
        if None != inference_cache  : self.inference_cache   = inference_cache
        if None != C                : self.C                 = C
        if None != tol              : self.tol               = tol
        if None != njobs            : self.njobs             = njobs
        if None != save_every       : self.save_every        = save_every
        if None != max_iter         : self.max_iter          = max_iter

    def load(self, expiration_timestamp=None):
        """
        Load myself from disk
        If an expiration timestamp is given, the model stored on disk must be fresher than timestamp
        return self or raise a ModelException
        """
        Model.load(self, expiration_timestamp)
        self.ssvm = self._loadIfFresh(self.getModelFilename(), expiration_timestamp, lambda x: SaveLogger(x).load())
        return self
    
    # --- TRAIN / TEST / PREDICT ------------------------------------------------
    def train(self, lGraph, bWarmStart=True, expiration_timestamp=None, verbose=0):
        """
        Train a CRF model using the list of labelled graph as training
        if bWarmStart if True, try to continue from previous training, IF the stored model is older than expiration_timestamp!!
            , otherwise, starts from scratch
        return nothing
        """
        traceln("\t- computing features on training set")
        traceln("\t\t #nodes=%d  #edges=%d "%Graph.getNodeEdgeTotalNumber(lGraph))
        chronoOn()
        lX, lY = self.transformGraphs(lGraph, True)
        traceln("\t\t #features nodes=%d  edges=%d "%(lX[0][0].shape[1], lX[0][2].shape[1]))
        traceln("\t [%.1fs] done\n"%chronoOff())

        traceln("\t- retrieving or creating model...")
        self.ssvm = None
        sModelFN = self.getModelFilename()
        if bWarmStart:
            try:
                self.ssvm = self._loadIfFresh(sModelFN, expiration_timestamp, lambda x: SaveLogger(x).load())
                traceln("\t- warmstarting!")
            except Exception as e:
                self.ssvm = None
                traceln("\t- Cannot warmstart: %s"%e.message)
            #self.ssvm is either None or containing a nice ssvm model!!
        
        if not self.ssvm:
            traceln("\t- creating a new SSVM-trained CRF model")
            
            traceln("\t\t- computing class weight:")
            clsWeights = self.computeClassWeight(lY)
            traceln("\t\t\t%s"%clsWeights)
            
            crf = EdgeFeatureGraphCRF(inference_method='ad3', class_weight=clsWeights)
    
            self.ssvm = OneSlackSSVM(crf
                                , inference_cache=self.inference_cache, C=self.C, tol=self.tol, n_jobs=self.njobs
                                , logger=SaveLogger(sModelFN, save_every=self.save_every)
                                , max_iter=self.max_iter                                        
                                , show_loss_every=10, verbose=verbose)
            bWarmStart = False
        chronoOn()
        traceln("\t - training graph-based model")
        traceln("\t\t solver parameters:"
                    , " inference_cache=",self.inference_cache
                    , " C=",self.C, " tol=",self.tol, " n_jobs=",self.njobs)
        self.ssvm.fit(lX, lY, warm_start=bWarmStart)
        traceln("\t [%.1fs] done (graph-based model is trained) \n"%chronoOff())
        
        traceln(self.getModelInfo())
        
        #cleaning useless data that takes MB on disk
        if not bWarmStart:
            self.ssvm.alphas = None  
            self.ssvm.constraints_ = None
            self.ssvm.inference_cache_ = None    
            traceln("\t\t(model made slimmer. Not sure you can efficiently warm-start it later on. See option -w.)")        
        #the baseline model(s) if any
        self._trainBaselines(lX, lY)
        
        #do some garbage collection
        del lX, lY
        gc.collect()
        return 

    def _ssvm_ad3plus_predict(self, lX, lConstraints):
        """
        Since onlt ad3+ is able to deal with constraints, we use it!
        but training must have been done with ad3 or ad3+
        """
        assert self.ssvm.model.inference_method in ['ad3', 'ad3+'], "AD3+ is the only inference method supporting those constraints. Training with ad3 or ad3+ is required"
        
        #we use ad3+ for this particular inference
        _inf = self.ssvm.model.inference_method
        self.ssvm.model.inference_method = "ad3+"
        lY = self.ssvm.predict(lX, constraints=lConstraints)
        self.ssvm.model.inference_method = _inf            
        return lY
    
    #no need to define def save(self):
    #because the SSVM is saved while being trained, and the attached baeline models are saved by the parent class
                    
    def test(self, lGraph, lsDocName=None):
        """
        Test the model using those graphs and report results on stderr
        if some baseline model(s) were set, they are also tested
        Return a Report object
        """
        assert lGraph
        lLabelName   = lGraph[0].getLabelNameList()
        bConstraint  = lGraph[0].getPageConstraint()
        
        traceln("\t- computing features on test set")
        chronoOn()
        lX, lY = self.transformGraphs(lGraph, True)
        traceln("\t\t #features nodes=%d  edges=%d "%(lX[0][0].shape[1], lX[0][2].shape[1]))
        traceln("\t\t #nodes=%d  #edges=%d "%Graph.getNodeEdgeTotalNumber(lGraph))
        traceln("\t [%.1fs] done\n"%chronoOff())
        
        traceln("\t- predicting on test set")
        chronoOn()
        if bConstraint:
            lConstraints = [g.instanciatePageConstraints() for g in lGraph]
            lY_pred = self._ssvm_ad3plus_predict(lX, lConstraints)
        else:
            lY_pred = self.ssvm.predict(lX)
        traceln("\t [%.1fs] done\n"%chronoOff())
        
        tstRpt = TestReport(self.sName, lY_pred, lY, lLabelName, lsDocName=lsDocName)
        
        lBaselineTestReport = self._testBaselines(lX, lY, lLabelName, lsDocName=lsDocName)
        tstRpt.attach(lBaselineTestReport)
        
        #do some garbage collection
        del lX, lY
        gc.collect()
        
        return tstRpt

    def testFiles(self, lsFilename, loadFun):
        """
        Test the model using those files. The corresponding graphs are loaded using the loadFun function (which must return a singleton list).
        It reports results on stderr
        
        if some baseline model(s) were set, they are also tested
        
        Return a Report object
        """
        lX, lY, lY_pred  = [], [], []
        lLabelName   = None
        bConstraint  = None
        traceln("\t- predicting on test set")
        
        for sFilename in lsFilename:
            [g] = loadFun(sFilename) #returns a singleton list
            [X], [Y] = self.transformGraphs([g], True)

            if lLabelName == None:
                lLabelName = g.getLabelNameList()
                traceln("\t\t #features nodes=%d  edges=%d "%(X[0].shape[1], X[2].shape[1]))
            else:
                assert lLabelName == g.getLabelNameList(), "Inconsistency among label spaces"
            n_jobs = self.ssvm.n_jobs
            self.ssvm.n_jobs = 1
            if g.getPageConstraint():
                lConstraints = g.instanciatePageConstraints()
                [Y_pred] = self._ssvm_ad3plus_predict([X], [lConstraints])
            else:
                #since we pass a single graph, let force n_jobs to 1 !!
                [Y_pred] = self.ssvm.predict([X])
            self.ssvm.n_jobs = n_jobs

            lX     .append(X)
            lY     .append(Y)
            lY_pred.append(Y_pred)
            g.detachFromDOM()
            del g   #this can be very large
            gc.collect() 
        traceln("\t done")

        tstRpt = TestReport(self.sName, lY_pred, lY, lLabelName, lsDocName=lsFilename)
        
        lBaselineTestReport = self._testBaselinesEco(lX, lY, lLabelName, lsDocName=lsFilename)
        tstRpt.attach(lBaselineTestReport)
        
#         if True:
#             #experimental code, not so interesting...
#             node_transformer, _ = self.getTransformers()
#             try:
#                 _testable_extractor_ = node_transformer._testable_extractor_
#                 lExtractorTestReport = _testable_extractor_.testEco(lX, lY)
#                 tstRpt.attach(lExtractorTestReport)
#             except AttributeError:
#                 pass
        
        #do some garbage collection
        del lX, lY
        gc.collect()
        
        return tstRpt

    def predict(self, graph):
        """
        predict the class of each node of the graph
        return a numpy array, which is a 1-dim array of size the number of nodes of the graph. 
        """
        [X] = self.transformGraphs([graph])
        bConstraint  = graph.getPageConstraint()
        
        traceln("\t\t #features nodes=%d  edges=%d "%(X[0].shape[1], X[2].shape[1]))
        n_jobs = self.ssvm.n_jobs
        self.ssvm.n_jobs = 1
        if bConstraint:
            [Y] = self._ssvm_ad3plus_predict([X], [graph.instanciatePageConstraints()])
        else:
            [Y] = self.ssvm.predict([X])
        self.ssvm.n_jobs = n_jobs
            
        return Y

    def getModelInfo(self):
        """
        Get some basic model info
        Return a textual report
        """
        s =  "_crf_ Model: %s\n" % self.ssvm
        s += "_crf_ Number of iterations: %s\n" % len(self.ssvm.objective_curve_)
        if len(self.ssvm.objective_curve_) != len(self.ssvm.primal_objective_curve_):
            s += "_crf_ WARNNG: unextected data, result below might be wrong!!!!\n"
        last_objective, last_primal_objective  = self.ssvm.objective_curve_[-1], self.ssvm.primal_objective_curve_[-1]
        s += "_crf_ final primal objective: %f gap: %f\n" % (last_primal_objective, last_primal_objective - last_objective)
    
        return s
    
# --- MAIN: DISPLAY STORED MODEL INFO ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    try:
        sModelDir, sModelName = sys.argv[1:3]
    except:
        print "Usage: %s <model-dir> <model-name>"
        print "Display some info regarding the stored model"
        
    mdl = Model_SSVM_AD3(sModelName, sModelDir)
    print "Loading %s"%mdl.getModelFilename()
    if False:
        mdl.load()  #loads all sub-models!!
    else:
        mdl.ssvm = mdl._loadIfFresh(mdl.getModelFilename(), None, lambda x: SaveLogger(x).load())

    print mdl.getModelInfo()
    
    import matplotlib.pyplot as plt
    plt.plot(mdl.ssvm.loss_curve_)
    plt.ylabel("Loss")
    plt.show()

    
    
