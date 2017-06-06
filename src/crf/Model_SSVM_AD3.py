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
import sys, os, types
import gc
import cPickle
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV  #0.18.1 REQUIRES NUMPY 1.12.1 or more recent
    
from pystruct.utils import SaveLogger
from pystruct.learners import OneSlackSSVM
from pystruct.models import EdgeFeatureGraphCRF

try: #to ease the use without proper Python installation
    from common.trace import traceln
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
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

    #Config when training a baseline predictor on edges
    dEdgeGridSearch_LR_conf   = {'C':[0.01, 0.1, 1.0, 10.0] }  #Grid search parameters for LR edge baseline method training
    dEdgeGridSearch_LR_n_jobs = 1                              #Grid search: number of jobs
    
    def __init__(self, sName, sModelDir):
        """
        a CRF model, that uses SSVM and AD3, with a name and a folder where it will be stored or retrieved from
        """
        Model.__init__(self, sName, sModelDir)
        self.ssvm = None
        self.bGridSearch = False
        self._EdgeBaselineModel = None
        
    def configureLearner(self, inference_cache=None, C=None, tol=None, njobs=None, save_every=None, max_iter=None):
        #NOTE: we might get a list in C tol max_iter inference_cache  (in case of gridsearch)
        if None != inference_cache  : self.inference_cache   = inference_cache
        if None != C                : self.C                 = C
        if None != tol              : self.tol               = tol
        if None != njobs            : self.njobs             = njobs
        if None != save_every       : self.save_every        = save_every
        if None != max_iter         : self.max_iter          = max_iter
        
        self.bGridSearch = types.ListType in [type(v) for v in [self.inference_cache, self.C, self.tol, self.max_iter]]

    # --- UTILITIES -------------------------------------------------------------
    def getEdgeBaselineFilename(self):
        return os.path.join(self.sDir, self.sName+"_edge_baselines.pkl")

    def _getNbFeatureAsText(self):
        """
        return the number of node features and the number of edge features as a textual message
        """
        return "#features nodes=%d  edges=%d "%self._tNF_EF
        
    # --- EDGE BASELINE -------------------------------------------------------------
    def getEdgeModel(self):
        """
        Logisitic regression model for edges
        """
        return GridSearchCV( LogisticRegression(class_weight='balanced') 
                                                , self.dEdgeGridSearch_LR_conf
                                                , n_jobs=self.dEdgeGridSearch_LR_n_jobs) #1
        
    def _getEdgeXEdgeY(self, lX, lY):
        """
        return X,Y for each edge
        The edge label is in [0, ntype^2-1]
        """
        X_flat = np.vstack( edge_features for (_, edge_features, _) in lX )
        
        l_Y_flat = []
        for X_graph, Y_node in zip(lX, lY):
            _, edges, _ = X_graph
            l_Y_flat.append( Y_node[edges[:,0]]*self._nbClass + Y_node[edges[:,1]] )
        Y_flat = np.hstack(l_Y_flat)
        
        return X_flat, Y_flat
    
    def _trainEdgeBaseline(self, lX, lY):
        """
        Here we train a logistic regression model to predict the pair of labels of each edge.
        This code assume single type
        """
        X_flat, Y_flat = self._getEdgeXEdgeY(lX, lY)
        
        with open("edgeXedgeY_flat.pkl", "wb") as fd: cPickle.dump((X_flat, Y_flat), fd)
        
        self._EdgeBaselineModel = self.getEdgeModel()
        chronoOn()
        traceln("\t - training edge baseline model: %s"%str(self._EdgeBaselineModel))
        self._EdgeBaselineModel.fit(X_flat, Y_flat)
        traceln("\t [%.1fs] done\n"%chronoOff())
        del X_flat, Y_flat
        return True

    def _testEdgeBaselines(self, lX, lY, lLabelName=None, lsDocName=None):
        """
        test the edge baseline model, 
        return a test report list (a singleton for now)
        """
        lTstRpt = []
        if self._EdgeBaselineModel:
            if lsDocName: assert len(lX) == len(lsDocName), "Internal error"
            
            lEdgeLabelName = [ "%s_%s"%(lbl1, lbl2) for lbl1 in lLabelName for lbl2 in lLabelName ] if lLabelName else None
            lTstRpt = []
            X_flat, Y_flat = self._getEdgeXEdgeY(lX, lY)
            chronoOn("_testEdgeBaselines")
            Y_pred_flat = self._EdgeBaselineModel.predict(X_flat)
            traceln("\t\t [%.1fs] done\n"%chronoOff("_testEdgeBaselines"))
            lTstRpt.append( TestReport(str(self._EdgeBaselineModel), Y_pred_flat, Y_flat, lEdgeLabelName, lsDocName=lsDocName) )
                
            del X_flat, Y_flat, Y_pred_flat
        return lTstRpt                                                                              
    
    # --- TRAIN / TEST / PREDICT ------------------------------------------------
    def _computeModelCaracteristics(self, lX):
        """
        We discover dynamically the number of features. Pretty convenient for developer.
        Drawback: if the feature extractor code changes, predicting with a stored model will crash without beforehand catch
        """
        self._tNF_EF = (lX[0][0].shape[1], lX[0][2].shape[1]) #number of node features,  number of edge features
        return self._tNF_EF
        
    def _getCRFModel(self, clsWeights):
        if self._nbClass: #should always be the case, when used from DU_CRF_Task
            #if some class is not represented, we still train and do not crash
            crf = EdgeFeatureGraphCRF(inference_method='ad3', class_weight=clsWeights, n_states=self._nbClass)
        else:
            crf = EdgeFeatureGraphCRF(inference_method='ad3', class_weight=clsWeights)       
        return crf 

        
    def train(self, lGraph, bWarmStart=True, expiration_timestamp=None, verbose=0):
        """
        Train a CRF model using the list of labelled graph as training
        if bWarmStart if True, try to continue from previous training, IF the stored model is older than expiration_timestamp!!
            , otherwise, starts from scratch
        return nothing
        """
        if self.bGridSearch:
            return self.gridsearch(lGraph, verbose=verbose)
    
        traceln("\t- computing features on training set")
        traceln("\t\t #nodes=%d  #edges=%d "%Graph.getNodeEdgeTotalNumber(lGraph))
        chronoOn()
        lX, lY = self.get_lX_lY(lGraph)
        self._computeModelCaracteristics(lX)    #we discover here dynamically the number of features of nodes and edges
        traceln("\t\t %s"%self._getNbFeatureAsText())
        traceln("\t [%.1fs] done\n"%chronoOff())
        
        traceln("\t- retrieving or creating model...")
        self.ssvm = None
        sModelFN = self.getModelFilename()
        if bWarmStart:
            try:
                self.ssvm = self._loadIfFresh(sModelFN, expiration_timestamp, lambda x: SaveLogger(x).load())
                traceln("\t- warmstarting!")
                #we allow to change the max_iter of the model
                try:
                    self.ssvm.max_iter #to make sure we do something that makes sense...
                    if self.ssvm.max_iter != self.max_iter:
                        traceln("\t- changing max_iter value from (stored) %d to %d"%(self.ssvm.max_iter, self.max_iter))
                        self.ssvm.max_iter = self.max_iter
                except AttributeError:
                    traceln("\t- cannot access or change the max_iter value")
                    
                try:
                    self.ssvm.n_jobs #to make sure we do something that makes sense...
                    if self.ssvm.n_jobs != self.njobs:
                        traceln("\t- changing n_jobs value from (stored) %d to %d"%(self.ssvm.n_jobs, self.njobs))
                        self.ssvm.n_jobs = self.njobs
                except AttributeError:
                    traceln("\t- cannot access or change the n_jobs value")

            except Exception as e:
                self.ssvm = None
                traceln("\t- Cannot warmstart: %s"%e.message)
            #self.ssvm is either None or containing a nice ssvm model!!
        
        if not self.ssvm:
            traceln("\t- creating a new SSVM-trained CRF model")
            
            traceln("\t\t- computing class weight:")
            clsWeights = self.computeClassWeight(lY)
            traceln("\t\t\t%s"%clsWeights)
            
            crf = self._getCRFModel(clsWeights)
    
            self.ssvm = OneSlackSSVM(crf
                                , inference_cache=self.inference_cache, C=self.C, tol=self.tol, n_jobs=self.njobs
                                , logger=SaveLogger(sModelFN, save_every=self.save_every)
                                , max_iter=self.max_iter                                        
                                , show_loss_every=10, verbose=verbose)
            bWarmStart = False

        chronoOn()
        traceln("\t- training graph-based model")
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

    def gridsearch(self, lGraph, verbose=0):
        """
        do a grid search instead of a normal training
        """
        traceln("--- GRiD SEARCH FOR CRF MODEL ---")
        traceln("\t- computing features on training set")
        traceln("\t\t #nodes=%d  #edges=%d "%Graph.getNodeEdgeTotalNumber(lGraph))
        chronoOn()
        lX, lY = self.get_lX_lY(lGraph)
        self._computeModelCaracteristics(lX)    #we discover here dynamically the number of features of nodes and edges
        traceln("\t\t %s"%self._getNbFeatureAsText())
        traceln("\t [%.1fs] done\n"%chronoOff())

        dPrm = {}
        dPrm['C']               = self.C                if type(self.C)               == types.ListType else [self.C]
        dPrm['tol']             = self.tol              if type(self.tol)             == types.ListType else [self.tol]
        dPrm['inference_cache'] = self.inference_cache  if type(self.inference_cache) == types.ListType else [self.inference_cache]
        dPrm['max_iter']        = self.max_iter         if type(self.max_iter)        == types.ListType else [self.max_iter]

        traceln("\t- creating a SSVM-trained CRF model")
        
        traceln("\t\t- computing class weight:")
        clsWeights = self.computeClassWeight(lY)
        traceln("\t\t\t%s"%clsWeights)

        crf = self._getCRFModel(clsWeights)
        
        self._ssvm = OneSlackSSVM(crf
                            #, inference_cache=self.inference_cache, C=self.C, tol=self.tol
                            , n_jobs=self.njobs
                            #, logger=SaveLogger(sModelFN, save_every=self.save_every)
                            #, max_iter=self.max_iter                                        
                            , show_loss_every=10
#                            , verbose=verbose)
                            , verbose=1)
        
        self._gs_ssvm = GridSearchCV(self._ssvm, dPrm, n_jobs=1, verbose=verbose) 
        self.ssvm = None
        
        chronoOn()
        traceln("\t - training by grid search a graph-based model")
        traceln("\t\t solver parameters for grid search:"
                    , " inference_cache=",self.inference_cache
                    , " C=",self.C, " tol=",self.tol, " n_jobs=",self.njobs
                    , " max_iter=", self.max_iter)
        self._gs_ssvm.fit(lX, lY)
        traceln("\t [%.1fs] done (graph-based model is trained with best parameters, selected by grid search) \n"%chronoOff())

        self.ssvm = self._gs_ssvm.best_estimator_ #Estimator that was chosen by the search

        try:
            #win32
            dBestParams = self._gs_ssvm.best_params_ 
        except:
            #do not know how to get this... in 
            dBestParams = { 'C'             : self.ssvm.C,
                           'inference_cache': self.ssvm.inference_cache,
                           'max_iter'       : self.ssvm.max_iter,
                           'tol'            : self.ssvm.tol 
                           }
            
        self.storeBestParams(dBestParams)
        traceln("\t", "- "*20)
        traceln("\tBest parameters: ",  dBestParams)
        traceln("\t", "- "*20)
        
        try:
            self.ssvm.alphas = None  
            self.ssvm.constraints_ = None
            self.ssvm.inference_cache_ = None    
            traceln("\t\t(model made slimmer. Not sure you can efficiently warm-start it later on. See option -w.)")        
        except Exception as e:
            traceln("\t\t(COULD NOT make the model slimmer. Got exception: %s"%str(e))        

        logger=SaveLogger(self.getModelFilename())
        logger(self.ssvm)  #save this model!
        
        traceln(self.getModelInfo())
        
        #Also save the details of this grid search
        sFN = self.getModelFilename()[:-4] + "GridSearchCV.pkl"
        try:
            self.gzip_cPickle_dump(sFN, self._gs_ssvm)
            traceln("\n\n--- GridSearchCV details: (also in %s)"%sFN)
            traceln("--- Best parameters set found on development set:")
            traceln(self._gs_ssvm.best_params_)
            traceln("--- Grid scores on development set:")
            means = self._gs_ssvm.cv_results_['mean_test_score']
            stds = self._gs_ssvm.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, self._gs_ssvm.cv_results_['params']):
                traceln("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
            traceln("--- ---")            
        except Exception as e:
            traceln("WARNING: error while dealing with the GridSearchCV object.")
            traceln(e)

        #the baseline model(s) if any
        self._trainBaselines(lX, lY)
        
        #do some garbage collection
        del lX, lY
        gc.collect()
        return 

    # --- Load / Save ------------------------------------------------
    def load(self, expiration_timestamp=None):
        """
        Load myself from disk
        If an expiration timestamp is given, the model stored on disk must be fresher than timestamp
        return self or raise a ModelException
        """
        Model.load(self, expiration_timestamp)
        self.ssvm = self._loadIfFresh(self.getModelFilename(), expiration_timestamp, lambda x: SaveLogger(x).load())
        try:
            self._EdgeBaselineModel = self._loadIfFresh(self.getEdgeBaselineFilename(), expiration_timestamp, self.gzip_cPickle_load)
            self.setNbClass(self.ssvm.model.n_states) #required the compute the edge label
        except:
            self._EdgeBaselineModel = None
        return self
    
    def save(self):
        """
        Save a trained model
        """
        self.gzip_cPickle_dump(self.getEdgeBaselineFilename(), self._EdgeBaselineModel)
        return Model.save(self)


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
        
        traceln("- computing features on test set")
        chronoOn("test")
        lX, lY = self.get_lX_lY(lGraph)
        
        traceln("\t #nodes=%d  #edges=%d "%Graph.getNodeEdgeTotalNumber(lGraph))
        self._computeModelCaracteristics(lX)    #we discover here dynamically the number of features of nodes and edges
        traceln("\t %s"%self._getNbFeatureAsText())
        traceln("[%.1fs] done\n"%chronoOff("test"))
        
        traceln("- predicting on test set")
        chronoOn("test2")
        if bConstraint:
            lConstraints = [g.instanciatePageConstraints() for g in lGraph]
            lY_pred = self._ssvm_ad3plus_predict(lX, lConstraints)
        else:
            lY_pred = self.ssvm.predict(lX)
        traceln(" [%.1fs] done\n"%chronoOff("test2"))
        
        tstRpt = TestReport(self.sName, lY_pred, lY, lLabelName, lsDocName=lsDocName)
        
        lBaselineTestReport = self._testBaselines(lX, lY, lLabelName, lsDocName=lsDocName)
        tstRpt.attach(lBaselineTestReport)
        
        tstRpt.attach(self._testEdgeBaselines(lX, lY, lLabelName, lsDocName=lsDocName))
        
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
        traceln("- predicting on test set")
        chronoOn("testFiles")
        for sFilename in lsFilename:
            [g] = loadFun(sFilename) #returns a singleton list
            [X], [Y] = self.get_lX_lY([g])

            if lLabelName == None:
                lLabelName = g.getLabelNameList()
                traceln("\t #nodes=%d  #edges=%d "%Graph.getNodeEdgeTotalNumber([g]))
                self._computeModelCaracteristics([X])    #we discover here dynamically the number of features of nodes and edges
                traceln("\t %s"%self._getNbFeatureAsText())
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
        traceln("[%.1fs] done\n"%chronoOff("testFiles"))

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
        [X] = self.get_lX([graph])
        bConstraint  = graph.getPageConstraint()
        traceln("\t\t #nodes=%d  #edges=%d "%Graph.getNodeEdgeTotalNumber([graph]))
        self._computeModelCaracteristics([X])    #we discover here dynamically the number of features of nodes and edges
        traceln("\t\t %s"%self._getNbFeatureAsText())
        
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
    try:
        sModelDir, sModelName = sys.argv[1:3]
    except:
        print "Usage: %s <model-dir> <model-name>"%sys.argv[0]
        print "Display some info regarding the stored model"
        exit(1)
        
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

    
    
