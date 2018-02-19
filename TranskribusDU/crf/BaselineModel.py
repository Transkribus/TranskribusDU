from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from .Model import Model
from common.trace import traceln
from .TestReport import TestReport


class BaselineModel(Model):
    '''
    A simple Baseline Model
    '''
    def setBaselineModelList(self, mdlBaselines):
        """
        set one or a list of sklearn model(s):
        - they MUST be initialized, so that the fit method can be called at train time
        - they MUST accept the sklearn usual predict method
        - they SHOULD support a concise __str__ method
        They will be trained with the node features, from all nodes of all training graphs
        """
        #the baseline model(s) if any
        if type(mdlBaselines) in [list, tuple]:
            raise Exception('Baseline Model can only use a single Model')
        else:
            self._lMdlBaseline = [mdlBaselines]
        return


    # --- TRAIN / TEST / PREDICT ------------------------------------------------
    def train(self, lGraph, bWarmStart=True, expiration_timestamp=None,baseline_id=0):
        """
        Return a model trained using the given labelled graphs.
        The train method is expected to save the model into self.getModelFilename(), at least at end of training
        If bWarmStart==True, The model is loaded from the disk, if any, and if fresher than given timestamp, and training restarts

        if some baseline model(s) were set, they are also trained, using the node features

        """
        traceln("\t- computing features on training set")
        lX, lY = self.transformGraphs(lGraph, True)
        traceln("\t done")

        return self._trainBaselines(lX,lY)

    def test(self, lGraph):
        """
        Test the model using those graphs and report results on stderr

        if some baseline model(s) were set, they are also tested

        Return a Report object
        """
        assert lGraph
        traceln("\t- computing features on test set")
        lX, lY = self.transformGraphs(lGraph, True)
        return self._testBaselines(lX,lY)

    def testFiles(self, lsFilename, loadFun):
        """
        Test the model using those files. The corresponding graphs are loaded using the loadFun function (which must return a singleton list).
        It reports results on stderr

        if some baseline model(s) were set, they are also tested

        Return a Report object
        """
        lX, lY, lY_pred  = [], [], []
        lLabelName   = None
        traceln("\t- predicting on test set")

        for sFilename in lsFilename:
            [g] = loadFun(sFilename) #returns a singleton list
            X, Y = self.transformGraphs([g], True)

            if lLabelName == None:
                lLabelName = g.getLabelNameList()
                #traceln("\t\t #features nodes=%d  edges=%d "%(X[0].shape[1], X[2].shape[1]))
            else:
                assert lLabelName == g.getLabelNameList(), "Inconsistency among label spaces"

            X_node = [node_features for (node_features, _, _) in X]
            Y_pred = self.predictBaselines(X_node[0])

            lY.append(Y[0])
            traceln(" saving the first baseline predictions ....")
            lY_pred.append(Y_pred[0]) #Choose with Y_pred is a list of predictions of feach model


            g.detachFromDOM()
            del g   #this can be very large
            del X,X_node


        traceln("\t done")
        tstRpt = TestReport(self.sName, lY_pred, lY, lLabelName)
        del lX, lY
        return tstRpt
