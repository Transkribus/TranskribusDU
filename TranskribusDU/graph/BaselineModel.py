# -*- coding: utf-8 -*-

"""
    Baseline model (non structured) ML
    
    Copyright Xerox(C) 2018, 2019  Hervé Déjean, Jean-Luc Meunier, Animesh Prasad

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





from common.trace import traceln
from from common.TestReport import TestReport
from graph.GraphModel import GraphModel


class BaselineModel(GraphModel):
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
            if self.bConjugate: g.computeEdgeLabels()
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
