

from Model import Model,ModelException
from common.trace import traceln
import types


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
        if type(mdlBaselines) in [types.ListType, types.TupleType]:
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
