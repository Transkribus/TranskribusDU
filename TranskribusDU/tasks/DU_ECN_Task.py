from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys, os, glob, datetime
from optparse import OptionParser

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV  # 0.18.1 REQUIRES NUMPY 1.12.1 or more recent

try:  # to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import TranskribusDU_version

from common.chrono import chronoOn, chronoOff

import crf.Model

from xml_formats.PageXml import MultiPageXml
import crf.FeatureDefinition


from gcn.DU_Model_ECN import DU_Model_ECN

from tasks.DU_CRF_Task import DU_CRF_Task


class DU_ECN_Task(DU_CRF_Task):

    def __init__(self, sModelName, sModelDir, dLearnerConfig={}, sComment=None
                 , cFeatureDefinition=None, dFeatureConfig={},cModelClass=DU_Model_ECN,
                 ):
        super(DU_ECN_Task, self).__init__(sModelName,sModelDir,dLearnerConfig,sComment,cFeatureDefinition,dFeatureConfig)

        self.cModelClass = cModelClass
        assert issubclass(self.cModelClass, crf.Model.Model), "Your model class must inherit from crf.Model.Model"

    def isTypedCRF(self):
        """
        if this a classical CRF or a Typed CRF?
        """
        return False

    def predict(self, lsColDir, docid=None):
        """
        Return the list of produced files
        """
        self.traceln("-" * 50)
        self.traceln("Predicting for collection(s):", lsColDir)
        self.traceln("-" * 50)

        if not self._mdl: raise Exception("The model must be loaded beforehand!")

        # list files
        if docid is None:
            _, lFilename = self.listMaxTimestampFile(lsColDir, self.sXmlFilenamePattern)
        # predict for this file only
        else:
            try:
                lFilename = [os.path.abspath(os.path.join(lsColDir[0], docid + MultiPageXml.sEXT))]
            except IndexError:
                raise Exception("a collection directory must be provided!")

        DU_GraphClass = self.getGraphClass()

        lPageConstraint = DU_GraphClass.getPageConstraint()
        if lPageConstraint:
            for dat in lPageConstraint: self.traceln("\t\t%s" % str(dat))

        chronoOn("predict")
        self.traceln("- loading collection as graphs, and processing each in turn. (%d files)" % len(lFilename))
        du_postfix = "_du" + MultiPageXml.sEXT

        #Creates a tf.Session and load the model checkpoints
        session=self._mdl.restore()
        lsOutputFilename = []
        for sFilename in lFilename:
            if sFilename.endswith(du_postfix): continue  #:)
            chronoOn("predict_1")
            lg = DU_GraphClass.loadGraphs([sFilename], bDetach=False, bLabelled=False, iVerbose=1)
            # normally, we get one graph per file, but in case we load one graph per page, for instance, we have a list
            if lg:
                for g in lg:
                    doc = g.doc
                    if lPageConstraint:
                        self.traceln("\t- prediction with logical constraints: %s" % sFilename)
                    else:
                        self.traceln("\t- prediction : %s" % sFilename)
                    Y = self._mdl.predict(g,session)

                    g.setDomLabels(Y)
                    del Y
                del lg

                MultiPageXml.setMetadata(doc, None, self.sMetadata_Creator, self.sMetadata_Comments)
                sDUFilename = sFilename[:-len(MultiPageXml.sEXT)] + du_postfix
                doc.write(sDUFilename,
                          xml_declaration=True,
                          encoding="utf-8",
                          pretty_print=True
                          # compression=0,  #0 to 9
                          )

                lsOutputFilename.append(sDUFilename)
            else:
                self.traceln("\t- no prediction to do for: %s" % sFilename)

            self.traceln("\t done [%.2fs]" % chronoOff("predict_1"))
        self.traceln(" done [%.2fs]" % chronoOff("predict"))
        session.close()
        return lsOutputFilename
