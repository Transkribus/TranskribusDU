# -*- coding: utf-8 -*-

"""
    TestReport of a multi-class classifier

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
import types, time

import numpy as np

from sklearn.metrics import confusion_matrix
from util.metrics import confusion_classification_report, confusion_list_classification_report, confusion_accuracy_score, confusion_PRFAS

class TestReport:
    """
    A class that encapsulates the result of a classification test
    """
    
    def __init__(self, name, l_Y_pred, l_Y, lsClassName, lsDocName=None):
        """
        takes a test name, a prediction and a groundtruth or 2 lists of that stuff
        optionally the list f class names
        optionally the list document names
        compute:
        - if lsClassName: the sub-list of seen class (in either prediction or GT)
        
        NOTE:   Baseline models may not know the number of classes, nor the docnames...
                But since these TestReport are attachd to th emain one, we have the info!
        """
        #Content of this object (computed below!!)
        self.t = time.time()
        self.name          = name
        self.lsDocName     = lsDocName
        self.lsClassName   = lsClassName
        self.nbClass       = len(self.lsClassName) if self.lsClassName else None
        self.lBaselineTestReport = []       #TestReport for baseline methods, if any
        
        if type(l_Y_pred) == types.ListType:
            self.l_Y_pred = l_Y_pred
            self.l_Y      = l_Y
        else:
            self.l_Y_pred = [l_Y_pred]
            self.l_Y      = [l_Y]

        #Computing the classes that are observed 
        aSeenCls = np.unique(self.l_Y_pred[0])
        for _Y in self.l_Y_pred[1:]: aSeenCls = np.unique( np.hstack([aSeenCls, np.unique(_Y)]) )
#         #... or that were expected to be observed
#         for _Y in self.l_Y:          aSeenCls = np.unique( np.hstack([aSeenCls, np.unique(_Y)]) )
        aSeenCls.sort()
        self.lSeenCls = aSeenCls.tolist()
        
        # --- 
        assert type(l_Y_pred) == type(l_Y), "Internal error: when creating a test report, both args must be of same type (list of np.array or np.array)"
        if l_Y is None:
            assert l_Y is None and l_Y_pred is None, "Internal error"
        else:
            assert len(l_Y) == len(l_Y_pred), "Internal error"
            if lsDocName: assert len(l_Y) == len(lsDocName), "Internal error"
        # --- 
        
    
    def attach(self, loTstRpt):
        """
        attach this testReport or list of TestReport to the current TestReport (typically the results of the baseline(s) )
        """
        if type(loTstRpt) == types.ListType:
            self.lBaselineTestReport.extend(loTstRpt)
        else:
            self.lBaselineTestReport.append(loTstRpt)
        return self.lBaselineTestReport
    
    # ------------------------------------------------------------------------------------------------------------------
    def getTestName(self):              return self.name
    def getNbClass(self):               return self.nbClass
    def getClassNameList(self):         return self.lsClassName
    def getDocNameList(self):           return self.lsDocName
    def getBaselineTestReports(self):   return self.lBaselineTestReport
    
    def getConfusionMatrix(self):
        """
        Return a confusion matrix covering all classes (not only those observed in this particular test)
        if you have the flat Y and Y_pred, pass them!
        """
        labels   = range(self.nbClass) if self.nbClass else None
        confumat = reduce(np.add, [confusion_matrix(_Y, _Y_pred, labels) for _Y, _Y_pred in zip(self.l_Y, self.l_Y_pred)])
        return confumat
    
    def getConfusionMatrixByDocument(self):
        """
        List of confusion matrices, one for each document in the test set
        """
        labels   = range(self.nbClass) if self.nbClass else None
        return [ confusion_matrix(_Y, _Y_pred, labels) for _Y, _Y_pred in zip(self.l_Y, self.l_Y_pred) ]
    
    def getClassificationReport(self, aConfuMat=None):
        """
        return the score and the (textual) classification report
        """
        if aConfuMat is None: aConfuMat = self.getConfusionMatrix()
        if self.lsClassName:
            sClassificationReport = confusion_classification_report(aConfuMat, target_names=self.lsClassName)
        else:
            sClassificationReport = confusion_classification_report(aConfuMat)
        
        fScore = confusion_accuracy_score(aConfuMat)    
        
        return fScore, sClassificationReport

    # ------------------------------------------------------------------------------------------------------------------
    def toString(self, bShowBaseline=True, bBaseline=False):
        """
        return a nicely indented test report
        if bShowBaseline, includes any attached report(s), with proper indentation
        if bBaseline: report for a baseline method
        """
        if bBaseline:
            sSpace  = " "*8
            sSepBeg = "\n" + sSpace + "~" * 30
            sTitle  = " BASELINE "
            sSepEnd = "~"*len(sSepBeg)
        else:
            sSepBeg = "--- " +  time.asctime(time.gmtime(self.t)) + "---------------------------------"
            sTitle  = "TEST REPORT FOR"
            sSpace  = ""
            sSepEnd = "-"*len(sSepBeg)
        
        aConfuMat = self.getConfusionMatrix()
        np_dFmt = np.get_printoptions()
        np.set_printoptions(threshold=100*100, linewidth=100*20)
        s1 = str(aConfuMat)
        np.set_printoptions(np_dFmt)
        
        if self.lsClassName:
            iMaxClassNameLen = max([len(s) for s in self.lsClassName])            
            lsLine = s1.split("\n")    
            sFmt = "%%%ds  %%s"%iMaxClassNameLen     #producing something like "%20s  %s"
            assert len(lsLine) == len(self.lsClassName), "Internal error: expected one line per class name"
            s1 = "\n".join( [sFmt%(sLabel, sLine) for (sLabel, sLine) in zip(self.lsClassName, lsLine)])    

        fScore, s2 = self.getClassificationReport(aConfuMat)
        
        s3 = "(unweighted) Accuracy score = %.2f     trace=%d  sum=%d"% (fScore, aConfuMat.trace(), np.sum(aConfuMat))
        
        if bShowBaseline:
            if self.lBaselineTestReport:
                sBaselineReport = "".join( [o.toString(False, True) for o in self.lBaselineTestReport] )
            else:
                sBaselineReport = sSpace + "(No Baseline method to report)"
        else:
            sBaselineReport = ""
        
        sReport = """%(space)s%(sSepBeg)s 
%(space)s%(sTitle)s: %(name)s

%(space)s  Line=True class, column=Prediction
%(s1)s

%(space)s%(s3)s

%(s2)s
%(sBaselineReport)s
%(space)s%(sSepEnd)s
""" % {"space":sSpace, "sSepBeg":sSepBeg
       , "sTitle":sTitle, "name":self.name
       , "s1":s1, "s2":s2, "s3":s3, "sBaselineReport":sBaselineReport, "sSepEnd":sSepEnd}    

        #indent the baseline    
        if bBaseline: sReport = '\n'.join(['\t'+_s for _s in sReport.split('\n')])
        
        return sReport
    
    def __str__(self):
        """
        return a nicely formatted string containing all the info of this test report object
        """
        return self.toString()


class TestReportConfusion(TestReport):
    """
    A class that encapsulates the result of a classification test, by using the confusion matrices
    """
    
    def __init__(self, name, laConfuMat, lsClassName, lsDocName=None):
        """
        takes a test name, a list of confusiopn matrices
        optionally the list f class names
        optionally the list document names
        compute:
        - if lsClassName: the sub-list of seen class (in either prediction or GT)
        
        NOTE:   Baseline models may not know the number of classes, nor the docnames...
                But since these TestReport are attachd to th emain one, we have the info!
        """
        #We set Y_pred and Y_true to None
        TestReport.__init__(self, name, None, None, lsClassName=lsClassName, lsDocName=lsDocName)
        
        assert self.nbClass, "Internal error: TestReportConfusion needs to know the number of classes."
        #we instead store the confusion matrix
        self.laConfuMat = laConfuMat
        
        for aConfMat in self.laConfuMat: 
            assert aConfMat.shape==(self.nbClass, self.nbClass), "Internal Error: TestReportConfusion classes expect same size confusion matrices."

            
    @classmethod
    def newFromReportList(cls, name, loTestRpt):
        """
        Aggregate all those TestReports into a single TestReport object, which is returned
        """
        if not(loTestRpt): raise ValueError("ERROR: cannot aggregate empty list of TestReport objects")

        o0 = loTestRpt[0]
        oret = TestReportConfusion(name, [], lsClassName=o0.lsClassName, lsDocName=[] if o0.lsDocName else None)
        
        #we also build one TestConfusionReport per baseline method
        oret.lBaselineTestReport = [TestReportConfusion(bsln.name, [], lsClassName=o0.lsClassName, lsDocName=None) for bsln in o0.getBaselineTestReports()]
            
        for oTestRpt in loTestRpt:
            assert oTestRpt.nbClass == oret.nbClass , "Internal Error: cannot aggregate TestReport with heterogeneous number of classes."
            assert len(oTestRpt.getBaselineTestReports()) == len(oret.lBaselineTestReport), "Error: cannot aggregate TestReport with heterogeneous number of baselines."
            for oBsln, oBsln0 in zip(oTestRpt.getBaselineTestReports(), o0.getBaselineTestReports()):
                assert oBsln.name == oBsln0.name, "Error: cannot aggregate TestReport with heterogeneous baselines." 
            oret.accumulate(oTestRpt)
            
        return oret

    @classmethod
    def newFromYYpred(cls, name, Y, Y_pred, lsClassName, lsDocName=None):
        """
        Creator
        """
        confumat = confusion_matrix(Y, Y_pred, range(len(lsClassName)))
        return TestReportConfusion(name, [confumat], lsClassName, lsDocName=lsDocName)

    def getConfusionMatrix(self):
        """
        Return a confusion matrix covering all classes (not only those observed in this particular test)
        if you have the flat Y and Y_pred, pass them!
        """
        return reduce(np.add, self.laConfuMat)
        
    def getConfusionMatrixByDocument(self):
        return self.laConfuMat
    

    def getClassificationReport(self, aConfuMat=None):
        """
        return the score and the (textual) classification report
        """
        if aConfuMat is None: aConfuMat = self.getConfusionMatrix()
        if self.lsClassName:
            sClassificationReport = confusion_list_classification_report(self.laConfuMat, target_names=self.lsClassName)
        else:
            sClassificationReport = confusion_list_classification_report(self.laConfuMat)
        
        fScore = confusion_accuracy_score(aConfuMat)    
        
        return fScore, sClassificationReport        

#     def getClassificationReportByDoc(self):
#         """
#         Return a textual report by file
#         """
#         report = ""
#         for aConfuMat in self.laConfuMat:
#             p, r, f1, a, s = confusion_PRFAS(aConfuMat)
#             report += 
            
    def accumulate(self, oTestRpt):
        """
        Absorb this TestReport object into the current one.
        """
        if self.nbClass:
            assert not(oTestRpt.nbClass is None)    , "Internal Error: cannot aggregate a TestReport with unknown number of classes into this TestReport with known number of classes."
            assert self.nbClass == oTestRpt.nbClass, "Internal Error: cannot aggregate a TestReport with a number of classes into this TestReport object."

        self.laConfuMat.append(oTestRpt.getConfusionMatrix())

        #aggregate the results per baseline method        
        for i, oBslnRpt in enumerate(oTestRpt.lBaselineTestReport):
            self.lBaselineTestReport[i].accumulate(oBslnRpt)

        if self.lsDocName:
            assert oTestRpt.lsDocName, "Internal Error: one object has no list of document name. Cannot aggregate with doc name."
            self.lsDocName.extend(oTestRpt.getDocNameList())
            
            
            
            
            