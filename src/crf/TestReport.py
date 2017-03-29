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

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

class TestReport:
    """
    A class that encapsulates the result of a classification test
    """
    
    def __init__(self, name, l_Y_pred, l_Y, lsClassName=None, lsDocName=None):
        """
        takes a test name, a prediction and a groundtruth or 2 lists of that stuff
        optionnally the list f class names
        compute:
        - if lsClassName: the sub-list of seen class (in either prediction or GT)
        - the confusion matrix
        - the classification report
        - the global accuracy score
        
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

        #Computing the classes that are observed or that were expected to be observed
        aSeenCls = np.unique(self.l_Y_pred[0])
        for _Y in self.l_Y_pred[1:]: aSeenCls = np.unique( np.hstack([aSeenCls, np.unique(_Y)]) )
        for _Y in self.l_Y:          aSeenCls = np.unique( np.hstack([aSeenCls, np.unique(_Y)]) )
        aSeenCls.sort()
        self.lSeenCls = aSeenCls.tolist()
        
        # --- 
        assert type(l_Y_pred) == type(l_Y), "Internal error: when creating a test report, both args must be of same type (list of np.array or np.array)"
        assert len(l_Y) == len(l_Y_pred), "Internal error"
        if lsDocName: assert len(l_Y) == len(lsDocName), "Internal error"
        # --- 
        
#         self.fScore = None                  #global accuracy score
#         self.aConfusionMatrix = None        #confusion matrix (np.array)
#         self.laConfusionMatrix = None
#         self.sClassificationReport = ""     #formatted P/R/F per class
#         self.lsSeenClassName = []           #sub-list of class names seen in that test, ordered by class index, if any class name list was provided. 
        


    
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
    def getTestName(self):          return self.name
    def getNbClass(self):           return self.nbClass
    def getClassNameList(self):     return self.lsClassName
    def getSeenClassNameList(self): return [clsName for (i, clsName) in enumerate(self.lsClassName) if i in self.lSeenCls] if self.lsClassName else None
    def getDocNameList(self):       return self.lsDocName
    
    def getFlat_Y_YPred(self):
        """
        Might be a good idea to cache those values, which are useful for further calls, and del them afterward.
        """
        return np.hstack(self.l_Y), np.hstack(self.l_Y_pred)
    
    def getConfusionMatrix(self, tY_Ypred = None):
        """
        Return a confusion matrix covering all classes (not only those observed in this particular test)
        if you have the flat Y and Y_pred, pass them!
        """
        Y, Y_pred = tY_Ypred if tY_Ypred else self.getFlat_Y_YPred()
        confumat = confusion_matrix(Y, Y_pred, labels=(range(self.nbClass) if self.nbClass else None))
        if not tY_Ypred: del Y, Y_pred
        return confumat
    
    def getConfusionMatrixByDocument(self):
        """
        List of confusion matrices, one for each document in the test set
        """
        return [ confusion_matrix(_Y, _Y_pred, range(self.nbClass) if self.nbClass else None) for _Y, _Y_pred in zip(self.l_Y, self.l_Y_pred) ]
    
    def getClassificationReport(self, tY_Ypred = None):
        """
        return the score and the (textual) classification report
        """
        Y, Y_pred = tY_Ypred if tY_Ypred else self.getFlat_Y_YPred()
        if self.lsClassName:
            #we need to include all clasname that appear in the dataset or in the predicted labels (well... I guess so!)
            lsSeenClassName = self.getSeenClassNameList()
            sClassificationReport = classification_report(Y, Y_pred, target_names=lsSeenClassName)
        else:
            sClassificationReport = classification_report(Y, Y_pred)
        
        fScore = accuracy_score(Y, Y_pred)    
        if not tY_Ypred: del Y, Y_pred
        
        return fScore, sClassificationReport

    # ------------------------------------------------------------------------------------------------------------------
    def toString(self, bShowBaseline=True, bBaseline=False):
        """
        return a nicely indented test report
        if bShowBaseline, includes any attached report(s), with proper indentation
        if bBaseline: report for a baseline method
        """
        #not nice because of formatting of numbers possibly different per line
#         if lsClassName:
#             #Let's show the class names
#             s1 = ""
#             for sLabel, aLine in zip(lsSeenClassName, a):
#                 s1 += "%20s %s\n"%(sLabel, aLine)
#         else:
#             s1 = str(a)
        if bBaseline:
            sSepBeg = "\n" + "-" * 30
            sTitle  = " BASELINE "
            sSpace  = " "*8
        else:
            sSepBeg = "--- " +  time.asctime(time.gmtime(self.t)) + "---------------------------------"
            sTitle  = "TEST REPORT FOR"
            sSpace  = ""
        sSepEnd = "-"*len(sSepBeg)
        
        tY_Ypred = self.getFlat_Y_YPred()
        
        s1 = str(self.getConfusionMatrix(tY_Ypred))
        
        if self.lsClassName:
            lsSeenClassName = self.getSeenClassNameList()
            iMaxClassNameLen = max([len(s) for s in lsSeenClassName])            
            lsLine = s1.split("\n")    
            sFmt = "%%%ds  %%s"%iMaxClassNameLen     #producing something like "%20s  %s"
            assert len(lsLine) == len(lsSeenClassName)
            s1 = "\n".join( [sFmt%(sLabel, sLine) for (sLabel, sLine) in zip(lsSeenClassName, lsLine)])    

        fScore, s2 = self.getClassificationReport(tY_Ypred)
        
        s3 = "(unweighted) Accuracy score = %.2f"% fScore
        
        if bShowBaseline:
            if self.lBaselineTestReport:
                sBaselineReport = "".join( [o.toString(False, True) for o in self.lBaselineTestReport] )
            else:
                sBaselineReport = sSpace + "(No Baseline method to report)"
        else:
            sBaselineReport = ""
        
        return """%(space)s%(sSepBeg)s 
%(space)s%(sTitle)s: %(name)s

%(space)s  Line=True class, column=Prediction
%(s1)s

%(s2)s
%(space)s%(s3)s
%(sBaselineReport)s
%(space)s%(sSepEnd)s
""" % {"space":sSpace, "sSepBeg":sSepBeg
       , "sTitle":sTitle, "name":self.name
       , "s1":s1, "s2":s2, "s3":s3, "sBaselineReport":sBaselineReport, "sSepEnd":sSepEnd}        
        
    def __str__(self):
        """
        return a nicely formatted string containing all the info of this test report object
        """
        return self.toString()

