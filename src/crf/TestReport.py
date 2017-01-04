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
    
    def __init__(self, name, l_Y_pred, l_Y, lsClassName=None):
        """
        takes a test name, a prediction and a groundtruth or 2 lists of that stuff
        optionnally the list f class names
        compute:
        - if lsClassName: the sub-list of seen class (in either prediction or GT)
        - the confusion matrix
        - the classification report
        - the global accuracy score
        
        """
        #Content of this object (computed below!!)
        self.t = time.time()
        self.name = name
        self.fScore = None                  #global accuracy score
        self.aConfusionMatrix = None        #confusion matrix (np.array)
        self.sClassificationReport = ""     #formatted P/R/F per class
        self.lsSeenClassName = []           #sub-list of class names seen in that test, ordered by class index, if any class name list was provided. 
        
        self.lBaselineTestReport = []       #TestReport for baseline methods, if any
        
        # --- 
        assert type(l_Y_pred) == type(l_Y), "Internal error: when creating a tets report, both args must be of same type (list of np.array or np.array)"
        if type(l_Y_pred) == types.ListType:
            Y_pred = np.hstack(l_Y_pred)
            Y      = np.hstack(l_Y)
        else:
            Y_pred = l_Y_pred
            Y      = l_Y
        
        #we need to include all clasname that appear in the dataset or in the predicted labels (well... I guess so!)
        if lsClassName:
            setSeenCls = set()
            for _Y in [Y, Y_pred]:
                setSeenCls = setSeenCls.union( np.unique(_Y).tolist() )
            self.lsSeenClassName = [ cls for (i, cls) in enumerate(lsClassName) if i in setSeenCls]
            
            
        self.aConfusionMatrix = confusion_matrix(Y, Y_pred)

        if self.lsSeenClassName:
            self.sClassificationReport = classification_report(Y, Y_pred, target_names=self.lsSeenClassName)
        else:
            self.sClassificationReport = classification_report(Y, Y_pred)
        
        self.fScore = accuracy_score(Y, Y_pred)
    
    def attach(self, loTstRpt):
        """
        attach this testReport or list of TestReport to the current TestReport (typically the results of the baseline(s) )
        """
        if type(loTstRpt) == types.ListType:
            self.lBaselineTestReport.extend(loTstRpt)
        else:
            self.lBaselineTestReport.append(loTstRpt)
        return self.lBaselineTestReport
    
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
        
        s1 = str(self.aConfusionMatrix)
        if self.lsSeenClassName:
            iMaxClassNameLen = max([len(s) for s in self.lsSeenClassName])            
            lsLine = s1.split("\n")    
            sFmt = "%%%ds  %%s"%iMaxClassNameLen     #producing something like "%20s  %s"
            assert len(lsLine) == len(self.lsSeenClassName)
            s1 = "\n".join( [sFmt%(sLabel, sLine) for (sLabel, sLine) in zip(self.lsSeenClassName, lsLine)])    

        s2 = self.sClassificationReport
        
        s3 = "(unweighted) Accuracy score = %.2f"% self.fScore
        
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

s%(s2)s
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

