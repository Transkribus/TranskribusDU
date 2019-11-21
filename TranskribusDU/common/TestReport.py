# -*- coding: utf-8 -*-

"""
    TestReport of a multi-class classifier

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import time
import os
from functools import reduce

import numpy as np


from sklearn.metrics import confusion_matrix
from util.metrics import confusion_classification_report, confusion_list_classification_report, confusion_accuracy_score


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
        
        if type(l_Y_pred) == list:
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
#             if lsDocName: assert len(l_Y) == len(lsDocName), "Internal error"
        # --- 
        
    
    def attach(self, loTstRpt):
        """
        attach this testReport or list of TestReport to the current TestReport (typically the results of the baseline(s) )
        """
        if type(loTstRpt) == list:
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

    @classmethod
    def compareReport(cls,r1,r2,bDisplay=False):
        """    
            compare two reports using  
                scipy.stats.wilcoxon : Calculate the Wilcoxon signed-rank test.

        """
        from  scipy.stats import wilcoxon
        from scipy.stats import  mannwhitneyu
         
        lConfMat1 = r1.getConfusionMatrixByDocument()
        nbClasses = len(lConfMat1[0])
        print ('nb classes: %d'%nbClasses)
        eps=1e-8
        lP1=np.ndarray((len(lConfMat1),nbClasses),dtype=float)
        lR1=np.ndarray((len(lConfMat1),nbClasses),dtype=float)
        lF1=np.ndarray((len(lConfMat1),nbClasses),dtype=float)
        for i,conf in enumerate(lConfMat1):
            lP1[i]=np.diag(conf)/(eps+conf.sum(axis=0))
            lR1[i]=np.diag(conf)/(eps+conf.sum(axis=1))
            lF1[i]= 2*lP1[-1]*lR1[-1]/(eps+lP1[-1]+lR1[-1])
        
        lConfMat2 = r2.getConfusionMatrixByDocument()
        lP2=np.ndarray((len(lConfMat2),nbClasses),dtype=float)
        lR2=np.ndarray((len(lConfMat2),nbClasses),dtype=float)
        lF2=np.ndarray((len(lConfMat2),nbClasses),dtype=float)
        for i,conf2 in enumerate(lConfMat2):
            lP2[i]=np.diag(conf2)/(eps+conf2.sum(axis=0))
            lR2[i]=np.diag(conf2)/(eps+conf2.sum(axis=1))
            lF2[i]= 2*lP2[-1]*lR2[-1]/(eps+lP2[-1]+lR2[-1])
        
        if bDisplay:
            for cl in range(nbClasses):
                print ('class %s'%cl)
                r1 = lP1[:,cl]
                r2 = lP2[:,cl]
                print ('\tPrecision:')
                print('\t',wilcoxon(r1,r2, zero_method="zsplit"))  

                r1 = lR1[:,cl]
                r2 = lR2[:,cl]
                print ('\tRecall:')
                print('\t',wilcoxon(r1,r2, zero_method="zsplit"))  
                r1 = lF1[:,cl]
                r2 = lF2[:,cl]
                print ('\tF1:')
                print('\t',wilcoxon(r1,r2, zero_method="zsplit"))                        
        
    
    def getDetailledReport(self):
        """
            return Precision, Recall, F1 per label for each docucment
            0 5577_001.mpxml [ 0.882  0.922  0.976  1.     0.   ] [ 0.904  0.981  0.988  0.771  0.   ] [ 0.893  0.951  0.982  0.871  0.   ]
            1 5577_007.mpxml [ 0.969  0.952  0.992  1.     0.   ] [ 0.969  0.981  0.969  1.     0.   ] [ 0.969  0.967  0.981  1.     0.   ]
            
        """
        sReport = "\n Detailed Reporting \n" + "-"*20 +"\n"
        lConfMat = self.getConfusionMatrixByDocument()
        eps=1e-8
        # numpy bug!!
        np.set_printoptions(precision = 3, suppress = True)
        for i,conf in enumerate(lConfMat):
            p   = np.diag(conf)/(eps+conf.sum(axis=0))
            r   = np.diag(conf)/(eps+conf.sum(axis=1))
            f1  = 2*p*r/(eps+p+r)
            #sReport += "%d\t%s\t%s %s %s\n"%(i,os.path.basename(self.getDocNameList()[i]),p,r,f1)
            sReport += "%d\t%s %s %s\t%s\n"%(i, p,r,f1, os.path.basename(self.getDocNameList()[i]))
            
        return sReport
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
        
        s3 = "(unweighted) Accuracy score = %.2f %%    trace=%d  sum=%d"% (100*fScore, aConfuMat.trace(), np.sum(aConfuMat))
        
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



class RankingReport(TestReport):
    """
    A ranking Report which include Mean Average Precision and possibly other ranking measures
    """

    def __init__(self, name, l_Y_pred, l_Y, lsClassName=None):
        TestReport.__init__(self,name,l_Y_pred,l_Y,lsClassName)
        self.average_precision=[]

    def toString(self, bShowBaseline=True, bBaseline=False):
        #report=super(RankingReport,self).toString(bShowBaseline,bBaseline)
        report=TestReport.toString(self,bShowBaseline,bBaseline)
        report+="-" * 30+'\n'
        report+='     RANKING MEASURES  \n'
        report+="-" * 30+'\n'
        report+="Mean Average Precision\n"
        for i,avgpi in self.average_precision:
            report+='\t'+str(i)+':'+str(avgpi)+'\n'
        report+="-" * 30+'\n'
        return report


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
        oret.lBaselineTestReport = [TestReportConfusion(bsln.name, [], lsClassName=bsln.lsClassName, lsDocName=None) 
                                    for bsln in o0.getBaselineTestReports()]
            
        for oTestRpt in loTestRpt:
            assert len(oTestRpt.getBaselineTestReports()) == len(oret.lBaselineTestReport), "Error: cannot aggregate TestReport with heterogeneous number of baselines."
            for oBsln, oBsln0 in zip(oTestRpt.getBaselineTestReports(), o0.getBaselineTestReports()):
                assert oBsln.name    == oBsln0.name    , "Error: cannot aggregate TestReport with heterogeneous baselines." 
                assert oBsln.nbClass == oBsln0.nbClass , "Internal Error: cannot aggregate TestReport with heterogeneous number of classes."
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
            sClassificationReport = confusion_list_classification_report(self.laConfuMat, target_names=self.lsClassName, digits=3)
        else:
            sClassificationReport = confusion_list_classification_report(self.laConfuMat, digits=3)
        
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
            
            
            
            
def test_test_report():
    lsClassName = ['OTHER', 'catch-word', 'header', 'heading', 'marginalia', 'page-number']
    Y = np.array([0,  2, 3, 2, 5], dtype=np.int32)

    f, _ = TestReport("test", Y, np.array([0,  2, 3, 2, 5], dtype=np.int32), None).getClassificationReport()
    assert (f - 1.0) < 0.01
    f, _ = TestReport("test", Y, np.array([0,  2, 3, 2, 5], dtype=np.int32), lsClassName).getClassificationReport()
    assert (f - 1.0) < 0.01
    f, _ = TestReport("test", Y, np.array([0,  2, 3, 2, 2], dtype=np.int32), lsClassName).getClassificationReport()
    assert (f - 0.8) < 0.01
    f, _ = TestReport("test", Y, np.array([0,  2, 3, 2, 4], dtype=np.int32), lsClassName).getClassificationReport()
    assert (f - 0.8) < 0.01
