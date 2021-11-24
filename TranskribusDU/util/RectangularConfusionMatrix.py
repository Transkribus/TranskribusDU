"""
    A rectangular confusion matrix, for the case when the model can invent labels.
    Typically useful when using a regression model to predict row numbers
    , for instance. :)
    
    USAGE:
    - create a RectangularConfusionMatrixFamily
    - get confusion matrices from the family object
    
    JL Meunier
    Copyright 2020 Naver Labs Europe
    April 2020
"""
import time
import math
from collections import defaultdict
import functools

import numpy as np


class RectangularConfusionMatrixException(Exception):
    pass


class RectangularConfusionMatrix:
    """
    A rectangular confusion matrix, for the case when the model can invent labels.
    Typically useful when using a regression model to predict row numbers
    , for instance. :)
    """
    def __init__(self, family):
        """
        takes a test name
        , a list of groundtruth labels
        """
        self._family = family
        # dictionary d[Y_true][Y_pred] == count
        self._d = defaultdict(lambda : defaultdict(int))
        
    def observe(self, Y_true, Y_pred):
        """
        record one observation, or a sequence of observations
        """
        try:
            self._family._seenIter(Y_true, Y_pred)
            for Y_true, Y_pred in zip(iter(Y_true), iter(Y_pred)):
                self._d[Y_true][Y_pred] += 1
        except TypeError:
            self._family._seen(Y_true, Y_pred)
            self._d[Y_true][Y_pred] += 1
            
    def numpy(self):
        if self._family._bRotten:
            raise RectangularConfusionMatrixException("Family normalization required")
        return self._a
    
    def sum(self, Y_true=None):
        if self._family._bRotten:
            raise RectangularConfusionMatrixException("Family normalization not yet done")
        if Y_true is None:
            return self._a.sum()
        else:
            return self._a[self._family._dY_true_index[Y_true],:].sum()

    def trace(self):
        """
        trace of this rectangular matrix.... i.e. count of valid decisions
        """
        if self._family._bRotten:
            raise RectangularConfusionMatrixException("Family normalization required")
        return self._a[self._family._aRowIndex, self._family._aColIndex].sum()

    def accuracy(self):
        if self._family._bRotten:
            raise RectangularConfusionMatrixException("Family normalization required")
        fSum = self.sum()
        if fSum == 0:
            # what to do???
            return 0.0
        else:
            return self.trace() / fSum

    def PRFS(self, Y_true):
        """
        precision, recall, f1, support score for the given class
        """
        if self._family._bRotten:
            raise RectangularConfusionMatrixException("Family normalization required")
        f = self._family
        try:
            return self._PRFS(f._dY_true_index[Y_true], f._dY_pred_index[Y_true])   
        except KeyError:
            raise RectangularConfusionMatrixException("Unknown true label: %s"%Y_true)

    def _PRFS(self, i, j):
        """
        precision, recall, f1 score, support for the given class
        """
        nOk = self._a[i,j]
        nTotTrue = self._a[i,:].sum()
        nTotPred = self._a[:,j].sum()
        fP = nOk/(1e-8+nTotPred)
        fR = nOk/(1e-8+nTotTrue)
        fF1 = 2*fP*fR / (1e-8+fP+fR)
        return (fP, fR, fF1, nTotTrue)
    
    def _normalize(self, nbTrueClass, dY_true_index, nbPredClass, dY_pred_index):
        """
        return a "normalized rectangular matrix
        """
        a = np.zeros((nbTrueClass, nbPredClass), dtype=np.int64)
        
        for Y_true in self._d.keys():
            i = dY_true_index[Y_true]
            ai = a[i]
            for Y_pred, cnt in self._d[Y_true].items():
                ai[dY_pred_index[Y_pred]] = cnt
        self._a = a
        return self._a


class RectangularConfusionMatrixFamily(RectangularConfusionMatrix):
    """
    Reflect a family of rectangular confusion matrix that share the same true
    and predicted labels
    """
    def __init__(self, name):
        RectangularConfusionMatrix.__init__(self, self)
        
        self.name = name
        
        # all known memebers of this family"
        self._lMember = list()
        # default possible outputs
        self.setY_true  = set()
        self.setY_pred  = set()
        
        # after normalization, index to true and predicted labels
        self._bRotten   = True  # normalization is required
        self._aRowIndex = None #index of true Ys in rows of _a
        self._aColIndex = None # index of true Ys in columns of _a
        
        
    def createRectangularConfusionMatrix(self):
        """
        return a new empty confusion matrix belonging to this family
        """
        self._bRotten = True  # normalization is required
        o = RectangularConfusionMatrix(self)
        self._lMember.append(o)
        return o
    
    def normalize(self):
        """
        normalize the family to a common shape
        return the normalized shape, (1-size-fit-all)
        """
        self._bRotten = False  # normalization is done!
        lY_true = self.getYTrueList()
        lY_pred = self.getYPredList()
        nbTrueClass = len(lY_true)
        nbPredClass = len(lY_pred)
        self._dY_true_index = { Y_true:i for i,Y_true in enumerate(lY_true) }
        self._dY_pred_index = { Y_pred:i for i,Y_pred in enumerate(lY_pred) }
        self._aRowIndex = np.arange(nbTrueClass)
        self._aColIndex = np.array([self._dY_pred_index[_Y] 
                                    for _Y in lY_true], dtype=np.int64)
        
        for o in self._lMember:
            o._normalize(  nbTrueClass, self._dY_true_index
                         , nbPredClass, self._dY_pred_index)
        if self._lMember:
            self._a = functools.reduce(np.add, (o._a for o in self._lMember))
            return self.getNormalizedShape()
        else:
            self._a = np.zeros((nbTrueClass,nbTrueClass))
            return (nbTrueClass, nbTrueClass)
    
    @classmethod
    def _weighted_Avg_SD(cls, a, axis=None, weights=None):
        avg = np.average(a, axis=axis, weights=weights)
        variance = np.average((a-avg)**2, axis=axis, weights=weights)
        return (avg, np.sqrt(variance))
        
    def meanAccuracy(self):
        """
        return the average and standard deviation of accuracies over th efamily members
        """
        aAcc = np.array([o.accuracy() for o in self._lMember])
        return np.average(aAcc), np.std(aAcc)
        
    def meanPRF(self, Y_true, bWeighted=False):
        """
        average precision, recall, f1 score for the given class
        and standard deviation of each
        return a tuple of triplets:
            (P, R, F1), (sdP, sdR, sdF1)
        """
        iRowTrue = self._dY_true_index[Y_true]
        iColTrue = self._dY_pred_index[Y_true]
        aPRFS = np.array([o._PRFS(iRowTrue, iColTrue) for o in self._lMember])
        aPRF = aPRFS[:,0:3]
        if bWeighted:
            weights = aPRFS[:,3]
            return self._weighted_Avg_SD(aPRF, axis=0, weights=weights)
        else:
            return (np.average(aPRF, axis=0), np.std(aPRF, axis=0))

    def getYTrueList(self): 
        return sorted(list(self.setY_true))
    def getYPredList(self): 
        return sorted(list(self.setY_pred.union(self.setY_true)))
    
    def getNormalizedShape(self):
        if self._bRotten:
            raise RectangularConfusionMatrixException("Family normalization required")
        return (len(self.getYTrueList()), len(self.getYPredList()))

    def confusion_classification_report(self, digits=3):
        if self._bRotten:
            raise RectangularConfusionMatrixException("Family normalization required")
  
        true_labels = self.getYTrueList()
        pred_labels = self.getYPredList()
        
        last_line_heading = 'avg / total'
    
        true_name_width = max(len(str(cn)) for cn in true_labels)
        pred_name_width = max(len(str(cn)) for cn in pred_labels)
        
        width = max(pred_name_width, len(last_line_heading), digits)
    
        headers = ["precision", "recall", "f1-score", "support"]
        fmt = '%% %ds' % width  # first column: class name
        fmt += '  '
        fmt += ' '.join(['% 9s' for _ in headers])
        fmt += '\n'
    
        headers = [""] + headers
        report = fmt % tuple(headers)
        report += '\n'
        eps=1e-8
        aPRFS = np.array([self.PRFS(_v) for _v in true_labels])
        for i, label in enumerate(true_labels):
            values = [true_labels[i]]
            for ii, v in enumerate(self.PRFS(label)):
                values += ["{0:0.{1}f}".format(v, digits)] if ii < 3 else ["%d"%v]
            #values += ["{0}".format(self.sum(label))]
            report += fmt % tuple(values)
    
        report += '\n'
    
        # compute averages
        values = [last_line_heading]
        for v in (np.average(aPRFS[:,0], weights=aPRFS[:,3]),
                  np.average(aPRFS[:,1], weights=aPRFS[:,3]),
                  np.average(aPRFS[:,2], weights=aPRFS[:,3])):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ['%d'%np.sum(aPRFS[:,3])]
        report += fmt % tuple(values)
        return report

    # -------------------------------------------------------------
    def report(self):
        return self.toString()
    
    def toString(self):
        """
        return a nicely indented test report, as TestReport does
        """
        if self._bRotten:
            raise RectangularConfusionMatrixException("Family normalization required")
        sSepBeg = "--- " +  time.asctime(time.gmtime(time.time())) + "---------------------------------"
        sTitle  = "TEST REPORT FOR"
        sSpace  = ""
        sSepEnd = "-"*len(sSepBeg)
        
        np_dFmt = np.get_printoptions()
        np.set_printoptions(threshold=1000*1000, linewidth=5000)
        a = - np.array(self.numpy()) # negative copy
        a[:, self._aColIndex] = -a[:, self._aColIndex]
        s1 = str(a)
        np.set_printoptions(np_dFmt)
        def nbdigit(n):
            if n == 0: return 1
            i = 1+int(math.log10(abs(n)))
            # if n < 0: i += 1   #ignore the minus sign
            return i    
        # count of invented label occurences
        iSum = self.sum()
        invented = iSum -self.numpy()[:, self._aColIndex].sum()
        
        iMaxLenTrue = max(len(str(v)) for v in self.getYTrueList())    
        iMaxLenPred = max(iMaxLenTrue, max(len(str(v)) for v in self.getYPredList()))
        lsLine = s1.split("\n")    
        sFmtTrue = "%%%ds  %%s"%iMaxLenTrue     #producing something like "%20s  %s"
        lY_true = self.getYTrueList()
        assert len(lsLine) == len(lY_true), "Internal error: expected one line per class name: %s"%s1
        s0 = " " +" "*(2+iMaxLenTrue)
        if False:
            lNbDigitByCol = [max(nbdigit(min(a[:,i])), nbdigit(max(a[:,i]))) for i in range(a.shape[1])] 
            for (n, s) in zip(lNbDigitByCol, self.getYPredList()):
                sFmtCol = " %%%ds" % (1+n)
                s0 = s0 + sFmtCol % s
        else:
            nbDigitCol = max(nbdigit(a.min()), nbdigit(a.max()))
            sFmtCol = "%%%ds" % (1+nbDigitCol)
            for s in self.getYPredList():
                s0 = s0 + sFmtCol % s
            
        s1 = "\n".join( [sFmtTrue%(sLabel, sLine) for (sLabel, sLine) in zip(lY_true, lsLine)])    

        s3 = "(unweighted) Accuracy score = %.2f %%    trace=%d  sum=%d   invented=%d   (%.1f %%)"% (
            100*self.accuracy(), self.trace(), iSum, invented, 100*invented/(1e-9+iSum))
        
        sReport = """%(space)s%(sSepBeg)s 
%(space)s%(sTitle)s: %(name)s

%(space)s  Line=True class, column=Prediction, - for invented labels
%(s0)s
%(s1)s

%(space)s%(s3)s

%(s2)s
%(space)s%(sSepEnd)s
""" % {"space":sSpace, "sSepBeg":sSepBeg
       , "sTitle":sTitle, "name":self.name
       , "s0":s0
       , "s1":s1
       , "s2":self.confusion_classification_report()
       , "s3":s3, "sSepEnd":sSepEnd}    

        return sReport
    
    def __str__(self):
        """
        return a nicely formatted string containing all the info of this test report object
        """
        return self.toString()

    # -------------------------------------------------------------
    def __len__(self):        return len(self._lMember)
    def __getitem__(self, i): return self._lMember[i]
    def __delitem__(self, i): del self._lMember[i]

    def _seen(self, Y_true, Y_pred):
        """
        whenever we see a predicted label, we include it in the list of possible predicted labels
        """
        self._bRotten = True
        self.setY_true.add(Y_true)
        self.setY_pred.add(Y_pred)
 
    def _seenIter(self, lY_true, lY_pred):
        """
        whenever we see a predicted label, we include it in the list of possible predicted labels
        """
        self._bRotten = True
        self.setY_true.update(lY_true)
        self.setY_pred.update(lY_pred)


#===============================================================================    
def test_RectangularConfusionFamily_1():
    import pytest 
    import math
    
    f = RectangularConfusionMatrixFamily("tyty")
    with pytest.raises(RectangularConfusionMatrixException):
        f.accuracy()
    with pytest.raises(RectangularConfusionMatrixException):
        f.PRFS(2)
    
    with pytest.raises(IndexError):
        f[2]
    assert f.normalize() == (0,0), f.normalize()
                                                   
    cf = f.createRectangularConfusionMatrix()
    cf.observe(1, 1)
    cf.observe(1, 2)
    with pytest.raises(RectangularConfusionMatrixException):
        cf.accuracy()           
    with pytest.raises(RectangularConfusionMatrixException):
        cf.PRFS(2)           
    
    f.normalize()
    assert (f[0].numpy() == np.array([  [1, 1]
                                      ])).all()
    assert cf.sum() == 2, cf.sum()          
    assert cf.trace() == 1, cf.trace()          
    assert (cf.accuracy() - 0.5) <= 0.001, cf.accuracy()          
                             
    cf.observe(2, 22)
    f.normalize()
    assert (f[0].numpy() == np.array([  [1, 1, 0]
                                      , [0, 0, 1]])).all()
    assert (cf.accuracy() - 1/3) <= 0.001, cf.accuracy()          
                                                   
    f.normalize()
    assert f.getNormalizedShape() == (2,3)
    
    cf.observe(1, 1)
    f.normalize()
    assert len(f) == 1
    assert (f[0].numpy() == np.array([  [2, 1, 0]
                                      , [0, 0, 1]])).all(), f[0].numpy()
    assert (cf.accuracy() - 1/2) <= 0.001, cf.accuracy()          
    
    cf.observe(1, 0)
    with pytest.raises(RectangularConfusionMatrixException):
        assert f.getNormalizedShape() == (2,4)
    f.normalize()
    assert f.getNormalizedShape() == (2,4)
    
    assert (f[0].numpy() == np.array([  [1, 2, 1, 0]
                                      , [0, 0, 0, 1]])).all()
    assert (cf.accuracy() - 2/5) <= 0.001, cf.accuracy()          
    
    cf2 = f.createRectangularConfusionMatrix()
    with pytest.raises(RectangularConfusionMatrixException):
        assert (cf.accuracy() - 2/5) <= 0.001, cf.accuracy()    
    
    f.normalize()        
    assert (cf.accuracy() - 2/5) <= 0.001, cf.accuracy()          
    assert (cf2.sum() == 0), cf2.sum()    
    cf2.accuracy()      
    cf2.observe(2,2)
    
    assert f.getYTrueList() == [1,2]
    f.normalize()        
    assert f.getYPredList() == [0,1,2,22]
    assert f.getNormalizedShape() == (2,4)
    
    f.normalize()        
    assert (cf.accuracy() - 2/5) <= 0.001, cf.accuracy()          
    assert (cf2.accuracy() - 1)  <= 0.001, cf2.accuracy()          


    assert (f[0].numpy() == np.array([  [1, 2, 1, 0]
                                      , [0, 0, 0, 1]])).all()
    assert (f[1].numpy() == np.array([  [0, 0, 0, 0]
                                      , [0, 0, 1, 0]])).all()
    assert (f.accuracy() - 3/6) <= 0.001, f.accuracy()  
    fAcc, fStd = f.meanAccuracy()
    assert (fAcc - 7/5) <= 0.001, f.meanAccuracy()  
    assert (fStd - 0.424264069) <= 0.001, f.meanAccuracy()  
    
    cf3 = f.createRectangularConfusionMatrix()
    cf3.observe(np.arange(1,3), np.arange(10,12))
    
    assert f.getYTrueList() == [1,2]
    assert f.getYPredList() == [0,1,2,10,11,22]
    f.normalize()        
    assert f.getYPredList() == [0,1,2,10,11,22], f.getYPredList()
    assert f.getNormalizedShape() == (2,6)
    
    assert (cf.accuracy()  - 2/5) <= 0.001, cf.accuracy()          
    assert (cf2.accuracy() - 1/1) <= 0.001, cf2.accuracy()          
    assert (cf3.accuracy() - 0/2) <= 0.001, cf3.accuracy()          
    
    assert (f.accuracy() - 3/8)  <= 0.001, f.accuracy()                     # 0.38
    fAcc, fStd = f.meanAccuracy()
    assert (fAcc - 14/10/3)  <= 0.001, f.meanAccuracy()   # 0.47     
    assert (fStd - 0.503322296) <= 0.001, f.meanAccuracy()  
    
    with pytest.raises(RectangularConfusionMatrixException):
        f.PRFS(3)

    assert (f[0].numpy() == np.array([  [1, 2, 1, 0, 0, 0]
                                      , [0, 0, 0, 0, 0, 1]])).all(), f[0].numpy()
    def ok(f,ref): return abs(f-ref) < 0.001
    def ok3(tf,tref):
        return (ok(tf[0],tref[0]) and ok(tf[1],tref[1]) and ok(tf[2],tref[2]))
    P,R,F,S = cf.PRFS(1)
    assert ok(P, 1) and ok(R, 0.5) and ok(F, 0.6666), (P,R,F)
    assert ok3((P,R,F), (1, 0.5, 0.666))
    assert ok3(cf.PRFS(1), (1, 0.5, 0.666))
    assert ok3(cf.PRFS(2), (0, 0, 0))

    assert ok3(cf2.PRFS(1), (0,0,0))
    assert ok3(cf2.PRFS(2), (1,1,1))
    
    assert ok3(cf3.PRFS(1), (0,0,0))
    assert ok3(cf3.PRFS(2), (0,0,0))

    assert ok3(f.PRFS(1), (1   ,0.4    ,0.571428571)) , f.PRFS(1)
    assert ok3(f.PRFS(2), (0.5 ,0.3333 ,0.4))         , f.PRFS(2)

    mPRF, sdPRF = f.meanPRF(1)
    assert ok3(mPRF, (0.3333, 0.5/3, 0.22222))    
    mPRF, sdPRF = f.meanPRF(2)
    assert ok3(mPRF, (0.3333, 0.3333, 0.3333))    
    
    
    print(f.numpy())
    
    print(f)

def test_RectangularConfusionFamily_2():
    import pytest 
    import math
    
    f = RectangularConfusionMatrixFamily("Node task 12")
    with pytest.raises(RectangularConfusionMatrixException):
        f.accuracy()
    cf = f.createRectangularConfusionMatrix()
    cf.observe(1, 1)
    f.normalize()
    print(f)
    
if __name__ == "__main__":
    test_RectangularConfusionFamily_1()
    test_RectangularConfusionFamily_2()
    
    
    
        