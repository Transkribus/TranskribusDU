# -*- coding: utf-8 -*-
"""

    Vertical Zones template class: virutal template representinf a 'canonical' layout having vertical zones
        Basic example: a n-columns page
    
    Hervé Déjean
    cpy Xerox 2016
    
    READ project
    
"""
from templateClass import templateClass
from numpy import dtype
import numpy as np
class verticalZonestemplateClass(templateClass):
    """
        a class  for vertical layout  (like columns) 
    """
    gID = 1
    def __init__(self):
        
        templateClass.__init__(self)

        self.ID = verticalZonestemplateClass.gID 
        verticalZonestemplateClass.gID += 1
        self._TH = 20
        self.bMirrored= False
        # list of X1 positions for the cts
        self.lX= []
        
        #list of the zone width
        self.lWidth= []
        
        # getSetofFeatures used for using this template
        # in templateClass
#         self.pattern = None
    
    def __str__(self): return 'VZTemp:'+ str(self.pattern)
    def __repr__(self): return 'VZTemp:'+ str(self.pattern)

    def __hash__(self):
        return hash(self.ID)
        
    def setPattern(self,p): 
        """
            an itemset
        """
        
        self.pattern = p
        self.bMirrored = len(self.pattern) == 2
        
        lSubWith=[]
        prev = 0
        for item in self.pattern:
            self.addXCut(item)
            w= item.getValue() - prev
            if item.getValue() not in self.lWidth:
                lSubWith.append(w)
            prev= item.getValue()
        self.lWidth.append(lSubWith)
    
    def setTH(self,t): self._TH = t
    def getTH(self): return self._TH
    
    def setXCuts(self,lx): self.lX = lx
    
    def addXCut(self,x): 
        if x not in self.lX:
            self.lX.append(x)
            self.lX.sort()    
    
    def getXCuts(self): return self.lX
    
    
    def isRegularGrid(self):
        """
            is the template a regular grid (N columns= page justification)
        """
        
        return False
        
    
    def findBestMatch(self,calibration,lRegCuts,lCuts):
        """
            find the best solution assuming reg=x
            dynamic programing (viterbi path)
        """
        
        def buildObs(calibration,lRegCuts,lCuts):
            N=len(lRegCuts)+1
            obs = np.zeros((N,len(lCuts)), dtype=np.float32)+1.0/(20*10)
            for i,refx in enumerate(lRegCuts):
                for j,x in enumerate(lCuts):
                    # "sigmpoid" if too larong:empty state
                    try:
#                         print refx, x, (x.getValue()-calibration), abs((x.getValue()-calibration)-refx.getValue())
                        if abs((x.getValue()-calibration)-refx.getValue()) < 20:
#                             print "\t",refx, x, (x.getValue()-calibration)
                            obs[i,j]= 1.0/abs(x.getValue()-calibration-refx.getValue())
                        else:
                            obs[-1,j] = 1.0/(20+1)
                    except ZeroDivisionError: obs[i,j] = 1.0
            return obs

        import numpy as np
        import spm.viterbi as viterbi
        
        # add 'missing' state 
        N =len(lRegCuts)+1
        transProb = np.zeros((N,N), dtype = np.float32)
        for i  in range(N-1):
#             for j in range(i,N):
                transProb[i,i+1]=1.0/(N-i)
        transProb[:,-1,]=1.0/(N-i)
        transProb[-1,:]=1.0/(N-i)        
        
        initialProb = np.ones(N)
        initialProb = np.reshape(initialProb,(N,1))
        
        obs = buildObs(calibration,lRegCuts,lCuts)

        d = viterbi.Decoder(initialProb, transProb, obs)
        states,score =  d.Decode(np.arange(len(lCuts)))
#         result = np.array(lCuts)[states].tolist()
        
        # return the best alignement with template
        return states, score
        
        
    def selectBestAnchor(self,lCuts):
        """
            select the best anchor and use width for defining the other?
        """
        lFinal=[]
        fShort = 9e9
        bestElt = None
        for i,(x,y) in enumerate(lCuts):
            if abs(x.getValue() - y.getValue()) < fShort:
                bestElt=(x,y)
                fShort = abs(x.getValue() - y.getValue()) 
        
        print 'BEST', bestElt
            
        
    def selectBestCandidat(self,lCuts):
        """
            if several x are selected for a 'state': take the nearest one
            possible improvement: consider width
        """
        lFinal=[]
        dBest = {}
        for x,y in lCuts:
            try:
                if abs(x.getValue() - dBest[x].getValue()) > abs(x.getValue() - y.getValue()):
                    dBest[x]=y
            except KeyError:
                dBest[x]=y
        for x,y in lCuts:
            lFinal.append((x,dBest[x]))
        return lFinal
            
    def registration(self,pageObject):
        """
            using lCuts (and width) for positioning the page
            return the registered values 
        """
        if pageObject.lf_XCut == []:
            return None,None,-1
        
        # define lwidth for the page        
        pageObject.lf_XCut.sort(key=lambda x:x.getValue())
#         print  pageObject, pageObject.lf_XCut
#         print self.getXCuts()
        
        ## define a set of interesting calibration
        lCalibration= [0,-25,25,-50,50]
        
        
        bestScore=0
        bestReg=None
        for calibration in lCalibration:
            reg, curScore = self.findBestMatch(calibration,self.getXCuts(),pageObject.lf_XCut)
#             print calibration, reg, curScore
            if curScore > bestScore:
                bestReg=reg;bestScore=curScore
        
#         print '\t',bestReg
        ltmp = self.getXCuts()[:]
        ltmp.append('EMPTY')
        lMissingIndex = filter(lambda x: x not in bestReg, range(0,len(self.getXCuts())+1))
        lMissing = np.array(ltmp)[lMissingIndex].tolist()
        lMissing = filter(lambda x: x!= 'EMPTY',lMissing)
        result = np.array(ltmp)[bestReg].tolist()
        
        lFinres= filter(lambda (x,y): x!= 'EMPTY',zip(result,pageObject.lf_XCut))

        lFinres =  self.selectBestCandidat(lFinres)
        print lFinres
#         self.selectBestAnchor(lFinres)
        if len(lFinres) >=  len(self.getXCuts()) * 0.50:
            return lFinres,lMissing,bestScore
        else:
            return None,None,-1
        
    
    
    
    
    
    