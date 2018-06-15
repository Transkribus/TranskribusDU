# -*- coding: utf-8 -*-
"""

    Hervé Déjean
    cpy Xerox 2016
    
    READ project
    
    treeTemplate Class
    
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import numpy as np
from scipy.optimize import linear_sum_assignment

from .templateClass import templateClass
from spm.frechet import frechetDist

class treeTemplateClass(templateClass):
    """
        a template structured as a tree
            
    """
    
    gID = 1
    def __init__(self):
        templateClass.__init__(self)
        
        self._templateType = None
        self.parent=None # link to parent template 
        self.pattern = None
        self.lChildren = None
        self.bKleenePlus= False
        
        
    def __str__(self):return 'treeTemplate:%s'%(self.getPattern())
    def __repr__(self):return 'treeTemplate:%s'%(self.getPattern())

    def getPattern(self): return self.pattern
    def setPattern(self,p): self.pattern = p 
 
    def setType(self,t): self._templateType = t
    def getType(self): return self._templateType
    
    def getParent(self): return self.parent
    def setParent(self,p): self.parent= p
    
    def addChild(self,c): 
        try:self.lChildren.append(c)
        except AttributeError: self.lChildren = [c]
    def getChildren(self): return self.lChildren
    
    def print_(self,level=1):
        print  ('*'* level,self.getPattern(), self.getChildren())
        if self.getChildren():
            for child in self.getChildren():
                child.print_(level+1)
        else:
            print ("\t terminal", self.getPattern())
    
    
    def buildTreeFromPattern(self,pattern):
        """
             create a tree structure corresponding to pattern
        """
        self.setPattern(pattern)
        if isinstance(pattern,list):
#             if  'list' not in map(lambda x:type(x).__name__, pattern):
#                 print "real terminal??", pattern            
            for child in self.getPattern():
#                 print 'child',child
                ctemplate  = treeTemplateClass()
#                 ctemplate.setPattern(child)
                ctemplate.buildTreeFromPattern(child)
                self.addChild(ctemplate)
                ctemplate.setParent(self)
#             print '--',self.getChildren()
        else:
            pass

    
    def getTerminalTemplates(self):
        """
            find the terminals templates 
        """
        # if a flat list of patterns: terminal
        if self.getChildren() is None:
            return [self]
        elif  'list' not in [type(x.getPattern()).__name__ for x in self.getChildren()]:
            return [self]
        else:
            lRes=[]
            for child in self.getChildren():
                lRes.extend(child.getTerminalTemplates())
            return lRes
            
    def findTemplatePartFromPattern(self,pattern):
        """
            convert a recursive list (pattern) into treeTemplate objects
        """
#         print self, pattern
        if pattern  == self.getPattern():
            return self
        else:
            if self.getChildren() is not None:
                ##??
                if 'list' not in [type(x.getPattern()).__name__ for x in self.getChildren()]:
#                 if  'list' not in map(lambda x:type(x.getPattern()).__name__, self.getChildren()):
#                     print '\t===', self, self.getChildren(), map(lambda x:x.getPattern(), self.getChildren())
                    for c in self.getChildren():
                        if c == pattern:
                            return c                
                else:
                    for child in self.getChildren():
                        res = child.findTemplatePartFromPattern(pattern)
                        if res is not None:
                            return res
            else: return None
        return None  
                
    
    def findBestMatch2(self,lRegCuts,lCuts):
        """
            best match using hungarian
            add a threshold!
        """
        cost_matrix=np.zeros((len(lRegCuts),len(lCuts)),dtype=float)
        
        for a,refx in enumerate(lRegCuts):
            for b,x in enumerate(lCuts):
                dist = refx.getDistance(x)
                cost_matrix[a,b]=dist
            
        r1,r2 = linear_sum_assignment(cost_matrix)
        
        ltobeDel=[]
        for a,i in enumerate(r2):
            #if cost is too high: cut the assignment?
#             print (a,i,lRegCuts[a],lCuts[i], cost_matrix[a,i], 'deleted')
            if cost_matrix[a,i] > 100:
                ltobeDel.append(a)
        r2 = np.delete(r2,ltobeDel)
        r2 = np.delete(r1,ltobeDel)
#         print ('\t',r1,r2,lRegCuts,lCuts)
        # score Fréchet distance etween two mapped sequences    
        return r1,r2,None
    
    def findBestMatch(self,lRegCuts,lCuts):
        """
            find the best solution assuming reg=x
            dynamic programing (viterbi path)
            
            
            
        """
        def buildObs(lRegCuts,lCuts):
            N=len(lRegCuts)+1
            obs = np.zeros((N,len(lCuts)), dtype=np.float16)
            for i,refx in enumerate(lRegCuts):
                for j,x in enumerate(lCuts):
                    # are features compatible?
                    if x.getType() == refx.getType():
                        ## numerical 
#                         print x, refx, abs(x.getValue()-refx.getValue()) , refx.getTH()
#                         print x, refx, "dist=%s"%x.getDistance(refx)
                        dist =  x.getDistance(refx)
                        if dist < refx.getTH():
                            obs[i,j]=  x.getWeight() * ( refx.getTH() - ( dist)) / refx.getTH()
#                             print x,refx, obs[i,j], ( refx.getTH() - ( abs(x.getValue()-refx.getValue()))) / refx.getTH(), x.getWeight(), refx.getTH(),  ( refx.getTH() - ( abs(x.getValue()-refx.getValue()))),abs(x.getValue()-refx.getValue()) 
                        elif dist < (refx.getTH() * 2 ):
                            obs[i,j]=  x.getWeight() *  ( (refx.getTH() * 2) - ( dist)) / ( refx.getTH() * 2 )
                        ## STRING
                        ### TODO
                        else:
                            # go to empty state
                            obs[-1,j] = 1.0
                        if np.isinf(obs[i,j]):
                            obs[i,j] = min(refx.getWeight(),64000)
                        if np.isnan(obs[i,j]):
                            obs[i,j]=10e-3
#                         print x, refx, refx.getWeight(), obs[i,j]
                    else:
                        obs[-1,j] = 1.0
#                     print x,refx, obs[i,j], refx.getWeight()
            if np.amax(obs) != 0:
                # elt with no feature obs=0
                return obs# / np.amax(obs)
            else:
                return obs

        import spm.viterbi as viterbi
        
        # add 'missing' state 
        N =len(lRegCuts)+1
        transProb = np.zeros((N,N), dtype = np.float16)
        for i  in range(N-1):
            transProb[i,i]=1.0
            transProb[i,i+1]=0.75
            try:transProb[i,i+2]=0.5
            except IndexError:pass  
        transProb[:,-1,]=1.0 
        transProb[-1,:]=1.0  
        
#         print transProb  
#         print transProb/transProb.sum(axis=1)[:,None]
        initialProb = np.ones(N)
        initialProb = np.reshape(initialProb,(N,1))
        
                
        obs = buildObs(lRegCuts,lCuts)
#         print lCuts 
#         print obs
        d = viterbi.Decoder(initialProb, transProb, obs)
        states,score =  d.Decode(np.arange(len(lCuts)))
#         print "dec",score, states 
#         print map(lambda x:(x,x.getCanonical().getWeight()),lCuts)
        print (states, type(states[0]))
#         for i,si in enumerate(states):
#             print lCuts[si],si
#             print obs[si,:]
        
        # return the best alignment with template
        return states, score                
             
             
    def computeScore(self,p,q):
        d =frechetDist(list(map(lambda x:(x.getValue(),0),p)),list(map(lambda x:(x.getValue(),0),q)))
        print (d,list(map(lambda x:(x.getValue(),0),p)),list(map(lambda x:x.getValue(),q)))
        if d == 0:
            return 1
        return 1/(frechetDist(list(map(lambda x:(x.getValue(),0),p)),list(map(lambda x:(x.getValue(),0),q))))
        
                
    def computeScoreold(self,patLen,lReg,lMissed,lCuts):
        """
            it seems better not to use canonical: thus score better reflects the page 
            
            also for REF 130  129 is better than 150
        """
#         print lReg
#         print map(lambda (r,x):x.getWeight(),lReg)
#         print lCuts
#         print map(lambda x:x.getWeight(),lCuts)
        fFound= 1.0 * sum(list(map(lambda rx:rx[0].getWeight(),lReg)))
        fTotal = 1.0 * sum(list(map(lambda x:x.getWeight(),lCuts)))
        fMissed = 1.0 * sum(list(map(lambda x:x.getWeight(),lMissed)))

        dist = sum(list(map(lambda xy: abs(xy[0].getValue()-xy[1].getValue()),lReg)))
        
#         print map(lambda (x,y): abs(x.getValue()-y.getValue()),lReg)
        if dist ==0:dist=1.0
#         print "# match:",len(set(map(lambda (r,x):r,lReg))), patLen,fFound, fTotal, dist 
        # how many of the lreg found:
        ff= 1.0*len(set(list(map(lambda rx:rx[0],lReg))))/patLen
        assert dist/patLen != 0
#         ff= 1/(dist)
#         ff=1.0
#         print ff, dist,fFound,fTotal, fMissed ,fFound/(fTotal + fMissed),ff*(fFound/(fTotal + fMissed))
        return  ff*(fFound/(fTotal + fMissed))
        
    def selectBestUniqueMatch(self,lFinres):
        """
            use weight to select one match
             [('x=46.0', 'x=46.0'), ('x=123.0', 'x=111.0'), ('x=334.0', 'x=282.0'), ('x=334.0', 'x=384.0'), ('x=453.0', 'x=453.0')]
            [('x=10.0', 69.54), ('x=46.0', 327.46632), ('x=111.0', 316.68936), ('x=282.0', 87.54), ('x=334.0', 335.23848000000004), ('x=384.0', 180.14136), ('x=453.0', 215.22768)]
        """
        kRef={}
        for ref,cut in lFinres:
            try:kRef[ref].append(cut)
            except KeyError: kRef[ref]=[cut]
        lUniqMatch=[]
        ll=list(kRef.keys())
        ll.sort(key=lambda x:x.getValue())
        for mykey in ll:
            kRef[mykey].sort(key=lambda x:x.getWeight(),reverse=True)
            lUniqMatch.append((mykey, kRef[mykey][0]))
            
        return lUniqMatch
              
    def registration(self,anobject):
        """
            'register': match  the model to an object
            can only a terminal template 
        """
        lobjectFeatures = anobject.lFeatureForParsing
#         lobjectFeatures = anobject._fullFeaturesx
#         print "?",anobject, lobjectFeatures, self
        # empty object
        if lobjectFeatures == []:
            return None,None,-1
        
#         print self.getPattern(), lobjectFeatures
        try:  self.getPattern().sort(key=lambda x:x.getValue())
        except: pass ## P3 < to be defined for featureObject
#         print self.getPattern(), anobject, lobjectFeatures
        foundReg,bestReg, _ = self.findBestMatch2(self.getPattern(), lobjectFeatures)

#         bestReg, _ = self.findBestMatch(self.getPattern(), lobjectFeatures)
#         print bestReg, curScore
        if bestReg != []:
            lFinres = list(zip([(lobjectFeatures[i]) for i in bestReg], ([self.getPattern()[i] for i in foundReg])))
#             print (lFinres)
#             score1 = self.computeScore(len(self.getPattern()), lFinres, [],lobjectFeatures)
            score1 = self.computeScore([self.getPattern()[i] for i in bestReg], lobjectFeatures)

            return lFinres,None,score1
        else:
            return None,None,-1
        
















    