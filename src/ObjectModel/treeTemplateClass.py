# -*- coding: utf-8 -*-
"""

    Hervé Déjean
    cpy Xerox 2016
    
    READ project
    
    treeTemplate Class
    
"""

from templateClass import templateClass
import numpy as np

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
        print  '*'* level,self.getPattern(), self.getChildren()
        if self.getChildren():
            for child in self.getChildren():
                child.print_(level+1)
    
    
    def buildTreeFromPattern(self,pattern):
        """
             create a tree structure corresponding to pattern
        """
        self.setPattern(pattern)
#         print 'creation:',self.getPattern()
        if isinstance(pattern,list):
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
            #terminal    
#     def buildTreeFromPattern(self,pattern):
#         """
#              create a tree structure corresponding to pattern
#         """
#         self.setPattern(pattern)
# #         print 'creation:',self.getPattern()
#         if isinstance(pattern,list):
#             if self.getChildren() is not None:
#                 if  'list' not in map(lambda x:type(x.getPattern()).__name__, self.getChildren()):
#                     #terminal
#                     ctemplate  = treeTemplateClass()
#                     ctemplate.setPattern(pattern)
#                     self.addChild(ctemplate)
#                     ctemplate.setParent(self)                
#                 else:
#                     for child in self.getPattern():
#         #                 print 'child',child
#                         ctemplate  = treeTemplateClass()
#         #                 ctemplate.setPattern(child)
#                         ctemplate.buildTreeFromPattern(child)
#                         self.addChild(ctemplate)
#                         ctemplate.setParent(self)
#         #             print '--',self.getChildren()
#         else:
#             pass
#             #terminal
            
    
    def getTerminalTemplates(self):
        """
            find the terminals templates 
        """
        # if a flat list of patterns: terminal
        if self.getChildren() is None:
            return [self]
        elif  'list' not in map(lambda x:type(x.getPattern()).__name__, self.getChildren()):
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
                if  'list' not in map(lambda x:type(x.getPattern()).__name__, self.getChildren()):
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
                
    
    
    def findBestMatch(self,calibration,lRegCuts,lCuts):
        """
            find the best solution assuming reg=x
            dynamic programing (viterbi path)
            
            score needs to be normalized (0,1)
        """
        def buildObs(calibration,lRegCuts,lCuts):
            N=len(lRegCuts)+1
            obs = np.zeros((N,len(lCuts)), dtype=np.float16)+ 0.0
            for i,refx in enumerate(lRegCuts):
                for j,x in enumerate(lCuts):
                    # are features compatible?
                    if x.getType() == refx.getType():
                        ## numerical 
                        if abs((x.getValue()-calibration)-refx.getValue()) < refx.getTH():
                            obs[i,j]=  x.getCanonical().getWeight() * ( refx.getTH() - ( abs(x.getValue()-calibration-refx.getValue()))) / refx.getTH()
                        elif abs((x.getValue()-calibration)-refx.getValue()) < (refx.getTH() * 2 ):
                            obs[i,j]=  x.getCanonical().getWeight() * ( (refx.getTH() * 2) - ( abs(x.getValue()-calibration-refx.getValue()))) / ( refx.getTH() * 2 )
                        
                        ## STRING
                        ### TODO
                        else:
                            # go to empty state
                            obs[-1,j] = 1.0
                        if np.isinf(obs[i,j]):
    #                             print i,j,score
                            obs[i,j]=64000
                        if np.isnan(obs[i,j]):
    #                             print i,j,score
                            obs[i,j]=10e-3
                    else:
                        obs[-1,j] = 1.0
#             print obs / np.amax(obs)                                                        
            return obs / np.amax(obs)

        import spm.viterbi as viterbi
        
        # add 'missing' state 
        N =len(lRegCuts)+1
        transProb = np.zeros((N,N), dtype = np.float16)
        for i  in range(N-1):
#             for j in range(i,N):
                transProb[i,i+1]=1.0 #/(N-i)
        transProb[:,-1,]=1.0 #/(N)
        transProb[-1,:]=1.0  #/(N)        
        initialProb = np.ones(N)
        initialProb = np.reshape(initialProb,(N,1))

                
        obs = buildObs(calibration,lRegCuts,lCuts)
        d = viterbi.Decoder(initialProb, transProb, obs)
        states,score =  d.Decode(np.arange(len(lCuts)))
#         print map(lambda x:(x,x.getCanonical().getWeight()),lCuts)
#         print states
#         for i,si in enumerate(states):
#             print lCuts[si],si
#             print obs[si,:]
        
        # return the best alignment with template
        return states, score                
                
    def computeScore(self,lReg,lCuts):
        fFound= 1.0 * sum(map(lambda (r,x):x.getCanonical().getWeight(),lReg))
        fTotal = 1.0 * sum(map(lambda x:x.getCanonical().getWeight(),lCuts))
        return  fFound/fTotal
                    
    def registration(self,anobject):
        """
            'register': match  the model to an object
            can only a terminal template 
        """
        
        lobjectFeatures = anobject.lX
#         print self.getPattern(), lobjectFeatures
        
        bestReg, curScore = self.findBestMatch(0,self.getPattern(),lobjectFeatures)
        
        ltmp = self.getPattern()[:]
        ltmp.append('EMPTY')
        lMissingIndex = filter(lambda x: x not in bestReg, range(0,len(self.getPattern())+1))
        lMissing = np.array(ltmp)[lMissingIndex].tolist()
        lMissing = filter(lambda x: x!= 'EMPTY',lMissing)
        result = np.array(ltmp)[bestReg].tolist()
        lFinres= filter(lambda (x,y): x!= 'EMPTY',zip(result,anobject.lX))
#         print lFinres
        if lFinres != []:
            #lFinres =  self.selectBestCandidat(lFinres)
        # for estimating missing?
#         self.selectBestAnchor(lFinres) 
            return lFinres,lMissing,self.computeScore(lFinres, anobject.lX)
        else:
            return None,None,-1
        
















    