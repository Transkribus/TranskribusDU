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
        else:
            print "\t terminal", self.getPattern()
    
    
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
                
    
    
    def findBestMatch(self,lRegCuts,lCuts):
        """
            find the best solution assuming reg=x
            dynamic programing (viterbi path)
            
            
            
        """
        from sklearn.preprocessing import normalize
        def buildObs(lRegCuts,lCuts):
            N=len(lRegCuts)+1
            obs = np.zeros((N,len(lCuts)), dtype=np.float16)
            for i,refx in enumerate(lRegCuts):
                for j,x in enumerate(lCuts):
                    # are features compatible?
                    if x.getType() == refx.getType():
                        ## numerical 
#                         print x, refx, abs(x.getValue()-refx.getValue()) , refx.getTH()
                        if abs(x.getValue()-refx.getValue()) < refx.getTH():
                            obs[i,j]=  x.getWeight() * ( refx.getTH() - ( abs(x.getValue()-refx.getValue()))) / refx.getTH()
#                             print x,refx, obs[i,j], ( refx.getTH() - ( abs(x.getValue()-refx.getValue()))) / refx.getTH(), x.getWeight(), refx.getTH(),  ( refx.getTH() - ( abs(x.getValue()-refx.getValue()))),abs(x.getValue()-refx.getValue()) 
                        elif abs(x.getValue()-refx.getValue()) < (refx.getTH() * 2 ):
                            obs[i,j]=  x.getWeight() * ( (refx.getTH() * 2) - ( abs(x.getValue()-refx.getValue()))) / ( refx.getTH() * 2 )
                        ## STRING
                        ### TODO
                        else:
                            # go to empty state
                            obs[-1,j] = 1.0
                        if np.isinf(obs[i,j]):
                            obs[i,j]=64000
                        if np.isnan(obs[i,j]):
                            obs[i,j]=10e-3
                    else:
                        obs[-1,j] = 1.0
#                     print x,refx, obs[i,j]
            if np.amax(obs) != 0:
                # elt with no feature obs=0
                return obs / np.amax(obs)
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
#         print map(lambda x:(x,x.getCanonical().getWeight()),lCuts)
#         print states
#         for i,si in enumerate(states):
#             print lCuts[si],si
#             print obs[si,:]
        
        # return the best alignment with template
        return states, score                
                
    def computeScore(self,patLen,lReg,lCuts):
        """
            it seems better not to use canonical: thus score better reflects the page 
            
            also for REF 130  129 is better than 150
        """
#         print lReg
#         print map(lambda (r,x):x.getWeight(),lReg)
#         print lCuts
#         print map(lambda x:x.getWeight(),lCuts)
        fFound= 1.0 * sum(map(lambda (r,x):x.getWeight(),lReg))

        fTotal = 1.0 * sum(map(lambda x:x.getWeight(),lCuts))
#         print "# match:",len(set(map(lambda (r,x):r,lReg))), patLen,fFound, fTotal
        # how many of the lreg found:
        ff= 1.0*len(set(map(lambda (r,x):r,lReg)))/patLen
        return  ff*(fFound/fTotal)
        
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
        ll=kRef.keys()
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
#         lobjectFeatures = anobject._fullFeatures
#         print "?",anobject, lobjectFeatures
        # empty object
        if lobjectFeatures == []:
            return None,None,-1
        
#         print self.getPattern(), lobjectFeatures
        self.getPattern().sort(key=lambda x:x.getValue())
#         print self.getPattern(), anobject, lobjectFeatures
        bestReg, curScore = self.findBestMatch(self.getPattern(),lobjectFeatures)
#         print bestReg
        ltmp = self.getPattern()[:]
        ltmp.append('EMPTY')
        lMissingIndex = filter(lambda x: x not in bestReg, range(0,len(self.getPattern())+1))
        lMissing = np.array(ltmp)[lMissingIndex].tolist()
        lMissing = filter(lambda x: x!= 'EMPTY',lMissing)
        result = np.array(ltmp)[bestReg].tolist()
        lFinres= filter(lambda (x,y): x!= 'EMPTY',zip(result,lobjectFeatures))
        if lFinres != []:
            score1 = self.computeScore(len(self.getPattern()),lFinres,lobjectFeatures)
            lFinres =  self.selectBestUniqueMatch(lFinres)
        # for estimating missing?
#         self.selectBestAnchor(lFinres) 
            return lFinres,lMissing,score1
        else:
            return None,None,-1
        
















    