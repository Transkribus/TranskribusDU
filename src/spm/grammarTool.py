# -*- coding: utf-8 -*-
"""

    cpy Xerox 2016
    
    Hervé Déjean
    XRCE
    READ project     
    
"""


# Adjustement of the PYTHONPATH to include /.../DS/src
import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))


from feature import featureObject, sequenceOfFeatures


"""
 class: RULESEQElement
     correspond to the sequence objects with a specific __eq__: inclusion is used 
"""

class RULESEQElement():
    
    def __init__(self,node=None):
        self._node = node
        self._lBasicFeatures = sequenceOfFeatures()


    def __hash__(self):
        return hash(self.getSetofFeatures())
    
    def __repr__(self):
        try:
            return len(self._node.getContent()) , " [" + self._node.name+self._node.getContent()[:20]+']'
        except AttributeError:
            # no node: use features?
            return str(self.getSetofFeatures())


    def __eq__(self,other):
        
        for x in self.getSetofFeatures().getSequences():
            if x in other.getSetofFeatures().getSequences():
                return True
        return False
         

    def setFeatureFunction(self,foo,TH=2):
        """
            select featureFunction that have to be used
            
        """
        self._lStructures = []
        self._featureFunction = foo
        self._featureFunctionTH=TH    
        

    def getMyStructures(self): 
        return self._lStructures
    
    def addStructure(self,s):
        self._lStructures.append(s)
        
    def computeSetofFeatures(self,TH=90):
        
        try:
            self._lBasicFeatures
        except AttributeError:
            self._lBasicFeatures=None
            
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR

        self._featureFunction(self._featureFunctionTH)
        
    def addFeature(self,f):
        if f not in self.getSetofFeatures().getSequences():
            f.addNode(self)
            self._lBasicFeatures.addFeature(f)
                        
    def getSetofFeatures(self):
        
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR
        x= sequenceOfFeatures()
        return x            


class sequenceGrammar():        
    
        
    def generateGrammarRules(self, featureType,grammarTree,level):
        """
            convert list of features into rules (for earleyParser module)
        """
        
        from  earleyParser import Rule, Production
        
        lrules=[]        
        for featurerule in grammarTree:
            if type(featurerule).__name__ == 'list':
                lrules.append(self.generateGrammarRules(featureType,featurerule,level+1))
            else:
#                 feature = featureType()
#                 feature.setName(featurerule.getName())
#                 feature.setType(featurerule.getType())
#                 feature.setValue(featurerule.getValue())
#                 feature.setTH(featurerule.getTH())

                # simply use featurerule ? featureType useless?
                elt = RULESEQElement()
                elt.addFeature(featurerule)
                elt.computeSetofFeatures()
                
#                 gramRule = Rule(featurerule.getName() + featurerule.getStringValue(), Production(elt))
                gramRule = Rule(featurerule, Production(elt))

                lrules.append(gramRule)
                
        mainRuleName='s%d'%level
        MainRule=Rule(mainRuleName,Production(*lrules))            
        MainRule2 = Rule(mainRuleName+'+',Production(MainRule))
        MainRule2.add(Production(MainRule2,MainRule))

        return MainRule2



    
    def getHListFromNode(self,node,lElts,dInvertDict):
        """
            transform the hierarchical Node into a recursive list of index/elt
        """
        import re
        
        lChildOut= []
#         print type(node.value), node.value
        #if terminal: one index:
#         print "?",node.value.name, node.value.start_column,node.value.end_column.index-1
        
        # store if not kleenePlus
        try:dInvertDict[node.value.name].append((node.value.name,node.value.start_column.index,node.value.end_column.index-1))
        except KeyError: dInvertDict[node.value.name]=[(node.value.name,node.value.start_column.index,node.value.end_column.index-1)]        
        
#         if (node.value.end_column.index - node.value.start_column.index)==1:
#             lChildOut.append((node.value.name,lElts[node.value.start_column.index]))
#         else:
        if node.children != []:
            for child in node.children:
                lChildOut.extend((child.value.name,self.getHListFromNode(child,lElts,dInvertDict)))
        else:
            lChildOut.append((node.value.name,(node.value.name,node.value.start_column.index,node.value.end_column.index-1)))

        return lChildOut
            
    def simplifyRes(self,lRes,name):
        """
        LR parsing of a+ needs to be flatten!
        """
        lFlatenRes=[]
        for resname,lres in lRes:
            print 'ss',resname,lres
            if resname[-1] == '+':
                lFlatenRes.extend(self.simplifyRes(lres,name))
            else:
                lFlatenRes.append([resname,lres])
        return lFlatenRes
    
        
    def parseSequence(self,featureType, grammar, lSeqElement):
        """
            apply grammar to lSEqelement
            return 3-ple (list of covered elements, dict , parsing tree) 
        """
        from  earleyParser import  parse, build_trees
        
        self.bDEBUG = False
        
        if self.bDEBUG:print "\n\n===== GRAMMAR: %s =============" % (grammar)
        myGrammar = self.generateGrammarRules(featureType,grammar,0)
        ## if just one rule: a+  : need optimisation!
#         print myGrammar.productions
        # # here start with the first possible match?  (minus x for fussy?) 
        lList = lSeqElement
        if self.bDEBUG:print 'start parsing...'
        res, lT = parse(myGrammar, lList)
#         print 'res:',res
#         return 
        lInvertDict= []
        
        lParsings = []
        if res == 0 :
            if self.bDEBUG:print "full res",res
            lNodes = build_trees(lT[0])
            ## just take the first one  for the moment ??
            ## take the shortest one!!
            lTMP=[]
            for eachres in lNodes:#[:1]:
                dInvertDict={}
                if self.bDEBUG:eachres.print_()
                lOut = self.getHListFromNode(eachres,lList,dInvertDict)
#                 try:print dInvertDict['s0']
#                 except KeyError:pass
                lTMP.append((lList,dInvertDict,lOut))
            if lTMP != []:
                lTMP.sort(key=lambda (x,y,z):len(str(x)))
                lParsings.append(lTMP[0])
#                 print lTMP[0]
                
        else: 
            i = 0
            lResults = []
            while res != 0 and i < len(lList):
#                 print i,map(lambda x:x,lList)
                if res == 0:
    #                         for t in build_trees(lT[0]):
    #                             t.print_()
    #                         print lList
                    lResults.append([(lT[0], lList)])
                else:
                    # partial
                    # get final index //07/09/2016 !!! lIndex not used
                    lastindex = 0
                    lIndex = {}
                    for s in lT:
                        # partial 
                        if s.name == myGrammar.name and s.completed():
                            lastindex = max(lastindex, s.end_column.index)
                            lResults.append((s, lList[s.start_column.index:lastindex]))
#                         # missing? ?? missing what?
#                         elif s.completed():
#                             try:lIndex[s.name] = max(lIndex[s.name], s.end_column.index)
#                             except:lIndex[s.name] = s.end_column.index
                    if lastindex == 0: 
                        lastindex = 1
                    lList = lList[lastindex:]
                    res, lT = parse(myGrammar, lList)
            if res == 0:
#                 print 'xxxxhere',lT, lList
                lResults.append((lT[0], lList))
            
            if self.bDEBUG:print 'final structures'
            ### find the best coverage: sort by length and take the longest ones first???
            ### find all non overlapping solutions and take the ones with the best coverage
            lParsings.extend(self.getSortedPartialParsing(lResults))
            
        # get coverage
        lFinalList  = self.getCoverage(lParsings)
#         for x,y in lTmp:
# #             print '\t',x[1],len(x[1])
# #             print '\t\t',len (self.simplifyRes(x[0],'s0'))
#             lFinalList.append((x,self.simplifyRes(x,'s0')))

        return lFinalList
        
    def getSortedPartialParsing(self,lResults):
        """
            get longest parsing
                and then take the smallest explanation ?
            ignore those covered
            take next 
            
        """
        
        
        from  earleyParser import  build_trees

        lParsings= []
        lResults.sort(key=lambda x:len(x[1]),reverse=True)
        for res in lResults:
            if self.bDEBUG:print 'partial res:',res
            lNodes= build_trees(res[0])
            lTMP=[]
            for eachres in lNodes:
                if self.bDEBUG:eachres.print_()
                dInvertDict={}
                lout= self.getHListFromNode(eachres,res[1],dInvertDict)
                if self.bDEBUG:print lout
#                 try:print dInvertDict['s0']
#                 except KeyError:pass
                lTMP.append((res[1],dInvertDict,lout))
            #take the shortest
            if lTMP !=[]:
                lTMP.sort(key=lambda (x,y,z):len(str(z)))
                lParsings.append(lTMP[0])
#                 print "?",lTMP[0]
                            
        ## merge them ??? 
        return lParsings
        
    def getCoverage(self,lParsings):
        """
            build the full coverage for a given grammar
            assume lParsings sorted in a reverse order
            simply take the first first, discard the next 'first'
        """
        lFirst=[]
        lRes=[]
        for lelt,keys,tree in lParsings:
            if lelt[0] in lFirst:continue
            else: 
                lFirst.append(lelt[0])
                lRes.append((lelt,keys,tree))
        return lRes    
         
        
        
    