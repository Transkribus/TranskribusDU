# -*- coding: utf-8 -*-
"""

     H. Déjean

    copyright Xerox 2016
    READ project 

    toolkit for mining structural (sequential) elements a document
    
    - get grammar for sequential/contiguous patterns
    - get frequent pattern (sub contiguous  a,(b,c)  freq=X 


    see Prefix Tree Acceptor (PTA)

    issue one : get a suffix pattern : a b+ c  : difficult to get c 
    see 2009: (Fase, br, (D,br) br)
    
    
    
    see viterby : https://github.com/phvu/misc/blob/master/viterbi/
"""

# Adjustement of the PYTHONPATH to include /.../DS/src
import sys, os.path
from jsonschema._validators import pattern
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import common.Component as Component
import config.ds_xml_def as ds_xml
from operator import itemgetter



class sequenceMiner(Component.Component):
    
    # DEFINE the version, usage and description of this particular component
    usage = "[-f N.N] "
    description = "description: mine sequence and generate grammars"
    usage = ""
    version = "0.1"
    name = "Sequence Minor"

    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "ContentAnalyzor", self.usage, self.version, self.description) 
        
        # SET here the default value of all parameters
    
        self.tag = ds_xml.sTEXT
        
        # contiguity threshold
        self._TH = 0 
        
        # maximal sequence size
        self.maxSeqLength = 2

        self.sdc = 0.1  # support different constraints
        # support # given by mis:
#         self.support = 2

        self.THRULES = 0.6
        
    def setTH(self,th):
        self._TH = th
        
    def getTH(self): return self._TH
         
    def setSDC(self,s): self.sdc = s
    def getSDC(self): return self.sdc
           
    def setMinSequenceLength(self,l): self.minSeqLength=1
    def setMaxSequenceLength(self,l): self.maxSeqLength = l
    def getMaxSequenceLength(self): return self.maxSeqLength
    
    
    def generateMSPSData(self,lseq,lfeatures,mis=0.01):
        """
        data.txt
         <{18, 23, 37, 44}{18, 46}{17, 42, 44, 49}> per line    
         
        param.txt 
            MIS(1) = 0.003
            SDC = 00.1
        

        """
        lMIS = { i:mis for i in lfeatures }
        db=[]
        for seq in lseq:
            tmpseq = []
            for item in seq:
                tmpitem = []
                if item.__class__.__name__ =='multiValueFeatureObject':  #featureObject
                    tmpitem.append(item)
                else:
                    for fea in item.getSetofFeatures().getSequences():
                        tmpitem.append(fea)
#                     tmpitemset.sort()
                if tmpitem != []:
                    tmpseq.append(tmpitem)
                else:
                    #add emptyfeature
                    pass
            db.append(tmpseq)
        return db,lMIS
        
        
        
    def generateItemsets(self,lElts):
        """
            generate itemset sequences of length 1 .. self.maxSeqLength
            
            return a list of list
        """
        lList = []

        lenElt=len(lElts)     
        j= 0 
        while j < lenElt:
            lList.append(lElts[j:j+self.maxSeqLength])
            j+=1
        return lList
           
    def generateSequentialRules(self,lPatterns,lenSeq=5):
        """
            generate sequential rules from  a list pa (pattern,support)
        """
        dP={}
        lRules= []
        for p,s in lPatterns:
            dP[str(p)] = s
        for pattern,support in lPatterns:
            if support > 2:
                if support > lenSeq :
                    for i , itemset in enumerate(pattern):
                        if len(itemset) > 1:
        #                     print 'itemset', itemset
                            for item in itemset:
                                newItemSet = itemset[:]
                                newItemSet.remove(item)
                                newPattern=pattern[:]
                                newPattern[i] = newItemSet
        #                         print 'tst rules', newPattern
                                if  str(newPattern) in dP:
                                    fConfidence = 1.0 *support / dP[str(newPattern)]
                                    if fConfidence> self.THRULES: 
                                        if self.bDebug:print 'RULE: %s => %s[%d] (%s/%s = %s)'%(newPattern, item,i,dP[str(newPattern)],support, fConfidence)
                                        lRules.append( (newPattern,item,i,pattern, fConfidence) )
        return lRules
    
    
    def applySetOfRules(self,lSeqRules,lPatterns):
        """
            apply a list of seqRule to a list of patterns
            associate the richer new pattern to initial pattern
        """
        dAncestor = {}
        lNewPatterns=[]
        for pattern, _ in lPatterns:
            lNewPatterns=[]
            print "**",pattern
            curpattern = pattern
            for rule in lSeqRules:
                print '\t',curpattern,'\t',rule
                newpattern = self.applyRule(rule, curpattern)
                print '\t\t-> ',newpattern
                if newpattern:
                    curpattern = newpattern[:]
            if curpattern not in lNewPatterns:
                lNewPatterns.append(curpattern)
            #take the longest extention
            #  lNewPatterns must contain more than pattern
            if len(lNewPatterns) > 1  and len(pattern) == 1:
                lNewPatterns.sort(key=lambda x:len(x))
                dAncestor[str(pattern)] =  lNewPatterns[-1]
                print pattern, lNewPatterns
            
        return dAncestor        
    
    def applyRule(self,rule,pattern):
        """
            try to apply seqrule to pattern
        """
        """
            1 apply all rules: get a new pattern: 
            go to 2 is ne pattern
        """
        
        lhs, elt, pos,rhs, _ = rule
        if len(lhs) == len(pattern):
#             if len
            if elt not in lhs[pos]:
                foo = map(lambda x:x[:],lhs)
                foo[pos].append(elt)
                foo[pos].sort()
                if foo == rhs:
                    return rhs
        return None
    
    def patternToMV(self,pattern):
        """
            convert a list of f as multivaluefeature
            keep mvf as it is 
        """
        from feature import multiValueFeatureObject 
        
        lmv= []
        for itemset in pattern:
            # if one item which is boolean: gramplus element
            if len(itemset)==1 and itemset[0].getType()==1:
                lmv.append(itemset[0])
            else:
                mv = multiValueFeatureObject()
                name= "multi" #'|'.join(i.getName() for i in itemset)
                mv.setName(name)
                mv.setValue(map(lambda x:x,itemset))
    #             print mv.getValue()
                lmv.append(mv)
#             print itemset, mv
        return lmv    
    def parseWithPattern(self,pattern,lSeq):
        """
        """
        from feature import multiValueFeatureObject , featureObject

        print 'parsing with... ',pattern
        ##" convert into multivalFO here to get features 
#         mvPattern = self.patternToMV(pattern)
#         print pattern, mvPattern
        lParsingRes = self.parseSequence(pattern,multiValueFeatureObject,lSeq)
        lns,lnf = self.generateKleeneItemsets(lSeq,lParsingRes,featureObject)
        print "loop0:",lns, lnf        
        
        
#     def testSubSumingPattern(self,lPatterns):
#         """
#             find CLOSED PATTERNS :: A frequent closed sequential pattern is a frequent sequential pattern such that it is not included in another sequential pattern having exactly the same support.
#             not really subsuming pattern, but the richest (more features)
#             if ([a,b,c],s1)  and ([a,b],s2)  and and s2 > TH*s1 - >ab subsumeabc
#         """
#         import itertools
#         
#         lDict={}
#         for pattern in lPatterns:
#             lDict[str(pattern[0])]= (pattern[1],list(itertools.chain(*pattern[0])))
# #             print pattern,  list(itertools.chain(*pattern[0]))
#         
#         #sort longest first 
#         sortedItems = map(lambda x: (x,len(lDict[x][1])) ,lDict)
#         sortedItems.sort(key = itemgetter(1),reverse=True)
#         ltobedel=[]
#         for i,(x,y) in enumerate(sortedItems):
#             lx=eval(x)
#             for (x2,y2) in sortedItems[i:]:
#                 lx2=eval(x2)
#                 if len(lx) == len(lx2):
#                     bSuperset=True
#                     k=0
#                     while k < len(lx):
#                         bSuperset = bSuperset and set(lx[k]).issuperset(set(lx2[k]))
#                         k+=1 
#                 else:
#                     bSuperset=False
#                 if  x != x2 and bSuperset:
#                     ltobedel.append(x2)    
#         
#         lcleanedList=[]
#         for p,supp, in lPatterns:
#             if str(p) not in ltobedel and supp>1:
# #                 print '-\t-\t',p,supp
#                 lcleanedList.append((p,supp))
#         
#         return lcleanedList
        
    def generateKleeneItemsets(self,lPages,lParsings,featureType):
        """
            detect if the pattern is +. if yes: parse the elements 
        """
        from ObjectModel.sequenceAPI import sequenceAPI
        from feature import sequenceOfFeatures
        lNewKleeneFeatures=[]
        lLNewKleeneSeq = []
        lLScoreKleenSeq=[]
        for gram,lFullList,ltrees in lParsings:
            ## create new feature
            ## collect all covered pages ans see if this is a kleene gram
            ## each lParsedPages: correspond to a s0+
            ##  fkleeneFactor = proportion of sequence wich are klenee
            ikleeneFactor=0
            gramLen=len(gram) * 1.0
            for lParsedPages, ktree,ltree in ltrees:
#                 print gram, lParsedPages, ltree
                if len(lParsedPages)> gramLen:
                    ikleeneFactor +=len(lParsedPages)
#                     print 'GRAM', gram, ikleeneFactor, len(ltrees)
                else:
                    ikleeneFactor-=len(lParsedPages)
#             print gram,ikleeneFactor
            if ikleeneFactor > 0:# and ikleeneFactor >= 0.5* len(ltrees):
                gramPlus = featureType()
                gramPlus.setName(str(gram)+"+")
                gramPlus.setType(1)
                gramPlus.setValue(True)
                lNewKleeneFeatures.append(gramPlus)
                i=0
                lNewSeq=[]
                bNew=False
                while i < len(lPages):
                    if not lPages[i] in lFullList:
                        lNewSeq.append(lPages[i])
                        bNew=False
                    elif lPages[i] in lFullList:
                        if not bNew:
                            bNew=True
                            kleeneElt = sequenceAPI()
                            seqOfF = sequenceOfFeatures()
                            seqOfF.addFeature(gramPlus)
                            kleeneElt._lBasicFeatures=seqOfF                            
                            lNewSeq.append(kleeneElt)
                                                    
                    i+=1
                lLNewKleeneSeq.extend(lNewSeq)
            
        #sort?
        return lLNewKleeneSeq,lNewKleeneFeatures
            
            
    
    
    def beginMiningSequences(self,sequences,lFeatures,lMIS):
        """
            for all possible sequences of length 1. max
                minf them
                if kleene: generate new sets of sequences
        """
        from msps import msps
        
        myMiner=msps()
        myMiner.bDEBUG = self.bDebug
        myMiner.setFeatures(lFeatures)
        myMiner.setMIS(lMIS)
        myMiner.setSDC(self.getSDC())
        lPatterns = myMiner.begin_msps(sequences)

        if lPatterns is None:
            return None
        # frequency correction:     
        lPatterns= map(lambda (p,s): (p, round(1.0*len(p)*s/self.getMaxSequenceLength())),lPatterns)    
        return lPatterns
        

#     def computePatternScores(self,lPatterns,N):
#         """
#             compute support, confidence
#             conf a->b supp(a,b)/supp(a)
#         
#         """
#         dPatternFreq={}
#         dPatternConf={}
#         for p,s in lPatterns:
#             if len(p):
#                 dPatternFreq[str(p)] = s/N
#         
#         for p,s in lPatterns:
#             if len(p) == 1 and len(p[0]) == 2:
#                 print p, dPatternFreq[str([p[0]])], dPatternFreq[str([[p[0][0]]])], dPatternFreq[str([p[0]])] / (1.0 * dPatternFreq[str([[p[0][0]]])])
#             
#         return dPatternFreq, dPatternConf
          
    def featureGeneration(self, lList, TH=2):
        """
            lList: global list : all elements 
            generate a set of features for the elements.
            merge duplicate features (but each feature points to a set of Element which created them)
            
            ??? ??? replace calibrated feature in the elt.getSetofFeatures() ()seqOfFeatures
            delete feature with a too low frequency 
            
            ?? REGROUP FEATURES PER node 
            when enriching, consider only features from other nodes?
        """
        from  feature import featureObject   
        lFeatures = {}
        for elt in lList:
            elt.computeSetofFeatures()
            if self.bDebug: print "\t",elt, str(elt.getSetofFeatures())
            for feature in elt.getSetofFeatures().getSequences():
                try:lFeatures[feature].append(feature)
                except KeyError: lFeatures[feature] = [feature]
        sortedItems = map(lambda x: (x, len(lFeatures[x])), lFeatures)
        sortedItems.sort(key=itemgetter(1), reverse=True)    
        lCovered = []
        lMergedFeatures = {}
        for i, (f, freq) in enumerate(sortedItems):
#             print f,freq,hash(f)
            if f.getID() not in map(lambda x: x.getID(), lCovered):
                lMergedFeatures[f] = lFeatures[f]
                lCovered.append(f)
                # usefull? YES, 2D equality does not work with HASH !!! 
                for ff, freq in sortedItems[i + 1:]:
                    # 2d equality
                    if f == ff and f.getID() != ff.getID():
#                         print '=',f,ff
                        for ff2 in lFeatures[ff]:
                            if ff2.getID() not in map(lambda x: x.getID(), lCovered):
                                # test with IF if ff2 not in list 
                                lMergedFeatures[f].append(ff2)
                                lCovered.append(ff2)
#                     else:
#                         print '<>',f,ff
                
        sortedItems = map(lambda x: (x, len(lMergedFeatures[x])), lMergedFeatures)
        sortedItems.sort(key=itemgetter(1), reverse=True)
        lCovered = []  
        kNewValue={}
        for f, freq in sortedItems:
#             print f,freq,lMergedFeatures[f]
            ## update the value if numerical feature: take the mean and not the most frequent!!
            if f.getType() == featureObject.NUMERICAL:
                lvalues=map(lambda x:x.getValue(),lMergedFeatures[f])
                lweights= map(lambda x:1.0*x/len(lMergedFeatures[f]),lvalues)
#                 print f,freq, lvalues, lweights
                import numpy as np
                try: kNewValue[f] =  round(np.average(lvalues,weights=lweights),0)

                except ZeroDivisionError: print lweights, lvalues, lMergedFeatures[f]
            if freq >= TH and f not in lCovered:
                f.setCanonical(f)
                for ff in lMergedFeatures[f]:
#                     print "\t",ff,freq,ff.getTH(),ff.getObjectName(), ff.getID() ,f.getID()
                    if ff not in lCovered  and ff.getID() != f.getID():
                        # replace the feature by the canonical one
                        try:
#                             print ff,  ff.getObjectName()
#                             print '\tOK\t',ff.getObjectName().getSetofFeatures()
                            ff.getObjectName().getSetofFeatures().updateFeature(f)
                        except AttributeError:
                            pass
#                             print ff, "NO OBJECT"
#                         print "\t%s: replace %s by %s" %(ff.getObjectName(),ff,f)
                        for n in ff.getNodes():
                            f.addNode(n)
#                            print "\t\t",n
                        ff.setCanonical(f)
#                     print "\t canonical", ff,f 
                if f not in lCovered:
                    lCovered.append(f)
#                 print "xx\t",f, f.getCanonical()
        nbdeleted = 0
        lCovered.sort(key=lambda x:len(x.getNodes()), reverse=True)
        
        ## compute weight
        map(lambda x:x.setWeight(len(x.getNodes())),lCovered) 
        
#         for x in lCovered:print x, x.getWeight()

        lTBDel = []
#         for i,f in enumerate(lCovered):
# #             print f, kNewValue[f], f.getNodes(), lCovered[:i]
#             try:
#                 kNewValue[f]
#                 if f in lCovered[:i]:
#                     lTBDel.append(f)
#     #                 print "removed:",f
#                 elif f.getType() == featureObject.NUMERICAL:
#                     f.setValue(kNewValue[f])
#             except KeyError:pass

        ## need to update again the features: remerge again?? or simply discard?
        lCovered = filter(lambda x:x.getID() not in map(lambda x:x.getID(),lTBDel),lCovered)
                
        for elt in lList:
            ltodel = []
#             print "x",elt, elt.getSetofFeatures()
            for f in elt.getSetofFeatures().getSequences():
                if len(f.getCanonical().getNodes()) >= TH:
                    ## need t oupdate again sience kNewValue has changed
                    if f.getType() == featureObject.NUMERICAL:
                        f.getObjectName().getSetofFeatures().updateFeature(f.getCanonical())
                    
#                     print elt, f, f.getID(),len(f.getNodes()),len(f.getCanonical().getNodes()),f.getCanonical() #,f.getNodes()
#                    for n in f.getCanonical().getNodes():
#                        print "\t\t",n
                else:
#                     print "remove %d %s %s %s\t%s"%(TH,elt,f,f.getID(),len(f.getCanonical().getNodes()))
                    ltodel.append(f)
            for f in ltodel:
                elt.getSetofFeatures().deleteFeature(f)
                nbdeleted += 1
            if self.bDebug: print "\t",elt, str(elt.getSetofFeatures())
#        print "nb f deleted:",nbdeleted

#         for f in lCovered:
#             print f
#             for n in f.getNodes():
#                 for fea in n.getSetofFeatures().getSequences():
#                     print '\t', fea.getOldValue()
                
        return lCovered
    
    
    def parseSequence(self,myPattern,myFeatureType,lSeqElements):
        """
    
            try 'partial parsing'
                for a structure (a,b)  ->  try (a,b)  || (a,any)   || (any,b)
                only for 'flat' n-grams
           
               notion of wildcard * match anything  wildCardFeature.__eq__: return True
               
               in a grammar: replace iteratively each term by wildwrd and match
                       then final completion? ???
        """
        
        from grammarTool import sequenceGrammar
        
        
        myGram = sequenceGrammar()
        lParsings = myGram.parseSequence(myFeatureType,myPattern, lSeqElements)
        lFullList=[]
        lFullObjects = []
        for x,y,z in lParsings:
            lFullList.extend(x)
            lFullObjects.extend(y)
        return (myPattern,lFullList,lParsings)


