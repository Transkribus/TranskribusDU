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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import common.Component as Component
import config.ds_xml_def as ds_xml
from common.trace import trace, traceln
from common.chrono import chronoOn, chronoOff
from operator import itemgetter

import copy
from compiler.ast import flatten


class sequenceMiner(Component.Component):
    
    # DEFINE the version, usage and description of this particular component
    usage = "[-f N.N] "
    description = "description: mine sequence and generate grammars"
    usage = ""
    version = "0.1"
    name = "Sequence Minor"

    kTAG = "tag"
    kNGRAM = "ngram"
    KLEENEPLUS = 2
    SINGLETON = 1
    BIGRAM = 2
    
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

    def setTH(self,th):
        self._TH = th
        
    def getTH(self): return self._TH
         
    def setSDC(self,s): self.sdc = s
    def getSDC(self): return self.sdc
           
    def setMinSequenceLength(self,l): self.minSeqLength=1
    def setMaxSequenceLength(self,l): self.maxSeqLength = l
    
    
    
    def xxxMineSequence(self,lObjects,featureType):
        """
            recursively mine a seuqence using:  
                Sequentiql pattern mining + rewritting rule (kleene)
                
        """
        ############ TO BE DONE OUTSIDE
#         prev=None
#         for o in lObjects:
#             o.resetFeatures()
#             ##  add the most frequent ones as well
#             o.setFeatureFunction(o.getSetOfMigratedFeatures,TH=10.0,lFeatureList=p.lf_XCut)
# #             p.setFeatureFunction(p.getSetOfVInfoFeatures,TH=25,lFeatureList=p.getVX2Info()+p.getVX1Info())
# 
#             if prev:
#                 prev.next = [p]
#             prev=p
#         if prev:
#             prev.next = [p]         
           
#         print "\t xxxq \t\tfeature Generation..."
        
        
        ## OUTSIDE
        seqGen.bDebug = False
        seqGen.setMaxSequenceLength(1)
        # used for parsing, not for mining
        # 
        seqGen.setSDC(0.2) # related to noise level        

        
        lSortedFeatures = self.featureGeneration(lObjects,2)
        ## just need of the longest sequence !!! 
        #   then store patterns by length
        #   then get kleenePatterns for seq=1; get new seq for them; generate blacklisted patterns ( [a,a] for length 2)
        lListOfSequence = self.generateItemsets(lObjects)
        lSeq, lMIS= self.generateMSPSData(lListOfSequence,lSortedFeatures,mis=0.5)
        # here iteration on sequencelength
        lLPatterns = self.beginMiningSequences(lSeq,lSortedFeatures,lMIS)
        ### for sequence >1 support is * by lenght:: support in given by parsing ! 
        lNewKleeneSequences = []
        lNewFeatures = []
        for i,lp in enumerate(lLPatterns):
            seqLen=i+1
            ## before kleene: filter out subsumed patterns (total inclusion)
            lMainPattners= seqGen.testSubSumingPattern(lp)
            lMainPattners.sort(key=lambda (x,y):y)
            for pattern,supp in lMainPattners:
#                 test kleene+
                lLNewKleeneSequence =[]
                lParsingRes = self.parseSequence([pattern[0]],featureObject,lPages)
                lns,lnf = seqGen.generateKleeneItemsets(lPages,lParsingRes,featureObject)
                if lns != []:
                    lNewKleeneSequences.append(lns) 
                    lNewFeatures.append(lnf)
        
        i=0
        print lNewKleeneSequences
        while i < len(lNewKleeneSequences): 
            lnewpatterns=lNewKleeneSequences.pop(0)
            lnewFeatures= lNewFeatures.pop(0)
            print "CURRENT PATTERN:",i,lnewpatterns
            del seqGen
            seqGen=sequenceMiner()
            seqGen.setMaxSequenceLength(2)         
            seqGen.setSDC(0.3)       
            seqGen.bDebug=False
            lListOfSequence = seqGen.generateItemsets(lnewpatterns)
            print lListOfSequence
            #if kleeneStar: replace s0+ by gram+
            lSeq, lMIS= seqGen.generateMSPSData(lListOfSequence,lSortedFeatures+ lnewFeatures,mis=0.1)
            # mine new seqeunces
            # new seqGen??
#             print 'my seq:',lSeq
            lnp=seqGen.beginMiningSequences(lSeq,lSortedFeatures+lNewFeatures,lMIS)
            print 'new res', len(lnp)
            for p in lnp[1:]:
                print p
            ## add only patterns with kleene pattners of length >=2 
            lNewKleeneSequences.extend(lnp)
            # rec! apply parsing ,....
            i+=1
        return     
    
    
    def r_miningSeqeunce(self,lSequence,lFzatures):
        """
            
        """
        
    
    def generateMSPSData(self,lseq,lfeatures,mis=0.01):
        """
        data.txt
         <{18, 23, 37, 44}{18, 46}{17, 42, 44, 49}> per line    
         
        param.txt 
            MIS(1) = 0.003
            SDC = 00.1
        

        """
        
        lMIS = { i:mis for i in lfeatures }

        lLsequences=[]
        
        # data just the 
        
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
                tmpseq.append(tmpitem)
            db.append(tmpseq)
        return db,lMIS
        
        
        
    def generateItemsets(self,lElts):
        """
            generate itemset sequences of length 1 .. self.maxSeqLength
            
            return a list of list
        """
        lList = []

        lenElt=len(lElts)        
        for i in range(self.maxSeqLength-1,self.maxSeqLength):
            tmpList=[]
            j= 0 
            while j < lenElt:
                tmpList.append(lElts[j:j+i+1])
                j+=1
            lList.append(tmpList)
#         print lList
        return lList[0]
    
    
    def testSubSumingPattern(self,lPatterns):
        """
            find CLOSED PATTERNS :: A frequent closed sequential pattern is a frequent sequential pattern such that it is not included in another sequential pattern having exactly the same support.
            not really subsuming pattern, but the richest (more features)
            if ([a,b,c],s1)  and ([a,b],s2)  and and s2 > TH*s1 - >ab subsumeabc
        """
        from operator import itemgetter 
        import itertools
        
        lDict={}
        for pattern in lPatterns:
            lDict[str(pattern[0])]= (pattern[1],list(itertools.chain(*pattern[0])))
            print pattern,  list(itertools.chain(*pattern[0]))
        
        #sort longest first 
        sortedItems = map(lambda x: (x,len(lDict[x][1])) ,lDict)
        sortedItems.sort(key = itemgetter(1),reverse=True)
        ltobedel=[]
        for i,(x,y) in enumerate(sortedItems):
            lx=eval(x)
            for (x2,y2) in sortedItems[i:]:
                lx2=eval(x2)
                if len(lx) == len(lx2):
                    bSuperset=True
                    k=0
                    while k < len(lx):
#                         print lx[k], lx2[k]
                        bSuperset = bSuperset and set(lx[k]).issuperset(set(lx2[k]))
                        k+=1 
                else:
                    bSuperset=False
                if  x != x2 and bSuperset:
#                     print 'included',lx,lx2, lDict[x][0], lDict[x2][0]
                    ltobedel.append(x2)    
        
        lcleanedList=[]
        for p,supp, in lPatterns:
            if str(p) not in ltobedel and supp>1:
#                 print '-\t-\t',p,supp
                lcleanedList.append((p,supp))
        
        return lcleanedList
        
    def generateKleeneItemsets(self,lPages,lParsings,featureType):
        """
            similar to generateItemsets but with 'rewritting' for kleene strcutures
             --> not better to genarted generateMSPSData  ([ [features] instates of [pqge])
             
            
            lParsings!
            
            gram    [['x=33.0', 'x=465.0']] 
            fulllist: [PAGE 2 89, PAGE 3 93, PAGE 4 86, PAGE 5 55, PAGE 6 81, PAGE 7 93, PAGE 8 76, PAGE 9 74] 
            parsing: [([PAGE 2 89, PAGE 3 93, PAGE 4 86, PAGE 5 55, PAGE 6 81, PAGE 7 93, PAGE 8 76, PAGE 9 74]
                {'s0+': [('s0+', 0, 7)], 's1': [('s1', 0, 1), ('s1', 2, 3), ('s1', 4, 5), ('s1', 6, 7)], 's0': [('s0', 0, 7)], 'x33.0': [('x33.0', 0, 0), ('x33.0', 2, 2), ('x33.0', 4, 4), ('x33.0', 6, 6)], 's1+': [('s1+', 0, 7), ('s1+', 0, 5), ('s1+', 0, 3), ('s1+', 0, 1)], 'x465.0': [('x465.0', 1, 1), ('x465.0', 3, 3), ('x465.0', 5, 5), ('x465.0', 7, 7)]}, ['s0', ['s1+', ['s1+', ['s1+', ['s1+', ['s1', ['x33.0', [('x33.0', ('x33.0', 0, 0))], 'x465.0', [('x465.0', ('x465.0', 1, 1))]]], 's1', ['x33.0', [('x33.0', ('x33.0', 2, 2))], 'x465.0', [('x465.0', ('x465.0', 3, 3))]]], 's1', ['x33.0', [('x33.0', ('x33.0', 4, 4))], 'x465.0', [('x465.0', ('x465.0', 5, 5))]]], 's1', ['x33.0', [('x33.0', ('x33.0', 6, 6))], 'x465.0', [('x465.0', ('x465.0', 7, 7))]]]]])]

        
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
            print ikleeneFactor
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
        # filter out freq of 1 
        for p,s in lPatterns:
            if s > 1:print p,s
        return lPatterns
        
    
    def pattern2Multifeatures(self,lpatterns):
        """
        ['x=35.0', 'x=465.0']  -> multiValueFeatureObject
        """
        from feature import  multiValueFeatureObject
        
        
          
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
#             print f,freq,f.getTH()
            ## update the value if numerical feature: take the mean and not the most frequent!!
            if f.getType() == featureObject.NUMERICAL:
                lvalues=map(lambda x:x.getValue(),lMergedFeatures[f])
                lweights= map(lambda x:1.0*x/len(lMergedFeatures[f]),lvalues)
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

        lTBDel = []
#         for i,f in enumerate(lCovered):
# #             print f, kNewValue[f], f.getNodes(), lCovered[:i]
#             if f in lCovered[:i]:
#                 lTBDel.append(f)
# #                 print "removed:",f
#             elif f.getType() == featureObject.NUMERICAL:
#                 try:f.setValue(kNewValue[f])
# #                 try: f.getCanonical().setValue(kNewValue[f])
#                 except KeyError:pass # lweight div by Zero

        ## need to update again the features: remerge again?? or simply discard?
        lCovered=filter(lambda x:x.getID() not in map(lambda x:x.getID(),lTBDel),lCovered)
                
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
    
    
    
    def addNewPrefix(self,length, myTrie, startFeature, lkeys, next):
        """
               update trie from testContinautionPrefix
        """
        if length >= self.maxSeqLength:
            return
        
        # # difference with old version: if one feature in all features == starFeature:stop
        # # old version: use maxSeqLength to stop       
        if next and startFeature in next.getSetofFeatures().getSequences():
            # # here: store : contiguous +=1
            next = None
            return None
        else:
#             print "add new ",next,lkeys
            for fea in next.getSetofFeatures().getSequences():
                updatedKey=self.copyList(lkeys)
                updatedKey.append(fea)
                try:
                    myTrie[unicode(updatedKey)]
                    if next not in myTrie[unicode(updatedKey)][len(updatedKey)-1]:
                        myTrie[unicode(updatedKey)][len(updatedKey)-1].append(next)
#                         print '\telt added',next,myTrie[unicode(updatedKey)][len(updatedKey)-1]
                except KeyError:
                    myTrie[unicode(updatedKey)]=self.copyList(myTrie[unicode(lkeys)])
                    myTrie[unicode(updatedKey)].append([next])
#                     print '\tNEW elt added',next,myTrie[unicode(updatedKey)]
#                     for x in myTrie[unicode(updatedKey)]:
#                         print '\t', x
        return unicode(updatedKey)



    
    def enrichPattern(self,myTrei,mypref):
        """
            for each pre fin mypref:
                for the nodes in mytrie[pypref]
                    mining pattern
                
                
            need to store information: (support # no-ext)/(suport # ext)
            # multi enrichement: how to deal with this?
            ##" assume all support > TH
        """
    
    def extendItem(self,length,myTrie,lkeys,node):
        """
            extend the itemset  
           
           use tuple for representing itemset extension
               (list for representing sequence )
           -> no, just list? yes
               
            Add in the last tuple
            
            !!!!! Need to quantify the frequency!!!!! IMPORTANT -> support
            ->from the list of nodes
        """
        if length == self.maxSeqLength:
            return 
            
        
        lCanonical = map(lambda x:x.getCanonical(),node.getSetofFeatures().getSequences())
        for fea in lCanonical:
            updatedKey=self.copyList(lkeys)  # copy?
            # get last item
            if fea not in updatedKey[-1]:
                updatedKey[-1].append(fea)
#                 print updatedKey,lkeys
                try:
                    myTrie[unicode(updatedKey)]
                    if node not in  myTrie[unicode(updatedKey)][len(updatedKey)-1]:
                        myTrie[unicode(updatedKey)][len(updatedKey)-1].append(node)
                except KeyError:
                    myTrie[unicode(updatedKey)]= self.copyList(myTrie[unicode(lkeys)])
                    #delete node which do not have fea!
                    for node in myTrie[unicode(updatedKey)][len(updatedKey)-1]:
                        if fea not in node.getSetofFeatures().getSequences():
                            myTrie[unicode(updatedKey)][len(updatedKey)-1].remove(node)
    #                 if node not in myTrie[unicode(updatedKey)][len(updatedKey)-1]:
    #                     myTrie[unicode(updatedKey)].append([node])
#                 self.extendItem(length+1, myTrie, updatedKey, node)
                
    
    def addBranch(self, length, myTrie, startFeature, lkeys, nextNode):
        """
        append a new prefix element
        
        case [a,b] and nodes(b) INTER nodes(a)  != {}
               
        """

        # too long
        if length == self.maxSeqLength:
            return
        
        # # difference with old version: if one feature in all features == starFeature:stop
        # # old version: use maxSeqLength to stop       
        if False and nextNode and startFeature in nextNode.getSetofFeatures().getSequences():
            pass
            # # here: store : contiguous +=1
#             print 'out!!'
#             nextNode = None
        else:
            #get canonical features:
            lCanonical = map(lambda x:x.getCanonical(),nextNode.getSetofFeatures().getSequences())
#             print '\t',lkeys, nextNode, lCanonical
            for fea in lCanonical:
                if fea not in startFeature:
                    updatedKey=self.copyList(lkeys)
                    updatedKey.append(fea)
#                     print '%s: %s: %s: %s \t%s' %(lkeys, fea, updatedKey, nextNode,self.getAllFlatElements(myTrie[unicode(lkeys)]))
                    bAdded=False
                    try:
                        myTrie[unicode(updatedKey)]
                        if nextNode not in self.getAllFlatElements(myTrie[unicode(updatedKey)]):
#                         if nextNode not in  myTrie[unicode(updatedKey)][len(updatedKey)-1]:
                            myTrie[unicode(updatedKey)][len(updatedKey)-1].append(nextNode)
                            bAdded=True
                    except KeyError:
#                         if nextNode not in self.getAllFlatElements(myTrie[unicode(lkeys)]):
#                             myTrie[unicode(updatedKey)]= self.copyList(myTrie[unicode(lkeys)])
                            ## share the values of lkeys  (which can be latter on updated)
                            myTrie[unicode(updatedKey)]= myTrie[unicode(lkeys)][:]

                            myTrie[unicode(updatedKey)].append([nextNode])
                            bAdded=True
                    if True or bAdded:
#                         print '\tadded:',nextNode ,nextNode not in self.getAllFlatElements(myTrie[unicode(lkeys)])
#                         print '\t\t', myTrie[unicode(lkeys)]
#                         print  '\t\t=', myTrie[unicode(updatedKey)]
                        for next in nextNode.next:
                            self.addBranch(length + 1, myTrie, startFeature+[fea], updatedKey, next)
#         print 'return'

    def updateFinalValues(self,myTrie,lPrefixes=None):
        """
            by construction after trie creation: 
                myTrie[ab][0] contains all value of myTrie[a]
                    -> only a preceding b  must occur in myTrie[ab]
                    
                if a belong to several positions?
                
                
            if one value set is empty: discard prefix
        """
        
        
        if lPrefixes is None:
            lPrefixes = myTrie.suffixes()
        lReturnedList= lPrefixes[:]
        
        ltobedeleted=[]
        for suffix in lPrefixes:
            if len( myTrie[suffix])>1:
                ipos= len(myTrie[suffix])-1
                while ipos >= 1:
                    lastElts = myTrie[suffix][ipos]
                    if lastElts != []:
                        while type(lastElts[-1]).__name__ == 'list':
                            lastElts = lastElts[-1]
                    mappedlastElts = map(lambda x:x,lastElts)
#                     mappedlastElts = self.getLastElements(myTrie[suffix])
                    prevElts= self.getAllFlatElements(myTrie[suffix][ipos-1])
                    short1pref=eval(suffix)[:-1]
#                     print suffix, mappedlastElts
#                     print '\t',prevElts
                    try: prevElts= self.getLastElements(myTrie[unicode(short1pref)])
                    except KeyError:prevElts=[]
                    tbd=[]
                    for prev in prevElts:
                        # if next is none: no next, so delete!
#                         print suffix, myTrie[suffix], prev,mappedlastElts
#                         print 'prev',set(prev.next)
#                         print 'maaped',  mappedlastElts
                        if prev.next == [] or len(set(prev.next).intersection(set( mappedlastElts)))==0:
                            tbd.append(prev)
                    if tbd is not []:
                        myTrie[suffix][ipos-1]=self.copyList(myTrie[suffix][ipos-1])
                    for val in tbd:
                        for elt in myTrie[suffix][ipos-1]:
                            if elt is val:
                                myTrie[suffix][ipos-1].remove(elt)
                                if myTrie[suffix][ipos-1]==[] or len(self.getLastElements(myTrie[suffix])) < self.support:
                                    if suffix not in ltobedeleted:
                                        ltobedeleted.append(suffix)
                    ipos -=1
        
        for p in ltobedeleted:
#             del(myTrie[p])
#             print 'del',p
            lReturnedList.remove(p)
    
        return lReturnedList
    
    def r_extendOnly(self,lFeatures):
        """
            basic  pattern extension
            
            
            -> in prefixSpan-> 
        """
        import datrie
        import string
        
        lGrammar = []
        
        ## TRIE CREATION
        trie = datrie.Trie(string.ascii_letters + string.digits + string.punctuation + ' ')
        if lFeatures == []:
            return None,None
        
        for feature in lFeatures:
            for node in feature.getCanonical().getNodes():
                try:
                    if node not in trie[unicode([[feature]])][0]:
                        trie[unicode([[feature]])][0].append(node)
                except KeyError:
#                     print 'creation',feature,node
                    trie[unicode([[feature]])]=[[node]]
                # order elements to avoid to have [a,b] and [b,a]!
                self.extendItem(1, trie, [[feature]],node)
                
        for suffix in trie.suffixes():
            if len(trie[suffix][0])>0:
                print suffix, len(trie[suffix][0]) #,trie[suffix][0]
#             if len(eval(suffix)[0]) == 1:
#                 for suffix2 in trie.suffixes():
#                     print 
#                     if eval(suffix)[0][0] == eval(suffix)[0]:
#                         print "x",suffix, suffix2
# #                     if suffix2 !='':
# #                         print "xx",suffix, suffix2, len(trie[suffix][0]),len(trie[suffix2][0])
    
        
        
    def r_prefTree(self,lFeatures,lSeq):
        """
            simulate prefixSpan 
            datrie structure needed: yes since 'itemset' are not defined: we have one sequence only
        """
        import datrie
        import string
        
        self.globalFeatures = lFeatures
        lFullNodeList={}
        lGrammar = []
        
        ## TRIE CREATION
        trie = datrie.Trie(string.ascii_letters + string.digits + string.punctuation + ' ')
        
        if lFeatures == []:
            return None,None
        
        # unigrams
        for feature in lFeatures:
            for node in feature.getCanonical().getNodes():
#                 print feature,node
                try:
                    if node not in trie[unicode([[feature]])][0]:
                        trie[unicode([[feature]])][0].append(node)
                except KeyError:
                    if self.bDebug:print 'creation',feature
                    trie[unicode([[feature]])] = [[node]]

        #KLEENE PLUS 
        # here??
        lCovered=[]
        lKleenePrefixes = []
        ### INCREMENTAL: first test unigram
        for suffix in trie.suffixes():
            nbInitialNodes, nbKleenelNodes = self.testPrefixContiguity(trie, suffix)
            lCovered.append(suffix)
#             print suffix, len(trie.suffixes()),nbKleenelNodes, nbInitialNodes
            if nbInitialNodes is not None:  
                lKleenePrefixes.append(suffix)
                lFullNodeList[suffix] = self.getAllFlatElements(trie[suffix])
                lGrammar.append(suffix)    


        ## HERE PERFORM extension for each element
        ### purpose: find correlation  
        
        
#         for pref in lKleenePrefixes:
#             print 'prefix:',pref         
        
        oldseqmax = self.maxSeqLength
        ### INCREMENTAL: length: 2..maxLength
        for maxLength in range(2,oldseqmax+1):
            self.maxSeqLength = maxLength+1
#             print '#prefixes',maxLength,len(trie.suffixes())
            for upref in trie.suffixes():
                self.prefixTree(maxLength, trie, upref, upref)
            
#             for pref in trie.suffixes():
#                 if len(eval(pref))==maxLength:
#                     # already done in prefix??
#                     self.updateFinalValues(trie,[pref])
#                 print 'prefix:',pref    
 
            ## test kleenePlus
            for suffix in trie.suffixes():
                ## take newest prefixes
                if True or len(eval(suffix)) == maxLength:
                    if suffix not in lCovered: 
                        nbInitialNodes, nbKleenelNodes = self.testPrefixContiguity(trie, suffix)
                        lCovered.append(suffix)
                        if nbInitialNodes is not None:
                            print suffix, nbInitialNodes,nbKleenelNodes
                            print trie[suffix]
                            lKleenePrefixes.append(suffix)
                            lGrammar.append(suffix)    


        

        # frequency if the rule: min (#firstnodes, #lastnodes)
        lGrammar.sort(key=lambda x:len(self.getAllFlatElements(trie[x])), reverse=True)
        
#         for rule in lGrammar:
#             print 
#             print rule
#             for lnodes in trie[rule]:
# #                 lnodes.sort(key=lambda x:x.getY())
#                 for node in lnodes:
#                     try:
#                         print node.getID(),
#                     except:
#                         print map(lambda x:x.getID(),node),
#                 print


        def replace(node,lFeatures,out):
            """ iterate tree in pre-order depth-first search order """
            #if type(node).__name__ =='tuple':
            #    for elt in node:
            if type(node).__name__ =='str':
                ## assume a single feature per node
                for feature in lFeatures:
                    if str(feature) == "'%s'"%node:
                        out.append(feature)                
            elif type(node).__name__ == 'list':
                newList=[]
                out.append(newList)
                for child in node:
                    replace(child,lFeatures,newList)
        
        # recast The features into
        ## add info about internal structure: test if subseq in lgrammar
        lFeatureGrammar = []
        for g in lGrammar:
#             print 'g:',g,lFeatures
            outGram = []
            replace(eval(g),lFeatures,outGram)
            lFeatureGrammar.append(outGram[0])   
#             print outGram[0]
#         lFeatureGrammar.sort(key=lambda x:len(x), reverse=True)
#         print 'final grammars:'
#         for f in lFeatureGrammar:
#             print f
        
        if lFeatureGrammar == []:
            lFeatureGrammar=None
        # return lGrammar as well? or need a parsing?  : how to parse  graph?: parse each branch? need to be acyclic
        return lFeatureGrammar, lGrammar
              
        
        ### grammar inference part
        
    
    
    def prefixTree(self,length,myTrie, lstartFeature, ukey):
        """
        """
        if length >=self.maxSeqLength:
            return
        
        
        lkey= eval(ukey)
        # get the sequences
        lNewPrefixes={}        
        
        print 'pattern', lkey
        print "\tlast:", lkey[-1]
        # get last nodes         ## need to add []     
#         lvalues=myTrie[unicode([lkey[-1]])]
        lvalues=myTrie[unicode([lkey[-1]])]

        lLast=self.getLastElements(myTrie[ukey])
        print lLast
        lnext =[ item for sublist in map(lambda x:x.next,lLast)  for item in sublist] 
#             print mypref#, lvalues,lLast,'\n\tnext:',lnext

        # template 1:  {_, x}
        ##  avoid [a,b] and [b,a]!
        ## if [a] +  -> all enriched are + as well
        for lastnode in lLast:
            lCanonical = map(lambda x:x.getCanonical(),lastnode.getSetofFeatures().getSequences())
            for fea in lCanonical:
                # select among frequent ones 
                if unicode(fea) in unicode(self.globalFeatures):
                    updatedKey=self.copyList(lkey)  # copy?
                    # get last item
                    ## TEST AT STRING level!!  'a' in 'aze'  no way to get back unicode(feautre)
                    if unicode(fea) not in unicode(updatedKey[-1]):
                        updatedKey[-1].append(fea)
                        updatedKey[-1].sort()
                        try:
                            myTrie[unicode(updatedKey)]
                            bCont=False
                        except KeyError:bCont=True
                        if bCont:
                            try:
                                lNewPrefixes[unicode(updatedKey)]
#                                 if node not in  lNewPrefixes[unicode(updatedKey)][len(updatedKey)-1]:
#                                     lNewPrefixes[unicode(updatedKey)][len(updatedKey)-1].append(node)
                            except KeyError:
                                lNewPrefixes[unicode(updatedKey)]= self.copyList(myTrie[ukey])
                                #delete node which do not have fea!
                                for node in lLast: #lNewPrefixes[unicode(updatedKey)][len(updatedKey)-1]:
                                    if fea not in node.getSetofFeatures().getSequences():
                                        try:
                                            lNewPrefixes[unicode(updatedKey)][len(updatedKey)-1].remove(node)
                                        except:print 'need to solve this'
#                                     else:
#                                         print fea, node, node.getSetofFeatures().getSequences()

        # template 2:  {30} {x}
        for nextNode in lnext:
            # new prefix
            lCanonical = map(lambda x:x.getCanonical(),nextNode.getSetofFeatures().getSequences())
            for fea in lCanonical:
                ## TEST AT STRING level!!  'a' in 'aze'
                if unicode(fea) not in lkey:
                    updatedKey=self.copyList(lkey)
                    updatedKey.append([fea])
                    uKey= unicode(updatedKey)
#                         print 'ukey',updatedKey, lNewPrefixes.has_key(uKey)
                    #not already covered
                    try:
                        myTrie[unicode(updatedKey)]
                        bCont=False
                    except KeyError:bCont=True
                    if bCont:                    
                        try:
                            lNewPrefixes[uKey]
    #                             print updatedKey, lNewPrefixes[uKey]
                            if nextNode not in self.getAllFlatElements(lNewPrefixes[uKey]):
    #                         if nextNode not in  myTrie[unicode(updatedKey)][len(updatedKey)-1]:
                                lNewPrefixes[uKey][len(updatedKey)-1].append(nextNode)
    #                                 print 'added node pref:',      updatedKey,     lNewPrefixes[uKey]       
                                   
                        except KeyError:
    #                         if nextNode not in self.getAllFlatElements(myTrie[unicode(lkeys)]):
    #                             myTrie[unicode(updatedKey)]= self.copyList(myTrie[unicode(lkeys)])
                                ## share the values of lkeys  (which can be latter on updated)
                                lNewPrefixes[uKey]= myTrie[ukey][:]
                                lNewPrefixes[uKey].append([nextNode])
    #                                 print 'new pref:',      updatedKey,     lNewPrefixes[uKey]       
        
        # test if pruning or not
        ltobekept={}
#         print lNewPrefixes
        for newPref in lNewPrefixes:
            print newPref, lNewPrefixes[newPref], len(eval(newPref))
            assert len(eval(newPref)) == len(lNewPrefixes[newPref])
            if len(lNewPrefixes[newPref][len(eval(newPref))-1]) >= self.support:
#                 print "new!!",newPref, lNewPrefixes[newPref]
                myTrie[newPref]=lNewPrefixes[newPref]
                self.updateFinalValues(myTrie,eval(newPref))
                #second test after cleaning
                        # support: conisder length of seq * #nodes in last = total number of elements!
                
                if len(myTrie[newPref][len(eval(newPref))-1])  * len(eval(newPref))>= self.support:
                    self.prefixTree(length+1, myTrie, lstartFeature, newPref)
                else:
                    del(myTrie[newPref])
#             else:
#                 print "deleted", newPref, lNewPrefixes[newPref][len(eval(newPref))-1]
    
        
    def parseForMining(self,myTrie,lFeatures,lkeys,lSeq):
        """
        
        """
        myFeatureType=lFeatures[0].getClass()
        myGram, = self.convertKeyToFeature(lkeys)
        lParsing = self.parseSequence(myGram, myFeatureType, lSeqElements)
        
        # convert parsings into datrie structure: need the parsing tree, search for 
         
    def convertKeyToFeature(self,gram):
        """
            convert datrie keys (strings) to features
        """
        def replace(node,lFeatures,out):
            """ iterate tree in pre-order depth-first search order """
            #if type(node).__name__ =='tuple':
            #    for elt in node:
            if type(node).__name__ =='str':
                ## assume a single feature per node
                for feature in lFeatures:
                    if str(feature) == "'%s'"%node:
                        out.append(feature)                
            elif type(node).__name__ == 'list':
                newList=[]
                out.append(newList)
                for child in node:
                    replace(child,lFeatures,newList)
        
        # recast The features into
        ## add info about internal structure: test if subseq in lgrammar
        outGram = []
        replace(eval(gram),lFeatures,outGram)
        return outGram[0]  

    def parseSequence(self,myGrammar,myFeatureType,lSeqElements):
        """
    
            try 'partial parsing'
                for a structure (a,b)  ->  try (a,b)  || (a,any)   || (any,b)
                only for 'flat' n-grams
           
               notion of wildcard * match anything  wildCardFeature.__eq__: return True
               
               in a grammar: replace iteratively each term by wildwrd and match
                       then final completion? ???
        """
        
        from grammarTool import sequenceGrammar
        
        seqGen = sequenceMiner()
        
        myGram = sequenceGrammar()
        lParsings = myGram.parseSequence(myFeatureType,myGrammar, lSeqElements)
        lFullList=[]
        lFullObjects = []
        for x,y,z in lParsings:
            lFullList.extend(x)
            lFullObjects.extend(y)
            print "list:",x
            print "objects",y
#             lFullList.sort(key=lambda x:x.getNumber())
            print 'max coverage: ', len( lFullList), lFullList
            ## handle hole in the sequence??? 
            lLParsings.append((Gram,lFullList,lParsings))

        return lLParsings   

    def getFirstElements(self,node):
        """
            get the first non list element from a recursive list
        """
        if type(node).__name__ =='list':
            if type(node[0]).__name__ != 'list':
                    return node
        if type(node).__name__ !='list':
            return node
        else:
            return self.getFirstElements(node[0])
    
    def getLastElements(self,node):
        """
            get the last non list element from a recursive list
        """
#         print node
        if type(node).__name__ =='list':
            if type(node[-1]).__name__ != 'list':
                    return node
        if type(node).__name__ !='list':
            return node
        else:
            return self.getLastElements(node[-1])
        
        
        
    def mergeSimilarSequences(self,myTrie,myprefix):
        """
            find out sequences with similar coverage (whihc can be used for enrichment?)
            similar coverage:  TH * prefix nodes 
        """
        TH = 0.9
        myNodes = self.getAllFlatElements(myTrie[myprefix])
        setMyNodes= set(myNodes)
        lenSetMyNodes = len(setMyNodes)

        for prefix in myTrie.suffixes():
            if prefix != myprefix:
                lnodes = self.getAllFlatElements(myTrie[prefix])
                setNodes= set(lnodes)
                if len(setNodes.intersection(setMyNodes)) >= TH * lenSetMyNodes and len(setNodes.intersection(setMyNodes)) <= (1+ (1-TH)) * lenSetMyNodes:
                        print 'similar:' , myprefix, prefix,  lenSetMyNodes, len(set(lnodes)), len(setNodes.intersection(setMyNodes))
        
    def testPrefixContinuation(self,myTrie,lPref):
        """
            for a pref: consider the nextElements prefixes np and test is pref+np exists. if not add it 
            
            a way to build  a,b+,e, when b very long
        """
        lKleenePref=[]
        for pref in lPref:
#             print "\ncont"
#             print pref
            # get first
            nextElts=[]
            for elt in self.getLastElements( myTrie[unicode(pref)]):
                nextElts+=elt.next 
#             nextElts = map(lambda x:x.next, self.getLastElements( myTrie[unicode(pref)]))
            myElts = self.getAllFlatElements( myTrie[unicode(pref)])
            length=len(pref)
            for elt in set(nextElts):
                if elt is not None:
                    # not enough to test the first: (a(b,c)+)   -get b!
#                     print elt ,myTrie[unicode(pref)][0]
                    if elt not in myElts:
                        newprefix = self.addNewPrefix(length + 1, myTrie, None, pref, elt)
                        if newprefix is not None:
                            lKleenePref.append(newprefix)
#         for su in myTrie.suffixes():
#             print su
#             for va in myTrie[su]:
#                 print '\t',va
#         print '-'*20
        return lKleenePref
             
    def copyList(self,l):
        """
            copy in a new list but do not copy terminals
        """
        
        if type(l).__name__ !='list':
            return l                
        elif type(l).__name__ == 'list':
            out = []
            for child in l:
                out.append(self.copyList(child))
        return out          
        
#     def getAllFlatValues(self,node):
#         """
#             flat list
#         """  
#         if type(node).__name__ !='list':
#             return [node]                
#         elif type(node).__name__ == 'list':
#             out = []
#             for child in node:
#                 out.extend(child)
#         return out        
           
    def getAllFlatElements(self,node):
        """
            get all nodes of prefix
        """
        return flatten(node)
    
        if type(node).__name__ != 'list': #=='str':
            return [node]                
        elif type(node).__name__ == 'list':
            out = []
            for child in node:
                out.extend(self.getAllFlatElements(child))
        return out
            
        
        
        
    def copyValues(self,trie,oldpref,newpref,addedPref,startnexPref):
        """
           when a subart of a prefix is replaced by a structure, update the value list accordingly
        
            but take care to "generate" real sequences! 
            so need to insert only true next
        """
        trie[unicode(newpref)]= [[] for i in range(len(newpref))]
        for i, val in enumerate(trie[unicode(oldpref)]):
#             print i,startnexPref, val
            if i == startnexPref:
                trie[unicode(newpref)][i]=self.copyList(trie[unicode(addedPref)])
            elif i < startnexPref:
                trie[unicode(newpref)][i]= self.copyList(val)
            elif i > startnexPref+len(eval(addedPref))-1:
                x =i - len(trie[unicode(oldpref)])
                trie[unicode(newpref)][x]= self.copyList(val)
#             print '\t',newpref,'///',trie[unicode(newpref)]
        assert trie[unicode(newpref)][-1] != []   
            
    def replacePrefix(self, trie, prefix):
        """
            search for prefix in branches and replace
            -> search for branches where prefix ends up is enough?
        
        
            useful?? abbbbbbc
            
            with test continuation: we can lilit to first/last postion?
            
            
            TEET ALSO LENGTH   <= lenght_max!
        """
        def find_sub_list(sl,l):
            results=[]
            sll=len(sl)
            for ind in (i for i,e in enumerate(l) if e==sl[0]):
                if l[ind:ind+sll]==sl:
                    results.append((ind,ind+sll-1))
        
            return results        
        
        lNewPref=[]
        lTobeDeleted=[]
        # using gramLenght?
        for pref in trie.suffixes():
            bToBeDel = False
            
            ## LEFT RIGHT
            extend=eval(pref) + eval(prefix)
            if trie.has_keys_with_prefix(unicode(extend)):
                # several similar values!!
                prefixPos1 =  find_sub_list(eval(prefix), extend)
                if len(prefixPos1) == 1:
                    sNewKey = eval(pref)
                    sNewKey.append(eval(prefix))
                    lNewPref.append(unicode(sNewKey))
                    self.copyValues(trie,extend,sNewKey,prefix,prefixPos1[0][0])
                    bToBeDel=True
                    
            ##RIGHT LEFT
            extend=eval(prefix) + eval(pref)
            if trie.has_keys_with_prefix(unicode(extend)):
                prefixPos1 =  find_sub_list(eval(prefix), extend)
#                 print 'xx2',pref, prefixPos1,extend
                if len(prefixPos1) == 1:
                    sNewKey = eval(pref)
                    sNewKey.insert(prefixPos1[0][0],(eval(prefix)))
                    # new maybe exists !!
                    lNewPref.append(unicode(sNewKey))
                    self.copyValues(trie,extend,sNewKey,prefix,prefixPos1[0][0])
                    bToBeDel=True

            
            if bToBeDel:
                lTobeDeleted.append(unicode(extend))
        
#         print prefix,lTobeDeleted
        for p in set(lTobeDeleted):
            try:
                print 'delreplace:',p
                del(trie[p])
            except KeyError:pass
        
        return lNewPref
    
    def testPrefixContiguity(self, trie, prefix, bDebug=False):
        """
            Prefix  A B C 
                if C.next == A > TH  -> (ABC)+
                
            Max theorique : (n-1)/n
            
            DOES NOT WORK WITH: (A)+,B   :> |A| too many!!
                -> need next included in A
                
            Take the smaller set and see it is included in the bigger
            
        """
        lPref = eval(prefix)
        if lPref == []:
            return []
#         if len(self.getLastElements(trie[prefix])) < 2:
#             return None,None
        
        # get first element of the prefix
#         print trie[prefix]
#         sInitialNodes = set(trie[prefix][0])
        sInitialNodes  = set(self.getFirstElements(trie[prefix]))
        if len(sInitialNodes) < 2:
            return None,None
        
        nextElts=[]
        for elt in self.getLastElements(trie[prefix]):
            nextElts+=elt.next 
        kleeneStarElt = sInitialNodes.intersection(set(nextElts))
#         print '***\t',prefix ,len(kleeneStarElt), len(sInitialNodes)
#         print '\t',trie[prefix]
#         print '\t\t',set(nextElts)
#         print '\t\t', self.getAllFlatElements(trie[prefix])
#         print '\t\t',kleeneStarElt
#         print '\t\t', sInitialNodes
#         print prefix, len(sInitialNodes), len(set(nextElts)),len(kleeneStarElt),min(len(nextElts),len(sInitialNodes)),len(kleeneStarElt) > max(1, self.getTH()* min(len(nextElts),len(sInitialNodes)))
#         print '\tss', len(set(nextElts)),len(nextElts)
    
        ## borderline case: abab
#         if len(kleeneStarElt) == 1 and len(sInitialNodes) == 2:
#             return min(len(nextElts),len(sInitialNodes)), len(kleeneStarElt)

        if len(kleeneStarElt) >= max(1.1, self.getTH()* min(len(set(nextElts)),len(sInitialNodes))):
#             print prefix, len(sInitialNodes),max(1.1, self.getTH()* min(len(nextElts),len(sInitialNodes))), len(kleeneStarElt)
#             print '\t\t',kleeneStarElt
#             print '\t\t', sInitialNodes        
            return min(len(nextElts),len(sInitialNodes)), len(kleeneStarElt)
        
        return None,None
       


        