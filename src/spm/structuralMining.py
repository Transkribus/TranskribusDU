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
from operator import itemgetter

import numpy as np
from  feature import featureObject, multiValueFeatureObject   


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
        
        # sequentiality TH
        self.fKleenePlusTH = 1.5
        # eq  threshold
        self._TH = 0 
        
        # maximal sequence size
        self.maxSeqLength = 1
        # min size 
        self.minSeqLength= 1
        self.sdc = 0.1  # support different constraints
        # support # given by mis:
#         self.support = 2

        self.objectLevel= None
        self.THRULES = 0.8
        
    def setTH(self,th):
        self._TH = th
        
    def getTH(self): return self._TH
    
    def setKleenePlusTH(self,th): self.fKleenePlusTH = th
    def getKleenePlusTH(self): return self.fKleenePlusTH
    
    def setSDC(self,s): self.sdc = s
    def getSDC(self): return self.sdc
           
    def setObjectLevel(self,o): self.objectLevel = o
    def getObjectLevel(self): return self.objectLevel
    
    def setMinSequenceLength(self,l): self.minSeqLength = l
    def getMinSequenceLength(self): return self.minSeqLength
    def setMaxSequenceLength(self,l): self.maxSeqLength = l
    def getMaxSequenceLength(self): return self.maxSeqLength
    
    
    def generateMSPSData(self,lseq,lfeatures,mis=0.01,L1Support=[]):
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
#                 print item, item.getSetofFeatures()
                if item.getCanonicalFeatures() is not None:
                    for fea in item.getCanonicalFeatures():
                        ## to avoid TH= 30 and 315 + 285 select! introduce issues with spm!
                        oldth=fea.getTH()
                        fea.setTH(1)
                        if fea in lfeatures and not fea in L1Support:
#                             print fea, lfeatures.index(fea)
                            tmpitem.append(fea)
                        fea.setTH(oldth)
                else:
                    pass
                    # an elt can have no feature: use EMPTYFeature?
                if tmpitem != []:
                    tmpseq.append(tmpitem)
                else:
                    pass
            db.append(tmpseq)
        return db,lMIS
        
        
        
    def generateItemsets(self,lElts):
        """
            generate itemset sequences of length 1 .. self.maxSeqLength
            return a list of list
            must be optimized in considering the frequency of elements.
                if iternediate structure (maxlength < maxlength cur): use them to delimit the itemset
                    ex: maxlength = 2 [249 132+] , [249 132+] , [249 132+], [249 132+], [249 132+]
                        -> 
        """
        lList = []

        lenElt=len(lElts)     
        j= 0 
        while j < lenElt:
            lList.append(lElts[j:j+self.maxSeqLength])
            j+=1
#             j+=self.maxSeqLength
        return lList
           
    def generateSequentialRules(self,lPatterns):
        """
            generate sequential rules from  a list pa (pattern,support)
        """
        dP={}
        lRules= []
        for p,s in lPatterns:
            dP[str(p)] = s
        for pattern,support in lPatterns:
            if support > 1:
                for i , itemset in enumerate(pattern):
                    if len(itemset) > 1:
                        for item in itemset:
                            newItemSet = itemset[:]
                            newItemSet.remove(item)
                            newPattern=pattern[:]
                            newPattern[i] = newItemSet
                            # need to sort 
#                             newPattern[i].sort()
                            if  str(newPattern) in dP:
                                fConfidence = 1.0 *support / dP[str(newPattern)]
#                                 if self.bDebug:print 'RULE: %s => %s[%d] (%s/%s = %s)'%(newPattern, item,i,dP[str(newPattern)],support, fConfidence)
                                if fConfidence >  self.THRULES: 
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
    
#     def patternToMV(self,pattern):
#         """
#             convert a list of f as multivaluefeature
#             keep mvf as it is 
#         """
#         from feature import multiValueFeatureObject 
#         
#         lmv= []
#         for itemset in pattern:
#             # if one item which is boolean: gramplus element
#             if len(itemset)==1 and itemset[0].getType()==1:
#                 lmv.append(itemset[0])
#             else:
#                 mv = multiValueFeatureObject()
#                 name= "multi" #'|'.join(i.getName() for i in itemset)
#                 mv.setName(name)
#                 mv.setValue(map(lambda x:x,itemset))
#     #             print mv.getValue()
#                 lmv.append(mv)
# #             print itemset, mv
#         return lmv    
    

    def treePatternToMV(self,pattern,dMtoSingleFeature,TH):
        """
            convert a list of f as multivaluefeature. This allows for multi-item itemset <a,b>
        """
        
        lmv= []
        for itemset in pattern:
#             print ":",pattern,'\t\t', itemset
            # if no list in itemset: terminal 
#             print map(lambda x:type(x).__name__ ,itemset)
            if  'list' not in map(lambda x:type(x).__name__ ,itemset):
                # if one item which is boolean: gramplus element
                # no longer used???
                if False and len(itemset)==1 and itemset[0].getType()==1:
                    lmv.append(itemset[0])
                else:
                    mv = multiValueFeatureObject()
                    name= "multi" #'|'.join(i.getName() for i in itemset)
                    mv.setName(name)
                    mv.setTH(TH)
                    mv.setValue(map(lambda x:x,itemset))
        #             print mv.getValue()
                    lmv.append( mv )
                    dMtoSingleFeature[str(mv)]=itemset
#                     print "$$$ ",mv, itemset
            # 
            else:
#                 print "\t->",list
                lmv.append(self.treePatternToMV(itemset, dMtoSingleFeature,TH))
#             print '\t',lmv
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
        
        
        
    def generateKleeneItemsets(self,lPages,lParsings,featureType):
        """
            detect if the pattern is +. if yes: parse the elements 
        """
        from ObjectModel.sequenceAPI import sequenceAPI
        lNewKleeneFeatures=[]
        lLNewKleeneSeq = []
        for gram,lFullList,ltrees in lParsings:
            ## create new feature
            ## collect all covered pages ans see if this is a kleene gram
            ## each lParsedPages: correspond to a s0+
            ##  fkleeneFactor = proportion of sequence wich are klenee
            ikleeneFactor=0
            gramLen=len(gram) * 1.0
            for lParsedPages, _,_ in ltrees:
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
                            kleeneElt._lBasicFeatures= [gramPlus]                       
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
        ## sort them for rule detection
        lPatterns = filter(lambda (p,s):len(p) >= self.getMinSequenceLength() and len(p) <= self.getMaxSequenceLength(),lPatterns)

        for p,s in lPatterns:
#             print p, len(p), self.getMinSequenceLength(), len(p) >= self.getMinSequenceLength(), len(p) <= self.getMaxSequenceLength()
            for i in p:
                i.sort(key=lambda x:x.getValue())
                
        # s= None !!!?? how 
        lPatterns =  filter(lambda (p,s):s is not None,lPatterns)    
        lPatterns= map(lambda (p,s): (p, round(1.0*len(p)*s/self.getMaxSequenceLength())),lPatterns)    

        return lPatterns
        

    def featureGeneration(self, lList, TH=2,featureType =featureObject) :
        """
            lList: global list : all elements 
            generate a set of features for the elements.
            merge duplicate features (but each feature points to a set of Element which created them)
            
            ??? ??? replace calibrated feature in the elt.getSetofFeatures() ()seqOfFeatures
            delete feature with a too low frequency 
            
            ?? REGROUP FEATURES PER node 
            when enriching, consider only features from other nodes?
        """
        ll=[]
        lFeatures = {}
        for elt in lList:
            elt._canonicalFeatures=None
#             if self.bDebug: print "\t",elt, str(elt.getSetofFeatures())
            for feature in elt.getSetofFeatures():
#                 print '\t\t',elt,feature
                ll.append(feature.getValue())
                try:lFeatures[feature].append(feature)
                except KeyError: lFeatures[feature] = [feature]
            
#         from sklearn import  mixture
#         print ll    
#         ll= np.array(ll).reshape(-1,1)
#         for n in range(1,20):
#             if len(ll)> n:
#                 gmm=mixture.GaussianMixture(n_components=n)
#                 gmm.fit(ll)
#                 print gmm.bic(ll)
#                 print gmm.means_
#                 print gmm.covariances_
                
                
        sortedItems = map(lambda x: (x, len(lFeatures[x])), lFeatures)
        sortedItems.sort(key=itemgetter(1), reverse=True)    
        
        lCovered = []
        lMergedFeatures = {}
        for i, (f, _) in enumerate(sortedItems):
#             print f, lFeatures[f], f.getID()
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
                                
        sortedItems = map(lambda x: (x, len(lMergedFeatures[x])), lMergedFeatures)
        sortedItems.sort(key=itemgetter(1), reverse=True)
#         print lList
#         print sortedItems
        lCanonicalFeatures = list()
        # creation of the canonical features
        for f, s in sortedItems:
#             print '\t',f,s, lMergedFeatures[f]
            ## update the value if numerical feature: take the mean and not the most frequent!!
            cf  = featureType()  # featureObject
            cf.setName(f.getName())
            cf.setType(f.getType())
            cf.setObjectName(None)
            cf.bCanonical = True
            cf.setTH(f.getTH())
            if f.getType() == featureObject.NUMERICAL:
                lvalues=map(lambda x:x.getValue(),lMergedFeatures[f])
                ## now values can be tuples!!
                try:             lweights = map(lambda x:1.0*x/len(lMergedFeatures[f]),lvalues)
                except TypeError:
                    lvalues=map(lambda x:x.getValue()[0],lMergedFeatures[f])
                    lweights = map(lambda x:1.0*x/len(lMergedFeatures[f]),lvalues)
                try:
                    myValue =  round(np.average(lvalues,weights=lweights),0)
                except ZeroDivisionError:
                    # Weights sum to zero, can't be normalized
                    myValue=f.getValue()
            else:
                myValue=f.getValue()
#             print '\t\t', myValue, lMergedFeatures[f]
            cf.setValue(myValue)
            cf.setCanonical(cf)
#             for ff in lMergedFeatures[f]:
#                 ff.setCanonical(cf)
            ## testif not already a cf which is in delta!!! 
            ## take the most frequent one: whihc is alomost in he list (sorte)
            if cf not in lCanonicalFeatures:
#                 indxf=lCanonicalFeatures.index(cf)
#                 print 'DUPLICATION', cf, lCanonicalFeatures[indxf]
#                 print 'new CF', cf
#                 sys.stdout.flush()
                lCanonicalFeatures.append(cf)
                for ff in lMergedFeatures[f]:
                    ff.setCanonical(cf)
#             else:
#                 indxf=lCanonicalFeatures.index(cf)
#                 print 'DUPLICATION', cf, lCanonicalFeatures[indxf]                
            
            
            # for normal features: need a'local' normalisation
            #here features are those of the elements
#             w=f.getWeight()
#             for ff in lMergedFeatures[f]:
#                 [f.addNode(ffn) for ffn in ff.getNodes()]
#                 w += sum(x.getHeight() for x in ff.getNodes())
#             ff.setWeight(w)
        
        for elt in lList:
            ltbd=[]
            for f in elt.getSetofFeatures():
#                 print elt, f, len(f.getNodes())
                if f.getCanonical() is None:
                    try:
                        indxf=lCanonicalFeatures.index(f)
                        f.setCanonical(lCanonicalFeatures[indxf])
                    except:pass 
                if f.getCanonical() is not None:
                    elt.addCanonicalFeatures(f.getCanonical())
                    assert f.getCanonical() is not None
                    for n in f.getNodes():
                        f.getCanonical().addNode(n)
            elt._lBasicFeatures = filter(lambda x:x not in ltbd,  elt.getSetofFeatures())
            if self.bDebug: print elt, elt.getSetofFeatures(), elt.getCanonicalFeatures()
        
        lCanonicalFeatures = filter(lambda x:len(x.getNodes()) > TH, lCanonicalFeatures)
        
        # create weights: # elt per cf / total
        ftotalElts=0.0
        for cf in lCanonicalFeatures:
            l = len(cf.getCanonical().getNodes())
            cf.setWeight(l)
            ftotalElts+=l
        for cf in lCanonicalFeatures:
            cf.setWeight(cf.getWeight()/ftotalElts)
        
        
        lCanonicalFeatures.sort(key=lambda x:x.getWeight(),reverse=True)
        lCanonicalFeatures = filter(lambda x:len(x.getNodes()) >= TH,lCanonicalFeatures)

        return lCanonicalFeatures
    
    
    
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
        for x,_ in lParsings:
            lFullList.extend(x)
        return (myPattern,lFullList,lParsings)



    def testTreeKleeneageTemplates(self,dTemplatesCnd,lElts,iterMax=5):
        """
            test a set of patterns
            select those which are kleene+ 
            Generate the transition probability accoridng to the treetemplate (to outsource!)
        
        """
        
        """
            resulting parsing can be used as prior for hmm/viterbi? yes
            but need to apply seqrule for completion or relax parsing
            
        """
        lFullPattern=[]
        lFullTemplates=[]
        dScoreFullTemplate={}
        lTemplates = []
        dScoredTemplates = {}
        for templateType in dTemplatesCnd.keys():
            iCpt=0
            iFound = 0
            lenMax = len(dTemplatesCnd[templateType])
            while iCpt < lenMax  and iFound < iterMax:
#             for _,_, mytemplate in dTemplatesCnd[templateType][:4]:
                _,_, mytemplate  = dTemplatesCnd[templateType][iCpt]
                ## need to test kleene+: if not discard?
                isKleenePlus ,parseRes,lseq = self.parseWithTreeTemplate(mytemplate,lElts,bReplace=False)
                ## assess and keep if kleeneplus
                if isKleenePlus:
                    ### if len=2 and [b,a] already exists: skip it !!
                    ### also if a bigger pattern contains it ???  (but not TA!!)
                    if len(mytemplate.getPattern())==2 and [mytemplate.getPattern()[1],mytemplate.getPattern()[0]] in  lFullPattern:
                        print 'mirrored already in', mytemplate
                    else:
                        lFullTemplates.append(mytemplate)
                        lFullPattern.append(mytemplate.getPattern())
                        iFound+= 1  
                        dScoreFullTemplate[mytemplate]=len(parseRes[1])                        
                        print 'add kleenedPlus template', mytemplate,len(parseRes[1])
                        ## traverse the tree template to list the termnimal pattern
    #                     print mytemplate,'\t\t', mytemplate,len(parseRes[1])
                        lterm= mytemplate.getTerminalTemplates()
                        for template in lterm:
                            if template not in lTemplates:
                                lTemplates.append(template)
                                dScoredTemplates[template] = len(parseRes[1])
                iCpt += 1
                
        lFullTemplates.sort(key=lambda x:dScoreFullTemplate[x],reverse=True)
#         return lFullTemplates
        #  transition matrix: look at  for template in page.getVerticalTemplates():
        N = len(lTemplates) + 1
        transProb = np.zeros((N,N), dtype = np.float16)
            
        dTemplateIndex = {}
        for i,template in enumerate(lTemplates):
#             print i, template
            dTemplateIndex[template]=i
        
            
        for i,elt in enumerate(lElts):
            # only takes the best ?
#             print elt, elt.getTemplates()
            ltmpTemplates = elt.getTemplates()
            if ltmpTemplates is None: ltmpTemplates=[]
            for template in ltmpTemplates:
                try:
                    nextElt=lElts[i+1]
                    lnextTemplates = nextElt.getTemplates()
                    if lnextTemplates is None: lnextTemplates=[]
                    for nextT in lnextTemplates:
#                         print i, template,nextT
                        ## not one: but the frequency of the template
                        try:
                            w = dScoredTemplates[template]
                            dTemplateIndex[nextT]
                        except KeyError:
                            #template from previous iteration: ignore
                            w= None
                        ## kleene 
                        if w is not None:
                            if nextT is template:
                                w +=  dScoredTemplates[template]
                            ##
                            if  (nextT == template) or (nextT.getParent() == template.getParent()):
                                                                                            # w+
                                transProb[dTemplateIndex[template],dTemplateIndex[nextT]] =   1 #+ transProb[dTemplateIndex[template],dTemplateIndex[nextT]]
                            if np.isinf(transProb[dTemplateIndex[template],dTemplateIndex[nextT]]):
                                transProb[dTemplateIndex[template],dTemplateIndex[nextT]] = 64000
                except IndexError:pass
        
        #all states point to None
#         transProb = np.zeros((N,N), dtype = np.float16)
#         for i in range(0,N):
#             transProb[i,i]=10
        transProb[:,-1] = 0.10 #/totalSum
        #last: None: to all
        transProb[-1,:] = 0.10 #/totalSum
        mmax =  np.amax(transProb)
        if np.isinf(mmax):
            mmax=64000
#         print transProb/mmax
    
        return lFullTemplates,lTemplates,transProb / mmax   
    
    def replaceEltsByParsing(self,lElts, lParsedElts,toprule,pattern):
        """
            replace the parsed sequence in lElts
            lTopRuleq = parsingNode() .  convert them as getObjectLevel
        """
        # index in lElts
        i = 0
        while i < len(lElts):
            if lElts[i] == lParsedElts[0]:
                del lElts[i:i+len(lParsedElts)]
                lElts.insert(i,toprule)
                i=len(lElts)
            i+=1
        return  lElts    
#     def replaceEltsByParsing(self,lElts, lParsedElts,pattern):
#         """
#             replace the parsed sequence in lElts
#         """
#         # index in lElts
#         i = 0
#         while i < len(lElts):
#             if lElts[i] ==lParsedElts[0]:
#                 del lElts[i:i+len(lParsedElts)]
#                 # create a new object. need to have similar featureGeneration properties
#                 newObject = self.getObjectLevel()() #objectClass()
#                 newObject.addAttribute("virtual",pattern)
# #                 newObject.setName(str(pattern))
#                 newObject.setObjectsList(lParsedElts)
# #                 newObject = lParsedElts[0]
#                 lElts.insert(i,newObject)
#                 i=len(lElts)
#             i+=1
#         return  lElts  
      
    def parseWithTreeTemplate(self,mytemplate,lElts,bReplace=False):
        """
            parse a set of lElts with a template
            since mutivalued itemset: need multiValueFeatureObject
        """

        PARTIALMATCH_TH = 1.0
        dMtoSingleFeature = {}
        mvPattern = self.treePatternToMV(mytemplate.getPattern(),dMtoSingleFeature, PARTIALMATCH_TH)
#         print mvPattern
#         print dMtoSingleFeature
        
        #need to optimize this double call        
        for elt in lElts:
                elt.setSequenceOfFeatures(elt.lFeatureForParsing)
#                 print elt, elt.lFeatureForParsing ,elt.getSetofFeatures()
                # an elt can have no feature
                try:
                    lf= elt.getCanonicalFeatures()[:]
                except: lf=[]
#                 print elt, lf 
                elt.resetFeatures()
                elt.setFeatureFunction(elt.getSetOfMutliValuedFeatures,TH = PARTIALMATCH_TH, lFeatureList = lf )
                elt.computeSetofFeatures()
#                 print elt, elt.getSetofFeatures()

        # what is missing is the correspondence between rule name (sX) and template element
        ## provide a template to seqGen ? instead if recursive list (mvpatern)
        
#         for e in lElts:
#             print "wwww",e, e.getSetofFeatures()
        lNewSequence = lElts[:]
        parsingRes = self.parseSequence(mvPattern,multiValueFeatureObject,lElts)
        isKleenePlus = False
        nbSeq = 0.0
        nbKleeneInSeq= 0.0 
        if parsingRes:
            self.populateElementWithTreeParsing(mytemplate,parsingRes,dMtoSingleFeature)
            _,_, lParse  = parsingRes
            for  lElts, rootNode in lParse:
#                 rootNode.print_(0)
                xx = rootNode.convertToObject(self.getObjectLevel())
                nbKleeneInSeq +=  rootNode.searchForRule("s0")
                nbSeq+=1
                if bReplace:
                    lNewSequence = self.replaceEltsByParsing(lNewSequence,lElts,xx,mytemplate.getPattern())            
            isKleenePlus = nbSeq > 0 and nbKleeneInSeq / nbSeq >= self.getKleenePlusTH()
            ### retrun nbKleeneInSeq / nbSeq and let decide at smpcomponent level (if pattern is mirrored: keep it if nbKleeneInSeq / nbSeq near 1.66 )
            print mytemplate, nbSeq, nbKleeneInSeq, nbKleeneInSeq / nbSeq
        return isKleenePlus,parsingRes, lNewSequence

    def populateElementWithTreeParsing(self,mytemplate,parsings,dMtoSingleFeature):
        """
            using the parsing result: store in each element its vertical template 
        """
        def treePopulation(node,lElts):
            """
                tree:
                    terminal node: tuple (tag,(tag,indexstart,indexend))
                
            """
            if node.isTerminal():
                    tag= node.getRule().name
                    # tag need to be converted in mv -> feature
                    subtemplate = mytemplate.findTemplatePartFromPattern(dMtoSingleFeature[tag])
                    node.getData().addTemplate(subtemplate)
#                     print node.getData(), node.getData().getTemplates()
            else:
                for subtree in node.getChildren(): 
                    treePopulation(subtree, lElts)
                    
        _, _, ltrees  = parsings
        ## each fragment of the parsing
        for lParsedPages,rootNode in ltrees:
            treePopulation(rootNode, lParsedPages)