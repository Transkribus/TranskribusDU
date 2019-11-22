#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

     H. Déjean

    copyright Naver Labs Europe 2018
    READ project 

    mine a document (itemset generation)
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0])))

from lxml import etree
import numpy as np

import common.Component as Component
from common.chrono import chronoOff , chronoOn

from spm.structuralMining import sequenceMiner
from spm.feature import featureObject 
from spm.feature import setOfPointsFeatureObject
from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSObjectClass import XMLDSObjectClass
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.XMLDSTOKENClass import XMLDSTOKENClass
from ObjectModel.XMLDSPageClass import XMLDSPageClass
from ObjectModel.treeTemplateClass import treeTemplateClass

from tasks.TwoDChunking import TwoDChunking

class pageVerticalMiner(Component.Component):
    """
        pageVerticalMiner class: a component to mine column-like page layout
    """
    
    
    #DEFINE the version, usage and description of this particular component
    usage = "" 
    version = "v.01"
    description = "description: page vertical Zones miner "

    kContentSize ='contentSize'
    
    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "pageVerticalMiner", self.usage, self.version, self.description) 
        
        
        # tag level
        self.sTag= XMLDSTEXTClass
        # TH for comparing numerical features for X
        self.THNUMERICAL = 30
        self.testTH = 30  # use dfor --test  BUT ALSO IN RUN !!
        # use for evaluation
        self.THCOMP = 10
        self.evalData= None
        
        self.bDomTag=True
        
        # TH for sequentiality detection (see structuralMining)
        self.fKleenPlusTH =1.5

        # pattern provided manually        
        self.bManual = False
        
        
        # number of items in an itemset
        self.nbSeqItem = 1
        
        # evaluation using the baseline
        self.baselineMode = 0
        
        # ignore existing regions
        self.bIgnoreRegions=True
        # do not use graphical lines
        self.bNOGline = False
        
        # only use GL
        self.bGLOnly= False
        
        # does the scanning introduce vertical shift? 
        self.bScanShift = False
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if "pattern" in dParams: 
            self.manualPattern = eval( dParams["pattern"])
            self.bManual=True  

        if "THNUM" in dParams:
            self.testTH =   dParams["THNUM"]

        if "KLEENETH" in dParams:
            self.fKleenPlusTH =   dParams["KLEENETH"]
        

        if 'baseline' in dParams:
            self.baselineMode = dParams['baseline']     

        if 'bIgnoreRegions' in dParams:
            self.bIgnoreRegions = dParams['bIgnoreRegions']
            
        if 'nogline' in dParams:
            self.bGLOnly = dParams['nogline']
        
        if 'glonly' in dParams:
            self.bNOGline = dParams['glonly']
            
        if 'tag' in dParams:
            self.sTag = dParams['tag']                              
            
    def minePageDimensions(self,lPages):
        """
            use page dimensions to build highest structure
            
            need iterations!
        """
        self.THNUMERICAL = 60  # 2 cml
        ## initialization for iter 0
        for page in lPages:
            page.setFeatureFunction(page.getSetOfFeaturesPageSize,self.THNUMERICAL)
            page.computeSetofFeatures()
            
        seqGen = sequenceMiner()
        seqGen.setMaxSequenceLength(1)
        seqGen.setObjectLevel(XMLDSPageClass)

        lSortedFeatures  = seqGen.featureGeneration(lPages,2)

        for _,p in enumerate(lPages):
            p.lFeatureForParsing=p.getSetofFeatures()
#             print (p, p.lFeatureForParsing)
        icpt=0
        lCurList=lPages[:]
        lTerminalTemplates=[]
        while icpt <=0:
            if icpt > 0: 
                #                           N ?
                seqGen.setMaxSequenceLength(1)
#                 print '***'*20
                seqGen.bDebug = False
                for elt in lCurList:
                    if elt.getSetofFeatures() is None:
                        elt.resetFeatures()
                        elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['virtual'],myLevel=XMLDSPageClass)
                        elt.computeSetofFeatures()
                        elt.lFeatureForParsing=elt.getCanonicalFeatures()
                    else:
                        elt.setSequenceOfFeatures(elt.lFeatureForParsing)
                lSortedFeatures  = seqGen.featureGeneration(lCurList,1)
            lmaxSequence = seqGen.generateItemsets(lCurList)
            seqGen.bDebug = False
            
            # mis very small since: space is small; some specific pages can be 
            ## replace by PrefiwScan
            lSeq, _ = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures + lTerminalTemplates,mis = 0.002)
            lPatterns = seqGen.miningSequencePrefixScan(lSeq,minSupport=0.01,maxPatternLength=3)
#             lPatterns = seqGen.beginMiningSequences(lSeq,lSortedFeatures,lMIS)
            if lPatterns is None:
                return [lPages]
            
            # ignore unigram:  covered by previous steps 
            if icpt < 3:
                lPatterns  = list(filter(lambda p_s:len(p_s[0][0])>1,lPatterns))
            lPatterns.sort(key=lambda x_y:x_y[1], reverse=True)

            seqGen.bDebug = False
            seqGen.THRULES = 0.8
            lSeqRules = seqGen.generateSequentialRules(lPatterns)
            _,dCP = self.getPatternGraph(lSeqRules)
            
            dTemplatesCnd = self.pattern2PageTemplate(lPatterns,dCP,icpt)
            
            #no new template: stop here
            if dTemplatesCnd == {}:
                icpt=9e9
                break
            _,lTerminalTemplates,_ = seqGen.testTreeKleeneageTemplates(dTemplatesCnd, lCurList)
        
#             print tranprob
#             self.pageSelectFinalTemplates(lTerminalTemplates,tranprob,lCurList)
            ## store parsed sequences in mytemplate 
            for templateType in dTemplatesCnd.keys():
                for _,_, mytemplate in dTemplatesCnd[templateType]:
#                     mytemplate.print_()
#                     _,lCurList = self.parseWithTemplate(mytemplate,lCurList,bReplace=True)
                    _,_,lCurList = seqGen.parseWithTreeTemplate(mytemplate, lCurList, bReplace=True)    
                    for elt in lCurList:
                        if elt.getSetofFeatures() is None:
                            elt.resetFeatures()
                            elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['virtual'],myLevel=XMLDSPageClass)
                            elt.computeSetofFeatures()
                            elt.lFeatureForParsing=elt.getSetofFeatures()
            
            icpt +=1
            if self.bDebug:self.printTreeView(lCurList)
        
        lList = self.getFlatStructure(lCurList)
#         print lList
        del seqGen        
        # return also the tree ; also organize elements per level/pattern
        return lList 
    
  
    def pattern2PageTemplate(self,lPatterns,dCA,step):
        """
            select patterns and convert them into appropriate templates.
            
            Need to specify the template for terminals; or simply the registration function ?
        """
        dTemplatesTypes = {}
        for pattern,support in filter(lambda x_y:x_y[1]>1,lPatterns):
            try:
                dCA[str(pattern)]
                bSkip = True
#                 print 'skip:',pattern
            except KeyError:bSkip=False
            # first iter: keep only length=1
            # test type of  patterns
            bSkip = bSkip or (step > 0 and len(pattern) == 1)
            bSkip = bSkip or (len(pattern) == 2 and pattern[0] == pattern[1])  
#             print pattern, bSkip          
            if not bSkip: 
                print ('========',pattern, support, self.isMirroredPattern(pattern))
                ## test is pattern is mirrored         
                template  = treeTemplateClass()
                template.setPattern(pattern)
                template.buildTreeFromPattern(pattern)
                template.setType('lineTemplate')
                try:dTemplatesTypes[template.__class__.__name__].append((pattern, support, template))
                except KeyError: dTemplatesTypes[template.__class__.__name__] = [(pattern,support,template)]                      
        
        return dTemplatesTypes
        
    def pattern2TAZonesTemplate(self,lPatterns,dCA):
        """
            TA patterns
            select patterns and convert them into appropriate templates.
        """
        dTemplatesTypes = {}
        for pattern,support in filter(lambda x_y:x_y[1]>1,lPatterns):
            bSkip=False
            try:
                dCA[str(pattern)]
                bSkip = True
#                 if len(pattern)==2 and len( pattern[0] ) == len( pattern[1]) == 2:
            except KeyError:bSkip=False
            if not bSkip:
    #             if ( len(pattern) == 1 and len(pattern[0])== 2  and pattern[0][0].getValue() !=pattern[0][1].getValue()) or (len(pattern)==2 and len( pattern[0] ) == len( pattern[1] ) == 2 ):
                if ( len(pattern) == 1 and len(pattern[0]) >= 1) :
    #                 print pattern, support
                    template  = treeTemplateClass()
                    template.setPattern(pattern)
                    template.buildTreeFromPattern(pattern)
                    template.setType('lineTemplate')
                    try:dTemplatesTypes[template.__class__.__name__].append((pattern, support, template))
                    except KeyError: dTemplatesTypes[template.__class__.__name__] = [(pattern,support,template)]                      
        
        
        for ttype in dTemplatesTypes.keys():
            dTemplatesTypes[ttype].sort(key=lambda x_y_t:len(x_y_t[0]), reverse=True)
        return dTemplatesTypes


    def isCorrectPattern(self,pattern):
        """
            if length = 1: at least 2 elements (one zone)
            if length =2: 
                - same number of elements
                - at least one width similar
        """
        if len(pattern) == 1:
                                        # stil useful?
            return len(pattern[0])>=2  and pattern[0][0].getValue() !=pattern[0][1].getValue()
         
        elif len(pattern) == 2:
#             if len(pattern[0]) != 2: return False
#             bOK =  len( pattern[0] ) == len( pattern[1] ) >= 2
            # same width: the longest width must be shared
            inv1 =  pattern[1][:]
            lcouple1= zip(inv1,inv1[1:])
            lw1= map(lambda x_y:abs(x_y[1].getValue()-x_y[0].getValue()),lcouple1)
            max1 =max(lw1)
            lcouple0= zip(pattern[0],pattern[0][1:])
            lw0= map(lambda x_y:abs(x_y[1].getValue()-x_y[0].getValue()),lcouple0)
            max0= max(lw0)
            ## all width similar???
#             print pattern, zip(lw0,lw1) , max0,max1, abs(max1 - max0) < self.THNUMERICAL*2
            return       abs(max1 - max0) < self.THNUMERICAL*2
            
    def pattern2VerticalZonesTemplate(self,lPatterns,dCA):
        """
            select patterns and convert them into appropriate templates.
            
            Need to specify the template for terminals; or simply the registration function ?
        """
        dTemplatesTypes = {}
        iNbClosed = 0
        for pattern,support in filter(lambda x_y:x_y[1]>1,lPatterns):
            bSkip=False
            try:
                dCA[str(pattern)]
                bSkip = True
#                 if len(pattern)==2 and len( pattern[0] ) == len( pattern[1]) == 2:
            except KeyError:bSkip=False

            # duplicated a,b
            bSkip = bSkip or (len(pattern) == 2 and pattern[0] == pattern[1])
            if not bSkip: 
                iNbClosed+=1
                if True:#( len(pattern) == 1 and len(pattern[0])>=2  and pattern[0][0].getValue() !=pattern[0][1].getValue()) or (len(pattern)==2 and len( pattern[0] ) == len( pattern[1] ) >= 2  ):
                    if  True : #self.isCorrectPattern(pattern):
                        template  = treeTemplateClass()
                        template.setPattern(pattern)
                        template.buildTreeFromPattern(pattern)
                        template.setType('VTemplate') # ??
#                         print (pattern)
#                         template.print_()
                        try:dTemplatesTypes[template.__class__.__name__].append((pattern, support, template))
                        except KeyError: dTemplatesTypes[template.__class__.__name__] = [(pattern,support,template)]                      
        
        print ("patterns2template: ", iNbClosed)
        for ttype in dTemplatesTypes.keys():
#             dTemplatesTypes[ttype].sort(key=lambda (x,y,t):len(x[0]), reverse=True)
            dTemplatesTypes[ttype].sort(key=lambda p_s_t:p_s_t[1], reverse=True)

        return dTemplatesTypes
    
    
    
    def computeObjectProfile(self,lPages):
        """
            Compute the following information:
            * for Pages
                - surface covered by lines
                - # lines per page : issue with nosy elements 
                - avg BB
            * for lines
                - avg width  (gaussian?)
                - avg height (gaussian)? would be strange
            
        """
        lPageProfiles={}
        lPageProfiles[self.kContentSize]=[]
        for page in lPages:
            lElts= page.getAllNamedObjects(XMLDSTEXTClass)
            surface= sum( (x.getWidth()*x.getHeight() for x in lElts))/ (page.getWidth()*page.getHeight())
            lPageProfiles[self.kContentSize].append(surface)
        return lPageProfiles
    
    
    def minePageVerticalFeature2D(self,lPages,lFeatureList,level=XMLDSTEXTClass):
        """
            Input:
                lPages: list of pages
                lFeatureList: list of features to be used 
                level: level ot be used
            Output: pages decorated with features
            
        """ 
        chronoOn()
        lLElts=[ [] for i in range(0,len(lPages))]
        
        
        # here use Vertical cluster
        for i,page in enumerate(lPages):
#             lElts = page._dVC[level.name]
            lElts= page.getAllNamedObjects(level)
            lElts.sort(key=lambda x:x.getX())
            lLElts[i]=lElts
        
        ### VerticalZones START
        for i,page, in enumerate(lPages):
            page._VX1X2Info=[]
            page._Vpartitions = []
            page.setX1X2([])
            page.resetFeatures()
            lElts= lLElts[i]
            for elt in lElts:
                elt.resetFeatures()
                elt._canonicalFeatures = None
                elt.setFeatureFunction(elt.getSetOfX1X2Attributes,20,lFeatureList=lFeatureList,myLevel=level)
                elt.computeSetofFeatures()
#                 print (page,elt.getX(),elt.getY(),elt.getSetofFeatures())
            
            ## select regular x                                 
            seqGen = sequenceMiner()
#             seqGen.bDebug  =self.bDebug
            _fullFeatures =   seqGen.featureGeneration(lElts,1)
            for fx in _fullFeatures:
#                 fx.setWeight(sum(x.getHeight() for x in fx.getNodes())/64000)
#                 fx.setWeight(sum(x.getHeight()  for x in fx.getNodes()))
                fx.setWeight(sum(x.getHeight()  * x.getWidth() for x in fx.getNodes()))
#                 fx.setWeight(len(fx.getNodes()))
#                 print(fx,fx.getWeight())
            page.setX1X2(_fullFeatures)
            del seqGen
        
        self.buildPartitions(lPages)
        print ('chronoFeature',chronoOff())
        
        return lPages    
        
    
    
    def checkPath(self,path):
        
        if len(path) <2: return []
        curmax = path[0].getValue()[-1]
        i = 1
        bOK = True
        while i < len(path) and bOK:
#             print (path[i],path[i].getValue())
            bOK = path[i].getValue()[0] >= curmax * 0.95 and path[i].getValue() 
            curmax = path[i].getValue()[-1] 
            i += 1
#         if bOK: print (path)
        return bOK
#         for i,xx2 in enumerate(path):
#             if xx2.getValue()[0] >= curmax * 0.95 and xx2.getValue()[-1] > cur.getValue()[-1] * 0.95:
#                 lRecList.extend(ln)
        
    def recPathGeneration(self,curSeq,lSeq):
        """
            input: list of (x1,x2)  features
            output: comb generation of all possible layout
        """
        cur,lRest = lSeq[0],lSeq[1:]
        curSeq.append(cur)
        lRecList=[curSeq]
        curmax=cur.getValue()[-1]
#         print ("x",cur,lRest,curmax)
        for i,xx2 in enumerate(lRest):
#             print (xx2, xx2.getValue()[0] , curmax)
            if xx2.getValue()[0] >= curmax * 0.95 and xx2.getValue()[-1] > cur.getValue()[-1] * 0.95:
                curmax= xx2.getValue()[-1]
                ln = self.recPathGeneration(curSeq[:],lRest[i:])
#                 print ('\t',ln[0][0].getValue()[0],ln[-1][-1].getValue()[-1],ln)
                lRecList.extend(ln)
#                 print (lRecList[-1])
        return lRecList
            
            
        
    def buildPartitions(self,lp):
        """
            input: list of pages with horizontal 2Dfeatures    
            output: list of possible horizontal compatible partitions (sequences of horizontal stuff)
        """
        import itertools
        for page in lp:
            lLSeq=[]
            page.resetFeatures()
#             print (page,len(page.getX1X2()))
            page.getX1X2().sort(key=lambda x:x.getValue()[0])
            if self.nbSeqItem == 1:
                [lLSeq.append([x]) for x in page.getX1X2() ]
            else: 
                for x in itertools.chain(
                        itertools.combinations(page.getX1X2(), self.nbSeqItem) , \
#                     itertools.combinations(page.getX1X2(), 2) , \
#                                         itertools.combinations(page.getX1X2(), 1) , \
#                                         itertools.combinations(page.getX1X2(), 3) , \
    #                                     itertools.combinations(page.getX1X2(), 6)
                                        ):
                    if self.checkPath(x):
                        lLSeq.append(x)
#                         print (x)
#             lLSeq=[]
#             for i,x in enumerate(page.getX1X2()):
#                 lLSeq.extend(self.recPathGeneration([], page.getX1X2()[i:]))
            lLSeq.sort (key=lambda lx: sum(map(lambda x:x.getWeight(),lx)),reverse=True)
            lSet= []
#             print (page, len(lLSeq))
            for ln in lLSeq[:50]:
#                 print (ln)
                if ln not in lSet:
                    if True: #sum(map(lambda x:x.getWeight(),ln))/len(ln)> page.getHeight()*0.125:
                        feature = setOfPointsFeatureObject()
                        feature.setName('seqV')
                        # here TH cannot be higher than the x2-x1
                        feature.setTH(30)  # 20
                        feature.addNode(page)
                        feature.setObjectName(page)
                        feature.setValue(list(map(lambda x:x.getValue(),ln)))
                        feature.setType(featureObject.COMPLEX)
                        feature.setWeight(sum(map(lambda x:x.getWeight(),ln))/len(ln))
                        feature.setWeight(1)

                        page.addFeature(feature)
                        lSet.append(ln)
            if page.getSetofFeatures() is not None:
                page._Vpartitions=page.getSetofFeatures()[:]
            else:page._Vpartitions= None
        
            
    def getFlatStructure(self,lElts,level=1):
        
        lRes=[]
        for elt in lElts:
            if elt.getAttribute('virtual'):
                lRes.append(self.getFlatStructure(elt.getObjects(),level+1))
            else:
                lRes.append([elt])
        try:
            if len(lRes) == len([ x for y in lRes for x in y]):
                lRes= [ x for y in lRes for x in y]
        except TypeError:pass
                 
        return lRes
    
    
    
    def printTreeView(self,lElts,level=0):
        """
            recursive 
        """
        for elt in lElts:
            if elt.getAttribute('virtual'):
                print ("  "*level, 'Node', elt.getAttribute('virtual'))
                self.printTreeView(elt.getObjects(),level+1)
#             else:
#                 print ("  "*level, elt.getContent(), elt.lFeatureForParsing)            
    
   
    
     
    def processWithTemplate(self,lPattern,lPages):
        """
            process sequence of pqges with given pattern
            create table 
        """
        
        
        ## artifical construction Motter12
#         lPattern= [ [41.0, 110,442.0] , [40,340,442] ]
        ## MO12 pattern2 
#         lPattern= [ [41.0, 110,250,340,420,442.0]]
        
        ## RRB 
#         lPattern= [ [15,97.0, 311.0]]
        #MF012
#         lPattern= [ [27.0, 361,430.0] , [27.0,86.0,430.0] ]
#         lPattern = [[19,104,277,371,470,],[19,60,104,371,470]]
#         #nn_n0171 (hub)  [['x=295.0', 'x=850.0'], ['x=34.0', 'x=572.0']]
#         lPattern = [ [34.0, 564.0, 738.0], [156.0, 339.0, 846.0] ]
#         lPattern = [[295,850,],[34,572]]
        #lib
#         lPattern = [[28,321],[144,449]]
        
#         lPattern = [ [144.0, 449]]
 
        lfPattern= []
        for itemset in lPattern:
            fItemset = []
            for item in itemset:
                f= featureObject()
                f.setName("x")
                f.setType(featureObject.NUMERICAL)
                f.setValue(item)
                f.setTH(self.THNUMERICAL)                
                fItemset.append(f)
                f.setWeight(64000)
            lfPattern.append(fItemset)
    
        pattern = lfPattern
        
        print (pattern)
        print (self.THNUMERICAL)
        
        maintemplate = treeTemplateClass()
        maintemplate.buildTreeFromPattern(pattern)
        
        ### in prodf: mytemplate given by page.getVerticalTemplates()
        mytemplate1 = treeTemplateClass()
        mytemplate1.setPattern(pattern[0])
        
        mytemplate2 = treeTemplateClass()
#         mytemplate2.setPattern(pattern [1])
        if len(lPattern)==2:
            mytemplate2.setPattern(pattern[1])
            mytemplate2.setParent(maintemplate)
            mytemplate1.setParent(maintemplate)

        else:
            mytemplate2.setPattern(pattern[0])
        
        # registration provides best matching
        ## from registration matched: select the final cuts
        lScores = []
        for i,p in enumerate(lPages):
            if i %2==0:
                mytemplate= mytemplate1
            else:
                mytemplate = mytemplate2
            p.lFeatureForParsing  = p.lf_XCut 
#             print p, p.lf_XCut
            sys.stdout.flush()
            registeredPoints, _, score = mytemplate.registration(p)
#             registeredPoints2, lMissing2, score2 = mytemplate2.registration(p)
#             print i,p,registeredPoints
            # if score1 == score 2 !!
            if score > 0 : # and score1 >= score2:
                lfinalCuts= list(map(lambda x_y:x_y[1],list(filter(lambda x_y:x_y[0] != 'EMPTY',registeredPoints))))
#                 print p,'final1:',lfinalCuts, lMissing, score, '\t\t',registeredPoints
                lScores.append(score)
                p.addVerticalTemplate(mytemplate)
                p.addVSeparator(mytemplate,lfinalCuts)
#             elif score2 > 0 and  score2 > score1:
#                 lfinalCuts= map(lambda x_y:x_y[1],filter(lambda (x,y): x!= 'EMPTY',registeredPoints2))
#                 print registeredPoints2
#                 print 'final2:',lfinalCuts, lMissing2
#                 p.addVerticalTemplate(mytemplate2)
#                 p.addVSeparator(mytemplate,lfinalCuts)
            else:
                print ('NO REGISTRATION')
            fscore= np.average(lScores)                
        print ('final score:', fscore)
        self.tagAsRegion(lPages)
        
        return 1
    
    
    
    
    def computePatternScore(self,pattern):
        """
            consider the frequency of the pattern and the weights of the features
        """
        fScore = 0
        #terminal
        if  not isinstance(pattern, list):
            fScore += pattern.getCanonical().getWeight()
        else:
            for child in pattern:
                fScore += self.computePatternScore(child)
#         print 'score:',pattern ,fScore
        return fScore         
    
    
    
    def filterNonRegularPatterns(self,lPatterns):
        """
            a template must provide a simplr way to compute the reading order
            when a template is unigram: no issue
            when a template is bigram: it has to be balanced (same number of vertical zones)
        """
        
        """
            it has an impact on getPatternGraph
            
            p = [
               ['seqV=[(83.04, 602.4), (612.0, 734.88)]'], 
               ['seqV=[(77.76, 204.0), (215.04, 740.64)]', 'seqV=[(215.04, 740.64)]']
               ] 2
        """
#         for p,f in lPatterns:
#             print (p, len(p))
        
        # for 2D zones 
        #         return  list(filter(lambda x_y: (len(x_y[0]) != 2) or (len(x_y[0][0])==len(x_y[0][1])) , lPatterns))

        xx =  list(filter(lambda x_y: len(x_y[0]) ==1 or (len(x_y[0][0]) == len(x_y[0][1]) == 1) , lPatterns))
        yy = list(filter(lambda x_y: len(x_y[0]) == 1 or (len(x_y[0][0][0].getValue())==len(x_y[0][1][0].getValue())) , xx))
#         for p in yy:
#             print ('??',p, p[0][0][0],p[0][1][0])
        return yy



    def highLevelSegmentation(self,lPages):
        """
            use: image size and content (empty pages)
        """
        
        lSubList= self.minePageDimensions(lPages)

        return lSubList

        # useful???? seems there is a 'batch size' latter on
        lNewSub=[]
        for lp in lSubList:
            lProfiles =  self.computeObjectProfile(lp)
            lS = self.segmentWithProfile(lp,lProfiles[self.kContentSize])
            lNewSub.extend(lS)

        
        return lNewSub
            

    def segmentWithProfile(self,lPages,lListSurface):
        """
            compute average surface, min, max
            and find a threshold for "empty" pages
        """
        mean = np.mean(lListSurface)
        std = np.std(lListSurface)
#         print mean, std
        lSubList=[]
        lcur=[]
        for i,surface in enumerate(lListSurface):
#             print i, surface, surface < mean-std*2
            if surface < mean-std*2:
                lcur.append(lPages[i])
                lSubList.append(lcur)
                lcur =[]
#                 print lSubList, lcur
            else:
                lcur.append(lPages[i])
        
        if lcur !=[]:
            lSubList.append(lcur)
        
        return lSubList
        
        
    def testHighSupport(self,sequences):
        """
            compute unigram support
        """
        # from mssp
        from collections import Counter
        import  itertools
        
        sequence_count = len(sequences)
 
        flattened_sequences = [ list(set(itertools.chain(*sequence))) for sequence in sequences ]
        support_counts = dict(Counter(item for flattened_sequence in flattened_sequences for item in flattened_sequence))
        actual_supports = {item:support_counts.get(item)/float(sequence_count) for item in support_counts.keys()}
#         for x in actual_supports: print (x,actual_supports[x])
        lOneSupport= [k for k,v in actual_supports.items() if v >= 0.998 ]
        return lOneSupport

#         lOneSupport=[]
#         dFeatures={}
#         for item in lElts:
#             if item.getCanonicalFeatures() is not None:
#                 for fea in item.getCanonicalFeatures():
#                     try:dFeatures[fea] += 1
#                     except KeyError:dFeatures[fea] = 1
#         
#         for key in dFeatures:
#             print key, dFeatures[key], 1.0* dFeatures[key]/len(lElts)
#             if 1.0* dFeatures[key]/len(lElts)> 0.80:
#                 lOneSupport.append(key)
#         return lOneSupport
        
        
        
#     def baselineSegmentation(self,lLPages):
#         """
#             select the n 'best' cuts per pages
#         """
#         
#         NBEST = self.baselineMode
#         
#         for lPages in lLPages:
#             
#             #Width 
#             self.minePageVerticalFeature(lPages, ['width'])
#             lT, lScore,score = self.processVSegmentation(lPages,[],bTAMode=True,iMinLen=1,iMaxLen=1)
#             print score#, lScore
#             # here segmentation of lPages if needed (put outside of this loop)
#             
#             lsubList = [lPages]
# 
#             
#             for p  in lPages:
#                 p.resetVerticalTemplate()
# 
#             # V positions
#             NBATCH = 100
#             for nbPage in range(0,len(lPages),NBATCH):
#                 
#                 print nbPage, nbPage + NBATCH
#                 sys.stdout.flush()
#                 print "LENGTH = 1"
#                 # global or local??
#                 self.minePageVerticalFeature(lPages[nbPage:nbPage+NBATCH], ['x','x2'])
#                 ## V ZONES
#                 # length 1
# 
#                 for _,p in enumerate(lPages[nbPage:nbPage+NBATCH]):
#                     p._lBasicFeatures=p.lf_XCut[:]
#         #             print p, map(lambda x:(x,x.getWeight()),p.lf_XCut)
#                 
#                 seqGen = sequenceMiner()
#                 seqGen.bDebug = False
#                 seqGen.setMinSequenceLength(1)
#                 seqGen.setMaxSequenceLength(1)
#                 seqGen.setObjectLevel(XMLDSPageClass)
#         
#                 ## sdc: support difference constraint 
#                 seqGen.setSDC(0.7) # before 0.6
#         
#                 chronoOn()
#                 lSortedFeatures = seqGen.featureGeneration(lPages[nbPage:nbPage+NBATCH],2)
#                 print 'featuring...',chronoOff()
#                 #  done in featureGeneration
#                 for cf in lSortedFeatures:
#                     cf.setWeight(sum(x.getHeight() for x in cf.getNodes()))
# #                     print cf, cf.getWeight()
#                 print lSortedFeatures                      
#                 for _,p in enumerate(lPages[nbPage:nbPage+NBATCH]):
# #                     p.lFeatureForParsing = p.getCanonicalFeatures()[:NBEST]
#                     p.lFeatureForParsing = p.lf_XCut[:NBEST]
# 
# #                     print p, map(lambda x: (x,x.getWeight()),p.lFeatureForParsing)
#                     template=treeTemplateClass()
#                     template.buildTreeFromPattern(p.lFeatureForParsing)
#                     p.addVerticalTemplate(template)
#                     p.addVSeparator(template,template.getPattern())
#             
#             self.tagAsRegion(lPages) 
        
        
      
    def contrastive_loss(self,y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        
        see https://github.com/fchollet/keras/issues/1866
        '''
        margin = 1
        return np.mean(y_true * np.square(y_pred) +
                      (1 - y_true) * np.square(np.maximum(margin - y_pred, 0)))

    def euclidean_distance(self,x,y):
        return np.sqrt(np.sum(np.square(x - y)))       
     
 
    def iterativeProcessVSegmentation(self, lLPages):
        """
            process lPages by batch 
           
           parameter: NBATCH,  THNUM?
           
           
           W unigram: if not issue!!! this is mandatory!!
               -> with table???
               very ince for segmenting: it means a different template ??
               
            for W /TA
                identify sequences where empty states are majoritary (By dichomomy? )
                
            collect information for cut !!  a la specbook!!!
        """
        
        MAXELT = 12
        
        setFullListTemplates= set() 
        for lPages in lLPages:
            
#             lsubList = [lPages]

            self.THNUMERICAL = self.testTH
            for p  in lPages:
                p.resetVerticalTemplate()

#             self.bDebug=True
            # V positions
            NBATCH = 50
            for nbPage in range(0,len(lPages),NBATCH):
#                 tt=lPages[nbPage:nbPage+NBATCH:2]
                sys.stdout.flush()
                print ("LENGTH = 1")
#                 self.minePageVerticalFeature(lPages[nbPage:nbPage+NBATCH], ['x','x2'],level=self.sTag)
#                 
                lT1=None
                score1 = None
                maxNbElt = 1
#                 self.minePageVerticalFeature(lPages[nbPage:nbPage+NBATCH], ['x'],level=self.sTag)
                ## V ZONES
                # length 1
#                 for nbelt in [0]:
                for nbelt in range(MAXELT,0,-1):
                    print ("##################### %s ################" %nbelt)
                    self.nbSeqItem = nbelt
                    self.minePageVerticalFeature2D(lPages[nbPage:nbPage+NBATCH], [],level=self.sTag)
                    lT1, lScore1,score1 = self.processVSegmentation(lPages[nbPage:nbPage+NBATCH],[],bTAMode=False,iMinLen=1,iMaxLen=1)
                    if lT1 is not None:
                        maxNbElt=nbelt
                        break
                    ## stop as soon as a template found?
                
#                 print( nbPage, nbPage+NBATCH, lT1, score1)
#                 print( '\t',lScore1)
                sys.stdout.flush()
                bTable = lT1 is not None and lT1 !=[] and len(lT1[0].getPattern()) > 6
                lOneSupport=[]
                print( "LENGTH = 2")
                bTable = False
                if len( lPages) > 1 and bTable or len(lOneSupport) > 18:
                    score2=0
                    lT2=None
                    lScore2=[]
                else:
                    if lT1 is not None: 
                        lNegativesPatterns=list(map(lambda x:x.getPattern(),lT1))
                    else: lNegativesPatterns=[]
#                     lT2, lScore2, score2 = self.processVSegmentation(lPages[nbPage:nbPage+NBATCH],lNegativesPatterns,bTAMode=False,iMinLen=2,iMaxLen=2)
#                     for nbelt in [0]:
                    for nbelt in range(MAXELT,maxNbElt-1,-1): #[12,11,10,9,8,7,6,5,4,3,2,1]:
                        print ("##################### %s ################" %nbelt)
                        self.nbSeqItem = nbelt
                        self.minePageVerticalFeature2D(lPages[nbPage:nbPage+NBATCH], [],level=self.sTag)
                        lT2, lScore2, score2 = self.processVSegmentation(lPages[nbPage:nbPage+NBATCH],lNegativesPatterns,bTAMode=False,iMinLen=2,iMaxLen=2)
                        if lT2 is not None: break
    #                 score2=0
    #                 lT2=None
    #                 lScore2=[]
                    # test if cut somewhere
                    #  segment and relearn : if better score: keep the cut 
                    print (nbPage, nbPage+NBATCH, lT2, score2)    
                    print ('\t',lScore2)
#                     for ii,(_,score) in enumerate(lScore2):
#                         print '\t',ii,score
                    
                    
                bTwo=False
                lT=None
                # update  
                if score1 is None: score1 = -1
                if score2 is None:score2 = -1
                if score2 > score1:
                    bTwo=True
                    ldeltemplateset=lT1
                    lT=lT2
                else:
                    ldeltemplateset=lT2
                    lT=lT1
                if ldeltemplateset:
                    for p in lPages[nbPage:nbPage+NBATCH]: 
    #                     print p.getVerticalTemplates()
                        for deltemplate in ldeltemplateset:
                            try:
                                p.getVerticalTemplates().remove(deltemplate)
                            except ValueError:pass  # page not associated  
                
                print ("#",lPages[nbPage:nbPage+NBATCH], bTwo , lT)
            
            
            ## final step: finetuning and creation of separator???
            ## consider tokens and find best line: even correction with new nomacs??
            

            # for visu
            if self.bDomTag:self.tagAsRegion(lPages)

            ## build full list of templates
            ### smooth: see bar 13685:  if subpatterns  (due to content break)
            for p in lPages:
                #  what is stored in VerticalTemplate: the terminal element: take the parent for the full template
                ## this is needed by the Viterbi/matching step
                for t in p.getVerticalTemplates():
                    parentTemplate = t.getParent()
                    setFullListTemplates.add(parentTemplate)
                    if parentTemplate is not None:
                        parentTemplate.bIsMirrored = self.isMirroredPattern(parentTemplate.getPattern())
        
#         print setFullListTemplates

        return setFullListTemplates
        

                    
    def processVSegmentation(self,lPages,lNegativePatterns,bTAMode= False,iMinLen=1, iMaxLen=1):
        """
            use Vertical bloc/text info to find vertical patterns at page level
            Should correspond to column-like zones    
            
        """
        seqGen = sequenceMiner()
        seqGen.bDebug = False
        seqGen.setMinSequenceLength(iMinLen)
        seqGen.setMaxSequenceLength(iMaxLen)
        seqGen.setObjectLevel(XMLDSPageClass)

        chronoOn()

        for _,p in enumerate(lPages):
            p._lBasicFeatures = p._Vpartitions
#             print (p,p._lBasicFeatures)                   
#         lSortedFeatures = seqGen.featureGeneration(lPages,2)
        lSortedFeatures = seqGen.featureGeneration(lPages,2,featureType=setOfPointsFeatureObject)
# 
#         for cf in lSortedFeatures:
#             print (cf, cf.getWeight(),cf.getNodes())
#             # weights must not be too large ecause theyv are used in np.obs  (max=64000)
# #             cf.setWeight(sum(x.getHeight() * x.getWidth() for x in cf.getNodes())/64000)
        for _,p in enumerate(lPages):
            try:p.lFeatureForParsing = p.getCanonicalFeatures()[:]
            except TypeError: p.lFeatureForParsing = None
            if self.bDebug:print (p, p.lFeatureForParsing)
        sys.stdout.flush()
        
        print ('featuring...',chronoOff())
        
        if lSortedFeatures == []:
            print ("No template found in this document")
            return None,None,-1
        
        seqGen.bDebug = False
        lmaxSequence = seqGen.generateItemsets(lPages)
        lSeq, _ = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures,mis = 0.2)
        # if one page: what to do ??
        if len(lPages)>1:
            lOneSupport  = self.testHighSupport(lSeq)
        else : lOneSupport=[]
        print ('L1OneSupport: ', lOneSupport)
        ## reference solution
        if True or lOneSupport == [] :
            # MIS also for patterns, not only item!!
            ## can be used to assess the noise level or here
            chronoOn()
            print( "generation...")
            sys.stdout.flush()
            chronoOn()
            lSeq, _ = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures,mis = 0.1,L1Support=[])
            sys.stdout.flush()
            ## if many MIS=1.0 -> table with many columns!
            ##actual supports: {'x=40.0': 0.5, 'x=473.0': 1.0, 'x=73.0': 0.75, 'x=558.0': 1.0, 'x=327.0': 1.0, 'x=145.0': 1.0, 'x=243.0': 1.0, 'x=1180.0': 0.25, 'x=726.0': 1.0, 'x=408.0': 1.0, 'x=886.0': 1.0, 'x=1027.0': 0.75, 'x=803.0': 1.0, 'x=952.0': 1.0, 'x=636.0': 1.0, 'x=1136.0': 1.0, 'x=839.0': 0.25}
    #         lFSortedFeatures = self.factorizeHighlyFrequentItems()
            
            lPatterns = seqGen.miningSequencePrefixScan(lSeq,maxPatternLength=iMinLen)
            if lPatterns is None:
                return None, None, -1
                
            print( "chrono mining", chronoOff())
            print( 'nb patterns: ',len(list(lPatterns)))
            sys.stdout.flush()
            lPatterns  = self.filterNonRegularPatterns(lPatterns)
            lPatterns.sort(key=lambda p_s:self.computePatternScore(p_s[0]), reverse=True)
#             lPatterns.sort(key=lambda (p,s):s, reverse=True)
            lPatterns = list(filter(lambda p_:p_[0] not in lNegativePatterns, lPatterns))
            sys.stdout.flush()
            
            ### GENERATE SEQUENTIAL RULES
            seqGen.bDebug = True 
            seqGen.THRULES = 0.80
            lSeqRules = seqGen.generateSequentialRules(lPatterns)
            _,dCP = self.getPatternGraph(lSeqRules)
            if bTAMode:
                dTemplatesCnd = self.pattern2TAZonesTemplate(lPatterns,dCP)
            else:
                dTemplatesCnd = self.pattern2VerticalZonesTemplate(lPatterns,dCP)
            
            print ('dTemplatesCnd:', len(dTemplatesCnd))
            chronoOn()
            seqGen.setKleenePlusTH(self.fKleenPlusTH)
#             seqGen.setKleenePlusTH(0)
            
            for p in lPages:
                p.resetTemplate()
            _, lVTemplates,tranprob = seqGen.testTreeKleeneageTemplates(dTemplatesCnd, lPages,iterMax=10)
            print("chronoParsing", chronoOff())
            
            ## merge if similar patterns (see testV/nn)
            ## usually +1 element 
        
        else:
            ### if length=2 how to build the pattern!!!!!
            ## TABLE:
            tableTemplate=treeTemplateClass()
            lOneSupport.sort(key=lambda x:x.getValue())
            tableTemplate.buildTreeFromPattern(lOneSupport)    
            lVTemplates= [tableTemplate]
            tranprob = np.ones((2,2), dtype = np.float16)


        chronoOn()
        lT, lScores, score= self.selectFinalTemplate(lVTemplates,tranprob,lPages)
        print ("%s \nchronoFinalViterbi (%s): score= %s" % (lT,chronoOff(), score))
        sys.stdout.flush() 
#         del seqGen
        
        
        return lT, lScores, score 
        
        
    
    def selectFinalTemplate(self,lTemplates,transProb,lPages):
        """
            apply viterbi to select best sequence of templates
        """
        
        import spm.viterbi as viterbi        


        if  lTemplates == []:
            return None,None,None
        
        def buildObs(lTemplates,lPages):
            """
                build observation prob
            """
            N = len(lTemplates) + 1
#             print 'N:',N
            obs = np.zeros((N,len(lPages)), dtype=np.float16) 
            for i,temp in enumerate(lTemplates):
#                 print (i,temp)
                for j,page in enumerate(lPages):
                    x, y, score= temp.registration(page)
#                     print ("?",page, page.lFeatureForParsing,i, temp, score,x,y)
                    if score <= 0:
                        score = 1e-3
                        obs[-1,j] = 1e-3
                    obs[i,j]= score
                    if np.isinf(obs[i,j]):
                        obs[i,j] = 64000
                    if np.isnan(obs[i,j]):
                        obs[i,j] = 0.0                        
#                     print i,j,page, temp,score 
                #add no-template:-1
            return obs / np.amax(obs)

        N= len(lTemplates) + 1
        # build transition score matrix
        ## use the support to initialize ?? why 
        initialProb = np.ones(N) * 1
        initialProb = np.reshape(initialProb,(N,1))
        obs = buildObs(lTemplates,lPages)
        d = viterbi.Decoder(initialProb, transProb, obs)
        states,fscore =  d.Decode(np.arange(len(lPages)))
#         print (fscore,states)
        np.set_printoptions(precision= 3, linewidth =1000)
#         print (transProb,obs)
        lTemplate=[]
        lScores=[]
        #assigned to each page the template assigned by viterbi
        for i,page, in enumerate(lPages):
            try:
                mytemplate= lTemplates[states[i]]
#                 if mytemplate not in lTemplate:
#                     lTemplate.append(mytemplate)
            except:# no template
                mytemplate = None
            if mytemplate is not None:
#                 page.resetVerticalTemplate()
                page.addVerticalTemplate(mytemplate)
                registeredPoints, _, score = mytemplate.registration(page)
#                 print (page, states[i], mytemplate, registeredPoints, score, page.lFeatureForParsing)
#                 print (page, states[i], mytemplate, registeredPoints, score)

                if registeredPoints:
                    # 0 template ;
                    registeredPoints.sort(key=lambda x_y:x_y[0].getValue())
                    lcuts = list(map(lambda ref_cut:ref_cut[0],registeredPoints))
                    print (page, score, lcuts,registeredPoints)
#                     print '\t', page.lFeatureForParsing,map(lambda x:x.getWeight(), page.lFeatureForParsing)
                    lScores.append((i,states[i],score))
                    page.addVSeparator(mytemplate,lcuts)
                    if mytemplate not in lTemplate:
#                         print ('\added: ',mytemplate,lTemplate)
                        lTemplate.append(mytemplate)
                else: lScores.append((i,states[i],0))
            else:
                print (page, states[i], mytemplate, None, -1)
#                 lScores.append((N,-1))
                lScores.append((i,N,0))

        fscore= np.average(list(map(lambda xyz:xyz[-1],lScores)))
        return lTemplate, lScores, fscore        
        

        
    def getPatternGraph(self,lRules):
        """
            create an graph which linsk exoannded patterns
            (a) -> (ab)
            (abc) -> (abcd)
           
           rule = (newPattern,item,i,pattern, fConfidence)
                   RULE: [['x=19.0', 'x=48.0', 'x=345.0'], ['x=19.0', 'x=126.0', 'x=345.0']] => 'x=464.0'[0] (22.0/19.0 = 0.863636363636)


            can be used for  tagging go up until no parent
            
            for balanced bigram:  extension only: odd bigrams are filtered out
        """
        dParentChild= {}
        dChildParent= {}
        for lhs, _, _, fullpattern, _ in lRules:
            try:dParentChild[str(fullpattern)].append(lhs)
            except KeyError:dParentChild[str(fullpattern)] = [lhs]
            try:dChildParent[str(lhs)].append(fullpattern)
            except KeyError:dChildParent[str(lhs)] = [fullpattern]                

        # for bigram: extend to grammy
        for child in dChildParent.keys():
            ltmp=[]
            if len(eval(child)) == 2:
                for parent in dChildParent[child]:
                    try:
                        ltmp.extend(dChildParent[str(parent)])
                    except KeyError:pass
            dChildParent[child].extend(ltmp)
        return dParentChild,  dChildParent
        
    def isMirroredPattern(self,pattern):
        """
        to be replace by a classification model!!
        
            test if a pattern is mirrored or not
            [A,B,C] [C,B,A]
            
            left must be different from right !
            
            
            AT LEAST 3 VALUES  -> 2 values define one region: if mirrored: 2 regions needed!
               -> NONO!! BAR: the body is detected, and margins are mirroerd
               
            take into account scan shift 
        """
        # deprecated if O,max are added to the features
        if self.bScanShift:
            MTH = 3
        else:
            MTH = 2
        if len(pattern) !=2 or len(pattern[0]) != len(pattern[1]) or (pattern[0] == pattern[1]) or len(pattern[0]) < MTH: 
            return False
        else:
            ## add zeros (margins) if len(pattern[0])  == 2??
            zeroFeature=featureObject()
            zeroFeature.setValue(0)
            
            inv1 =  pattern[1][:]
            inv1.reverse()
            lcouple1= zip(inv1,inv1[1:])
            lw1= map(lambda x_y:abs(x_y[1].getValue()-x_y[0].getValue()),lcouple1)
            lcouple0 = zip(pattern[0],pattern[0][1:])
            lw0 = map(lambda x_y:abs(x_y[1].getValue()-x_y[0].getValue()),lcouple0)
            final = set(map(lambda x_y: abs(x_y[0] -x_y[1]) < self.THNUMERICAL * 2,zip(lw0,lw1)))
            return  set(final) == set([True])
    
    
    
    # here ??
    def flattencontent(self,lPages,lTemplates):
        """
            build reading order according to templates
            
            for each template create a stream
            
            if bigram (mirrored) 
                take 0 as ref; and reverse 1
        """
        lTemplateRO={}
        for template in lTemplates:
            # how many stream?
            # unigram
            if template.getChildren() is None:
                N=len(template.getPattern())
            else:
                # get first child length
                N=len(template.getChildren()[0].getPattern())
            lTemplateRO[template]= [ [] for i in range(0,N+1)]
#             print (template, lTemplateRO[template])
            
#         return 
        for  page in lPages:
            for template in page.getVerticalTemplates():
                ## what is stored if the ternimal: need to access the parent
                page.lVerticalObjects[template]=[]
#                 page.createVerticalZones(template)
                
                parentTemplate= template.getParent()
                bReverse=False
                ## need a real test for  isMirrored(self)
#                 print (parentTemplate,parentTemplate.isMirrored())
                if parentTemplate.isMirrored():
                    bReverse = template.getPattern() == parentTemplate.getPattern()[1]
                lregions= page.getVerticalObjects(template)
                if bReverse:
                    lregions.reverse()
                print (page, bReverse)
                for i,region in enumerate(lregions):
                    print (region.getContent().encode('utf-8'))
                    lTemplateRO[parentTemplate][i].append(region)
#                     print page, i,  parentTemplate,len(lTemplateRO[parentTemplate][i]) #.getContent().encode('utf-8')

            
        ## need a 2D smoothing for line: need to find the line grid and match it against the page.    
        ## baseline method! if overlap :merge?
        ### build blocks for pageTemplate: get headers,...
        ## from block -> create lines (allow for merge and ordering) 
        # once blocks: pageTemplate

        ### now patterns with indentation
        for tem in lTemplateRO:
            print ("###################################", tem)
            lColumnElts=[ [] for i in range(len(lTemplateRO[tem]))]
            for i,lreg in enumerate(lTemplateRO[tem]):
#                 print "%d\t\t%s"%(i,"")
                for reg in lreg:
                    lElts= reg.getObjects()
                    lColumnElts[i].extend(lElts)
#                     for elt in lElts:
                        ## compute justification
#                         try:  print elt.getPage(), elt.getContent().encode('utf-8')[:30]
#                         except:pass
                
        return lColumnElts
                    
    
    ############################# DOM TAGGING ###################
    def tagDomWithBestTemplate(self,lPages):
        """
            Create (empty) REGIONS (for end-to-end; for GT: create table)
        """
        
        for page in lPages:
            if page.getNode():
                best = None
                bestRegisteredPoints =None
                lMissingBest= None
                bestScore = 0
                for mytemplate in page.getVerticalTemplates():
                    registeredPoints, lMissing, score= mytemplate.registration(page)
                    print( page, mytemplate, score)
                    if score > bestScore:
                        best = mytemplate
                        bestRegisteredPoints= registeredPoints
                        lMissingBest=lMissing 
                        bestScore = score
                print( page,best, bestScore) # bestRegisteredPoints, lMissingBest
                if best:
                    prevcut=0
                    for refcut,realcut in bestRegisteredPoints:
                        if realcut != prevcut:
                            region  = etree.Element('REGION')
                            region.set("x",str(prevcut))
                            region.set("y",'0')
                            region.set("height",str(page.getHeight()))
                            region.set("width", str(realcut.getValue() - prevcut))                              
                            region.set('points', '%f,%f,%f,%f,%f,%f,%f,%f'%(prevcut,0, realcut.getValue(),0 ,realcut.getValue(),page.getHeight(),prevcut,page.getHeight()))
                            page.getNode().append(region)
                            prevcut = realcut.getValue()
                    #final col
                    if prevcut != page.getWidth():
                        region  = etree.Element('REGION')
                        width = page.getWidth() - prevcut
                        region.set("x",str(prevcut))
                        region.set("y",'0')
                        region.set("height",str(page.getHeight()))
                        region.set("width", str(width))                              
                        region.set('points', '%f,%f,%f,%f,%f,%f,%f,%f'%(prevcut,0, page.getWidth(),0,page.getWidth(),page.getHeight(),prevcut,page.getHeight()))
                        page.getNode().append(region)
                        
                        

    
    def deleteRegions(self,page):
        """
            delete regions of the page
        """
        for region in page.getAllNamedObjects('REGION'):
            region.getNode().getparent().remove(region.getNode())
                
    def storeLineInRegions(self,page,lRegions):
        """
            assign lines to region
        """
        for line in page.getAllNamedObjects(XMLDSTEXTClass):
#             print line, lRegions
            myReg= line.bestRegionsAssignment(lRegions)
            if myReg:
                myReg.addObject(line)
#                 line.getNode().unlinkNode()
                myReg.getNode().append(line.getNode())
        
    def tagAsRegion(self,lPages):
        """
            create regions
            
            if bIgnoreRegions: delte previous regions
                and assign new regions to textlines
            
            if border page regions are missing :add them?
                or don't put them for tagging
        """
        for page in lPages:
            if page.getNode() is not None:
#                 if self.bIgnoreRegions:
#                     self.deleteRegions(page)
                lRegions=[]
                for template in page.getVerticalTemplates():
#                     print (page, template, template.getParent())
                    page.getdVSeparator(template).sort(key=lambda x:x.getValue())
#                     print (page.getdVSeparator(template))
                    page.getNode().set('template',str(list(map(lambda x:x.getValue(),page.getdVSeparator(template)))))
#                     print (page,page.getNode().get('template'))
                    if template.getParent() is not None and len(template.getParent().getPattern())==2:
                        pos = -1
                        if template.getPattern() == template.getParent().getPattern()[0]:
                            pos = 0
                        elif template.getPattern() == template.getParent().getPattern()[1]:
                            pos = 1
                        else:
                            raise 'template index issue'
                        page.getNode().set('reftemplate',str((pos,list(map(lambda x:x.getValue(),template.getParent().getPattern()[0])),list(map(lambda x:x.getValue(),template.getParent().getPattern()[1])))))
                    else:
                        # sinlge: add () for comparison/evaluation
                        page.getNode().set('reftemplate',str((0,(list(map(lambda x:x.getValue(),template.getPattern()))))))

                    prevcut = 1
                    lCuts=[prevcut]
                    lRegions=[]
                    print (page, page.getdVSeparator(template))
                    for lcut in page.getdVSeparator(template):
                        for (x1,x2) in lcut.getValue(): 
                            print ("\t",page,lcut.getNodes(),x2)
                            newReg= XMLDSObjectClass()
                            domNode  = etree.Element('REGION')
                            domNode.set("x",str(x1))
                            YMinus= 1
                            domNode.set("y",str(YMinus))
                            domNode.set("height",str(page.getHeight()-2 * YMinus))
                            domNode.set("width", str(x2 - x1))
                            newReg.setNode(domNode)
                            page.getNode().append(domNode)
                            newReg.setDimensions(x1,YMinus, page.getHeight()-2 * YMinus,x2-x1)
                            lRegions.append(newReg)
                
                if lRegions != []:
                    if self.bIgnoreRegions:
                        self.deleteRegions(page)
                    self.storeLineInRegions(page,lRegions)
    
    def tagDomAsTable(self,lPages):
        """
            create a table object:
            table zone: page
            columns: the created vertical zones
        """
        for page in lPages:
            if page.getNode():
                # if several template ???
                for template in page.getVerticalTemplates():
                    ### create a table
                    tableNode = etree.Element('TABLE')
                    tableNode.set('x','0')
                    tableNode.set('y','0')
                    tableNode.set('height',str(page.getHeight()))
                    tableNode.set('width',str(page.getWidth()))

                    page.getNode().append(tableNode) 
                    page.getdVSeparator(template).sort(key=lambda x:x.getValue())
#                     print page.getdVSeparator(template)
                    prevcut=0
                    for i,cut in enumerate(page.getdVSeparator(template)):
                        cellNode  = etree.Element('CELL')
                        cellNode.set("x",str(prevcut))
                        cellNode.set("y",'0')
                        cellNode.set("irow","0")
                        cellNode.set("icol",str(i))
                        cellNode.set("height",str(page.getHeight()))
                        cellNode.set("width", str(cut.getValue() - prevcut))                            
#                         cellNode.set('points', '%f,%f,%f,%f,%f,%f,%f,%f'%(cut,0, cut.getValue(),0 ,cut.getValue(),page.getHeight(),cut,page.getHeight()))
                        tableNode.append(cellNode)
                        prevcut  = cut.getValue()      
            
            
    def testCliping(self,lPages):
        """
            all in the name
            
        """
        for page in lPages:
            region=XMLDSObjectClass()
            region.addAttribute('x', 0)
            region.addAttribute('y', 0)
            region.addAttribute('height', page.getAttribute('height'))
            region.addAttribute('width', 110)
            print (region.getX(),region.getY(),region.getWidth(),region.getHeight())
            print (page.getAttributes(), page.getX2())
            lObjects = page.clipMe(region)
            region.setObjectsList(lObjects)       
        
        
    def cleanInput(self,lPages):
        """
            Delete Otokens which are too close to x=0 x=max
            
            Does not touch the dom!!
        """
        for page in lPages:
            
            for txt in page.getAllNamedObjects(XMLDSTEXTClass):
                ltobd=[]
                for word in txt.getAllNamedObjects(XMLDSTOKENClass):
                    if word.getX() < 5:
                        ltobd.append(word)
                    elif word.getX2() +5 > page.getWidth():
                        ltobd.append(word)
                    elif word.getWidth() < 10:
                        ltobd.append(word)
                # resize text
                for x in ltobd:
#                     print x.getAttribute('id')
                    txt.getObjects().remove(x)
                    
                # resize DOM  node as well??
                if len(txt.getAllNamedObjects(XMLDSTOKENClass)) == 0:
                    try:
                        page.getObjects().remove(txt)
                    except ValueError:
                        print(txt)
                else:
                    txt.resizeMe(XMLDSTOKENClass)
            
    
    def generateTestOutput(self,lPages):
        """
            create a run XML file
        """
        
        root = etree.Element('DOCUMENT')
        self.evalData = etree.ElementTree(root)
        for page in lPages:
            domp=etree.Element('PAGE')
            domp.set('number',page.getAttribute('number'))
            root.append(domp)
            if page.getNode().get('template'):
                domp.set('template',page.getNode().get('template'))
            if page.getNode().get('reftemplate'):
                domp.set('reftemplate',page.getNode().get('reftemplate'))
        
        return self.evalData
    
     
    
    def buildVerticalChunks(self,lPages):
        """
            build vertical chunks based on position ( only one next top/bottom possible elements)
        """
        Vctool= TwoDChunking()
        Vctool.run2DChunkingPerPage(self.ODoc,XMLDSTEXTClass)
        # get Vertical clusters and create zones
    
    #--- RUN ---------------------------------------------------------------------------------------------------------------
    
    def loadDSDoc(self,doc):
        """
        
        """
        self.doc= doc
        self.ODoc = XMLDSDocument()
        
        chronoOn()
        self.ODoc.loadFromDom(self.doc,listPages=range(self.firstPage,self.lastPage+1))        
        self.lPages= self.ODoc.getPages()
#         self.lPages= self.lPages[:1] 
#         self.cleanInput(self.lPages)
        print('chronoloading:', chronoOff())
        sys.stdout.flush()

    def run(self):
        """
            for a set of pages, associate each page with several vertical zones  aka column-like elements
            Populate the vertical zones with page elements (text)

            indicate if bigram page template (mirrored pages)
             
        """
        
        if self.bManual:
#             self.tagWithTemplate(self.manualPattern,self.lPages)
            self.THNUMERICAL = 30
#             level=XMLDSTEXTClass
            self.minePageVerticalFeature(self.lPages, ['x','x2'],level=self.sTag)
            self.processWithTemplate(self.manualPattern,self.lPages)

        else:
            chronoOn()
            # first mine page size!!
            ## if width is not the 'same' , then  initial values are not comparable (x-end-ofpage)
            lSubPagesList = self.highLevelSegmentation(self.lPages)
            ## need to regroup similar 
             
#             self.arrayApproach(lSubPagesList)
#             return
            if self.baselineMode > 0:
                # not implemented
                self.baselineSegmentation(lSubPagesList)
            else:
                # build basic elements: page elements used to get pagefeatures  
                self.buildVerticalChunks(self.lPages)
#                 self.buildLeadingChunks(self.lPages)
                lTemplates = self.iterativeProcessVSegmentation(lSubPagesList)
#                 self.flattencontent(self.lPages,lTemplates)
                return self.ODoc, lTemplates, self.lPages
                ## in this package?? current yes
#             self.processVSegmentation(self.lPages)
            print( 'chronoprocessing: ', chronoOff())
            
#         self.addTagProcessToMetadata(self.doc)
        
        return self.doc 

    #--- TESTS -------------------------------------------------------------------------------------------------------------    
    #
    # Here we have the code used to test this component on a prepared testset (see under <ROOT>/test/common)
    # Do: python ../../src/common/TypicalComponent.py --test REF_TypicalComponent/
    #
    
    def testComparePageVertical(self,runElt,refElt):
        """
            input:  <SeparatorRegion x="51.36" y="7.44" height="764.4" width="2.88"/>
        """
        self.THNUMERICAL = 30.0
        ## x=XX
        return abs(runElt - refElt) < (self.THNUMERICAL *2.0)
  
    
        
        
    def testTemplateType(self,srefData,srunData, bVisual):
        """
            run PAGE @template
            ref PAGE @refteemplate 
        """
        
        
        cntOk = cntErr = cntMissed = 0
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))

        lRun = []
        if RunData:
            lpages = RunData.xpath('//%s' % ('PAGE'))
            for page in lpages:
                if page.get('reftemplate'):
                    lRun.append(eval(page.get('reftemplate')))
                else:lRun.append([])

        lRef = []
        lPages = RefData.xpath('//%s' % ('PAGE'))
        for page in lPages:
            if page.get('reftemplate'):
                lRef.append(eval(page.get('reftemplate')))
            else: lRef.append([])

        runLen = len(lRun)
        refLen = len(lRef)
        
        assert runLen == refLen
        ltisRefsRunbErrbMiss= list()
        for i in range(0,len(lRef)):
            if lRun[i] != []:
                runLen = len(lRun[i])
            else:
                runLen=0
            if lRef[i] != []:
                refLen = len(lRef[i])
            else:
                refLen=0
#             print i, refLen, runLen
            if runLen == refLen:
                cntOk += 1
                ltisRefsRunbErrbMiss.append( (i,  lRef[i],lRun[i], False, False) )
            else:
                cntErr+=1
                cntMissed+=1
                ltisRefsRunbErrbMiss.append( (i,  lRef[i],lRun[i], True, True) )

        ltisRefsRunbErrbMiss.sort(key=lambda x_y_z_t_u:x_y_z_t_u[0])
        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)        
        
   
   
    def testRUNREFVerticalSegmentation(self,srefData,srunData, bVisual):
        """
            Test found run-template and run-reftemplate
        """ 
        
        cntOk = cntErr = cntMissed = 0
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))    
         
        lRun = []
        if RunData:
            lpages = RunData.xpath('//%s' % ('PAGE'))
            for page in lpages:
                if page.get('template'):
                    lRun.append(eval(page.get('template')))
                else:lRun.append([])

        lRef = []
            #### NO LONGER REFDATA!!!
        lPages = RunData.xpath('//%s' % ('PAGE'))
        for page in lPages:
            if page.get('reftemplate'):
                    lRef.append(eval(page.get('reftemplate')))
            else: lRef.append([])
 
        ltisRefsRunbErrbMiss= list()
        for i in range(0,len(lRef)):
            lRefCovered = []
            runLen = len(lRun[i])

            if lRef[i]==[]:
                refLen=0
                refElt=None
                posref=None
            else:
                posref=lRef[i][0]
                refLen= len(lRef[i][posref+1])
            curRun = curRef = 0
            while curRun <= runLen - 1:  # or curRef <= refLen -1:
                bErr, bMiss = False, False
                try:
                    runElt = lRun[i][curRun]
                except IndexError: runElt = None
    #             print '___',curRun,runElt
                curRef = 0
                bFound = False
                while not bFound and curRef <= refLen - 1:
                    try: refElt = lRef[i][posref+1][curRef]
                    except IndexError: refElt = None
    #                 self.compareString(runElt,runElt)
                    if runElt and refElt not in lRefCovered and self.testComparePageVertical(runElt, refElt):
                        bFound = True
                        lRefCovered.append(refElt)
                        resRef=refElt
                    else:
                        curRef += 1
                if bFound:
                    if bVisual:print( "FOUND:", runElt, ' -- ', lRefCovered[-1])
                    cntOk += 1
                    curRun += 1
                else:
                    resRef=''
                    curRun += 1
                    cntErr += 1
                    bErr = True
#                     bMiss = True
                    if bVisual:print ("ERROR:", runElt)
                ltisRefsRunbErrbMiss.append( (i, resRef, runElt,bErr, bMiss) )
             
            if posref is not None:
                for ref in lRef[i][posref+1]:
                    if ref not in lRefCovered:
                        ltisRefsRunbErrbMiss.append( (i, ref, '',False, True) )
                        # add missed elements!
                        cntMissed += 1 #len(lRef[i][posref+1]) - len(lRefCovered)
 
 
        ltisRefsRunbErrbMiss.sort(key=lambda x_y_z_t_u:x_y_z_t_u[0])

        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)  
    
    def testREFVerticalSegmentation(self,srefData,srunData, bVisual):
        """
            Test found reftemplate and reftemplate
        """ 
        cntOk = cntErr = cntMissed = 0
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))        
         
        lRun = []
        if RunData is not None:
            lpages = RunData.xpath('//%s' % ('PAGE'))
            for page in lpages:
                if page.get('reftemplate'):
                    lRun.append(eval(page.get('reftemplate')))

        lRef = []
        lPages = RefData.xpath('//%s' % ('PAGE'))
        for page in lPages:
            if page.get('reftemplate'):
                lRef.append(eval(page.get('reftemplate')))
            else: lRef.append([])
 
        ltisRefsRunbErrbMiss= list()
        for i in range(0,len(lRef)):
            lRefCovered = []
            if lRun[i] ==[]:
                runLen=0
            else:
                posrun = lRun[i][0]
                runLen = len(lRun[i][posrun+1])
            if lRef[i]==[]:
                refLen=0
                refElt=None
                posref=None
            else:
                posref=lRef[i][0]
                refLen= len(lRef[i][posref+1])
            curRun = curRef = 0
            while curRun <= runLen - 1:  # or curRef <= refLen -1:
                bErr, bMiss = False, False
                try:
                    runElt = lRun[i][posrun+1][curRun]
                except IndexError: runElt = None
    #             print '___',curRun,runElt
                curRef = 0
                bFound = False
                while not bFound and curRef <= refLen - 1:
                    try: refElt = lRef[i][posref+1][curRef]
                    except IndexError: refElt = None
    #                 self.compareString(runElt,runElt)
                    if runElt and refElt not in lRefCovered and self.testComparePageVertical(runElt, refElt):
                        bFound = True
                        lRefCovered.append(refElt)
                        resRef=refElt
                    else:
                        curRef += 1
                if bFound:
                    if bVisual:print ("FOUND:", runElt, ' -- ', lRefCovered[-1])
                    cntOk += 1
                    curRun += 1
                else:
                    resRef=''
                    curRun += 1
                    cntErr += 1
                    bErr = True
#                     bMiss = True
                    if bVisual:print ("ERROR:", runElt)
                ltisRefsRunbErrbMiss.append( (i, resRef, runElt,bErr, bMiss) )
             
            if posref is not None:
                for ref in lRef[i][posref+1]:
                    if ref not in lRefCovered:
                        ltisRefsRunbErrbMiss.append( (i, ref, '',False, True) )
                        # add missed elements!
#                         print 'missed', len(lRef[i][posref+1]) , len(lRefCovered), lRef[i][posref+1], lRefCovered
                        cntMissed += 1#(len(lRef[i][posref+1]) - len(lRefCovered))
 
        ltisRefsRunbErrbMiss.sort(key=lambda x_y_z_t_u:x_y_z_t_u[0])

        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)                 
                 
    def testVerticalSegmentation(self,srefData,srunData, bVisual):
        """
            Test found cuts and reftemplate
             
        """
 

        cntOk = cntErr = cntMissed = 0
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))   

        lRun = []
        if RunData:
            lpages = RunData.xpath('//%s' % ('PAGE'))
            for page in lpages:
                if page.get('template'):
                    lRun.append(eval(page.get('template')))
                else:lRun.append([])

        lRef = []
        lPages = RefData.xpath('//%s' % ('PAGE'))
        for page in lPages:
            if page.get('reftemplate'):
                    lRef.append(eval(page.get('reftemplate')))
            else: lRef.append([])
 
        ltisRefsRunbErrbMiss= list()
        for i in range(0,len(lRef)):
            lRefCovered = []
            runLen = len(lRun[i])
            if lRef[i]==[]:
                refLen=0
                refElt=None
                posref=None
            else:
                posref=lRef[i][0]
                refLen= len(lRef[i][posref+1])
            curRun = curRef = 0
            while curRun <= runLen - 1:  # or curRef <= refLen -1:
                bErr, bMiss = False, False
                try:runElt = lRun[i][curRun]
                except IndexError: runElt = None
    #             print '___',curRun,runElt
                curRef = 0
                bFound = False
                while not bFound and curRef <= refLen - 1:
                    try: refElt = lRef[i][posref+1][curRef]
                    except IndexError: refElt = None
    #                 self.compareString(runElt,runElt)
                    if runElt and refElt not in lRefCovered and self.testComparePageVertical(runElt, refElt):
                        bFound = True
                        lRefCovered.append(refElt)
                        resRef=refElt
                    else:
                        curRef += 1
                if bFound:
                    if bVisual:print( "FOUND:", runElt, ' -- ', lRefCovered[-1])
                    cntOk += 1
                    curRun += 1
                else:
                    resRef=''
                    curRun += 1
                    cntErr += 1
                    bErr = True
#                     bMiss = True
                    if bVisual:print ("ERROR:", runElt)
                ltisRefsRunbErrbMiss.append( (i, resRef, runElt,bErr, bMiss) )
             
            if posref is not None:
                for ref in lRef[i][posref+1]:
                    if ref not in lRefCovered:
                        ltisRefsRunbErrbMiss.append( (i, ref, '',False, True) )
                        # add missed elements!
                    cntMissed += (len(lRef[i][posref+1]) - len(lRefCovered))
 
 
        ltisRefsRunbErrbMiss.sort(key=lambda x_y_z_t_u:x_y_z_t_u[0])

        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)        
                
                
    
    def testRun(self, filename, outFile=None):
        """
        testRun is responsible for running the component on this file and returning a string that reflects the result in a way
        that is understandable to a human and to a program. Nicely serialized Python data or XML is fine
        """
        
        doc = self.loadDom(filename)
        self.loadDSDoc(doc)
        self.doc= doc
        self.run()
#         doc.freeDoc()
        self.generateTestOutput(self.lPages)

        if outFile: self.writeDom(doc)
        return etree.tostring(self.evalData,encoding='unicode')
#         return etree.tostring(self.evalData,encoding='utf-8',xml_declaration=True)
    
    def testCompare(self, srefData, srunData, bVisual=False):
        """
        Our comparison is very simple: same or different. N
        We anyway return this in term of precision/recall
        If we want to compute the error differently, we must define out own testInit testRecord, testReport
        """
        dicTestByTask = dict()
        dicTestByTask['VREFzones']= self.testREFVerticalSegmentation(srefData, srunData,bVisual)
#         dicTestByTask['Vzones']= self.testVerticalSegmentation(srefData, srunData,bVisual)
        dicTestByTask['VRUNREFzones']= self.testRUNREFVerticalSegmentation(srefData, srunData,bVisual)
#         dicTestByTask['templateType']= self.testTemplateType(srefData, srunData,bVisual)
        return dicTestByTask
        


#--- MAIN -------------------------------------------------------------------------------------------------------------    
#
# In case we want to use this component from a command line
#
# Do: python TypicalComponent.py -i toto.in.xml
#
if __name__ == "__main__":
    
    
    docM = pageVerticalMiner()

    #prepare for the parsing of the command line
    docM.createCommandLineParser()
    docM.add_option("-f", "--first", dest="first", action="store", type="int", help="first page number", metavar="NN")
    docM.add_option("-l", "--last", dest="last", action="store", type="int", help="last page number", metavar="NN")
    docM.add_option("-t", "--tag", dest="tag", action="store", type="string", help="tag level", metavar="S")
    docM.add_option("--pattern", dest="pattern", action="store", type="string", help="pattern to be applied", metavar="[]")
    docM.add_option("--TH", dest="THNUM", action="store", type="int", help="TH as eq delta", metavar="NN")
    docM.add_option("--KTH", dest="KLEENETH", action="store", type="float", help="TH for sequentiality", metavar="NN")
    docM.add_option("--baseline", dest="baseline", type='int', default=0, action="store", help="baseline method",metavar="N")
    docM.add_option("--ignoreRegion", dest="bIgnoreRegions", default=True, action="store_true", help="Ignore existing TextRegions")
    docM.add_option("--nogl", dest="nogline",  action="store_true",default=False ,help="no graphical line used")
    docM.add_option("--glonly", dest="glineonly",  action="store_true",default=False ,help="graphical line only (no text)")
        
    #parse the command line
    dParams, args = docM.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    docM.setParams(dParams)
    
    doc = docM.loadDom()
    docM.loadDSDoc(doc)
#     docM.bDebug = True
    docM.run()
    
    if doc and docM.getOutputFileName() != "-":
        docM.writeDom(doc, True)
