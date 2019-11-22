#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

     H. Déjean

    copyright Xerox 2016
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

from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSObjectClass import XMLDSObjectClass
from ObjectModel.XMLDSGRAHPLINEClass import XMLDSGRAPHLINEClass
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.XMLDSTOKENClass import XMLDSTOKENClass
from ObjectModel.XMLDSPageClass import XMLDSPageClass
from ObjectModel.treeTemplateClass import treeTemplateClass
from ObjectModel.XMLDSCELLClass import XMLDSTABLECELLClass

# from spm.spm2 import PrefixSpan

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
        # use for evaluation
        self.testTH = 30  # use dfor --test  BUT ALSO IN RUN !!
        self.THCOMP = 10
        self.evalData= None
        
        self.bDomTag=True
        
        # TH for sequentiality detection (see structuralMining)
        self.fKleenPlusTH =1.5

        # pattern provided manually        
        self.bManual = False
        
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
#                     _,lCurList = self.parseWithTemplate(mytemplate,lCurList,bReplace=True)
                    _,_,lCurList = seqGen.parseWithTreeTemplate(mytemplate, lCurList, bReplace=True)    
                    for elt in lCurList:
                        if elt.getSetofFeatures() is None:
                            elt.resetFeatures()
                            elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['virtual'],myLevel=XMLDSPageClass)
                            elt.computeSetofFeatures()
                            elt.lFeatureForParsing=elt.getSetofFeatures()
            
            icpt +=1
#             if self.bDebug:self.printTreeView(lCurList)
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
#                 if ( len(pattern) == 1 and len(pattern[0])>=2  and pattern[0][0].getValue() !=pattern[0][1].getValue()) or (len(pattern)==2 and len( pattern[0] ) == len( pattern[1] ) >= 2  ):
                if (len(pattern[0]) == 3  ):

                    if  self.isCorrectPattern(pattern):
                        ## alos width must be similar :['x=115.0', 'x=433.0'], ['x=403.0', 'x=433.0']] not possible !! 
                        template  = treeTemplateClass()
                        template.setPattern(pattern)
                        template.buildTreeFromPattern(pattern)
                        template.setType('VTemplate') # ??
                        try:dTemplatesTypes[template.__class__.__name__].append((pattern, support, template))
                        except KeyError: dTemplatesTypes[template.__class__.__name__] = [(pattern,support,template)]                      
        
        print ("closed-patterns: ", iNbClosed)
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
            Output: pages decorted with features
            
        """ 
        
        chronoOn()
        ## COMMUN PROCESSING
        ### DEPENDS ON OBJECT LEVEL !! TEXT/TOKEN!!
        lLElts=[ [] for i in range(0,len(lPages))]
        for i,page in enumerate(lPages):
            lElts= page.getAllNamedObjects(level)
            lElts.sort(key=lambda x:x.getY())
            lLElts[i]=lElts
        ### VerticalZones START
        for i,page, in enumerate(lPages):
            page._VX1X2Info=[]
            lElts=[]
            if not self.bGLOnly:
                lElts= lLElts[i]
                for elt in lElts:
                    elt.resetFeatures()
                    elt._canonicalFeatures = None
                    elt.setFeatureFunction(elt.getSetOfX1X2Attributes,50,lFeatureList=lFeatureList,myLevel=level)
                    elt.computeSetofFeatures()
            
            # GRAPHICAL LINES 
            gl = []
            if not self.bNOGline:
                for graphline in page.getAllNamedObjects(XMLDSGRAPHLINEClass):
                    graphline.resetFeatures()
                    graphline._canonicalFeatures = None
                    if graphline.getHeight() > graphline.getWidth() and graphline.getHeight() > 30:  #30 twice the height of a line
                        gl.append(graphline)
                        # create a feature
                        f = featureObject()
                        f.setType(featureObject.NUMERICAL)
                        f.setTH(self.THNUMERICAL)
                        f.setWeight(graphline.getHeight()/64000)
                        f.setName("x")
                        f.setObjectName(graphline)
                        f.addNode(graphline)
                        f.setValue(round(graphline.getX()))
                        graphline.addFeature(f)
                        page.setVGLFeatures(f)

            ## select regular x                                 
            seqGen = sequenceMiner()
#             seqGen.bDebug  =self.bDebug
            _fullFeatures =   seqGen.featureGeneration(lElts+gl,1)
            for fx in _fullFeatures:
                fx.setWeight(sum(x.getHeight() for x in fx.getNodes())/64000)
                fx.setWeight(len(fx.getNodes()))
#                 print(fx,fx.getWeight())
            page.setX1X2(_fullFeatures)
            del seqGen
        self.buildPartitions(lPages)
        
        print ('chronoFeature',chronoOff())
        
        return lPages    
        
    
    def minePageVerticalFeature2(self,lPages,lFeatureList,level=XMLDSTEXTClass):
        """
            get page features for  vertical zones: find vertical regular vertical Blocks/text structure
            
        """ 
        
        import util.TwoDNeighbourhood as TwoDRel
        
        chronoOn()
        ## COMMUN PROCESSING
        ### DEPENDS ON OBJECT LEVEL !! TEXT/TOKEN!!
        lVEdge = []
        lLElts=[ [] for i in range(0,len(lPages))]
        for i,page in enumerate(lPages):
            page.resetFeatures()
            page._canonicalFeatures=None
            lElts= page.getAllNamedObjects(level)

            for e in lElts:
                e.next=[]

            lElts.sort(key = lambda x:x.getY())
            lLElts[i]=lElts
            lVEdge = TwoDRel.findVerticalNeighborEdges(lElts)
            for  a,b in lVEdge:
                a.next.append( b )

        ### VerticalZones START
        for i,page, in enumerate(lPages):
            page._VX1Info=[]
            lElts=[]
            if not self.bGLOnly:
                lElts= lLElts[i]
                for elt in lElts:
                    elt.resetFeatures()
                    elt._canonicalFeatures = None
                    elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=lFeatureList,myLevel=level)
#                     elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=lFeatureList,myLevel=level)

    #                 elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=lFeatureList,myLevel=XMLDSTOKENClass)
     
                    elt.computeSetofFeatures()
                    ## rename all as 'x'
                    [x.setName('x') for x in elt.getSetofFeatures()]
        
            ## select regular x                                 
            seqGen = sequenceMiner()
#             seqGen.bDebug  =self.bDebug
            _fullFeatures =   seqGen.featureGeneration(lElts,1)
            for fx in _fullFeatures:
                fx.setWeight(sum(x.getHeight() for x in fx.getNodes())/64000)
                # for article
                fx.setWeight(len(fx.getNodes()))
                
#             lKleendPlus = self.getKleenePlusFeatures(lElts)
            page.setVX1Info(_fullFeatures)
#             page.setVX1Info(_fullFeatures)
            del seqGen
        self.buildVZones(lPages)
        print ('chronoFeature',chronoOff())
        
        return lPages    
    
        
    def minePageVerticalFeature(self,lPages,lFeatureList,level=XMLDSTEXTClass):
        """
            get page features for  vertical zones: find vertical regular vertical Blocks/text structure
            
        """ 
        
        import util.TwoDNeighbourhood as TwoDRel
        
        chronoOn()
        ## COMMUN PROCESSING
        ### DEPENDS ON OBJECT LEVEL !! TEXT/TOKEN!!
        lVEdge = []
        lLElts=[ [] for i in range(0,len(lPages))]
        for i,page in enumerate(lPages):
            page.resetFeatures()
            page._canonicalFeatures=None
            lElts= page.getAllNamedObjects(level)
#             lElts= page.getAllNamedObjects(XMLDSTOKENClass)

            for e in lElts:
                e.next=[]

            lElts.sort(key=lambda x:x.getY())
            lLElts[i]=lElts
            lVEdge = TwoDRel.findVerticalNeighborEdges(lElts)
            for  a,b in lVEdge:
                a.next.append( b )

        ### VerticalZones START
        for i,page, in enumerate(lPages):
            page._VX1Info=[]
            lElts=[]
            if not self.bGLOnly:
                lElts= lLElts[i]
                for elt in lElts:
                    elt.resetFeatures()
                    elt._canonicalFeatures = None
                    elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=lFeatureList,myLevel=level)
    #                 elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=lFeatureList,myLevel=XMLDSTOKENClass)
                    
                    elt.computeSetofFeatures()
                    ## rename all as 'x'
                    [x.setName('x') for x in elt.getSetofFeatures()]
            
            # GRAPHICAL LINES 
            gl = []
            if not self.bNOGline:
                for graphline in page.getAllNamedObjects(XMLDSGRAPHLINEClass):
                    
                    # Y 
#                     graphline.resetFeatures()
#                     graphline._canonicalFeatures = None
#                     if graphline.getHeight() > graphline.getWidth(): # and graphline.getHeight() > 30:  #30 twice the height of a line
# 
#                         gl.append(graphline)
#                         # create a feature
#                         f = featureObject()
#                         f.setType(featureObject.NUMERICAL)
#                         f.setTH(self.THNUMERICAL)
#                         f.setWeight(graphline.getHeight()/64000)
#                         f.setName("x")
#                         f.setObjectName(graphline)
#                         f.addNode(graphline)
#                         f.setValue(round(graphline.getX()))
#                         graphline.addFeature(f)
#                         page.setVGLFeatures(f)
                    #X 
                    graphline.resetFeatures()
                    graphline._canonicalFeatures = None
                    if graphline.getWidth() > graphline.getHeight(): # and graphline.getHeight() > 30:  #30 twice the height of a line

                        gl.append(graphline)
                        # create a feature
                        f = featureObject()
                        f.setType(featureObject.NUMERICAL)
                        f.setTH(self.THNUMERICAL)
                        f.setWeight(graphline.getWidth())
                        f.setName("x")
                        f.setObjectName(graphline)
                        f.addNode(graphline)
                        f.setValue(round(graphline.getY()))
                        graphline.addFeature(f)
                        page.setVGLFeatures(f)                    
                    
            
            ## select regular x                                 
            seqGen = sequenceMiner()
#             seqGen.bDebug  =self.bDebug
            _fullFeatures =   seqGen.featureGeneration(lElts+gl,1)
            for fx in _fullFeatures:
                fx.setWeight(sum(x.getHeight() for x in fx.getNodes())/64000)
                # for article
                fx.setWeight(len(fx.getNodes()))
            
            lKleendPlus = self.getKleenePlusFeatures(lElts)
            page.setVX1Info(lKleendPlus)
#             page.setVX1Info(_fullFeatures)
            del seqGen
        self.buildVZones(lPages)
        
        print ('chronoFeature',chronoOff())
        
        return lPages
            
    def getKleenePlusFeatures(self,lElts):
        """
            select KleenePlus elements based on .next (only possible for unigrams)
        """   
        dFreqFeatures={}
        dKleenePlusFeatures = {}
        
        lKleenePlus=[]
        for elt in lElts:
            if elt.getCanonicalFeatures() is not None:
                for fea in elt.getCanonicalFeatures():
                    if len(fea.getNodes())>0:
                        try:dFreqFeatures[fea] +=1
                        except KeyError:dFreqFeatures[fea] = 1
                        for nextE in elt.next:
                            if fea in nextE.getSetofFeatures():
                                try:
                                    dKleenePlusFeatures[fea].append((elt,nextE))
        #                             dKleenePlusFeatures[fea].append(elt)                            
                                except KeyError:
                                    dKleenePlusFeatures[fea]=[(elt,nextE)]
        #                             dKleenePlusFeatures[fea].append(nextE)
        for fea in dFreqFeatures:
            try:
                dKleenePlusFeatures[fea]
#                 print ("###",fea, len(set(dKleenePlusFeatures[fea])), dFreqFeatures[fea] ) #, dKleenePlusFeatures[fea]
                if len(set(dKleenePlusFeatures[fea])) >= 0.5 *  dFreqFeatures[fea]:
                    lKleenePlus.append(fea) 
                    fea.setKleenePlus(1.0*len(set(dKleenePlusFeatures[fea])) / dFreqFeatures[fea])
            except KeyError:
                pass
        return  lKleenePlus
    
    
    def recPathGeneration(self,curSeq,lSeq):
        """
            input: list of (x1,x2)  features
            output: comb genration of all possible layout
        """
        cur,lRest = lSeq[0],lSeq[1:]
        curSeq.append(cur)
        lRecList=[curSeq]
        curmax=cur.getValue()[-1]
#         print ("x",cur,lRest,curmax)
        for i,xx2 in enumerate(lRest):
#             print (xx2, xx2.getValue()[0] , curmax)
            if xx2.getValue()[0] > curmax:
                curmax= xx2.getValue()[-1]
                ln = self.recPathGeneration(curSeq[:],lRest[i:])
#                 print ('\t',ln)
                lRecList.extend(ln)
#                 print (lRecList[-1])
        return lRecList
            
            
        
    def buildPartitions(self,lp):
        """
            input: list of pages with horizontal 2Dfeatures
            output: list of possible horizontal compatible partitions (sequences of horizontal stuff)
        """
        from spm.feature import setOfPointsFeatureObject
        for page in lp:
            page._Vpartitions=[]
#             page.getX1X2().sort(key=lambda x:x.getWeight(),reverse=True)
            page.getX1X2().sort(key=lambda x:x.getValue()[0])
            lLSeq=[]
            for i,x in enumerate(page.getX1X2()):
                lLSeq.extend(self.recPathGeneration([], page.getX1X2()[i:]))
            lLSeq.sort (key=lambda lx: sum(map(lambda x:x.getWeight(),lx)),reverse=True)
            for ln in lLSeq:
                feature = setOfPointsFeatureObject()
                feature.setName('seqV')
                feature.setTH(5)
                feature.addNode(page)
                feature.setObjectName(page)
                feature.setValue(list(map(lambda x:x.getValue(),ln)))
                feature.setType(featureObject.COMPLEX)
                page.addFeature(feature)    
                
        
    
    def buildVZones(self,lp):
        """
            store vertical positions in each page
        """
        
        for _, p in enumerate(lp):
            p.lf_XCut=[]
            p.getVX1Info().sort(key=lambda x:x.getWeight(),reverse=True)
#             print (p,  p.getVX1Info(),p.getVGLFeatures(), p.lf_XCut) 
            for fi in p.getVX1Info():
                if fi not in p.lf_XCut:
#                     l = sum(x.getWidth()*x.getHeight() for x in fi.getNodes())
                    l = sum(x.getHeight() for x in fi.getNodes())
                    fi.setWeight(l)
                    p.lf_XCut.append(fi)
#                 else:
#                     print  'skip!',p, fi, fi.getWeight()
            p.getVGLFeatures().sort(key=lambda x:x.getWeight(),reverse=True)
            for fi in p.getVGLFeatures():
                if fi not in p.lf_XCut:
                    l = sum(x.getWidth()*x.getHeight() for x in fi.getNodes())
                    fi.setWeight(l)
                    p.lf_XCut.append(fi)
#                 else:
#                     print  'skip!',p, fi, fi.getWeight()            
            
            p.lf_XCut.sort(key=lambda x:x.getWeight(),reverse=True)
            p.lf_XCut = p.lf_XCut   #[:15]
            p.lf_XCut.sort(key=lambda x:x.getValue())
            
#             if self.bDebug : print p, map(lambda x:(x.getTH(),x.getCanonical().getValue(),x.getCanonical().getWeight()),p.lf_XCut)          
    
    def getTerminals(self,node):
        """
            get terminal objects
        """
        if not node.getAttribute('virtual'):
            return [node]
        lReturn=[]
        for obj in node.getObjects():
            lReturn.extend(self.getTerminals(obj))
            
            
        return  lReturn
    
        
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
        """
        
        return  list(filter(lambda x_y: (len(x_y[0]) != 2) or (len(x_y[0][0])==len(x_y[0][1])) , lPatterns))



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
        print(actual_supports.items())        
        lOneSupport= [k for k,v in actual_supports.items() if v >= 0.5 ]
        return lOneSupport
      


    def createTableTemplate(self,lTemplate):
        """
            create a pxml for each template
        """
        from xml_formats.PageXml import PageXml
        from util.unitConversion import convertDot2Pixel

        
        page= self.lPages
        for temp in lTemplate:
            pageXmlDoc,pageNode = PageXml.createPageXmlDocument(creatorName='NLE', filename = "%s_%s"% (self.getInputFileName(),'tt.pxml'), imgW = convertDot2Pixel(self.dpi,page.getWidth()), imgH = convertDot2Pixel(self.dpi,page.getHeight()))

            tablenode= PageXml.createPageXmlNode('TableRegion')
            pageNode.append(tablenode)
            # creata a table and one cell per column
            for child in temp.getChildren():
                cellnode = PageXml.createPageXmlNode('TableCell')
                tablenode.append(cellnode)
                print (child.getPattern().getValue())
                
        
            
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
        setFullListTemplates= set() 
        for lPages in lLPages:
            
#             lsubList = [lPages]

            self.THNUMERICAL = self.testTH
            for p  in lPages:
                p.resetVerticalTemplate()

#             self.bDebug=True
            # V positions
            NBATCH = 3
            for nbPage in range(0,len(lPages),NBATCH):
#                 print nbPage, nbPage + NBATCH
#                 print lPages[nbPage:nbPage+NBATCH]
                sys.stdout.flush()
                print ("LENGTH = 1", self.sTag)
#                 self.minePageVerticalFeature(lPages[nbPage:nbPage+NBATCH], ['x','x2'],level=self.sTag)
                self.minePageVerticalFeature2(lPages[nbPage:nbPage+NBATCH], ['x','xxxx2'],level=self.sTag)
                
                ## V ZONES
                # length 1
                lT1, lScore1, score1 = self.processVSegmentation(lPages[nbPage:nbPage+NBATCH],[],bTAMode=False,iMinLen=1,iMaxLen=1)
                print( nbPage, nbPage + NBATCH, lT1, score1)
                print( '\t',lScore1)
                sys.stdout.flush()

                bTable = True #lT1 is not None and lT1 !=[] and len(lT1[0].getPattern()) > 6
                
                
#                 if lT1 is not None:
#                     self.createTableTemplate(lT1)
#                     ss
                ##"  If parsing structure K+ has good coverage: skip length2?
                # if high enough score: skip len=2?
    #             sys.stdout.flush()
                lOneSupport=[]
                print( "LENGTH = 2")
                if len( lPages) > 1 and bTable or len(lOneSupport) > 18:
                    score2=0
                    lT2=None
                    lScore2=[]
                else:
                    if lT1 is not None: 
                        lNegativesPatterns=list(map(lambda x:x.getPattern(),lT1))
                    else: lNegativesPatterns=[]
                    lT2, lScore2, score2 = self.processVSegmentation(lPages[nbPage:nbPage+NBATCH],lNegativesPatterns,bTAMode=False,iMinLen=2,iMaxLen=2)
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
            
#             self.correctionStep(lPages)

            # for visu
#             if self.bDomTag:self.tagDomAsTable(lPages)
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
#         self.bDebug = False
        for _,p in enumerate(lPages):
            p._lBasicFeatures=p.lf_XCut[:]
#             print (p, list(map(lambda x:(x,x.getTH()),p.getSetofFeatures())))
#             print p, map(lambda x:(x,x.getTH()),p.lf_XCut)
        
        seqGen = sequenceMiner()
        seqGen.bDebug = False
        seqGen.setMinSequenceLength(iMinLen)
        seqGen.setMaxSequenceLength(iMaxLen)
        seqGen.setObjectLevel(XMLDSPageClass)

        chronoOn()

        print ('featuring...',chronoOff())
        lSortedFeatures = seqGen.featureGeneration(lPages,2)
        for cf in lSortedFeatures:
            # weights must not be too large ecause theyv are used in np.obs  (max=64000)
#             cf.setWeight(sum(x.getHeight() * x.getWidth() for x in cf.getNodes())/64000)
            cf.setWeight(sum(x.getHeight() for x in cf.getNodes()) / 64000)
#             cf.setWeight(1)
#             cf.setWeight(len(cf.getNodes()))
#             print (cf, cf.getWeight(),len(cf.getNodes())) # map(lambda x:x.getX(),cf.getNodes())
#         print lSortedFeatures                      
        for _,p in enumerate(lPages):
            p.lFeatureForParsing = p.getCanonicalFeatures() 
            if self.bDebug:print (p, p.lFeatureForParsing)
        sys.stdout.flush()

        if lSortedFeatures == []:
            print ("No template found in this document")
            return None,None,-1
        
        seqGen.bDebug = False
        lmaxSequence = seqGen.generateItemsets(lPages)
        lSeq, _ = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures,mis = 0.2)
        # if one page: what to do ??
        if len(lPages) > 1:
            lOneSupport = self.testHighSupport(lSeq)
        else : lOneSupport=[]
        print ('L1OneSupport: ', lOneSupport)
        ## reference solution
        if len(lOneSupport) < 0:
            # MIS also for patterns, not only item!!
            ## can be used to assess the noise level or here
            chronoOn()
            print( "generation...")
            sys.stdout.flush()
            lSeq, _ = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures,mis = 0.1,L1Support = [])
            sys.stdout.flush()
            ## if many MIS=1.0 -> table with many columns!
            ##actual supports: {'x=40.0': 0.5, 'x=473.0': 1.0, 'x=73.0': 0.75, 'x=558.0': 1.0, 'x=327.0': 1.0, 'x=145.0': 1.0, 'x=243.0': 1.0, 'x=1180.0': 0.25, 'x=726.0': 1.0, 'x=408.0': 1.0, 'x=886.0': 1.0, 'x=1027.0': 0.75, 'x=803.0': 1.0, 'x=952.0': 1.0, 'x=636.0': 1.0, 'x=1136.0': 1.0, 'x=839.0': 0.25}
    #         lFSortedFeatures = self.factorizeHighlyFrequentItems()
            
            lPatterns = seqGen.miningSequencePrefixScan(lSeq)
            
            if lPatterns is None:
                return None, None, -1
                
#             lPatterns = seqGen.beginMiningSequences(lSeq,lSortedFeatures,lMIS)
            print( "chronoTraining", chronoOff())
            print( 'nb patterns: ',len(list(lPatterns)))
            sys.stdout.flush()
            lPatterns  = self.filterNonRegularPatterns(lPatterns)
            lPatterns.sort(key=lambda p_s:self.computePatternScore(p_s[0]), reverse=True)
#             lPatterns.sort(key=lambda (p,s):s, reverse=True)

            lPatterns = list(filter(lambda p_:p_[0] not in lNegativePatterns, lPatterns))
            
#             if self.bDebug:
#                 for p,s in lPatterns:
#                     if s >= 1: 
#                         print p,s, self.computePatternScore(p)
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
            
    #         print 'patterns:', dTemplatesCnd
            chronoOn()
#             seqGen.setKleenePlusTH(self.fKleenPlusTH)
            seqGen.setKleenePlusTH(1.6)

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

        for p in lPages:
            p.lFeatureForParsing = p.lf_XCut
        ## for eac n-cut categories:
            # template with 2 , 3 ,4, 5 .. cuts
        chronoOn()
#         for i,t in enumerate(lVTemplates):
#             print i,lPages[-2], t, t.registration(lPages[-2])
#             print i,lPages[-1], t, t.registration(lPages[-1])
#             print 
           
        lT, lScores, score= self.selectFinalTemplate(lVTemplates,tranprob,lPages)
        print ("chronoFinalViterbi: %s score= %s" % ( chronoOff(), score))
         
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
                for j,page in enumerate(lPages):
                    x, y, score= temp.registration(page)
#                     print (page, i, page.lf_XCut,temp, score,x,y)
                    if score < 0:
                        score= 0
#                         # no template
#                         print ('\t xxx',score)
#                         obs[-1,j]=1.0
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
        np.set_printoptions(precision= 3, linewidth =1000)
        print ("viterbi scores:",fscore)
#         print (transProb)
#         print (obs)
        lTemplate=[]
        lScores=[]
        #assigned to each page the template assigned by viterbi
        for i,page, in enumerate(lPages):
            try:
                mytemplate= lTemplates[states[i]]
                if mytemplate not in lTemplate:
                    lTemplate.append(mytemplate)
            except:# no template
                mytemplate = None
            if mytemplate is not None:
#                 page.resetVerticalTemplate()
                page.addVerticalTemplate(mytemplate)
                registeredPoints, _, score = mytemplate.registration(page)
#                 print ("??",page, states[i], mytemplate, registeredPoints, score, page.lFeatureForParsing)
                if registeredPoints:
                    registeredPoints.sort(key=lambda x_y:x_y[1].getValue())
                    lcuts = list(map(lambda ref_cut:ref_cut[1],registeredPoints))
                    print (page, score, lcuts)
#                     print page, score, lcuts, map(lambda x:x.getWeight(), lcuts),registeredPoints
#                     print '\t', page.lFeatureForParsing,map(lambda x:x.getWeight(), page.lFeatureForParsing)

                    page.addVSeparator(mytemplate,lcuts)
                    lScores.append((states[i],score))
            else:
#                 lScores.append((N,-1))
                lScores.append((N,0))

#         for t in lTemplate:
#             print t, t.getParent()
        fscore= np.average(list(map(lambda x_y:x_y[1],lScores)))
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
    
    
    ##########CORRECTION STEP #################
    def correctionStep(self,lPages):
        """
            correct segmentation wrt template
            - according to the template:
                look at token levels the bets cuts?
                resegment text if needed
                
            
            still need to consider graphline 
        """
        
        for page in lPages:
#             lText = page.getAllNamedObjects(XMLDSTEXTClass)
            lTokens = page.getAllNamedObjects(XMLDSTOKENClass)
#             xTokenHisto = Counter(map(lambda x:round(x.getX()),lTokens))
#             print page, xTokenHisto.most_common(15)
            prevcuts= page.lf_XCut 
            self.minePageVerticalFeature([page], ['x','x2'], level=XMLDSTOKENClass)
            page.lFeatureForParsing  = page.lf_XCut
            if lTokens != []:
                for template in page.getVerticalTemplates():
                    sys.stdout.flush()
                    registeredPoints, lMissing, score = template.registration(page)
                    if registeredPoints is not None:
                        lfinalCuts= map(lambda x_y:x_y[1],filter(lambda x: x[0]!= 'EMPTY',registeredPoints))
#                         print page,registeredPoints, lMissing, score
                        page.resetVSeparator(template)
                        page.addVSeparator(template,lfinalCuts)
    
    
    ####   delete page break ##########################
    
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
                page.createVerticalZones(template)
                
                parentTemplate= template.getParent()
                bReverse=False
                ## need a real test for  isMirrored(self)
#                 print (parentTemplate,parentTemplate.isMirrored())
                if parentTemplate.isMirrored():
                    bReverse = template.getPattern() == parentTemplate.getPattern()[1]
                lregions= page.getVerticalObjects(template)
                if bReverse:
                    lregions.reverse()
#                 print (page, bReverse)
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
#             print (line,myReg)
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
#                     print page.getdVSeparator(template)
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
#                     print page, page.getdVSeparator(template)
                    for cut in page.getdVSeparator(template):
                        newReg= XMLDSObjectClass()
                        domNode  = etree.Element('REGION')
                        domNode.set("x",str(prevcut))
                        ## it is better to avoid
                        YMinus= 1
                        domNode.set("y",str(YMinus))
                        domNode.set("height",str(page.getHeight()-2 * YMinus))
                        domNode.set("width", str(cut.getValue() - prevcut))
                        lCuts.append(cut.getValue() )
                        newReg.setNode(domNode)
                        page.getNode().append(domNode)
                        newReg.setDimensions(prevcut,YMinus, page.getHeight()-2 * YMinus,cut.getValue() - prevcut)
#                         print newReg.getX(),newReg.getY(),newReg.getHeight(),newReg.getWidth(),cut.getValue() - prevcut
                        lRegions.append(newReg)
                        prevcut  = cut.getValue()
                
#                 if lRegions != []:
#                     if self.bIgnoreRegions:
#                         self.deleteRegions(page)
#                     self.storeLineInRegions(page,lRegions)
    
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
                    prevcut=1
                    lCells = []
                    for i,cut in enumerate(page.getdVSeparator(template)):
                        newCell= XMLDSTABLECELLClass()
                        cellNode  = etree.Element('CELL')
                        cellNode.set("x",str(prevcut))
                        cellNode.set("y",'1')
                        cellNode.set("row","0")
                        cellNode.set("col",str(i))
                        cellNode.set("height",str(page.getHeight()))
                        cellNode.set("width", str(cut.getValue() - prevcut))                            
                        cellNode.set('points', '%f,%f,%f,%f,%f,%f,%f,%f'%(prevcut,1, cut.getValue(),1 ,cut.getValue(),page.getHeight(),prevcut,page.getHeight()))
                        newCell.setNode(cellNode)
                        newCell.fromDom(cellNode)
                        tableNode.append(cellNode)
                        lCells.append(newCell)
#                         print (newCell.getX(),newCell.getWidth(),newCell.getY(),newCell.getHeight()),
#                         newCell.setDimensions(prevcut,1, page.getHeight()-2 * 1,cut.getValue() - prevcut)

                        prevcut  = cut.getValue()   
      
                    if lCells != []:
                        self.storeLineInRegions(page,lCells)
                    
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
#             lSubPagesList = self.highLevelSegmentation(self.lPages)
            lSubPagesList = [self.lPages]
            ## need to regroup similar 
             
#             self.arrayApproach(lSubPagesList)
#             return
            if self.baselineMode > 0:
                # not implemented
                self.baselineSegmentation(lSubPagesList)
            else:
                lTemplates = self.iterativeProcessVSegmentation(lSubPagesList)
                return self.ODoc, lTemplates, self.lPages
                ## in this package?? current yes
                self.flattencontent(self.lPages,lTemplates)
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
                else:
                    lRun.append([])

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
    docM.bDebug = True
    docM.run()
    
    if doc and docM.getOutputFileName() != "-":
        docM.writeDom(doc, True)

        
