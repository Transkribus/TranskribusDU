# -*- coding: utf-8 -*-
"""

     H. Déjean

    copyright Xerox 2016
    READ project 

    mine a document (itemset generation)
"""

import sys, os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import common.Component as Component
from common.trace import traceln

from structuralMining import sequenceMiner

from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.XMLDSTOKENClass import XMLDSTOKENClass
from ObjectModel.XMLDSPageClass import XMLDSPageClass
    
    
class pageVerticalMiner(Component.Component):
    """
        pageVerticalMiner class: a component to mine column-like page layout
    """
    
    
    #DEFINE the version, usage and description of this particular component
    usage = "" 
    version = "v.01"
    description = "description: page vertical Zones miner "

    
    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "pageVerticalMiner", self.usage, self.version, self.description) 
        
        
        # use for evaluation
        self.THCOMP = 10
        self.evalData= None
        
        self.lGridPopulation = {}
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        

    
    def processVerticalZones(self,lPages):
        """
            get page features for  vertical zones: find regular vertical Blocks/text structure
            
        """ 
        
        import util.TwoDNeighbourhood as TwoDRel
        
        
        ## COMMUN PROCESSING
        ### DEPENDS ON OBJECT LEVEL !! TEXT/TOKEN!!
        lGrammars = []
        lVEdge = []
        lLElts=[ [] for i in range(0,len(lPages))]
        for i,page in enumerate(lPages):
            lElts= page.getAllNamedObjects(XMLDSTEXTClass)
            for e in lElts:
                e.next=[]
            ## filter elements!!!
            lElts = filter(lambda x:min(x.getHeight(),x.getWidth()) >10,lElts)
            lElts.sort(key=lambda x:x.getY())
            lLElts[i]=lElts
            lVEdge = TwoDRel.findVerticalNeighborEdges(lElts)
            for  a,b in lVEdge:
                a.next.append( b )
            

        ### VerticalZones START
        for i,page in enumerate(lPages):
            lElts= lLElts[i]
            for elt in lElts:
#                 elt.setFeatureFunction(elt.getSetOfListedFeatures,10,lFeatureList=['x'],myLevel=XMLDSTOKENClass)
                elt.setFeatureFunction(elt.getSetOfListedAttributes,20,lFeatureList=['x'],myLevel=XMLDSTEXTClass)
            seqGen = sequenceMiner()
            seqGen.bDebug = False
          
#             traceln("feature Generation...")
            seqGen.setMaxSequenceLength(1)
            seqGen.setTH(0.50)
            ## select minimal frequency for feature
            lSortedFeatures = seqGen.featureGeneration(lElts,2)
#             print page,lSortedFeatures
              
#             print 'start mining of %d elements'% len(lElts)
            linfoHPage, lFeaturedStructure = seqGen.r_prefTree( lSortedFeatures,lElts)
#             self.tagDom(linfoHPage,page,'LEFTBORDER')
            if linfoHPage is not None:
                page.setVX1Info(linfoHPage)
#             self.parseSequence(linfoHPage, lElts)
#         ### VerticalZones XC
#         for i,page in enumerate(lPages):
#             lElts= lLElts[i]
#             for elt in lElts:
#                 elt.resetFeatures()
# #                 elt.setFeatureFunction(elt.getSetOfListedFeatures,10,lFeatureList=['x'],myLevel=XMLDSTOKENClass)
#                 elt.setFeatureFunction(elt.getSetOfListedAttributes,10,lFeatureList=['xc'],myLevel=XMLDSTEXTClass)
#             
#             seqGen = sequenceMiner()
#             seqGen.bDebug = False
#    
#             seqGen.setMaxSequenceLength(1)
#             seqGen.setTH(0.50)
#             ## select minimal frequency for feature
#             lSortedFeatures = seqGen.featureGeneration(lElts,2)
#             print page,lSortedFeatures
#                 
#             print 'start mining of %d elements'% len(lElts)
#             linfoHPage, lFeaturedStructure = seqGen.r_prefTree( lSortedFeatures,lElts)
#             self.tagDom(linfoHPage,page,label="CENTERBORDER") 
#             if linfoHPage is not None:
#                 page.setVXCInfo(linfoHPage)    
# #             self.parseSequence(linfoHPage, lElts)            
#          
        ### VerticalZones ENd
        for i,page in enumerate(lPages):
            lElts= lLElts[i]
            for elt in lElts:
                elt.resetFeatures()
                elt.setFeatureFunction(elt.getSetOfListedAttributes,20,lFeatureList=['x2'],myLevel=XMLDSTEXTClass)
                
            seqGen = sequenceMiner()
            seqGen.bDebug = self.bDebug
                
            seqGen.setMaxSequenceLength(1)
            seqGen.setTH(0.50)
            ## select minimal frequency for feature
            lSortedFeatures = seqGen.featureGeneration(lElts,2)
#             print page, lSortedFeatures
                    
#             print 'start mining of %d elements'% len(lElts)
            linfoHPage, lFeaturedStructure = seqGen.r_prefTree( lSortedFeatures,lElts)
#             self.tagDom(linfoHPage,page,'RIGHTBORDER')
            if linfoHPage is not None:
                page.setVX2Info(linfoHPage)    
#                        
#         ### VerticalZones W
#         for i,page in enumerate(lPages):
#             lElts= lLElts[i]
#             for elt in lElts:
#                 elt.resetFeatures()
# #                 elt.setFeatureFunction(elt.getSetOfListedFeatures,10,lFeatureList=['x'],myLevel=XMLDSTOKENClass)
#                 elt.setFeatureFunction(elt.getSetOfListedAttributes,10,lFeatureList=['width'],myLevel=XMLDSTEXTClass)
#               
#             seqGen = sequenceMiner()
#             seqGen.bDebug = False
#               
#             seqGen.setMaxSequenceLength(1)
#             seqGen.setTH(0.50)
#             ## select minimal frequency for feature
#             lSortedFeatures = seqGen.featureGeneration(lElts,2)
#             print lSortedFeatures
#                   
#             print 'start mining of %d elements'% len(lElts)
#             linfoHPage, lFeaturedStructure = seqGen.r_prefTree( lSortedFeatures,,lElts))
#             if linfoHPage:
#                 ## parse; get sequences of elements; take sequences longer than one!
#                 from feature import featureObject
#                 lLParsings = self.parseSequence(linfoHPage,featureObject,lElts)
#                 for gram,lElts, lres in lLParsings:
#                     self.computeXFromW(linfoHPage,page,lElts)
#             if linfoHPage is not None:
# #                 print linfoHPage
#                 page.setVWInfo(linfoHPage)    
        
        ## build zones:
        ## FIND THE PAGE ZONE FIRST!!
        ## CREATE BO !
        self.buildVZones(lPages)
            
#         return     
        # now page level using previous features
        ##  page uses info stored before
        ## MUST  USE BusObject
        data = self.processVSegmentation(lPages)
#         for i,g in enumerate(lGrammars):
#             print i+1,g
        
        #  CREATE ZONES AND POLUPATEL THEM
        
        ###link inter-page zones
        
        ### BUILD LINEGRID PER ZONE
#         self.populatePageZones()
#         self.buildLineGrid()
        
        return data
            
    def buildVZones(self,lp):
        """
            REASONING NEEDED !!!!!!!!!!! PYCLIPS ?
            if X2 = create similar X1!
            zones 
                feature  = [x1,x2]
            
            full page justification segmentaition : [x11,x12], [x21,x22],....  and x12=x21 
            
            x2 : if right-aligned; OK , otherwise ??
            x1: if left-aliogned OK, otherwise ?? 
            build W features: invariant to shift!
            a sequence of w !!
            
        """
        from feature import featureObject
        import itertools
        
#         print 'BUILD XCUTS'
        for p in lp:
            #add 0
            p.lf_XCut=[]
            for lfi in p.getVX1Info() + p.getVX2Info():
                for fi in lfi:
                    if fi not in   p.lf_XCut:
                        p.lf_XCut.append(fi[0])
            
            p.lf_XCut.sort(key=lambda x:x.getValue())
#             print p, p.lf_XCut
            ## alternatives: build verticalZonefeatureObject
        
        
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
    
    def processVSegmentation(self,lPages):
        """
            use Vertical bloc/text info to find vertical patterns at page level
            Should correspond to column-like zones    
            
        """
        from ObjectModel.verticalZonesTemplateClass import verticalZonestemplateClass
        from ObjectModel.doublePageTemplateClass import doublePageTemplateClass
        from ObjectModel.singlePageTemplate import singlePageTemplateClass

        from feature import featureObject, multiValueFeatureObject
         
         

        for p in lPages:
            p.resetFeatures()
            p.setFeatureFunction(p.getSetOfMigratedFeatures,TH=20.0,lFeatureList=p.lf_XCut)
           
#         print "\t xxxq \t\tfeature Generation..."
        seqGen = sequenceMiner()
        seqGen.bDebug = False
        seqGen.setMaxSequenceLength(2)
        # used for parsing, not for mining
        # 
        seqGen.setSDC(0.3) # related to noise level        
        # INIIAL
        ## select minimal frequency for feature
        lSortedFeatures = seqGen.featureGeneration(lPages,2)
#         for f in lSortedFeatures:f.setType(1)
        ## just need of the longest sequence !!! 
        lmaxSequence = seqGen.generateItemsets(lPages)
        lSeq, lMIS = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures,mis=0.2)
        # here iteration on sequencelength
        lPatterns = seqGen.beginMiningSequences(lSeq,lSortedFeatures,lMIS)
        ### for sequence >1 support is * by lenght 
#         lPatterns.sort(key=lambda x:x[1],reverse=True)
        lNewKleeneSequences = []
        lNewFeatures = []

        lMainPattners= seqGen.testSubSumingPattern(lPatterns)
        lMainPattners.sort(key=lambda (x,y):y)
        
        for p in lPages:
            lf= p.getSetofFeatures().getSequences()[:]
            p.resetFeatures()
            p.setFeatureFunction(p.getSetOfMutliValuedFeatures,TH=0.25,lFeatureList=lf)
            p.computeSetofFeatures()
#             print p, p.getSetofFeatures()
        
        for pattern,supp in lMainPattners[:1]:
            print 'parsing with... ',pattern,supp
            ##" convert into multivalFO here to get features 
            mvPattern = self.patternToMV(pattern)
            print pattern, mvPattern
            lParsingRes = self.parseSequence(mvPattern,multiValueFeatureObject,lPages)
            lns,lnf = seqGen.generateKleeneItemsets(lPages,lParsingRes,featureObject)
            print "loop0:",lns, lnf
            if lns != []:
                lNewKleeneSequences.append(lns) 
                lNewFeatures.extend(lnf)
                ## NEED TO STORE PATTERN 
                ## NEED TO 'recompose'
                ##   a dict to find mvpattern<->pattern
#                 p.addVerticalTemplate(pattern)
        
        ## for the moment enough to have a flat structure?
        i=0
#         print lNewKleeneSequences
        if lNewKleeneSequences == []:
            ## ???relax stuff???
            pass
        while i < 1:
            lnewseq=lNewKleeneSequences.pop(0)
            lnewFeatures= lNewFeatures
            print "CURRENT SEQUENCE:",i,lnewseq
            print "CURRENT FEATURES:",i,lSortedFeatures+ lnewFeatures
            print "\tinit",lSortedFeatures
            del seqGen
            
            # or iterate on lnewseq and reset if pageObject? 
            #                       don't touch the sequenceAPI object
            for p in lPages:
                p.resetFeatures()
                p.setFeatureFunction(p.getSetOfMigratedFeatures,TH=20.0,lFeatureList=p.lf_XCut)  
                p.computeSetofFeatures()          
            seqGen=sequenceMiner()
            seqGen.setMaxSequenceLength(4)         
            seqGen.setSDC(0.8)       
            seqGen.bDebug=False
            longestSequence = seqGen.generateItemsets(lnewseq)
#             print longestSequences
            #if kleeneStar: replace s0+ by gram+
            ltmp = lSortedFeatures[:]
            ltmp.extend(lnewFeatures)
            lSeq, lMIS= seqGen.generateMSPSData(longestSequence,ltmp,mis=0.2)
            print '\n',lSeq,'\n'
            # mine new sequences
            # new seqGen??
            lnp=seqGen.beginMiningSequences(lSeq,lSortedFeatures+lNewFeatures,lMIS)
#             lMainPattners= seqGen.testSubSumingPattern(lnp)
            print 'new res'
            for p in lPages:
                lf= p.getSetofFeatures().getSequences()[:]
                p.resetFeatures()
                p.setFeatureFunction(p.getSetOfMutliValuedFeatures,TH=0.5,lFeatureList=lf)
                p.computeSetofFeatures()
#                 print p, p.getSetofFeatures()            
            for p,sup in lnp:
                if sup > 2 and len(p) > 2:
                    mvPattern = self.patternToMV(p)
                    print p,sup, mvPattern
                    lParsingRes = self.parseSequence(mvPattern,multiValueFeatureObject,lnewseq)
                    lns,lnf = seqGen.generateKleeneItemsets(lnewseq,lParsingRes,featureObject)
                    if lns != []:
                        print p,lns,"\t\t",lnf
                        lNewKleeneSequences.append(lns) 
                        lNewFeatures.extend(lnf)     
                        ## NEED TO STORE p
                        ## from the final mvpattern: need to go back to get all hierarchical
#                         p.addVerticalTemplate(p)           
            i+=1
        return 
    
        ## keep only some 'grammars' with good coverage
        ## parse and store patterns associated to a page 
        ##  find a way to score pattern/grammars : just coverage
        
        
#                 # create template: how to automate the mapping!!
#                 ### if bigrams= create doublepageTemplate then verticaZonetemplate
#                 ## if hierarchical?? 
#                 if len(Gram) == 2:
#                     dpTemplate = doublePageTemplateClass()
#                     # instantiate left/right page: split Gram ??
#                 else:
#                     pageTemplate = singlePageTemplateClass()
#                     vzTemplate   = verticalZoneTemplateClass()
#                     pageTemplate.addMainZone(vzTemplate)
#                     vzTemplate.createFromSeqOfFeatures(Gram)
#                 ## Gram: can contains several templates: need a doublePageTemple which contains the VerticalTemplates??
#                 # add lcuts
#                 # or simply addTemplate
#                 for p in lFullList:
# #                     print p,Gram, p.getVerticalModels()
#                     ### 
#                     p.addTemplate(pageTemplate)
        
        
        for p in lPages:
            for model in p.getVerticalTemplates():
                # kind of registration for non noisy pages
                # MUST use template now.
                self.collectVerticalZones(p,model,model)
            print p.lVSeparator
        
        ## need models??? currently use features as model
        ## create zone objects and populate them
        for p in lPages:
            for model in p.getVerticalTemplates():
                ## creation of the zones and populations with subElements 
                p.createVerticalZones(model)
            
        ## DOM tagging
        ## no longer grammars but parsed sequences!!
        self.tagDOMVerticalZones(lPages,lGrammars)            
    
        
    def collectVerticalZones(self,page,model,lFeaturedStructure):
        """
            match a model against the page to get 'real' values  (not model-values which are 'generic')
        """
        if lFeaturedStructure is None:return

        import numpy as np
        
        for featurerule in lFeaturedStructure:
            if type(featurerule).__name__ == 'list':
                self.collectVerticalZones(page,model,featurerule)
            else:
                lElts= featurerule.getNodes()
                for fea in page.getSetofFeatures().getSequences():
                    if featurerule == fea:
                        myValues = fea.getOldValue()
                        if type(myValues).__name__ == 'list' :
                            for myfeatureX in lmyValues:
#                                     x = myfeatureX[0].getValue()
                                if myfeatureX not in page.lVSeparator:
                                    page.lVSeparator[str(model)].append(myfeatureX)
                        else:
                            if myValues not in page.lVSeparator:
                                page.lVSeparator[str(model)].append(myValues)                                
                                    
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
        
        seqGen = sequenceMiner()
        
        myGram = sequenceGrammar()
        lLParsings = []
        lParsings = myGram.parseSequence(myFeatureType,myPattern, lSeqElements)
        lFullList=[]
        lFullObjects = []
        for x,y,z in lParsings:
            lFullList.extend(x)
            lFullObjects.extend(y)
        lLParsings.append((myPattern,lFullList,lParsings))

        return lLParsings
        

                    
    def tagDOMVerticalZones(self,lPages,lFeaturedStructure):
        """
            depends on the feature semantics
            get feature structure
            go down to the 'real instances' and a given element; tag it 
            
            
            regression = np.polyfit(x, y, 1)
            http://www.scipy-lectures.org/intro/summary-exercises/optimize-fit.html

            
        """
        if lFeaturedStructure is None:return

        import numpy as np
        
        for featurerule in lFeaturedStructure:
            if type(featurerule).__name__ == 'list':
                self.tagDOMVerticalZones(lPages,featurerule)
            else:
#                 print featurerule, featurerule.getNodes()
                import libxml2                
                
                #### NOW FOR VERTICALZONEFEATURE ONLY
                # elts = page
#                 print "\n%s\n"%featurerule
                lElts= featurerule.getNodes()
#                 for pelt in lElts:
                for pelt in lPages:

#                     print pelt.getNumber(), pelt.getSetofFeatures()
                    for fea in pelt.getSetofFeatures().getSequences():
                        if featurerule == fea:
                            lmyValues = fea.getOldValue()
                            for myfeatureX in [lmyValues]:
                                lX = []
                                lY = []
#                                 print '!!!',myfeatureX, myfeatureX[0].getValue()
                                for elt in pelt.getAllNamedObjects(XMLDSTEXTClass):
#                                     if abs(float(elt.getAttribute("x")) - myfeatureX[0].getValue()) < myfeatureX[0].getTH():
                                    if abs(float(elt.getAttribute("x")) - myfeatureX) < featurerule.getTH():

                                        lX.append(float(elt.getAttribute('x')))
                                        lY.append(float(elt.getAttribute('y')))
#                                 print lX, lY
                                if len(lX) < 3:
                                    ## missing value:
                                        ##   [0].getValue()
                                    b = myfeatureX
                                    ymax = myfeatureX
#                                     print pelt, 'MISSING' ,myfeatureX ,b, ymax
                                else:
                                    a,b = np.polyfit(lY, lX, 1)
                                    y0=  b
                                    ymax = a*pelt.getHeight()+b
#                                     print pelt, a,b, y0, ymax
                                    
                                verticalSep  = libxml2.newNode('PAGEBORDER')
                                verticalSep.setProp('points', '%f,%f,%f,%f'%(b,0,ymax,pelt.getHeight()))
    #                             verticalSep.setProp('x',str(fea.getOldValue()))
    #                             verticalSep.setProp('y',str(y0))
    #                             verticalSep.setProp('width','2')
    #                             verticalSep.setProp('height',str(ymax))
#                                 print pelt.getNumber(),verticalSep
                                pelt.getNode().addChild(verticalSep)
#             return
            
            
    def createColumnSequences(self):
        """
            create a sequence of line from concatenated columns
            
            order: same order ot mirrored
            take the first 100 pages: since (a,b) -> mirrored
        """    
        
    
            
    def testCliping(self,lPages):
        """
            all in the name
            
        """
        from ObjectModel.XMLDSObjectClass import XMLDSObjectClass 
        for page in lPages:
            region=XMLDSObjectClass()
            region.addAttribute('x', 0)
            region.addAttribute('y', 0)
            region.addAttribute('height', page.getAttribute('height'))
            region.addAttribute('width', 110)
            print        region.getX(),region.getY(),region.getWidth(),region.getHeight()
            print page.getAttributes(), page.getX2()
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
                        print txt
                else:
                    txt.resizeMe(XMLDSTOKENClass)
            
    
    def generateTestOutput(self,lPages):
        """
            create a run XML file
        """
        
        import libxml2
        self.evalData = libxml2.newDoc('1.0')
        root = libxml2.newNode('DOCUMENT')
        self.evalData.setRootElement(root)
        for page in lPages:
            domp=libxml2.newNode('PAGE')
            domp.setProp('number',page.getAttribute('number'))
            root.addChild(domp)
            for sep in page.lVSeparator:
                print page.lVSeparator
                domsep= libxml2.newNode('SeparatorRegion')
                domp.addChild(domsep)
#                 domsep.setProp('x', str(sep[0].getValue()))
                domsep.setProp('x', str(sep[0]))
        
    
    
    #--- RUN ---------------------------------------------------------------------------------------------------------------    
    def run(self, doc):
        """
            for a set of pages, associate each page with several vertical zones  aka column-like elements
            Populate the vertical zones with page elements (text)
             
             
        """
        self.doc= doc
        # use the lite version
        self.ODoc = XMLDSDocument()
        
        self.ODoc.loadFromDom(self.doc,listPages=range(self.firstPage,self.lastPage+1))        
        self.lPages= self.ODoc.getPages()   
        print self.lPages
        

        rundata =self.processVerticalZones(self.lPages)
    
        return rundata

    #--- TESTS -------------------------------------------------------------------------------------------------------------    
    #
    # Here we have the code used to test this component on a prepared testset (see under <ROOT>/test/common)
    # Do: python ../../src/common/TypicalComponent.py --test REF_TypicalComponent/
    #
    
    
    def testComparePageVertical(self,runElt,refElt,tag):
        """
            input:  <SeparatorRegion x="51.36" y="7.44" height="764.4" width="2.88"/>
        """
        return abs(float(runElt.prop('x')) - float(refElt.prop('x'))) < self.THCOMP 
  
        
        
    def testVerticalSegmentation(self,srefData,srunData, bVisual):
        """
            GT: rectangle or separator?
        
            GT: very basic one: 
                list of X cuts : parameter : delta X
                better one:
                y = ax + b  : how to compute similarities??  see how baseline computation is done ?
    
    
            ref: <PAGE number='' lcuts='X1 X2 x3'>
            need of page X1 X2 ???  
             <PAGE number="8" imageFilename="M_Otterskirchen_012_0007.jpg" width="488.4" height="774.24">
    <REGION type="other" x="1.2" y="5.04" height="767.04" width="56.64"/>
    <REGION type="other" x="50.64" y="6.48" height="765.6" width="323.76"/>
    <REGION type="other" x="366.72" y="4.56" height="765.12" width="118.08"/>
    <SeparatorRegion x="51.36" y="7.44" height="764.4" width="2.88"/>
    <SeparatorRegion x="360.96" y="4.32" height="766.32" width="15.12"/>
  </PAGE>

        """
        import libxml2

        cntOk = cntErr = cntMissed = 0
        RefData = libxml2.parseMemory(srefData.strip("\n"), len(srefData.strip("\n")))
        try:
            RunData = libxml2.parseMemory(srunData.strip("\n"), len(srunData.strip("\n")))
        except:
            RunData = None
            return (cntOk, cntErr, cntMissed)        
        
        lRun = []
        if RunData:
            ctxt = RunData.xpathNewContext()
            lpages = ctxt.xpathEval('//%s' % ('PAGE'))
            for page in lpages:
                xpath  = "./%s" % ("SeparatorRegion")
                ctxt.setContextNode(page)
                lSep = ctxt.xpathEval(xpath)
                lRun.append(lSep)
            ctxt.xpathFreeContext()

    #         print '----',self.getInputFileName()

        lRef = []
        ctxt = RefData.xpathNewContext()
        lPages = ctxt.xpathEval('//%s' % ('PAGE'))
        for page in lPages:
            xpath  = "./%s" % ("SeparatorRegion")
            ctxt.setContextNode(page)
            lSep = ctxt.xpathEval(xpath)
            lRef.append(lSep)
        ctxt.xpathFreeContext()            

        runLen = len(lRun)
        refLen = len(lRef)
        
        assert runLen == refLen
        ltisRefsRunbErrbMiss= list()
        for i in range(0,len(lRef)):
            lRefCovered = []
            runLen = len(lRun[i])
            refLen= len(lRef[i])
            curRun = curRef = 0
            while curRun <= runLen - 1:  # or curRef <= refLen -1:
                bErr, bMiss = False, False
                try:runElt = lRun[i][curRun]
                except IndexError: runElt = None
    #             print '___',curRun,runElt
                curRef = 0
                bFound = False
                while not bFound and curRef <= refLen - 1:
                    try: refElt = lRef[i][curRef]
                    except IndexError: refElt = None
    #                 self.compareString(runElt,runElt)
                    if runElt and refElt not in lRefCovered and self.testComparePageVertical(runElt, refElt,"SeparatorRegion"):
                        bFound = True
                        lRefCovered.append(refElt)
                        resRef=refElt.prop('x')
                    else:
                        curRef += 1
                if bFound:
                    if bVisual:print "FOUND:", runElt, ' -- ', lRefCovered[-1]
                    cntOk += 1
                    curRun += 1
                else:
                    resRef=''
                    curRun += 1
                    cntErr += 1
                    bErr = True
#                     bMiss = True
                    if bVisual:print "ERROR:", runElt
                ltisRefsRunbErrbMiss.append( (i+1, runElt.prop('x'),resRef, bErr, bMiss) )
            
            for ref in lRef[i]:
                if ref not in lRefCovered:
                    ltisRefsRunbErrbMiss.append( (i+1, '',ref.prop('x'), False, True) )
                    
            # add missed elements!
            ltisRefsRunbErrbMiss.sort(key=lambda (x,y,z,t,u):x)
            cntMissed += len(lRef[i]) - len(lRefCovered)


            for x in lRef[i]:
                if x not in lRefCovered:
                    print "MISSED:", x
        # traceln("%d\t%d\t%d" % (cntOk,cntErr,cntMissed))
        return (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss)        
                
        
    
    def testRun(self, filename, outFile=None):
        """
        testRun is responsible for running the component on this file and returning a string that reflects the result in a way
        that is understandable to a human and to a program. Nicely serialized Python data or XML is fine
        """
        doc = self.loadDom(filename)
        self.run(doc)
#         doc.freeDoc()
        self.generateTestOutput(self.lPages)
#         print self.evalData.serialize('utf-8',1)

        if outFile: self.writeDom(doc)
        return self.evalData.serialize('utf-8',1)
    
    def testCompare(self, srefData, srunData, bVisual=False):
        """
        Our comparison is very simple: same or different. N
        We anyway return this in term of precision/recall
        If we want to compute the error differently, we must define out own testInit testRecord, testReport
        """
        dicTestByTask = dict()
#         dicTestByTask['Region']= self.testVerticalSegmentation(Self, srefData, srunData,'REGION')
        dicTestByTask['Separator']= self.testVerticalSegmentation(srefData, srunData,bVisual)
    
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
        
    #parse the command line
    dParams, args = docM.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    docM.setParams(dParams)
    
    doc = docM.loadDom()
    docM.run(doc)
    if doc and docM.getOutputFileName() != "-":
        docM.writeDom(doc, True)

