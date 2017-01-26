# -*- coding: utf-8 -*-
"""

     H. Déjean

    copyright Xerox 2016
    READ project 

    mine a document (itemset generation)
"""

import sys, os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import libxml2
import numpy as np
import common.Component as Component

from structuralMining import sequenceMiner
from feature import featureObject, multiValueFeatureObject 


from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSGRAHPLINEClass import XMLDSGRAPHLINEClass
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.XMLDSTOKENClass import XMLDSTOKENClass
#from ObjectModel.XMLDSPageClass import XMLDSPageClass
from ObjectModel.singlePageTemplateClass import singlePageTemplateClass
from ObjectModel.doublePageTemplateClass import doublePageTemplateClass
from ObjectModel.verticalZonesTemplateClass import verticalZonestemplateClass    
    
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
        
        
        # TH for comparing numerical features for X
        self.THNUMERICAL= 30
        # use for evaluation
        self.THCOMP = 10
        self.evalData= None
        
        self.bManual = False
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if dParams.has_key("pattern"): 
            self.manualPattern = eval( dParams["pattern"])
            self.bManual=True  


        
    
    
    def minePageVerticalFeature(self,lPages):
        """
            get page features for  vertical zones: find vertical regular vertical Blocks/text structure
            
        """ 
        
        import util.TwoDNeighbourhood as TwoDRel
        
        
        ## COMMUN PROCESSING
        ### DEPENDS ON OBJECT LEVEL !! TEXT/TOKEN!!
        lVEdge = []
        lLElts=[ [] for i in range(0,len(lPages))]
        for i,page in enumerate(lPages):
            lElts= page.getAllNamedObjects(XMLDSTEXTClass)
            for e in lElts:
                e.next=[]
            ## filter elements!!!
            lElts = filter(lambda x:min(x.getHeight(),x.getWidth()) > 10,lElts)
            lElts = filter(lambda x:x.getHeight() > 10,lElts)

            lElts.sort(key=lambda x:x.getY())
            lLElts[i]=lElts
            lVEdge = TwoDRel.findVerticalNeighborEdges(lElts)
            for  a,b in lVEdge:
                a.next.append( b )

        ### VerticalZones START
        for i,page, in enumerate(lPages):
#             print page
            lElts= lLElts[i]
            for elt in lElts:
#                 elt.setFeatureFunction(elt.getSetOfListedAttributes,20,lFeatureList=['width'],myLevel=XMLDSTEXTClass)
                elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['x','x2'],myLevel=XMLDSTEXTClass)
   
            seqGen = sequenceMiner()
#             seqGen.bDebug = self.bDebug
            _  = seqGen.featureGeneration(lElts,2)
            lKleendPlus = self.getKleenePlusFeatures(lElts)
            page.setVX1Info(lKleendPlus)
            del seqGen

        self.buildVZones(lPages)
        
        return lPages
            
    def getKleenePlusFeatures(self,lElts):
        """
            select KleenePlus elements based on .next (only possible for unigrams)
        """   
        dFreqFeatures={}
        dKleenePlusFeatures = {}
        
        lKleenePlus=[]
        for elt in lElts:
            for fea in elt.getSetofFeatures().getSequences():
                try:dFreqFeatures[fea] +=1
                except KeyError:dFreqFeatures[fea] = 1
                for nextE in elt.next:
                    if fea in nextE.getSetofFeatures().getSequences():
                        try:
                            dKleenePlusFeatures[fea].append(nextE)
                            dKleenePlusFeatures[fea].append(elt)                            
                        except KeyError:
                            dKleenePlusFeatures[fea]=[elt]
                            dKleenePlusFeatures[fea].append(nextE)
        for fea in dFreqFeatures:
            try:
                dKleenePlusFeatures[fea]
                if len(set(dKleenePlusFeatures[fea])) >= 0.5 *  dFreqFeatures[fea]:
                    lKleenePlus.append(fea) 
#                     print fea,len(set(dKleenePlusFeatures[fea])) , dFreqFeatures[fea]
            except:
                pass
        return  lKleenePlus
    
    def buildVZones(self,lp):
        """
            store vertical positions in each page
        """
        
        for p in lp:
            p.lf_XCut=[]
            for fi in p.getVX1Info() + p.getVX2Info():
                if fi not in p.lf_XCut:
                    l = sum(x.getHeight() for x in fi.getNodes())
                    fi.setWeight(l)
                    p.lf_XCut.append(fi)
            for graphline in p.getAllNamedObjects(XMLDSGRAPHLINEClass):
                if graphline.getHeight() > graphline.getWidth() and graphline.getHeight() > 50:
                    # create a feature
                    f = featureObject()
                    f.setType(featureObject.NUMERICAL)
                    f.setTH(self.THNUMERICAL)
                    f.setValue(graphline.getY())
                    f.setWeight(graphline.getHeight())
                    f.setName("x")
                    f.setValue(round(graphline.getX()))
                    if f not in p.lf_XCut:
                        p.lf_XCut.append(f)
                
            p.lf_XCut.sort(key=lambda x:x.getValue())
            
#             for x in  p.lf_XCut  : print x,x.getWeight()            
            if self.bDebug :print p,  p.lf_XCut            
        
    def patternToMV(self,pattern,TH):
        """
            convert a list of f as multivaluefeature
            keep mvf as it is
            
            important to define the TH  value for multivaluefeature
        """
        
        lmv= []
        for itemset in pattern:
            # if one item which is boolean: gramplus element
            if len(itemset)==1 and itemset[0].getType()==1:
                lmv.append(itemset[0])
            else:
                mv = multiValueFeatureObject()
                name= "multi" #'|'.join(i.getName() for i in itemset)
                mv.setName(name)
                mv.setTH(TH)
                mv.setValue(map(lambda x:x,itemset))
    #             print mv.getValue()
                lmv.append(mv)
#             print itemset, mv
        return lmv
    
    
    
    def processWithTemplate(self,lPattern,lPages):
        """
            process sequence of pqges with given pattern
            create table 
        """
        
        
        ## artifical construction Motter12
#         lPattern= [ [41.0, 380,442.0] , [40,110,442] ]
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
            lfPattern.append(fItemset)
    
        pattern = lfPattern
        
        print pattern
        
        ### in prodf: mytemplate given by page.getVerticalTemplates()
        mytemplate = verticalZonestemplateClass()
        mytemplate.setPattern(pattern[0])

        mytemplate2 = verticalZonestemplateClass()
#         mytemplate2.setPattern(pattern [1])
        mytemplate2.setPattern(pattern[0])
        
        # registration provides best matching
        ## from registration matched: select the final cuts
        for i,p in enumerate(lPages):
            print p, p.lf_XCut
            registeredPoints1, lMissing1, score1 = mytemplate.registration(p)
            registeredPoints2, lMissing2, score2 = mytemplate2.registration(p)
            print i,p,registeredPoints1, registeredPoints2, score1
            # if score1 == score 2 !!
            if score1 > 0 and score1 >= score2:
                lfinalCuts= map(lambda (x,y):y,filter(lambda (x,y): x!= 'EMPTY',registeredPoints1))
                print p,'final1:',lfinalCuts, lMissing1
                p.addVerticalTemplate(mytemplate)
                p.addVSeparator(mytemplate,lfinalCuts)
            elif score2 > 0 and  score2 > score1:
                lfinalCuts= map(lambda (x,y):y,filter(lambda (x,y): x!= 'EMPTY',registeredPoints2))
                print registeredPoints2
                print 'final2:',lfinalCuts, lMissing2
                p.addVerticalTemplate(mytemplate)
                p.addVSeparator(mytemplate,lfinalCuts)
            else:
                print 'NO REGISTRATION'
        
        self.tagDomAsTable(lPages)
        
        return 1
    
    
    def testKleeneageTemplates(self,dTemplatesCnd,lPages):
        """
            test a set of patterns
            select those which are kleene+ 
        """
        
        """
            resulting parsing can be used as prior for hmm/viterbi? yes
            but need to apply seqrule for completion or relax parsing
            
        """
        lTemplates = []
        dScoredTemplates = {}
        for templateType in dTemplatesCnd.keys():
            for _,_, mytemplate in dTemplatesCnd[templateType][:5]:
                
                ## need to test kleene+: if not discard?
                parseRes = self.parseWithTemplate(mytemplate,lPages)
                
                ## assess and keep if kleeneplus
                if parseRes:
                    parseRes[1].sort()
                    print mytemplate,'\t\t',len(parseRes[1]),parseRes[1]
                    if mytemplate.isMirrored():
                        lTemplates.append(mytemplate.leftPage)
                        dScoredTemplates[mytemplate.leftPage] = len(parseRes[1])
                        lTemplates.append(mytemplate.rightPage)
                        dScoredTemplates[mytemplate.rightPage] = len(parseRes[1])
                        
                    else:
                        lTemplates.append(mytemplate.mainZone)
                        dScoredTemplates[mytemplate.mainZone] = len(parseRes[1])

        
        #  transition matrix: look at  for template in page.getVerticalTemplates():
        N = len(lTemplates) + 1
        transProb = np.zeros((N,N), dtype = np.float16)
            
        dTemplateIndex= {}
        for i,template in enumerate(lTemplates):
            dTemplateIndex[template]=i
        for t in dTemplateIndex:
            print dTemplateIndex[t],t, t.parent
            
#         for page in lPages:
#             for t in page.getVerticalTemplates():
#                 print page , dTemplateIndex[t]
        for i,page in enumerate(lPages):
            # only takes the best ?
            for template in page.getVerticalTemplates():
#                 print page, template
                try:
                    nextPage=lPages[i+1]
                    lnextTemplates = nextPage.getVerticalTemplates()
                    for nextT in lnextTemplates:
#                         print i, template,dScoredTemplates[template], nextT,nextT is template
                        ## not one: but the frequency of the template
                        w = dScoredTemplates[template]
                        ## kleene 
                        if nextT is template:
                            w +=  dScoredTemplates[template]
                        if template.parent.isMirrored() and nextT.parent.isMirrored() and template.parent == nextT.parent:
# #                             print nextPage, dTemplateIndex[template],dTemplateIndex[nextT],transProb[dTemplateIndex[template],dTemplateIndex[nextT]], template, nextT
                            w += dScoredTemplates[template]
                        # if transition 
                        ## ...
                        transProb[dTemplateIndex[template],dTemplateIndex[nextT]] = w +  transProb[dTemplateIndex[template],dTemplateIndex[nextT]]
                        if np.isinf(transProb[dTemplateIndex[template],dTemplateIndex[nextT]]):
                            transProb[dTemplateIndex[template],dTemplateIndex[nextT]] = 64000
                except IndexError:pass
        
#         transProb /= totalSum
        #all states point to None
        transProb[:,-1] = 1.0 #/totalSum
        #last: None: to all
        transProb[-1,:] = 1.0 #/totalSum
        mmax =  np.amax(transProb)
        if np.isinf(mmax):
            mmax=64000
        return lTemplates,transProb / mmax
    
    def parseWithTemplate(self,mytemplate,lPages):
        """
            parse a set of pages with pattern
        """

        pattern = mytemplate.getPattern()
        
        dMtoSingleFeature= {}
        PARTIALMATCH_TH = 1.0
        mvPattern = self.patternToMV(pattern, PARTIALMATCH_TH)
        for i,pp in enumerate(mvPattern):
            dMtoSingleFeature[mvPattern[i]] = pattern[i]

        #need to optimize this double call        
        for p in lPages:
            p.resetFeatures()
            p.setFeatureFunction(p.getSetOfMigratedFeatures,TH = self.THNUMERICAL,lFeatureList=p.lf_XCut)        
            p.computeSetofFeatures()    
        for p in lPages:
            lf= p.getSetofFeatures().getSequences()[:]
            p.resetFeatures()
            p.setFeatureFunction(p.getSetOfMutliValuedFeatures,TH = PARTIALMATCH_TH, lFeatureList = lf)
            p.computeSetofFeatures()            

        seqGen = sequenceMiner()
        
        
        parsingRes = seqGen.parseSequence(mvPattern,multiValueFeatureObject,lPages)
        if parsingRes:
            # needed for computing transition matrix
            self.populateElementWithParsing(mytemplate,parsingRes,dMtoSingleFeature)
            
        return parsingRes
    
    def processVSegmentation(self,lPages):
        """
            use Vertical bloc/text info to find vertical patterns at page level
            Should correspond to column-like zones    
            
        """
        
        """
            generate a n-best candidates for the various templates
            -singlePage
                onecol
                ncols
                regulargrid
            -mirroredpage
                onecol
                ncols
                regulargrid
            
            test: if i want a singlepagesinglecol: whic best
                        
            
        """  
        pageWidth=0.0
        for p in lPages:
            p.resetFeatures()
            p.setFeatureFunction(p.getSetOfMigratedFeatures,TH=self.THNUMERICAL,lFeatureList=p.lf_XCut)
            pageWidth += p.getWidth()
        pageWidth /=len(lPages)

        seqGen = sequenceMiner()
        seqGen.bDebug = False
        seqGen.setMaxSequenceLength(2)
        ## sdc: depends on the templates: if small is ok: one single template, onepage?? 
        seqGen.setSDC(0.66) # related to noise level       AND STRUCTURES (if many columns) 

        lSortedFeatures = seqGen.featureGeneration(lPages,2)
        if lSortedFeatures == []:
            print "No template found in this document"
            return
        seqGen.bDebug = False
        lmaxSequence = seqGen.generateItemsets(lPages)
        lSeq, lMIS = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures,mis = 0.1)
        lPatterns = seqGen.beginMiningSequences(lSeq,lSortedFeatures,lMIS)
        
        ## cleaning? :         [a,b] [b,a], [a],[a,a]  
        lPatterns.sort(key=lambda (x,y):y, reverse=True)
        #filer too short patterns
        lPatterns  = filter(lambda (p,s):self.getPatternWidth(p) > pageWidth * 0.25,lPatterns)
        if self.bDebug:
            for p,s in lPatterns:
                if s > 1: print p,s
                
        ### GENERATE SEQUENTIAL RULES
        seqGen.bDebug=True
        lSeqRules = seqGen.generateSequentialRules(lPatterns)
        
        ### APPLY RULESSSS setMaxSequenceLength=1 enough for tagging?
        ### or simulate the 'gain' that the rule could bring to some pattern!
        _,dCP = self.getPatternGraph(lSeqRules)
#         if self.bDebug:print dCP
        #select N(5)-best candidates for each pattern
        dTemplatesCnd = self.analyzeListOfPatterns(lPatterns,dCP)
        if self.bDebug:print dTemplatesCnd
#         dAncestors = seqGen.applySetOfRules(lSeqRules, lPatterns)
        
        ## DECORATE PAGES WITH ANCESTOR TEMPLATE
        ### %updated: level of noise 
#         self.updateWithAncestor(lPages,dPC)
        # parse with templates to collect proba  for matching 
        lVTemplates,tranprob = self.testKleeneageTemplates(dTemplatesCnd, lPages)
#         print lVTemplates
        
        ## final decision: viterbi for assigning a template to an element using registration score
        self.selectFinalTemplate(lVTemplates,tranprob,lPages)
        
        # tagging for visualization
        self.tagDomAsTable(lPages)
#         self.tagDomWithBestTemplate(lPages)
        
        del seqGen
        
        return 1 
    
    

    def updateWithAncestor(self,lPages,dCP):
        """
            replace multiValueFeature by Ancestor feature

        """
        
        for page in lPages:
            print page
            mvPattern = page.getSetofFeatures().getSequences()
            print mvPattern
            try:
                dCP[str(mvPattern)]
                print mvPattern, 'replaced by: ', dCP[str(mvPattern)]
            except KeyError:pass
        
        
        
    def selectFinalTemplate(self,lTemplates,transProb,lPages):
        """
            apply viterbi to select best sequence of templates
        """
        import spm.viterbi as viterbi        

        if  lTemplates == []:
            return None
        
        def buildObs(lTemplates,lPages):
            """
                build observation prob
            """
            N = len(lTemplates) + 1
            obs = np.zeros((N,len(lPages)), dtype=np.float16) 
            for i,temp in enumerate(lTemplates):
                for j,page in enumerate(lPages):
                    _, _, score= temp.registration(page)
                    if score == -1:
                        score= 0.0
                    obs[i,j]= score
                    if np.isinf(obs[i,j]):
                        obs[i,j] = 64000
                    if np.isnan(obs[i,j]):
                        obs[i,j] = 0.0                        
#                     print i,j,page, temp,score 
                #add no-template:-1
            return obs / np.amax(obs)

        
        N= len(lTemplates) + 1.0
        # build transition score matrix
        ## use the support to initialize ?? why 
        initialProb = np.ones(N) * 1
        initialProb = np.reshape(initialProb,(N,1))
        obs = buildObs(lTemplates,lPages)
        d = viterbi.Decoder(initialProb, transProb, obs)
        states,score =  d.Decode(np.arange(len(lPages)))

        print states, score
        
        #assigned to each page the template assigned by viterbi
        for i,page, in enumerate(lPages):
            mytemplate= lTemplates[states[i]]
            page.resetVerticalTemplate()
            page.addVerticalTemplate(mytemplate)
            registeredPoints, lMissing, score= mytemplate.registration(page)
            if registeredPoints:
                print registeredPoints, lMissing , score
                if lMissing != []:
                    registeredPoints.extend(zip(lMissing,lMissing))
                registeredPoints.sort(key=lambda (x,y):y.getValue())
                lcuts = map(lambda (ref,cut):cut,registeredPoints)
                page.addVSeparator(mytemplate,lcuts)
            
        return score        
        
    
    def getPatternWidth(self,pattern):
        """
            return the width of a pattern
        """
        # if bigramm
        return min ( (pattern[0][-1].getValue() - pattern[0][0].getValue()), pattern[-1][-1].getValue() - pattern[-1][0].getValue())
    
    def analyzeListOfPatterns(self,lPatterns,dCA):
        """
            group pattern by family
            unigram, bigrann,...
            and by size (# item in itemset)
            and by  mirrored ([<a>,<B>] [<b>,<a>])
            
            family tree: a ; a,b ; a,b,c 
            shiftedpages! same page but leftright shift  (bozen)
            
            need to add kleentest?
            
            for each template: keep soltion with length 2,3,4 + elements anyway?
            
            
        """
        dTemplatesTypes={}
        for pattern,support in filter(lambda (x,y):y>1,lPatterns):
            ## here first test width of the pattern
            ## if too narrow, skip it!
            ### singlepage:
#             print pattern
            if len(pattern) == 1:
                try:
                    dCA[str(pattern)]
                    bSkip = True
                except KeyError:bSkip=False
                if not bSkip:                
                    # singlecol
                    icolMin=2
                    icolMax=2
                    template =  self.isVZonePattern(pattern,icolMin,icolMax)
                    if template:
                        try:dTemplatesTypes[template.__class__.__name__].append((pattern,support,template))
                        except KeyError: dTemplatesTypes[template.__class__.__name__] = [(pattern,support,template)]                
                      
                    ## test multicol
                    icolMin=3
                    if len(pattern) == 1:
                        template =  self.isVZonePattern(pattern,icolMin)
                        if template:
                            try:dTemplatesTypes[template.__class__.__name__].append((pattern,support,template))
                            except KeyError: dTemplatesTypes[template.__class__.__name__] = [(pattern,support,template)]              

            #double page
            if self.isMirroredPattern(pattern):
                bSkip=False
                try:
                    # skip if one of the ancestor is mirrored: or simply one ancestor ?? 
                    dCA[str(pattern)]
                    for parent in dCA[str(pattern)]:
                        if self.isMirroredPattern(parent):
#                             print pattern, 'skipped,',parent
                            bSkip = True
                except KeyError:bSkip=False
#                 if len(pattern[0]) != 3:
#                     bSkip=True
                if not bSkip:
                    template=doublePageTemplateClass()
                    template.fromPattern(pattern)
                    try:dTemplatesTypes[template.__class__.__name__].append((pattern,support,template))
                    except KeyError: dTemplatesTypes[template.__class__.__name__] = [(pattern,support,template)]
#                 
#                 ## test if translated  a= b(+-)X
#                     # generate only one singlepage : shift done during the registration?? but then issue with parsing?
#                     # ==> better to have a specific template
                
                
            
        print '-'*30
        for classType in  dTemplatesTypes.keys():
            print classType
            for t in dTemplatesTypes[classType]:
                print '\t',t
        
        return dTemplatesTypes
        
    def getPatternGraph(self,lRules):
        """
            create an graph which linsk exoannded patterns
            (a) -> (ab)
            (abc) -> (abcd)
           
           rule = (newPattern,item,i,pattern, fConfidence)
                   RULE: [['x=19.0', 'x=48.0', 'x=345.0'], ['x=19.0', 'x=126.0', 'x=345.0']] => 'x=464.0'[0] (22.0/19.0 = 0.863636363636)


            can be used for  tagging go up unitl no paretn
        """
        dParentChild= {}
        dChildParent= {}
        for lhs, rhs, itemsetIndex, fullpattern, fConfidence in lRules:
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
            test if a pattern is mirrored or not
            [A,B,C] [C,B,A]
            
            left must be different from right !
        """
        if len(pattern) != 2 or len(pattern[0]) != len(pattern[1]) or (pattern[0] == pattern[1]): 
            return False
        else:
            inv1 =  pattern[1][:]
            inv1.reverse()
            lcouple1= zip(inv1,inv1[1:])
            lw1= map(lambda (x,y):abs(y.getValue()-x.getValue()),lcouple1)
            lcouple0= zip(pattern[0],pattern[0][1:])
            lw0= map(lambda (x,y):abs(y.getValue()-x.getValue()),lcouple0)
            final = set(map(lambda (x,y): abs(x-y) < self.THNUMERICAL * 1,zip(lw0,lw1)))
            return  set(final) == set([True])
    
    def isVZonePattern(self,pattern,iNbCol=3,iNbColMax=None):
        """
            is pattern a regular zone pattern:
                one or two elements, each with the same number of cuts, at least 2
            return corresponding template class
        """

        if iNbColMax == None:
            linterval=range(iNbCol,20)
        else: 
            linterval=range(iNbCol,iNbCol+1)
        if len(pattern) > 2:
            return None
        # at least 3 elements
        if len(pattern) == 2 and pattern[0] != pattern[1] and  len(pattern[0])  == len(pattern[1])  and len(pattern[0]) > 2 and len(pattern[0]) in linterval:
            bitemplate=doublePageTemplateClass()
            bitemplate.setPattern(pattern)
            bitemplate.leftPage = verticalZonestemplateClass()
            bitemplate.leftPage.setType('leftpage')
            bitemplate.leftPage.setPattern(pattern[0])
            bitemplate.leftPage.setParent(bitemplate)
            
            bitemplate.rightPage = verticalZonestemplateClass()
            bitemplate.rightPage.setType('rightpage')
            bitemplate.rightPage.setPattern(pattern[1])
            bitemplate.rightPage.setParent(bitemplate)
            
            #regular grid or not: tested in verticalZonestemplateClass
            
            return bitemplate
        
        elif len(pattern) == 1 and len(pattern[0]) in linterval:
            template = singlePageTemplateClass()
            template.mainZone = verticalZonestemplateClass()
            template.mainZone.setPattern(pattern[0])
            template.mainZone.setParent(template)
            return template
    
    def populateElementWithParsing(self,template,parsings,dMtoSingleFeature):
        """
            using the parsing result: store in each element its vertical template 
        """
        def treePopulation(node,lElts):
            """
                tree:
                    terminal node: tuple (tag,(tag,indexstart,indexend))
                
            """
            if type(node[0]).__name__  == 'tuple':
                    _,(tag,start,end) = node[0]
                    # tag need to be converted in mv -> featreu
                    subtemplate = template.findTemplatePartFromPattern(dMtoSingleFeature[tag])
                    lElts[start].addVerticalTemplate(subtemplate) 
                    ## correct here !!!
#                     print "xxx",lElts[start],tag,subtemplate, lElts[start].getVerticalTemplates()
            else:
                for subtree in node[1::2]:
                    treePopulation(subtree, lElts)
                    
        gram, lFullList, ltrees  = parsings
        ## each fragment of the parsing
        for lParsedPages, ktree,ltree in ltrees:
            treePopulation(ltree, lParsedPages)

    def tagPartialElements(self,lPages,dMtoSingleFeature):
        """
            decorate  objects with vertical separators
        """
        for page in lPages:
            # to be sure the features are the correct ones
            page.resetFeatures()
            page.setFeatureFunction(page.getSetOfMigratedFeatures,TH = 10.0,lFeatureList=page.lf_XCut)
#             print page, page.lf_XCut
            page.computeSetofFeatures()                
            for template in page.getVerticalTemplates():
                pattern=template.getPattern()
#                 lTemplatefeatures = dMtoSingleFeature[mvfeature]
#                 print '\t', lTemplatefeatures
#                 for Tempfeature in lTemplatefeatures:
                for Tempfeature in pattern:
                    if Tempfeature not in  page.lf_XCut: #getSetofFeatures().getSequences():
#                         print '\t\tadded:',Tempfeature
                        page.lf_XCut.append(Tempfeature)
#             print '\t',page, page.lf_XCut



    
    
    def tagDomWithBestTemplate(self,lPages):
        """
            Create (empty) REGIONS (for end-to-end; for GT: create table)
        """
        
        x = 0
        for page in lPages:
            if page.getNode():
                best = None
                bestRegisteredPoints =None
                lMissingBest= None
                bestScore = 0
                for mytemplate in page.getVerticalTemplates():
                    registeredPoints, lMissing, score= mytemplate.registration(page)
                    print page, mytemplate, score
                    if score > bestScore:
                        best = mytemplate
                        bestRegisteredPoints= registeredPoints
                        lMissingBest=lMissing 
                        bestScore = score
                print page,best, bestScore # bestRegisteredPoints, lMissingBest
                if best:
                    prevcut=0
                    for refcut,realcut in bestRegisteredPoints:
                        if realcut != prevcut:
                            region  = libxml2.newNode('REGION')
                            region.setProp("x",str(prevcut))
                            region.setProp("y",'0')
                            region.setProp("height",str(page.getHeight()))
                            region.setProp("width", str(realcut.getValue() - prevcut))                              
                            region.setProp('points', '%f,%f,%f,%f,%f,%f,%f,%f'%(prevcut,0, realcut.getValue(),0 ,realcut.getValue(),page.getHeight(),prevcut,page.getHeight()))
                            page.getNode().addChild(region)
                            prevcut = realcut.getValue()
                    #final col
                    if prevcut != page.getWidth():
                        region  = libxml2.newNode('REGION')
                        width = page.getWidth() - prevcut
                        region.setProp("x",str(prevcut))
                        region.setProp("y",'0')
                        region.setProp("height",str(page.getHeight()))
                        region.setProp("width", str(width))                              
                        region.setProp('points', '%f,%f,%f,%f,%f,%f,%f,%f'%(prevcut,0, page.getWidth(),0,page.getWidth(),page.getHeight(),prevcut,page.getHeight()))
                        page.getNode().addChild(region)
                        
                        
    def tagDOMVerticalZones(self,lPages):
        """
            decorate dom objects with vertical separators
            
            simply use lf_XCut
        """
        for page in lPages:
#             for x in set(page.lf_XCut):
#                 verticalSep  = libxml2.newNode('PAGEBORDER')
#                 verticalSep.setProp('points', '%f,%f,%f,%f'%(x.getValue(),0,x.getValue(),page.getHeight()))
#                 page.getNode().addChild(verticalSep)
            if True and page.getNode() is not None:
                page.resetFeatures()
                page.setFeatureFunction(page.getSetOfMigratedFeatures,TH=10.0,lFeatureList=page.lf_XCut)
                page.computeSetofFeatures()                
                for template in page.getVerticalTemplates():
#                     print page, mvfeature
                    lfeatures = template.getPattern()
                    for feature in lfeatures:
#                         print page, feature, page.getSetofFeatures().getSequences()
                        ## several fea can match feature: take the best one !(nearest?)
                        verticalSep  = libxml2.newNode('CENTERBORDER')
                        verticalSep.setProp('points', '%f,%f,%f,%f'%(feature.getOldValue(),0,feature.getOldValue(),page.getHeight()))
                         
                        page.getNode().addChild(verticalSep)
                        for fea in page.getSetofFeatures().getSequences():
                            if feature == fea:
                                verticalSep  = libxml2.newNode('PAGEBORDER')
                                verticalSep.setProp('points', '%f,%f,%f,%f'%(fea.getOldValue(),0,fea.getOldValue(),page.getHeight()))
                                lmyValues = fea.getOldValue()
                                page.getNode().addChild(verticalSep)
                                for myfeatureX in [lmyValues]:
                                    lX = []
                                    lY = []
    #                                 print '!!!',myfeatureX, myfeatureX[0].getValue()
                                    for elt in page.getAllNamedObjects(XMLDSTEXTClass):
    #                                     if abs(float(elt.getAttribute("x")) - myfeatureX[0].getValue()) < myfeatureX[0].getTH():
                                        if abs(float(elt.getAttribute("x")) - myfeatureX) < feature.getTH():
     
                                            lX.append(float(elt.getAttribute('x')))
                                            lY.append(float(elt.getAttribute('y')))
    #                                 print lX, lY
                                    if len(lX) < 3:
                                        b = myfeatureX
                                        ymax = myfeatureX
                                    else:
                                        a,b = np.polyfit(lY, lX, 1)
                                        y0=  b
                                        ymax = a*page.getHeight()+b
                                         
                                    verticalSep  = libxml2.newNode('PAGEBORDER')
                                    verticalSep.setProp('points', '%f,%f,%f,%f'%(b,0,ymax,page.getHeight()))
        #                             verticalSep.setProp('x',str(fea.getOldValue()))
        #                             verticalSep.setProp('y',str(y0))
        #                             verticalSep.setProp('width','2')
        #                             verticalSep.setProp('height',str(ymax))
#                                     print page.getNumber(),verticalSep
                                    page.getNode().addChild(verticalSep)
    #             return
            
            
    
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
                    tableNode = libxml2.newNode('TABLE')
                    tableNode.setProp('x','0')
                    tableNode.setProp('y','0')
                    tableNode.setProp('height',str(page.getHeight()))
                    tableNode.setProp('width',str(page.getWidth()))

                    page.getNode().addChild(tableNode) 
                    page.getdVSeparator(template).sort(key=lambda x:x.getValue())
#                     print page.getdVSeparator(template)
                    prevcut=0
                    for i,cut in enumerate(page.getdVSeparator(template)):
                        cellNode  = libxml2.newNode('CELL')
                        cellNode.setProp("x",str(prevcut))
                        cellNode.setProp("y",'0')
                        cellNode.setProp("irow","0")
                        cellNode.setProp("icol",str(i))
                        cellNode.setProp("height",str(page.getHeight()))
                        cellNode.setProp("width", str(cut.getValue() - prevcut))                            
#                         cellNode.setProp('points', '%f,%f,%f,%f,%f,%f,%f,%f'%(cut,0, cut.getValue(),0 ,cut.getValue(),page.getHeight(),cut,page.getHeight()))
                        tableNode.addChild(cellNode)
                        prevcut  = cut.getValue()      
            
            
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

            indicate if bigram page template (mirrored pages)
             
        """
        self.doc= doc
        # use the lite version
        self.ODoc = XMLDSDocument()
        
        self.ODoc.loadFromDom(self.doc,listPages=range(self.firstPage,self.lastPage+1))        
        self.lPages= self.ODoc.getPages()   
#         self.cleanInput(self.lPages)
        
        
        # first mine page size!!
        ## if width is not the 'same' , then  initial values are not comparable (x-end-ofpage)

        self.minePageVerticalFeature(self.lPages)

        if self.bManual:
            self.processWithTemplate(self.manualPattern,self.lPages)
        else:
            self.processVSegmentation(self.lPages)
        
        self.addTagProcessToMetadata(self.doc)
        
        return self.doc 

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
    docM.add_option("--pattern", dest="pattern", action="store", type="string", help="pattern to be applied", metavar="[]")
        
    #parse the command line
    dParams, args = docM.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    docM.setParams(dParams)
    
    doc = docM.loadDom()
    docM.run(doc)
    if doc and docM.getOutputFileName() != "-":
        docM.writeDom(doc, True)

