# -*- coding: utf-8 -*-
"""

     H. Déjean

    copyright Xerox 2017
    READ project 

    mine a set of lines (itemset generation)
    
"""

import sys, os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import libxml2

import numpy as np

import warnings
warnings.filterwarnings('ignore')

import common.Component as Component
 
from structuralMining import sequenceMiner

from feature import featureObject 

from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.XMLDSTEXTClass  import XMLDSTEXTClass
from ObjectModel.treeTemplateClass import treeTemplateClass

    
class lineMiner(Component.Component):
    """
        pageVerticalMiner class: a component to mine column-like page layout
    """
    
    
    #DEFINE the version, usage and description of this particular component
    usage = "" 
    version = "v.01"
    description = "description: line miner "

    
    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "lineMiner", self.usage, self.version, self.description) 
        
        
        # TH for comparing numerical features for X
        self.THNUMERICAL= 20
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



    def mainLineMining(self,lPages):
        """
            mine with incremental length
            
        """
        import util.TwoDNeighbourhood as TwoDRel
        
        
        lVEdge = []
        lLElts=[ [] for i in range(0,len(lPages))]
        for i,page in enumerate(lPages):
            lElts= page.getAllNamedObjects(XMLDSTEXTClass)
            for e in lElts:
                e.lnext=[]
            ## filter elements!!!
            lElts = filter(lambda x:min(x.getHeight(),x.getWidth()) > 10,lElts)
            lElts = filter(lambda x:x.getHeight() > 10,lElts)
            lElts = filter(lambda x:x.getX() < 500,lElts)

            lElts.sort(key=lambda x:x.getY())
            lLElts[i]=lElts
            lVEdge = TwoDRel.findVerticalNeighborEdges(lElts)
            for  a,b in lVEdge:
                a.lnext.append( b )      
                  
        for i,page, in enumerate(lPages):
            lElts= lLElts[i]
            for elt in lElts:
                ### need of more abstract features: justified, center, left, right  + numerical 
                elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['x','xx2','xxc','text'],myLevel=XMLDSTEXTClass)
                elt.computeSetofFeatures()
#                 print elt.getSetofFeatures()
            seqGen = sequenceMiner()
            seqGen.setMaxSequenceLength(1)
            seqGen.setSDC(0.7) # related to noise level       AND STRUCTURES (if many columns)          
            _  = seqGen.featureGeneration(lElts,2)
            seqGen.setObjectLevel(XMLDSTEXTClass)

            # for registration: needs to be replaced by ._lRegValues
            print "sequence of elements and their features:"

            for elt in lElts:
                elt.lFeatureForParsing=elt.getSetofFeatures()
                print elt, elt.lFeatureForParsing
            
            lTerminalTemplates=[]
            lCurList=lElts[:]
            for iLen in range(1,3):
                lCurList,lTerminalTemplates = self.mineLineFeature(seqGen,lCurList,lTerminalTemplates,iLen)                  
            
            del seqGen

    def mineLineFeature(self,seqGen,lCurList,lTerminalTemplates,iLen):
        """
            get a set of lines and mine them
        """ 
        seqGen.setMinSequenceLength(iLen)
        seqGen.setMaxSequenceLength(iLen)
        
        print '***'*20, iLen
        seqGen.bDebug = False
        for elt in lCurList:
            if elt.getSetofFeatures() is None:
                elt.resetFeatures()
                elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['virtual'],myLevel=XMLDSTEXTClass)
                elt.computeSetofFeatures()
                elt.lFeatureForParsing=elt.getSetofFeatures()
            else:
                elt.setSequenceOfFeatures(elt.lFeatureForParsing)
#                 print elt, elt.getSetofFeatures()
        lSortedFeatures  = seqGen.featureGeneration(lCurList,2)
        for cf in lSortedFeatures:
            cf.setWeight(len((cf.getNodes())))
        for x in lCurList:
            print x, x.getCanonicalFeatures()
        lmaxSequence = seqGen.generateItemsets(lCurList)
        seqGen.bDebug = False
        
        lSeq, lMIS = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures + lTerminalTemplates,mis = 0.01)
        lPatterns = seqGen.beginMiningSequences(lSeq,lSortedFeatures,lMIS)
        if lPatterns is None:
            return lCurList, lTerminalTemplates
                    
        lPatterns.sort(key=lambda (x,y):y, reverse=True)
        
        print "List of patterns and their support:"
        for p,support  in lPatterns:
            if support >= 1: 
                print p, support
        seqGen.THRULES = 0.95
        lSeqRules = seqGen.generateSequentialRules(lPatterns)
        
        " here store features which are redundant and consider only the core feature"
        _,dCP = self.getPatternGraph(lSeqRules)
        
        dTemplatesCnd = self.analyzeListOfPatterns(lPatterns,dCP)
        lFullTemplates,lTerminalTemplates,tranprob = seqGen.testTreeKleeneageTemplates(dTemplatesCnd, lCurList)
        
        ## here we have a graph; second : is it useful here to correct noise??
        ## allows for selecting templates ?
        self.selectFinalTemplates(lTerminalTemplates,tranprob,lCurList)
        
        ## store parsed sequences in mytemplate
        ### patterns are competing: generate a set of parsing ??? 
        for mytemplate in lFullTemplates[:1]: # dTemplatesCnd.keys():
#                     for _,_, mytemplate in dTemplatesCnd[templateType][:1]:
                mytemplate.print_()
#                         print '___'*30
#                         print lCurList
                isKleenePlus,_,lCurList = seqGen.parseWithTreeTemplate(mytemplate,lCurList,bReplace=True)
                for elt in lCurList:
                    if elt.getSetofFeatures() is None:
                        elt.resetFeatures()
                        elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['virtual'],myLevel=XMLDSTEXTClass)
                        elt.computeSetofFeatures()
                        elt.lFeatureForParsing=elt.getSetofFeatures()
                    else:
                        elt.setSequenceOfFeatures(elt.lFeatureForParsing)                        
#                 self.printTreeView(lCurList)        
                            
#                 print 'curList:',lCurList
#                 print len(lCurList)
        print "final hierarchy"
        self.printTreeView(lCurList)
#             lRegions= self.getRegionsFromStructure(page,lCurList)
                # store all interation
#             lPageRegions.append((page,lRegions,lCurList))
                
        return lCurList, lTerminalTemplates
    def mineLineFeatureBasic(self,lPages):
        """
            get a set of lines and mine them
        """ 
        
        import util.TwoDNeighbourhood as TwoDRel
        
        lPageRegions = []
        
        lVEdge = []
        lLElts=[ [] for i in range(0,len(lPages))]
        for i,page in enumerate(lPages):
            lElts= page.getAllNamedObjects(XMLDSTEXTClass)
            for e in lElts:
                e.lnext=[]
            ## filter elements!!!
            lElts = filter(lambda x:min(x.getHeight(),x.getWidth()) > 10,lElts)
            lElts = filter(lambda x:x.getHeight() > 10,lElts)
            lElts = filter(lambda x:x.getX() > 100,lElts)

            lElts.sort(key=lambda x:x.getY())
            lLElts[i]=lElts
            lVEdge = TwoDRel.findVerticalNeighborEdges(lElts)
            for  a,b in lVEdge:
                a.lnext.append( b )

        for i,page, in enumerate(lPages):
            lElts= lLElts[i]
            for elt in lElts:
                elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['x','x2','xc','text'],myLevel=XMLDSTEXTClass)
                elt.computeSetofFeatures()
#                 print elt.getSetofFeatures()
            seqGen = sequenceMiner()
            seqGen.setMaxSequenceLength(1)
            seqGen.setSDC(0.5) # related to noise level       AND STRUCTURES (if many columns)          
            lSortedFeatures  = seqGen.featureGeneration(lElts,2)
            seqGen.setObjectLevel(XMLDSTEXTClass)

            # for registration: needs to be replaced by ._lRegValues
            print "sequence of elements and their features:"

            for elt in lElts:
                elt.lFeatureForParsing=elt.getSetofFeatures()
                print elt, elt.lFeatureForParsing
            icpt=0
            lTerminalTemplates = []
            
            lCurList= lElts
#             print len(lCurList)
            while icpt < 2:
    #             seqGen.bDebug = False
                ## generate sequences
                if icpt > 0:
                    seqGen.setMinSequenceLength(2)
                    seqGen.setMaxSequenceLength(2)
                    
                    print '***'*20, icpt
                    seqGen.bDebug = False
                    for elt in lCurList:
                        if elt.getSetofFeatures() is None:
                            elt.resetFeatures()
                            elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['virtual'],myLevel=XMLDSTEXTClass)
                            elt.computeSetofFeatures()
                            elt.lFeatureForParsing=elt.getSetofFeatures()
                        else:
                            elt.setSequenceOfFeatures(elt.lFeatureForParsing)
                            print elt, elt.getSetofFeatures()
                    lSortedFeatures  = seqGen.featureGeneration(lCurList,2)
                    
                lmaxSequence = seqGen.generateItemsets(lCurList)
                seqGen.bDebug = False
                
                lSeq, lMIS = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures + lTerminalTemplates,mis = 0.2)
                lPatterns = seqGen.beginMiningSequences(lSeq,lSortedFeatures,lMIS)            
                lPatterns.sort(key=lambda (x,y):y, reverse=True)
                
                print "List of patterns and their support:"
                for p,support  in lPatterns:
                    if support >= 1: 
                        print p, support
                seqGen.THRULES = 0.95
                lSeqRules = seqGen.generateSequentialRules(lPatterns)
                
                " here store features which are redundant and consider only the core feature"
                _,dCP = self.getPatternGraph(lSeqRules)
                
                dTemplatesCnd = self.analyzeListOfPatterns(lPatterns,dCP,icpt)
                lFullTemplates,lTerminalTemplates,tranprob = seqGen.testTreeKleeneageTemplates(dTemplatesCnd, lCurList)
#                 lFullTemplates = seqGen.testTreeKleeneageTemplates(dTemplatesCnd, lCurList)
#                 print tranprob
#                 print lTerminalTemplates
                
                ## here we have a graph; second : is it useful here to correct noise??
                ## allows for selecting templates ?
                self.selectFinalTemplates(lTerminalTemplates,tranprob,lCurList)
                
                ## store parsed sequences in mytemplate
                ### patterns are competing: generate a set of parsing ??? 
                for mytemplate in lFullTemplates[:1]: # dTemplatesCnd.keys():
#                     for _,_, mytemplate in dTemplatesCnd[templateType][:1]:
                        mytemplate.print_()
#                         print '___'*30
#                         print lCurList
                        isKleenePlus,_,lCurList = seqGen.parseWithTreeTemplate(mytemplate,lCurList,bReplace=True)
                        for elt in lCurList:
                            if elt.getSetofFeatures() is None:
                                elt.resetFeatures()
                                elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['virtual'],myLevel=XMLDSTEXTClass)
                                elt.computeSetofFeatures()
                                elt.lFeatureForParsing=elt.getSetofFeatures()
                            else:
                                elt.setSequenceOfFeatures(elt.lFeatureForParsing)                        
#                 self.printTreeView(lCurList)        
                                
                icpt +=1
#                 print 'curList:',lCurList
#                 print len(lCurList)
            print "final hierarchy"
            self.printTreeView(lCurList)
#             lRegions= self.getRegionsFromStructure(page,lCurList)
                # store all interation
#             lPageRegions.append((page,lRegions,lCurList))
                
        return lPageRegions
    
            
    def selectFinalTemplates(self,lTemplates,transProb,lElts):
        """
            apply viterbi to select best sequence of templates
        """
        import spm.viterbi as viterbi        

        if  lTemplates == []:
            return None
        
        def buildObs(lTemplates,lElts):
            """
                build observation prob
            """
            N = len(lTemplates) + 1
            obs = np.zeros((N,len(lElts)), dtype=np.float16) +10e-3
            for i,temp in enumerate(lTemplates):
                for j,elt in enumerate(lElts):
                    # how to dela with virtual nodes
                    try:
                        _, _, score= temp.registration(elt)
                    except : score =1
                    if score == -1:
                        score= 0.0
                    obs[i,j]= score
                    if np.isinf(obs[i,j]):
                        obs[i,j] = 64000
                    if np.isnan(obs[i,j]):
                        obs[i,j] = 0.0                        
#                     print i,j,elt,elt.lX, temp,score 
                #add no-template:-1
            return obs / np.amax(obs)

        
        N= len(lTemplates) + 1.0
        
        initialProb = np.ones(N) 
        initialProb = np.reshape(initialProb,(N,1))
        obs = buildObs(lTemplates,lElts)
        
        np.set_printoptions(precision= 3, linewidth =1000)
#         print "transProb"
#         print transProb
#         print 
#         print obs
        
        d = viterbi.Decoder(initialProb, transProb, obs)
        states,score =  d.Decode(np.arange(len(lElts)))

        # add empty template (last one in state)
        lTemplates.append(None)
        print states, score

        #assign to each elt the template assigned by viterbi
        for i,elt, in enumerate(lElts):
#             try: print elt,elt.lX,  lTemplates[states[i]]
#             except: print elt, elt.lX, 'no template'
            mytemplate= lTemplates[states[i]]
            elt.resetTemplate()
            if mytemplate is not None:
                elt.addTemplate(mytemplate)
                try:
                    registeredPoints, lMissing, score= mytemplate.registration(elt)
                except:
                    registeredPoints = None
                if registeredPoints:
#                     print registeredPoints, lMissing , score
                    if lMissing != []:
                        registeredPoints.extend(zip(lMissing,lMissing))
                    registeredPoints.sort(key=lambda (x,y):y.getValue())
                    lcuts = map(lambda (ref,cut):cut,registeredPoints)
                    ## store features for the final parsing!!!
#                     print elt, lcuts
#                     elt.addVSeparator(mytemplate,lcuts)

        # return the new list with kleenePlus elts for next iteration
        ## reparse ?? YES using the featureSet given by viterbi  -> create an objectClass per kleenePlus element: objects: sub tree
#         print lTemplates[0]
#         self.parseWithTemplate(lTemplates[0], lElts)
        # elt = template
        
        return score     
    
    
    def createItemSetFromNext(self,lElts,iLen):
        """
            create itemset of length iLen using .lnext structures 
        """
    
    def getKleenePlusFeatures(self,lElts):
        """
            select KleenePlus elements based on .next (only possible for unigrams)
        """   
        dFreqFeatures={}
        dKleenePlusFeatures = {}
        
        lKleenePlus=[]
        for elt in lElts:
            for fea in elt.getSetofFeatures():
                try:dFreqFeatures[fea] +=1
                except KeyError:dFreqFeatures[fea] = 1
                for nextE in elt.next:
                    if fea in nextE.getSetofFeatures():
                        try:dKleenePlusFeatures[fea].append((elt,nextE))
                        except KeyError:dKleenePlusFeatures[fea]=[(elt,nextE)]
        for fea in dFreqFeatures:
            try:
                dKleenePlusFeatures[fea]
                if len(dKleenePlusFeatures[fea]) >= 0.5 *  dFreqFeatures[fea]:
                    lKleenePlus.append(fea) 
            except:
                pass
        return  lKleenePlus
    
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
    
    def analyzeListOfPatterns(self,lPatterns,dCA,):
        """
            select patterns with no ancestor
            other criteria ?
            
            
            if many with similar frequency:  sort using  computePatternScore?
        """
        # reorder lPatterns considering feature weights and # of elements (for equally frequent patterns)
#         lPatterns.sort(key=lambda (x,y):self.computePatternScore(x),reverse=True)
#         for x,y in lPatterns:
#             print x,y,self.computePatternScore(x)
             
        dTemplatesTypes = {}
        for pattern,support in filter(lambda (x,y):y>1,lPatterns):
            try:
                dCA[str(pattern)]
                bSkip = True
            except KeyError:bSkip=False
#             if step > 0 and len(pattern) == 1:
#                 bSkip=True
            if not bSkip:   
                template  = treeTemplateClass()
                template.setPattern(pattern)
                template.buildTreeFromPattern(pattern)
                template.setType('lineTemplate')
                try:dTemplatesTypes[template.__class__.__name__].append((pattern, support, template))
                except KeyError: dTemplatesTypes[template.__class__.__name__] = [(pattern,support,template)]                      
        
        return dTemplatesTypes
        
        
    
    def processWithTemplate(self,lPattern,lPages):
        """
            apply a known pattern 
        """
        
        def convertStringtoPattern(xcur):
            ## need to integrate the 'virtual level'
            lRes=[]
            for elt in xcur:
                if isinstance(elt,list):
                    lRes.extend([convertStringtoPattern(elt)])
                else:
                    try:
                        float(elt)
                        f= featureObject()
                        f.setName("x")
                        f.setType(featureObject.NUMERICAL)
                        f.setValue(elt)
                        f.setObjectName(elt)
                        f.setWeight(1)
                        f.setTH(self.THNUMERICAL)                
                    except:
                        f= featureObject() 
                        f.setName("f")
                        f.setType(featureObject.EDITDISTANCE)
                        f.setValue(elt)
                        f.setTH(100.0)                
                    lRes.append(f)                    
            
            return lRes
    
        lfPattern =convertStringtoPattern(lPattern)
#         print lfPattern
        # create a template from lfPattern!
        mytemplate = treeTemplateClass()
#         mytemplate.setPattern(lfPattern)
        mytemplate.buildTreeFromPattern(lfPattern)
        mytemplate.print_()
        for page in lPages:
            seqGen = sequenceMiner()
            seqGen.setObjectLevel(XMLDSTEXTClass)

            lElts= page.getAllNamedObjects(XMLDSTEXTClass)
            for elt in lElts:
                elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['x','x2','text'],myLevel=XMLDSTEXTClass)
                elt.computeSetofFeatures()
                elt.lFeatureForParsing=elt.getSetofFeatures()    
            
            lParsing,lNewSeq = seqGen.parseWithTreeTemplate(mytemplate,lElts,bReplace=True)
            
            del  seqGen
            self.printTreeView(lNewSeq)
            # process lNewSeq: create the output data structure?
    
    def printTreeView(self,lElts,level=0):
        """
            move to structuralMining? 
        """
        for elt in lElts:
            if elt.getAttribute('virtual'):
                print "  "*level, 'Node', elt.getAttribute('virtual')
                self.printTreeView(elt.getObjects(),level+1)
            else:
                print "  "*level, elt, elt.getContent().encode('utf-8')
    
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

#         # for bigram: extend to grammy
#         for child in dChildParent.keys():
#             ltmp=[]
#             if len(eval(child)) == 2:
#                 for parent in dChildParent[child]:
#                     try:
#                         ltmp.extend(dChildParent[str(parent)])
#                     except KeyError:pass
#             dChildParent[child].extend(ltmp)
        return dParentChild,  dChildParent
        
        
    
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
        
    def getRegionsFromStructure(self,page,lTree):
        """
            tag the dom with what?
        """
        
        lZone=[]
        srx, sry  = 9e9, 9e9
        srx2,sry2 = 0 ,  0
        lSubZone=[]
        for elt in lTree:
            if elt.getAttribute('virtual'):
                rx,ry,rx2,ry2,lsub = self.getRegionsFromStructure(page,elt.getObjects())
                srx = min(srx,rx)
                sry = min(sry,ry)
                srx2= max(srx2,rx2)
                sry2= max(sry2,ry2)
                lSubZone.append([rx,ry,rx2,ry2,lsub])
            else:
                lZone.append(elt)
        # get BB of the zone
        fMinX , fMinY = srx  , sry
        fMaxX , fMaxY = srx2 , sry2
        for e in lZone:
            if e.getX2() > fMaxX:
                fMaxX= e.getX2()
            if e.getY2() > fMaxY:
                fMaxY= e.getY2()
            if e.getX() < fMinX:
                fMinX= e.getX()
            if e.getY() < fMinY:
                fMinY= e.getY()
        # has substructure
        if srx != 9e9:
            return [ fMinX,fMinY,fMaxX,fMaxY , lSubZone]
        else:
            return [ fMinX,fMinY,fMaxX,fMaxY, []]

    def tagDom(self,page,region):
        """
            tag page with region (x,y,x2,y2)
        """
        fMinX , fMinY,  fMaxX , fMaxY, ltail = region
        # new region node
        regionNode = libxml2.newNode('REGION')
        page.getNode().addChild(regionNode)
        regionNode.setProp('x',str(fMinX))
        regionNode.setProp('y',str(fMinY))
        regionNode.setProp('height',str(fMaxY-fMinY))
        regionNode.setProp('width',str(fMaxX - fMinX))
        print 
        print region
        print regionNode
        [self.tagDom(page,tail) for tail in ltail]
        
        return regionNode        
            
    #--- RUN ---------------------------------------------------------------------------------------------------------------    
    def run(self, doc):
        """
            take a set of line in a page and mine it
        """
        self.doc= doc
        # use the lite version
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(self.doc,listPages=range(self.firstPage,self.lastPage+1))        

        self.lPages= self.ODoc.getPages()   
        
        if self.bManual:
            self.processWithTemplate(self.manualPattern,self.lPages)
        else:
            
            self.mainLineMining(self.lPages)
#             lRes = self.mineLineFeature(self.lPages)
#             print lRes
            # returns the hierarchical set of elements (a list)
#             for page , region, tree in lRes:
#                 self.tagDom(page, region)
#                 return 
        
        self.addTagProcessToMetadata(self.doc)
        
        return self.doc 

    #--- TESTS -------------------------------------------------------------------------------------------------------------    
    #
    # Here we have the code used to test this component on a prepared testset (see under <ROOT>/test/common)
    # Do: python ../../src/common/TypicalComponent.py --test REF_TypicalComponent/
    #
    
    
 
        
        
#--- MAIN -------------------------------------------------------------------------------------------------------------    
#
# In case we want to use this component from a command line
#
# Do: python TypicalComponent.py -i toto.in.xml
#
if __name__ == "__main__":
    
    
    docM = lineMiner()

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

