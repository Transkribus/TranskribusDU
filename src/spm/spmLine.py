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

from feature import featureObject, multiValueFeatureObject 

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
        self.THNUMERICAL= 10
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


    def mineLineFeature(self,lPages):
        """
            get set of line and mine
            
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
#             print page
            lElts= lLElts[i]
            for elt in lElts:
                elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['x','x2','xc','text'],myLevel=XMLDSTEXTClass)
                elt.computeSetofFeatures()
            seqGen = sequenceMiner()
            seqGen.setMaxSequenceLength(1)
            seqGen.setSDC(0.66) # related to noise level       AND STRUCTURES (if many columns)          
            lSortedFeatures  = seqGen.featureGeneration(lElts,2)
            
            # for registration: needs to be replaced by ._lRegValues
            print "sequence of elements and their features:"
            for elt in lElts:
                elt.lX=elt.getSetofFeatures()
                print elt, elt.lX
            
            icpt=0
            lTerminalTemplates = []

            lCurList= lElts
#             print len(lCurList)
            while icpt < 2:
    #             seqGen.bDebug = False
                ## generate sequences
                if icpt > 0: 
                    seqGen.setMaxSequenceLength(3)
#                     print '***'*20
                    seqGen.bDebug = False
                    for elt in lCurList:
                        if elt.getSetofFeatures() is None:
                            elt.resetFeatures()
                            elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['virtual'],myLevel=XMLDSTEXTClass)
                            elt.computeSetofFeatures()
                            elt.lX=elt.getSetofFeatures()
                        else:
                            elt.setSequenceOfFeatures(elt.lX)
                    lSortedFeatures  = seqGen.featureGeneration(lCurList,1)
                    
                lmaxSequence = seqGen.generateItemsets(lCurList)
                seqGen.bDebug = False
                
                lSeq, lMIS = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures + lTerminalTemplates,mis = 0.2)
                lPatterns = seqGen.beginMiningSequences(lSeq,lSortedFeatures,lMIS)            
                lPatterns.sort(key=lambda (x,y):y, reverse=True)
    
                print "List of patterns and their support:"
                for p,support  in lPatterns:
                    if support > 1: #and len(p) == 4:
                        print p, support
                
                seqGen.bDebug = False
                seqGen.THRULES = 0.95
                lSeqRules = seqGen.generateSequentialRules(lPatterns)
                _,dCP = self.getPatternGraph(lSeqRules)
                
                dTemplatesCnd = self.analyzeListOfPatterns(lPatterns,dCP,icpt)
                lTerminalTemplates,tranprob = self.testKleeneageTemplates(dTemplatesCnd, lCurList)
    
#                 print tranprob
#                 print lTerminalTemplates
                
                ## here we have a graph ()
                self.selectFinalTemplates(lTerminalTemplates,tranprob,lCurList)
                
                ## store parsed sequences in mytemplate 
                for templateType in dTemplatesCnd.keys():
                    for _,_, mytemplate in dTemplatesCnd[templateType][:1]:
                        mytemplate.print_()
                        _,lCurList = self.parseWithTemplate(mytemplate,lCurList,bReplace=True)                
                
                icpt +=1
#                 print 'curList:',lCurList
#                 print len(lCurList)
            print "final hierarchy"
            self.printTreeView(lCurList)
            lRegions= self.getRegionsFromStructure(page,lCurList)
                # store all interation
            lPageRegions.append((page,lRegions,lCurList))
                
        return lPageRegions
    
            
    def testKleeneageTemplates(self,dTemplatesCnd,lElts):
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
            for _,_, mytemplate in dTemplatesCnd[templateType][:1]:
#                 mytemplate.print_()
                ## need to test kleene+: if not discard?
                parseRes,lseq = self.parseWithTemplate(mytemplate,lElts,bReplace=False)
#                 print len(lseq)
                ## assess and keep if kleeneplus
                if parseRes:
                    ## traverse the tree template to list the termnimal pattern
#                     print mytemplate,'\t\t', mytemplate,len(parseRes[1])
                    lterm= mytemplate.getTerminalTemplates()
#                     print 'terms: ',lterm
                    for template in lterm:
#                         template.print_()
                        if template not in lTemplates:
                            lTemplates.append(template)
                            dScoredTemplates[template] = len(parseRes[1])
        
        #  transition matrix: look at  for template in page.getVerticalTemplates():
        N = len(lTemplates) + 1
        transProb = np.zeros((N,N), dtype = np.float16)
            
        dTemplateIndex= {}
        for i,template in enumerate(lTemplates):
            dTemplateIndex[template]=i
        
#         for t in dTemplateIndex:
#             print dTemplateIndex[t],t.getPattern()
            
            
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
    
    
    def analyzeListOfPatterns(self,lPatterns,dCA,step):
        """
            select patterns with no ancestor
            other criteria ? 
        """
        dTemplatesTypes = {}
        for pattern,support in filter(lambda (x,y):y>1,lPatterns):
            try:
                dCA[str(pattern)]
                bSkip = True
            except KeyError:bSkip=False
            if step > 0 and len(pattern) == 1 :
                bSkip=True
            if not bSkip:                        
                template  = treeTemplateClass()
                template.setPattern(pattern)
                template.buildTreeFromPattern(pattern)
                template.setType('lineTemplate')
                try:dTemplatesTypes[template.__class__.__name__].append((pattern, support, template))
                except KeyError: dTemplatesTypes[template.__class__.__name__] = [(pattern,support,template)]                      
        
        return dTemplatesTypes
        
        
    def patternToMV(self,pattern,dMtoSingleFeature,TH):
        """
            convert a list of f as multivaluefeature
            keep mvf as it is
            
            important to define the TH  value for multivaluefeature
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
                    dMtoSingleFeature[mv]=itemset
#                     print "$$$ ",mv, itemset
            # 
            else:
#                 print "\t->",list
                lmv.append(self.patternToMV(itemset, dMtoSingleFeature,TH))
#             print '\t',lmv
        return lmv
    
    
    
    
    def processWithTemplate(self,lPattern,lPages):
        """
            recursive pattern now!
        """
        
        def convertStringtoPattern(xcur):
#             print xcur
            lRes=[]
            for elt in xcur:
                if isinstance(elt,list):
                    lRes.extend([convertStringtoPattern(elt)])
                else:
                    try:
                        float(elt)
                        f= featureObject()
                        f.setName("f")
                        f.setType(featureObject.NUMERICAL)
                        f.setValue(elt)
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
            lElts= page.getAllNamedObjects(XMLDSTEXTClass)
            for elt in lElts:
                elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['x','x2','text'],myLevel=XMLDSTEXTClass)
                elt.computeSetofFeatures()
                elt.lX=elt.getSetofFeatures()                
            lParsing,lNewSeq = self.parseWithTemplate(mytemplate,lElts,bReplace=True)
            self.printTreeView(lNewSeq)
            # process lNewSeq: create the output data structure?
    
    def parseWithTemplate(self,mytemplate,lElts,bReplace=False):
        """
            parse a set of lElts with a template
            since mutivalued itemset: need multiValueFeatureObject
        """

        def print_(head, tail,level = 0):
#             print ' ' * level,level, type(head),type(tail)
            if not isinstance(head,str):
                # terminal element
#                 print ' ' * level,head, tail[0][1]
                return tail[0][1]
            else:
                lres=[]
                if head[-1] == '+':
                    for subhead, subtail in zip(tail[0::2],tail[1::2]):
                        lres.extend(print_(subhead,subtail,level + 1))
#                     print " "*level, level, 'kleene+', head, len(lres)
#                     if len(lres) == 1:
#                         lres=lres[0]                         
                else:
#                     print ' ' * level,head
                    # create a hierarchical level
                    lres = [  ]
                    for subhead, subtail in zip(tail[0::2],tail[1::2]):
                        lres.append(print_(subhead,subtail,level + 1))
                return lres
                
        
        PARTIALMATCH_TH = 1.0
        dMtoSingleFeature = {}
        mvPattern = self.patternToMV(mytemplate.getPattern(),dMtoSingleFeature, PARTIALMATCH_TH)
#         print mvPattern
#         print dMtoSingleFeature
        
        #need to optimize this double call        
        for elt in lElts:
            try:
                elt.setSequenceOfFeatures(elt.lX)
#                 print elt, elt.lX ,elt.getSetofFeatures()
                lf= elt.getSetofFeatures()[:]
#                 print elt, lf 
                elt.resetFeatures()
                elt.setFeatureFunction(elt.getSetOfMutliValuedFeatures,TH = PARTIALMATCH_TH, lFeatureList = lf )
#                 print elt, elt.getSetofFeatures()
                elt.computeSetofFeatures()            
            except AttributeError:
                # lower recursive level; objetcclass: dont touch features
                pass
        seqGen = sequenceMiner()
        # what is missing is the correspondance between rule name (sX) and template element
        ## provide a template to seqGen ? instead if recursive list (mvpatern)
        
#         for e in lElts:
#             print "wwww",e, e.getSetofFeatures()
        lNewSequence = lElts[:]
        parsingRes = seqGen.parseSequence(mvPattern,multiValueFeatureObject,lElts)
        if parsingRes:
            self.populateElementWithParsing(mytemplate,parsingRes,dMtoSingleFeature)
            _, _, ltrees  = parsingRes
            for lParsedElts, _,ltree in ltrees:
                # s0 or s0+ as key, so alwas [ head, tail]
                lkleeneVersion = print_(ltree[0],ltree[1])
                # replace kleene+ elt by an object ?
                if bReplace:
                    lNewSequence = self.replaceEltsByParsing(lNewSequence,lkleeneVersion,lParsedElts,mytemplate.getPattern())
#                 print len(lNewSequence)
            
        return parsingRes, lNewSequence

    def populateElementWithParsing(self,mytemplate,parsings,dMtoSingleFeature):
        """
            using the parsing result: store in each element its vertical template 
        """
        def treePopulation(node,lElts):
            """
                tree:
                    terminal node: tuple (tag,(tag,indexstart,indexend))
                
            """
            if isinstance(node[0],tuple):
                    _,(tag,start,_) = node[0]
                    # tag need to be converted in mv -> feature
                    subtemplate = mytemplate.findTemplatePartFromPattern(dMtoSingleFeature[tag])
#                     print subtemplate, dMtoSingleFeature[tag]
#                     print  lElts[start], tag, subtemplate, mytemplate, dMtoSingleFeature[tag]
                    lElts[start].addTemplate(subtemplate)
            else:
                for subtree in node[1::2]: 
                    treePopulation(subtree, lElts)
                    
        _, _, ltrees  = parsings
        ## each fragment of the parsing
        for lParsedPages, _,ltree in ltrees:
            treePopulation(ltree, lParsedPages)    


    def replaceEltsByParsing(self,lElts, lParsing, lParsedElts,pattern):
        """
            replace the parsed sequence in lElts
        """
        # index in lElts
        i = 0
        while i < len(lElts):
            if lElts[i] ==lParsedElts[0]:
                del lElts[i:i+len(lParsedElts)]
                # create a new object. need to have similar featureGeneration properties
                newObject = XMLDSTEXTClass() #objectClass()
                newObject.addAttribute("virtual",pattern)
#                 newObject.setName(str(pattern))
                newObject.setObjectsList(lParsedElts)
#                 newObject = lParsedElts[0]
                lElts.insert(i,newObject)
                i=len(lElts)
            i+=1
#         print lElts
        return  lElts
        
        
        
    def printTreeView(self,lElts,level=0):
        """
            recursive 
        """
        for elt in lElts:
            if elt.getAttribute('virtual'):
                print "  "*level, 'Node'
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
            
            lRes = self.mineLineFeature(self.lPages)
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

