# -*- coding: utf-8 -*-
"""

     H. Déjean

    component to mine table to find out  horizontal cuts (hence rows)
    copyright Xerox 2017
    READ project 

"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import sys, os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

from lxml import etree
import numpy as np


import common.Component as Component
 
from .structuralMining import sequenceMiner

from .feature import featureObject 

from ObjectModel.xmlDSDocumentClass import XMLDSDocument
from ObjectModel.treeTemplateClass import treeTemplateClass
from ObjectModel.XMLDSCELLClass import XMLDSTABLECELLClass
from ObjectModel.XMLDSTABLEClass import XMLDSTABLEClass
    
class tableRowMiner(Component.Component):
    """
        tableRowMiner class: a component to mine table to find out  horizontal cuts (hence rows)
    """
    
    
    #DEFINE the version, usage and description of this particular component
    usage = "" 
    version = "v.01"
    description = "description: table row miner "

    
    #--- INIT -------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        """
        Always call first the Component constructor.
        """
        Component.Component.__init__(self, "tableRowMiner", self.usage, self.version, self.description) 
        
        
        # TH for comparing numerical features
        ## need to be fucntion of leading: when leading small THNUMERICAL small, lineHeighr reduced a well?
        self.THNUMERICAL = 25
        # use for evaluation
        self.THCOMP = 10
        self.evalData= None
        
        self.THHighSupport = 0.33
        
        self.bManual = False
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if "pattern" in dParams:
            self.manualPattern = eval( dParams["pattern"])
            self.bManual=True  
        if "thhighsupport" in dParams:
            self.THHighSupport = dParams["thhighsupport"] * 0.01




    def testHighSupport(self,sequences,th):
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
#         lOneSupport= [k for k,v in actual_supports.iteritems() if v >= 0.5 ]
        lOneSupport= [k for k,v in actual_supports.items() if v >= th ]
#         print(actual_supports.items() )
#         print (th,lOneSupport)
        return lOneSupport
    
    
    def createFeatureFromValue(self,elt,value,name):
        feature = featureObject()
        feature.setName(name)
        feature.setTH(self.THNUMERICAL)
        feature.addNode(elt)
        feature.setObjectName(elt)
        feature.setValue(float(value))
        feature.setType(featureObject.NUMERICAL)
        return feature
    
    def columnMining(self,table,thnum, th,predefinedCuts=[]):
        """
            for a table: take itemset=colmun, item=cell(Y) + separator
            - test: is a same rowgrid for all pages: row of fixed positions, size

        separator
        
        
        PREPRO: get the Y of overlaping separators for each col
            
        """
        if thnum != None:
            self.THNUMERICAL = thnum
        
        lElts=  table.getColumns() #getAllNamedObjects(XMLDSTABLECOLUMNClass)
        for elt in lElts:
            # how to add separator?
#             for c in elt.getCells():c.setHeight()
            elt.resetFeatures()
            elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['y'],myLevel=XMLDSTABLECELLClass)

            ## add predefinedCuts here
            elt.computeSetofFeatures()
            for prevFea in predefinedCuts:
                f = self.createFeatureFromValue(elt,round(prevFea), 'y')
                elt.addFeature(f)

        seqGen = sequenceMiner()
        seqGen.bDebug = False
        seqGen.setMaxSequenceLength(1)
#         seqGen.setSDC(0.7) # related to noise level       AND STRUCTURES (if many columns)          
#         _  = seqGen.featureGeneration(lElts,2) # more at token level, but at text level: freq=2
        seqGen.setObjectLevel(XMLDSTABLECELLClass)
        for elt in lElts:
            elt.lFeatureForParsing=elt.getSetofFeatures()
            elt.lFeatureForParsing.sort(key = lambda x:x.getValue())
#             print( elt, elt.lFeatureForParsing)
#             
        lSortedFeatures = seqGen.featureGeneration(lElts,2)
#         for f in lSortedFeatures:
#             print ("%s\s%s"%(f, f.getNodes()))


        lmaxSequence = seqGen.generateItemsets(lElts)
#         for elt in lElts:
#             print elt, elt.getCanonicalFeatures()
        lSeq, _ = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures,mis = 0.5)
        lOneSupport =self.testHighSupport(lSeq,th)
        lOneSupport.sort(key = lambda x:x.getValue())
        return lOneSupport
        
#         lTerminalTemplates=[]
#         lCurList=lElts[:]
#         lCurList,lTerminalTemplates = self.mineLineFeature(seqGen,lCurList,lTerminalTemplates)                  
#         print lTerminalTemplates
#         return


    def mainMining(self,lPages):
        """
            mine with incremental length
            
        """
        import util.TwoDNeighbourhood as TwoDRel
        
        
        lLElts=[ [] for i in range(0,len(lPages))]
        for i,page in enumerate(lPages):
            lElts= page.getAllNamedObjects(XMLDSTABLECELLClass) #+page.getAllNamedObjects(XMLDSGRAPHLINEClass)
            for e in lElts:
                e.lnext=[]
            ## filter elements!!!
            lElts = filter(lambda x:min(x.getHeight(),x.getWidth()) > 10,lElts)
            lElts = filter(lambda x:x.getHeight() > 10,lElts)

            lElts.sort(key=lambda x:x.getY())
            lLElts[i]=lElts
            for elt in lElts: TwoDRel.rotateMinus90deg(elt)              #rotate by 90 degrees and look for vertical neighbors :-)
            lHEdge = TwoDRel.findVerticalNeighborEdges(lElts)
            for elt in lElts: TwoDRel.rotatePlus90deg(elt)              
#             lVEdge = TwoDRel.findVerticalNeighborEdges(lElts)
            for  a,b in lHEdge:
                a.lnext.append( b )      

        for i,page, in enumerate(lPages):
            lElts= lLElts[i]
            for elt in lElts:
                elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['y'],myLevel=XMLDSTABLECELLClass)
                elt.computeSetofFeatures()
#                 print elt.getSetofFeatures()
            seqGen = sequenceMiner()
            seqGen.setMaxSequenceLength(1)
            seqGen.setSDC(0.7) # related to noise level       AND STRUCTURES (if many columns)          
            _  = seqGen.featureGeneration(lElts,2) # more at token level, but at text level: freq=2
            seqGen.setObjectLevel(XMLDSTABLECELLClass)
            
#             lKleendPlus = self.getKleenePlusFeatures(lElts)
#             print lKleendPlus
            
            for elt in lElts:
                elt.lFeatureForParsing=elt.getSetofFeatures()
#                 print elt, elt.lFeatureForParsing
#             
            lTerminalTemplates=[]
            lCurList=lElts[:]
            lCurList,lTerminalTemplates = self.mineLineFeature(seqGen,lCurList,lTerminalTemplates)                  
#             print lTerminalTemplates
            for mytemplate in lTerminalTemplates:
                page.addVerticalTemplate(mytemplate)
                page.addVSeparator(mytemplate,mytemplate.getPattern())            
            del seqGen
#         self.tagAsRegion(lPages)

    def mineLineFeature(self,seqGen,lCurList,lTerminalTemplates):
        """
            get a set of lines and mine them
        """ 
        seqGen.setMinSequenceLength(1)
        seqGen.setMaxSequenceLength(1)
        
        print ('***'*20)
        
        seqGen.bDebug = False
        for elt in lCurList:
            if elt.getSetofFeatures() is None:
                elt.resetFeatures()
                elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['virtual'],myLevel=XMLDSTABLECELLClass)
                elt.computeSetofFeatures()
                elt.lFeatureForParsing=elt.getSetofFeatures()
            else:
                elt.setSequenceOfFeatures(elt.lFeatureForParsing)
#                 print elt, elt.getSetofFeatures()
        lSortedFeatures  = seqGen.featureGeneration(lCurList,2)
        for cf in lSortedFeatures:
            cf.setWeight(len((cf.getNodes())))
            
        lmaxSequence = seqGen.generateItemsets(lCurList)
        seqGen.bDebug = False
        
        lSeq, _ = seqGen.generateMSPSData(lmaxSequence,lSortedFeatures + lTerminalTemplates,mis = 0.01)
        lPatterns = seqGen.miningSequencePrefixScan(lSeq)
#         lPatterns = seqGen.beginMiningSequences(lSeq,lSortedFeatures,lMIS)
        if lPatterns is None:
            return lCurList, lTerminalTemplates
                    
        lPatterns.sort(key=lambda xy:xy[0], reverse=True)
        
        print( "List of patterns and their support:")
        for p,support  in lPatterns:
            if support >= 1: 
                print( p, support)
        seqGen.THRULES = 0.95
        lSeqRules = seqGen.generateSequentialRules(lPatterns)
        
        " here store features which are redundant and consider only the core feature"
        
        dTemplatesCnd = self.analyzeListOfPatterns(lPatterns,{})
        lFullTemplates,lTerminalTemplates,tranprob = seqGen.testTreeKleeneageTemplates(dTemplatesCnd, lCurList,iterMax=40)
        

        return lCurList, lTerminalTemplates
    
            
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

        
        N= len(lTemplates) + 1
        
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
        print(states, score)

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
                    registeredPoints.sort(key=lambda xy:xy[0].getValue())
                    lcuts = map(lambda refcut:refcut[1],registeredPoints)
                    ## store features for the final parsing!!!
#                     print elt, lcuts
#                     elt.addVSeparator(mytemplate,lcuts)

        # return the new list with kleenePlus elts for next iteration
        ## reparse ?? YES using the featureSet given by viterbi  -> create an objectClass per kleenePlus element: objects: sub tree
#         print lTemplates[0]
#         self.parseWithTemplate(lTemplates[0], lElts)
        # elt = template
        
        return score     
    
    
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
                for nextE in elt.lnext:
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
        for pattern,support in filter(lambda xy:xy[1]>1,lPatterns):
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
            process sequence of pqges with given pattern
            create table 
        """
        
        
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
        
        print (pattern)
        
        ### in prodf: mytemplate given by page.getVerticalTemplates()
        mytemplate = treeTemplateClass()
        mytemplate.setPattern(pattern[0])

        
        # registration provides best matching
        ## from registration matched: select the final cuts
        for i,p in enumerate(lPages):
            p.lFeatureForParsing  = p.lf_XCut 
            sys.stdout.flush()
            registeredPoints1, lMissing1, score1 = mytemplate.registration(p)
            if score1 >= 0:
                lfinalCuts= map(lambda xy:xy[1],filter(lambda xy: xy[0]!= 'EMPTY',registeredPoints1))
                print( p,'final1:',lfinalCuts, lMissing1)
                p.addVerticalTemplate(mytemplate)
                p.addVSeparator(mytemplate,lfinalCuts)
            else:
                print( 'NO REGISTRATION')
        
        self.tagAsRegion(lPages)
        
        return 1
    
    def tagAsRegion(self,lPages):
        """
            create regions
        """
        for page in lPages:
            if page.getNode():
                # if several template ???
                for template in page.getVerticalTemplates():
                    page.getdVSeparator(template).sort(key=lambda x:x.getValue())
                    print (page.getdVSeparator(template))
                    page.getNode().setProp('template',str(page.getdVSeparator(template)))
                    XMinus = 1
                    prevcut = 10
#                     print page, page.getdVSeparator(template)
                    for cut in page.getdVSeparator(template):
                        cellNode  = etree.Element('REGION')
                        cellNode.set("y",str(prevcut))
                        ## it is better to avoid
                        YMinus= 10
                        cellNode.set("x",str(XMinus))
                        cellNode.set("width",str(page.getWidth()-2 * YMinus))
                        cellNode.set("height", str(cut.getValue() - prevcut))                            
                        page.getNode().addChild(cellNode)
                        prevcut  = cut.getValue()        
    
        
    
    def generateTestOutput(self,lPages):
        """
            create a run XML file
        """
        root = etree.Element('DOCUMENT')
        self.evalData = etree.ElementTree(root)
        for page in lPages:
            domp=etree.Element('PAGE')
            domp.set('number',page.getAttribute('number'))
            root.addChild(domp)
            for sep in page.lVSeparator:
                print (page.lVSeparator)
                domsep= etree.Element('SeparatorRegion')
                domp.append(domsep)
#                 domsep.setProp('x', str(sep[0].getValue()))
                domsep.set('x', str(sep[0]))
        

      
            
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
            
#             self.mainMining(self.lPages)
            for page in self.lPages:
                print("page")
                lTables = page.getAllNamedObjects(XMLDSTABLEClass)
                for table in lTables: 
                    self.columnMining(table)            
        
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
    
    
    docM = tableRowMiner()

    #prepare for the parsing of the command line
    docM.createCommandLineParser()
    docM.add_option("-f", "--first", dest="first", action="store", type="int", help="first page number", metavar="NN")
    docM.add_option("-l", "--last", dest="last", action="store", type="int", help="last page number", metavar="NN")
    docM.add_option("--pattern", dest="pattern", action="store", type="string", help="pattern to be applied", metavar="[]")
    docM.add_option("--thhighsupport", dest="thhighsupport", action="store", type="int", help="TH for high support", metavar="NN")
        
    #parse the command line
    dParams, args = docM.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    docM.setParams(dParams)
    
    doc = docM.loadDom()
    docM.run(doc)
    if doc and docM.getOutputFileName() != "-":
        docM.writeDom(doc, True)

