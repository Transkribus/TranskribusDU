#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

     H. Déjean

    copyright Naver Labs Europe 2018
    READ project 

    mine vertical separators for inducing a table template
    
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import sys, os.path
from shapely.geometry.linestring import LineString
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
from ObjectModel.XMLDSPageClass import XMLDSPageClass
from ObjectModel.treeTemplateClass import treeTemplateClass
from ObjectModel.XMLDSTABLEClass import XMLDSTABLEClass
from ObjectModel.XMLDSCELLClass import XMLDSTABLECELLClass
from ObjectModel.XMLDSTEXTClass import XMLDSTEXTClass
from ObjectModel.XMLDSTableColumnClass import XMLDSTABLECOLUMNClass
from ObjectModel.XMLDSTableRowClass import XMLDSTABLEROWClass

from xml_formats.Page2DS import primaAnalysis

from xml_formats.PageXml import PageXml
from util.geoTools import polygon2points
from util.geoTools import populateGeo
from util.geoTools import sPoints2tuplePoints
from shapely.ops import cascaded_union
from shapely.geometry import Polygon

from util.partitionEvaluation import evalPartitions
from util.partitionEvaluation import jaccard
from util.partitionEvaluation import iuo


class tableTemplateSep(Component.Component):
    """
        tableTemplateSep class: a component to mine page layout to infer a table template from separator
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
        Component.Component.__init__(self, "tableTemplateSep", self.usage, self.version, self.description) 
        
        self.docid= None
        
        # tag level
        self.sTag= XMLDSGRAPHLINEClass
        # TH for comparing numerical features for X
        self.THNUMERICAL = 10
        # use for evaluation
        self.THCOMP = 10
        self.evalData= None
        
        self.glength = 100
         
        self.bDomTag=True
        
        self.fKleenPlusTH =1.5

        # pattern provided manually        
        self.bManual = False
        
        
        self.bCreateRef = False
        self.bCreateRefCluster = False
        self.bEvalCluster=False
        
        self.do2DS = False
        
    def setParams(self, dParams):
        """
        Always call first the Component setParams
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        Component.Component.setParams(self, dParams)
        if "pattern" in dParams: 
            self.manualPattern = eval( dParams["pattern"])
            self.bManual=True  

        if "KLEENETH" in dParams:
            self.fKleenPlusTH =   dParams["KLEENETH"]
            
        if 'tag' in dParams:
            self.sTag = dParams['tag']       
        
        if 'createref' in dParams:
            self.bCreateRef = dParams['createref']   
                                       
        if "createrefCluster" in dParams:         
            self.bCreateRefCluster = dParams["createrefCluster"]
            
        if "evalCluster" in dParams:         
            self.bEvalCluster = dParams["evalCluster"]            

        if 'dsconv' in dParams:
            self.do2DS = dParams['dsconv']
            
        if 'glength' in dParams:
            self.glength = dParams['glength']            
            
            
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
#                         elt.setFeatureFunction(elt.getSetOfListedAttributes,self.THNUMERICAL,lFeatureList=['virtual'],myLevel=XMLDSPageClass)
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
        del seqGen        
        # return also the tree ; also organize elements per level/pattern
        return lList 
            
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
  
    
    

    def mergeFeatureH(self,Fone,Ftwo):
        """
            @input: two features
            @return: a merge representation of one and two 
        """
#         print('One',Fone,list(map(lambda x:x.getPoints(),Fone.getNodes())))
#         print('Two',Fone,list(map(lambda x:x.getPoints(),Ftwo.getNodes())))


        # get points, order them
        lPoints = []
        [lPoints.append(x.getPoints()) for x in Fone.getNodes()] 
        [lPoints.append(x.getPoints()) for x in Ftwo.getNodes()] 
        lPoints.sort(key=lambda x:x[0][0])
#         print (lPoints)
        [Fone.addNode(n) for n in Ftwo.getNodes()]
        return Fone
        # extrapolate 0 and last
 
    def mergeFeatureV(self,Fone,Ftwo):
        """
            @input: two features
            @return: a merge representation of one and two 
        """
#         print('One',Fone,list(map(lambda x:x.getPoints(),Fone.getNodes())))
#         print('Two',Fone,list(map(lambda x:x.getPoints(),Ftwo.getNodes())))


        # get points, order them
        lPoints = []
        [lPoints.append(x.getPoints()) for x in Fone.getNodes()] 
        [lPoints.append(x.getPoints()) for x in Ftwo.getNodes()] 
        lPoints.sort(key=lambda x:x[0][1])
#         print (lPoints)
        [Fone.addNode(n) for n in Ftwo.getNodes()]
        return Fone
        # extrapolate 0 and last        

    def processIntersection(self,pageH,pageW,lFeatures):
        """
            if a segment intersects more than one segment: ignore it
                # compute intersections: sort by freq
                # del most segment with most frequent intersections (>=1)
            remaining: segment with intersect <=1:
                merge with other intersecting segment
                
                
            final: polynomial from points from segments? 
        """
        lVLineStrings=[]
        lHLineStrings=[]
        lFinalFeature = []
        
        lH =list(filter(lambda x:x.getName()=='h',lFeatures))
        lV =list(filter(lambda x:x.getName()=='v',lFeatures))
    
#         return lH+lV
    
        for f in lFeatures:
            if f.getName() =='v':
                x2 = f.getValue() + f.b * pageH
                lVLineStrings.append(LineString( ((f.getValue(),0) ,(x2,pageH))))
            else:
                y2 = f.getValue() + f.b * pageW
                lHLineStrings.append(LineString( ((0,f.getValue()),(pageW,y2) )))
        
        # HORIZONTAL
        dInter={} 
        lNoIntersection=[]
        for i,l in enumerate(lHLineStrings):
            for j,ll in enumerate(lHLineStrings):
                if i != j and l.intersects(ll):
                    try:dInter[i].append(j)
                    except KeyError: dInter[i]=[j] 
            try:dInter[i]
            except: lNoIntersection.append(lH[i])
        
        
        #sort by len
        dSInter =sorted(dInter.items(), key=lambda item: len(item[1]),reverse=True)
        i=0
        bGO=True
        if dSInter != []:
            while bGO:            
                if len(dSInter[i][1]) > 1:
#                     print ('del',dSInter[i])
                    for j in dSInter[i][1]: dInter[j].remove(dSInter[i][0])
                    del(dInter[dSInter[i][0]])
    #                 try:[dSInter[j][1].remove(i) for j in dSInter[i]]
    #                 except: pass
                i+=1
                bGO = len(dSInter[i][1])>1 or i < len(dSInter)-1    
            
            lKeepThem=[]
            lSeen=[]
            for key, value in dInter.items():
#                 print (key,value[0])
                if value != [] and value[0] not in lSeen:
                    lKeepThem.append((key,value[0]))
                    lSeen.append(value[0])
                    lSeen.append(key)

            for one,two in lKeepThem:
#                 lFinalFeature.append(lH[one])
#                 lFinalFeature.append(lH[two])
                lFinalFeature.append(self.mergeFeatureH(lH[one],lH[two]))
        lFinalFeature.extend(lNoIntersection)
#         print ("??",lFinalFeature)

        # VERTICAL 
        dInter={} 
        lNoIntersection=[]
        for i,l in enumerate(lVLineStrings):
            for j,ll in enumerate(lVLineStrings):
                if i != j and l.intersects(ll):
                    try:dInter[i].append(j)
                    except KeyError: dInter[i]=[j] 
#                     print (l,ll,dInter[i])
            try:dInter[i]
            except: lNoIntersection.append(lV [i])
        
        
        #sort by len
        dSInter =sorted(dInter.items(), key=lambda item: len(item[1]),reverse=True)
        i=0
        bGO=True
        if dSInter != []:
            while bGO:            
                if len(dSInter[i][1]) > 1:
#                     print ('del',dSInter[i])
                    for j in dSInter[i][1]: dInter[j].remove(dSInter[i][0])
                    del(dInter[dSInter[i][0]])
    #                 try:[dSInter[j][1].remove(i) for j in dSInter[i]]
    #                 except: pass
                i+=1
                bGO = len(dSInter[i][1])>1 or i < len(dSInter)-1    
            
            lKeepThem=[]
            lSeen=[]
            for key, value in dInter.items():
#                 print (key,value[0])
                if value!= [] and  value[0] not in lSeen:
                    lKeepThem.append((key,value[0]))
                    lSeen.append(value[0])
                    lSeen.append(key)
    
            for one,two in lKeepThem:
#                 lFinalFeature.append(lV[one])
#                 lFinalFeature.append(lV[two])         
                lFinalFeature.append(self.mergeFeatureV(lV[one],lV[two]))
        lFinalFeature.extend(lNoIntersection)
        return lFinalFeature
        
        
        
        
    def minePageVerticalFeature(self,lPages,lFeatureList,level=XMLDSGRAPHLINEClass):
        """
            get page features for  vertical zones: find vertical regular vertical Blocks/text structure
            
            
            add : if intersection of lines: take the longest one
            
            
            put the length test after feature creation!!
        """ 
        chronoOn()
        
        
        glength = self.glength
        for _,page, in enumerate(lPages):
            page.origFeat = []
            # GRAPHICAL LINES 
            gl = []
            for graphline in page.getAllNamedObjects(level):
                graphline.resetFeatures()
                graphline._canonicalFeatures = None
                # vertical 
#                 if graphline.getHeight() > graphline.getWidth() and graphline.getHeight() > 30:
                if  graphline.getHeight() > glength and  graphline.getHeight()  > graphline.getWidth():

                    gl.append(graphline)
                    # create a feature
                    f = featureObject()
                    f.setType(featureObject.NUMERICAL)
                    f.setTH(self.THNUMERICAL)
#                     f.setTH(5)
                    f.setWeight(graphline.getHeight())
                    f.setName("v")
                    f.setObjectName(graphline)
                    f.addNode(graphline)
                    X = [float(xy.split(',')[0]) for xy in graphline.getAttribute('points').split(' ') ]
                    Y = [float(xy.split(',')[1]) for xy in graphline.getAttribute('points').split(' ') ]
#                     Y = [float(y) for y in graphline.getAttribute('points').split(',')[1::2]]

                    a, b = np.polynomial.polynomial.polyfit(Y,X,1)
#                     x2 = a + b * page.getHeight()
                    ## project to Y=0
                    f.setValue(round(a))
                    f.b= b
#                     f.setValue(round(graphline.getX()))
                    graphline.addFeature(f)
                    page.setVGLFeatures(f)
                
# #                 horizontal
                elif graphline.getWidth() >  glength and graphline.getWidth()  > graphline.getHeight() : #and graphline.getY() > 10 and  graphline.getY2() < page.getHeight() - 10  :  
                    gl.append(graphline)
                    # create a feature
                    f = featureObject()
                    f.setType(featureObject.NUMERICAL)
                    f.setTH(self.THNUMERICAL)
#                     f.setTH(5)
                    f.setWeight(graphline.getWidth())
                    f.setName("h")
                    f.setObjectName(graphline)
                    f.addNode(graphline)
                    X = [float(xy.split(',')[0]) for xy in graphline.getAttribute('points').split(' ') ]
                    Y = [float(xy.split(',')[1]) for xy in graphline.getAttribute('points').split(' ') ]

                    a, b = np.polynomial.polynomial.polyfit(X,Y,1)
                    f.b= b
                    f.setValue(round(a))
                    # project to X = 0
#                     f.setValue(round(graphline.getY()))
                    graphline.addFeature(f)
                    page.setVGLFeatures(f)                    
#                     
        self.buildVZones(lPages)
        
        print ('chronoFeature',chronoOff())
        
        return lPages
            
        
    
    def buildVZones(self,lp):
        """
            store vertical positions in each page
        """
        
        for _, p in enumerate(lp):
            p.lf_XCut=[]
            ltmp=[]
            p.getVGLFeatures().sort(key=lambda x:x.getWeight(),reverse=True)
            for fi in p.getVGLFeatures():
                if  fi not in ltmp:
                    ltmp.append(fi)
                else: 
                    [ltmp[ltmp.index(fi)].addNode(n) for n in fi.getNodes()]
            ltmp.sort(key=lambda x:x.getValue())
            for fi in ltmp:
                if fi.getName() == 'v':
                    totalLen = sum(x.getHeight() for x in fi.getNodes())
                    maxElt= max(fi.getNodes(),key=lambda x:x.getHeight())
                    fi._lnodes = set()
                    fi._lnodes.add(maxElt)                    
                else: 
                    totalLen = sum(x.getWidth() for x in fi.getNodes())
                    #keep the longest one: represenative enough
                    maxElt= max(fi.getNodes(),key=lambda x:x.getWidth())
#                     print (fi,maxElt.getWidth(),[x.getWidth() for x in fi.getNodes()])
                    fi._lnodes = set()
                    fi._lnodes.add(maxElt)
                if totalLen > self.glength:
                    p.lf_XCut.append(fi)

    def highLevelSegmentation(self,lPages):
        """
            use: image size and content (empty pages)
        """
        
        lSubList= self.minePageDimensions(lPages)

        return lSubList
     
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
                ## test is pattern is mirrored         
                template  = treeTemplateClass()
                template.setPattern(pattern)
                template.buildTreeFromPattern(pattern)
                template.setType('lineTemplate')
                try:dTemplatesTypes[template.__class__.__name__].append((pattern, support, template))
                except KeyError: dTemplatesTypes[template.__class__.__name__] = [(pattern,support,template)]                      
        
        return dTemplatesTypes
                
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
        
            
    def iterativeProcessVSegmentation(self, lLPages):
        """
            process lPages by batch
            
            
            for a page: generate a template with the previous, the next   (mirrored: n-2, p-2)    n-N, p-N
            Several templates per pages: sequence mining!
                needed: a good way to compute template against page (as usual)!
        """
        
        for lPages in lLPages:
#             print (lPages)
            self.THNUMERICAL = 10#ABP 
#             self.THNUMERICAL = 5 #NAF
            self.minePageVerticalFeature(lPages,None,level=self.sTag)
            self.cleanupPages(lPages)
            for page in lPages:
                lSelectedFeatures = self.processIntersection(page.getHeight(), page.getWidth(),page.lf_XCut)
                page.resetVerticalTemplate()  
                page.lf_XCut = lSelectedFeatures
                page._lBasicFeatures=page.lf_XCut[:]          
#             for p  in lPages:
#                 p.resetVerticalTemplate()
#                 p._lBasicFeatures=p.lf_XCut[:]
                
#             self.dtwMatching(lPages)
            self.separatorBaseline(lPages)
           
        
        
    def separatorBaseline(self,lPages):
        """
            consider vertical separators as col 
        """
        tableTemplate=treeTemplateClass()
        for p in lPages:
            p.addVerticalTemplate(tableTemplate)
            p.addVSeparator(tableTemplate,p.lf_XCut)
        
        self.tableRecognition(lPages)    
        
    def dtwMatching(self,lPages):
        """
            create vertical zones and match them
            
            When possible: compute with prev page and next page and take the best alignment
            
            
            test if too many elements with distance=0!!!
        """            
        from util.partitionEvaluation import  matchingPairs
        
        
        def distX(c1,c2): 
            o  = min(c1.getX2() , c2.getX2()) - max(c1.getX() , c2.getX())
            if o <= 0:
                return 1
            return 1 -  ((min(c1.getX2() , c2.getX2()) - max(c1.getX() , c2.getX())) / max(c1.getWidth() , c2.getWidth()))
        
        def distW(c1,c2):
            return abs(c1.getWidth() - c2.getWidth()) / max(c1.getWidth() , c2.getWidth())
        
        
        tableTemplate=treeTemplateClass()
        INC=0
        for i,p in enumerate(lPages):
            if i == len(lPages)-1:
                INC=-2
            # elif i == -1  : take prevoous
            l1 =[]
            for a,f in enumerate(p.lf_XCut):
                o = XMLDSObjectClass()
                o.setX(f.getValue())
                if a == len(p.lf_XCut) -1:
                    o.setName("%s_right"%(f.getValue()))
                    o.setWidth(p.getWidth() - o.getX())
                else:
                    o.setName("%s_%s"%(f.getValue(),p.lf_XCut[a+1].getValue()))
                    o.setWidth(p.lf_XCut[a+1].getValue() -o.getX())
                o.setContent(str(o.getWidth()))
                o.o = f
                l1.append(o)
            l2 =[]
#             print (i+1+INC)
            for a,f in enumerate(lPages[i+1+INC].lf_XCut):
                o= XMLDSObjectClass()
                o.setX(f.getValue())
                if a == len(lPages[i+1+INC].lf_XCut) -1:
                    o.setName("%s_right"%(f.getValue()))
                    o.setWidth(lPages[i+1+INC].getWidth() -o.getX())
                else:
                    o.setName("%s_%s"%(f.getValue(),lPages[i+1+INC].lf_XCut[a+1].getValue()))
                    o.setWidth(lPages[i+1+INC].lf_XCut[a+1].getValue() -o.getX())
                o.setContent(str(o.getWidth()))
                l2.append(o)        
#             print (p,list(map(lambda x:(x.getValue(),x.getWeight()),p.lf_XCut)))
            #create vzones
#             lmatches = matchingPairs(l1,l2,distW)
            lmatches = matchingPairs(l1,l2,distX)
            lcuts=[]
            for lc1,lc2 in lmatches:
                #[(573.0_596.0[None] 23.0, 0.17557251908396942), (596.0_703.0[None] 107.0, 0.8091603053435115)]
#                 print(p,lc1,lc2) 
                lc1.sort(key=lambda x:x[-1], reverse=True)
                bestcut = lc1[0]
                lcuts.append(bestcut[0].o)
#             print (p,lcuts)
            p.addVerticalTemplate(tableTemplate)
            p.addVSeparator(tableTemplate,lcuts)
        
        self.tagAsRegion(lPages)    
            
        

    def points2Feature(self,page,dir,v,x,y):
        """
            create FeatureObject for table borders
            
        """
        o = XMLDSObjectClass()
        o.addAttribute('points',"%s,%s %s,%s"%(x[0],x[1],y[0],y[1]))
        o.setPage(page)
        o.setX(x[0])
        o.setY(y[0])
        f = featureObject()
        f.setType(featureObject.NUMERICAL)
        f.setTH(self.THNUMERICAL)
        f.setName(dir)
        f.setValue(v)
        f.addNode(o)
        
        return f        
    def createTable(self,page,VSep,HSep):
        """
            
        """
        table= XMLDSTABLEClass()
        table.setPage(page)
        lSep=[]
        [lSep.append(n) for s in VSep+HSep for n in s.getNodes()]
        lP = []
        [ lP.append(n.toPolygon()) for n in lSep ]
        contour = cascaded_union([ p for p in lP if p.is_valid])                
        
        try:contourpoints =  contour.minimum_rotated_rectangle.exterior.coords
        except AttributeError: return None
         
        assert len(contourpoints) == 5
#         print (list(contourpoints))
        lX= list(contourpoints[:-1])
        lX.sort(key=lambda x:x[0])
        lY= list(contourpoints[:-1])
        lY.sort(key=lambda x:x[1])

        topleftx= min(x[0] for x in lX[:2])
        toplefty =  min(x[1] for x in lX[:2])
        toprightx = min(x[0] for x in lX[2:])
        toprighty = min(x[1] for x in lX[2:])

        bottomleftx = min(x[0] for x in lY[2:])
        bottomlefty = min(x[1] for x in lY[2:])      
        bottomrightx = max(x[0] for x in lY[:2])
        bottomrighty = max(x[1] for x in lY[2:])
        table.addAttribute('points','%f,%f %f,%f %f,%f %f,%f'%(topleftx,toplefty,toprightx,toprighty,bottomrightx,bottomrighty,bottomleftx,bottomlefty))
        table.setDimensions(topleftx, toplefty, bottomlefty - toplefty, toprightx-topleftx)
        table.firstV = self.points2Feature(page,'v',topleftx,(topleftx,toplefty),(bottomleftx,bottomlefty))
        table.lastV = self.points2Feature(page,'v',toprightx,(toprightx,toprighty),(bottomrightx,bottomrighty))
        table.firstH=self.points2Feature(page,'h',toplefty,(topleftx,toplefty),(toprightx,toprighty))
        table.lastH= self.points2Feature(page,'h',bottomlefty,(bottomleftx,bottomlefty),(bottomrightx,bottomrighty))
        return table
    
    def tableRecognition(self,lPages):
        """
           Assume one table per page
           
        """
        for page in lPages:
            if page.getNode() is not None:
                print ('processing page %s'%page)
                irow=0
                icol=0
                for template in page.getVerticalTemplates():
                    prevcut = 1
                    lCuts=[prevcut]
                    lastcutpointsV =""
                    lastcutpointsH =""
                    VSep = list(filter(lambda x:x.getName()=='v', page.getdVSeparator(template)))
                    HSep = list(filter(lambda x:x.getName()=='h', page.getdVSeparator(template)))
                    HSep.sort(key=lambda x:list(iter(x.getNodes()))[0].getY())
                    VSep.sort(key=lambda x:list(iter(x.getNodes()))[0].getX())
                    curTable = self.createTable(page,VSep,HSep)
                    if curTable is None: continue
                    ## need to add first V and Last from table
                    ## need to add first and last H from table
                    HSep.insert(0,curTable.firstH)
                    HSep.append(curTable.lastH)
                    VSep.insert(0,curTable.firstV)
                    VSep.append(curTable.lastV)
                    for cut in HSep +VSep : #page.getdVSeparator(template):
                        if cut.getName() == 'v':
                            newReg= XMLDSTABLECOLUMNClass(icol)
                            icol+=1
#                             newReg.setName('COL')
                            domNode  = etree.Element('COL')
                            domNode.set("x",str(prevcut))
                            ## it is better to avoid
                            YMinus = 1
                            domNode.set("y",str(YMinus))
                            domNode.set("height",str(page.getHeight()-2 * YMinus))
                            domNode.set("width", str(cut.getValue() - prevcut))
                            lCuts.append(cut.getValue() )
                            newReg.setNode(domNode)
                            
                            # top first 
                            pageNodes = list(filter(lambda x:x.getPage() == page, cut.getNodes()))
                            pageNodes.sort(key=lambda x:x.getY())
                            
                            niceNode = pageNodes[0]
                            X = [float(xy.split(',')[0]) for xy in niceNode.getAttribute('points').split(' ') ]
                            Y = [float(xy.split(',')[1]) for xy in niceNode.getAttribute('points').split(' ') ]
                            a, b = np.polynomial.polynomial.polyfit(Y,X,1)
                            x2 = a + b * page.getHeight()

                            if len(pageNodes) > 1:
                                niceNode = pageNodes[-1]
                                X = [float(xy.split(',')[0]) for xy in niceNode.getAttribute('points').split(' ') ]
                                Y = [float(xy.split(',')[1]) for xy in niceNode.getAttribute('points').split(' ') ]
                                a2, b2 = np.polynomial.polynomial.polyfit(Y,X,1)
                                x22 = a2 + b2 * page.getHeight()
                            else: x22=x2
                            
                            sPoints= "%s,%s " %(a,0)
                            for n in pageNodes:
                                sPoints += n.getAttribute('points')+" "
                            sPoints+= "%s,%s"%(x22,page.getHeight())
                            if lastcutpointsV != "":
                                hull = lastcutpointsV+" "+sPoints
                                newReg.addAttribute('points', hull)
                                newReg.setDimensions(prevcut,YMinus, page.getHeight()-2 * YMinus,cut.getValue() - prevcut)
                                if newReg.toPolygon().is_valid and newReg.getWidth()>5:
                                    domNode.set('points',hull)
                                    curTable.addColumn(newReg)
                                    newReg.setParent(curTable)
                                    xx =sPoints.split(' ')
                                    xx.reverse()
                                    lastcutpointsV = " ".join(list(x for x in xx))
                                else:icol-=1
                            else:
                                # table.firstV
                                icol-=1
#                                 print('first Vline:',sPoints)
                                xx =sPoints.split(' ')
                                xx.reverse()
                                lastcutpointsV = " ".join(list(x for x in xx))                            
                                
                            prevcut  = cut.getValue()
                            
                            
                        elif cut.getName() == 'h':
#                             print(page,cut,cut.getNodes())
                            newReg= XMLDSTABLEROWClass(irow)
                            irow+=1
                            domNode  = etree.Element('ROW')
                            XMinus= 1
                            domNode.set("x",str(XMinus))
                            ## it is better to avoid
                            domNode.set("y",str(prevcut))
                            domNode.set("width",str(page.getWidth()-2 * XMinus))
                            domNode.set("height", str(cut.getValue() - prevcut))
                            lCuts.append(cut.getValue() )
                            newReg.setNode(domNode)
                            
                            pageNodes = list(filter(lambda x:x.getPage() == page, cut.getNodes()))
                            #left first 
                            pageNodes.sort(key=lambda x:x.getX())     
                      
                            niceNode = pageNodes[0]
                            
                            X = [float(xy.split(',')[0]) for xy in niceNode.getAttribute('points').split(' ') ]
                            Y = [float(xy.split(',')[1]) for xy in niceNode.getAttribute('points').split(' ') ]
                            a, b = np.polynomial.polynomial.polyfit(X,Y,1)
                            y2 = a + b * page.getWidth()
                            
                            if len(pageNodes) >1:
                                niceNode2=pageNodes[-1]
                                X = [float(xy.split(',')[0]) for xy in niceNode2.getAttribute('points').split(' ') ]
                                Y = [float(xy.split(',')[1]) for xy in niceNode2.getAttribute('points').split(' ') ]
                                a2, b2 = np.polynomial.polynomial.polyfit(X,Y,1)
                                y22 = a2 + b2 * page.getWidth()                                
                            else:
                                y22=y2
                            
                            sPoints= "%s,%s " %(0,a)
                            for n in pageNodes:
                                sPoints += n.getAttribute('points')+" "
                            sPoints+= "%s,%s"%(page.getWidth(),y22)

                            if lastcutpointsH != "":
                                hull= lastcutpointsH + ' ' +sPoints 
                                newReg.addAttribute('points',hull)
                                newReg.setDimensions(XMinus,prevcut,cut.getValue() - prevcut, page.getWidth()-2 * XMinus)
                                if newReg.toPolygon().is_valid and newReg.getHeight()> 5:
                                    domNode.set('points',hull)
                                    curTable.addRow(newReg)
                                    curTable.setPage(page)
                                    newReg.setParent(curTable)
                                    xx =sPoints.split(' ')
                                    xx.reverse()
                                    lastcutpointsH = " ".join(list(x for x in xx))
                                else:
                                    irow-=1
#                                     print('back',irow)
                            else:
                                irow-=1
#                                 print('first Hline:',sPoints)
                                xx =sPoints.split(' ')
                                xx.reverse()
                                lastcutpointsH = " ".join(list(x for x in xx))                                     
                            
                            prevcut  = cut.getValue()                            
                
#                 TdomNode  = etree.Element(curTable.getName())
#                 curTable.setNode(TdomNode)
                if  curTable:
                    curTable.tagMe()
                    page.getNode().append(curTable.getNode())
                    for col in curTable.getColumns(): 
                        col.tagMe()
    #                     print (col.getParent(),col.getAttribute('points'))
                    for row in curTable.getRows():
                        row.tagMe()  
    #                     print (row.getParent(),row.getAttribute('points'))
#                     lEltValid = list(filter(lambda x:x.toPolygon().is_valid and x.toPolygon().area >0 , page.getAllNamedObjects(XMLDSTEXTClass)))
                    lEltValid=[]
                    curTable.reintegrateCellsInColRow(lObj=lEltValid)
                    curTable.buildNDARRAY()

        return None
    
    
                
                
    def generateTableTemplate(self,lT1):
        """
            create a table object:
            table zone: page
            columns: the created vertical zones
        """
        
        from util.unitConversion import convertDot2Pixel
         
        for template in lT1: 
            ### create a table: assumption: the full page
            page= self.lPages[0]
            dpi=300
#             pageXmlDoc,pageNode = PageXml.createPageXmlDocument(creatorName='NLE', filename = "%s_%s"% (self.getInputFileName(),'tt.pxml'), imgW = int(page.getWidth()), imgH = int(page.getHeight()))
            pageXmlDoc,pageNode = PageXml.createPageXmlDocument(creatorName='NLE', filename = "%s_%s"% (os.path.basename(self.getInputFileName()),'tt.pxml'), imgW = convertDot2Pixel(dpi,page.getWidth()), imgH = convertDot2Pixel(dpi,page.getHeight()))

            tableNode= PageXml.createPageXmlNode('TableRegion')            
            pageNode.append(tableNode)
            tableNode.set('id','TableRegion_1')
#             tableNode.set('x','0')
#             tableNode.set('y','0')
#             tableNode.set('height',str(page.getHeight()))
#             tableNode.set('width',str(page.getWidth()))
            coords = PageXml.createPageXmlNode('Coords')    
            tableNode.append(coords)
            hcuts= list(filter(lambda x:x.getName()=='h',template.getPattern()))[0:-1]
            vcuts= list(filter(lambda x:x.getName()=='v',template.getPattern()))
            print (hcuts,vcuts)
            # #hcuts[0].getValue()\
            #hcuts[-1].getValue()))
            if vcuts == []: return
            a,z,e,r,t,y,u,i = map(lambda x:convertDot2Pixel(dpi,x),(vcuts[0].getValue(), 0  
                                                                    ,vcuts[-1].getValue(),0 
                                                                    ,vcuts[-1].getValue(),page.getHeight()
                                                                    ,vcuts[0].getValue(),page.getHeight()-20))  
            coords.set('points','%d,%d %d,%d %d,%d %d,%d'%(a,z,e,r,t,y,u,i))
            print (template,template.getPattern())
#             print (hcuts,vcuts)
            
            prevrcut= 20
#             prevrcut=hcuts[0].getValue()
#             for i,rcut in enumerate([hcuts[0],hcuts[-1]])
            for i,rcut in enumerate([page.getHeight()-20]):
                prevccut= vcuts[0].getValue()
                for j,ccut in enumerate(vcuts[1:]):
#                     print (i,ccut,j,rcut)
                    cellNode  =  PageXml.createPageXmlNode('TableCell')
                    
                    cellNode.set("id","TableCell%d%d"%(i,j))
                    cellNode.set("row",str(i))
                    cellNode.set("col",str(j))

                    cellNode.set("colSpan","1")
                    cellNode.set("rowSpan","1")
                    
                    
                    for border in ["leftBorderVisible","rightBorderVisible","topBorderVisible","bottomBorderVisible"]:
                        cellNode.set(border,"true")
                    for border in ["topBorderVisible","bottomBorderVisible"]:
                        cellNode.set(border,"false")
                           
                    coords = PageXml.createPageXmlNode('Coords')       
                    cellNode.append(coords)  
                    ###    KEY!!!! point order is very important
                    a,z,e,r,t,y,u,o = map(lambda x:convertDot2Pixel(dpi,x),(prevccut,prevrcut, ccut.getValue(),prevrcut ,ccut.getValue(),rcut,prevccut,rcut))
#                     a,z,e,r,t,y,u,o = map(lambda x:convertDot2Pixel(dpi,x),(prevccut,prevrcut, ccut.getValue(),prevrcut ,ccut.getValue(),rcut.getValue(),prevccut,rcut.getValue()))
                    coords.set('points', '%d,%d %d,%d %d,%d %d,%d'%( a,z,u,o,t,y,e,r))
#                     print ('\t',prevccut,prevrcut, ccut.getValue(),prevrcut ,ccut.getValue(),rcut.getValue(),prevccut,rcut.getValue())
                     
                    corners = PageXml.createPageXmlNode('CornerPts')
                    corners.text = '0 1 2 3'
                    
                    cellNode.append(corners)   
                    tableNode.append(cellNode)
                    prevccut  = ccut.getValue() 
#                 prevrcut  = rcut.getValue()     


#             print(etree.tostring(pageXmlDoc, pretty_print=True))
            pageXmlDoc.write("tt0.xml")        
    


    def testGeometry(self, th, srefData, srunData, bVisual=False):
        """
            compare geometrical zones (dtw + iou)
            :param
            
            returns tuple (cntOk, cntErr, cntMissed,ltisRefsRunbErrbMiss
            
        """

        cntOk = cntErr = cntMissed = 0
        ltisRefsRunbErrbMiss = list()
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))
        
        lPages = RefData.xpath('//%s' % ('PAGE[@number]'))
        
        for ip,page in enumerate(lPages):
            lY=[]
            key=page.get('pagekey') 
            xpath = ".//%s" % ("COL")
            lcols = page.xpath(xpath)
            if len(lcols) > 0:
                for col in lcols:
                    xpath = ".//@points"
                    lpoints  = col.xpath(xpath) 
                    colgeo = cascaded_union([ Polygon(sPoints2tuplePoints(p)) for p in lpoints])
                    if lpoints != []:
                        lY.append(colgeo)
        
            if RunData is not None:
                lpages = RunData.xpath('//%s' % ('PAGE[@pagekey="%s"]' % key))
                lX=[]
                if lpages != []:
                    for page in lpages[0]:
                        xpath = ".//%s" % ("COL")
                        lcols = page.xpath(xpath)
                        if len(lcols) > 0:
                            for col in lcols:
                                xpath = ".//@points"
                                lpoints =  col.xpath(xpath)
                                if lpoints != []:
                                    lX.append(  Polygon(sPoints2tuplePoints(lpoints[0])))
                    lX = list(filter(lambda x:x.is_valid,lX))
                    ok , err , missed,lfound,lerr,lmissed = evalPartitions(lX, lY, th,iuo)
                    cntOk += ok 
                    cntErr += err
                    cntMissed +=missed
                    [ltisRefsRunbErrbMiss.append((ip, y1.bounds, x1.bounds,False, False)) for (x1,y1) in lfound]
                    [ltisRefsRunbErrbMiss.append((ip, y1.bounds, None,False, True)) for y1 in lmissed]
                    [ltisRefsRunbErrbMiss.append((ip, None, x1.bounds,True, False)) for x1 in lerr]

#                     ltisRefsRunbErrbMiss.append(( lfound, ip, ok,err, missed))
#                     print (key, cntOk , cntErr , cntMissed)
        return (cntOk , cntErr , cntMissed,ltisRefsRunbErrbMiss)         
    
    
    def testCluster(self, th, srefData, srunData, bVisual=False):
        """
        <DOCUMENT>
          <PAGE number="1" imageFilename="g" width="1457.52" height="1085.04">
            <TABLE x="120.72" y="90.72" width="1240.08" height="923.28">
              <COL>
                <TEXT id="line_1502076498510_2209"/>
                <TEXT id="line_1502076500291_2210"/>
                <TEXT id="line_1502076502635_2211"/>
                <TEXT id="line_1502076505260_2212"/>
        
            
            
            NEED to work at page level !!??
            then average?
        """
        cntOk = cntErr= cntMissed = 0
        RefData = etree.XML(srefData.strip("\n").encode('utf-8'))
        RunData = etree.XML(srunData.strip("\n").encode('utf-8'))

        lPages = RefData.xpath('//%s' % ('PAGE[@number]'))
        for page in lPages:
            lY=[]
            key=page.get('pagekey') 
            xpath = ".//%s" % ("COL")
            lcols = page.xpath(xpath)
            if len(lcols) > 0:
                for col in lcols:
#                     if col.get('width') and int(row.get('width'))>100:
                        xpath = ".//@id"
                        lid  = col.xpath(xpath) 
                        if lid != []:
                            lY.append(lid)
        
            if RunData is not None:
                lpages = RunData.xpath('//%s' % ('PAGE[@pagekey="%s"]' % key))
                lX=[]
                if lpages != []:
                    for page in lpages:
                        xpath = ".//%s" % ("COL")
                        lcols = page.xpath(xpath)
                        if len(lcols) > 0:
                            for col in lcols:
#                                 if row.get('width') and int(row.get('width'))>100:
                                    xpath = ".//@id"
                                    lid =  col.xpath(xpath)
                                    if lid != []:
                                        lX.append( lid)
    #                 assert len(lX) > 0
    #                 assert len(lY) > 0
#                     print(key,len(lY),len(lX))
                    ok , err , missed,lf,lerr,lmiss = evalPartitions(lX, lY, th,jaccard)
                    cntOk += ok 
                    cntErr += err
                    cntMissed +=missed
#                     print (key, cntOk , cntErr , cntMissed)
        ltisRefsRunbErrbMiss= list()
        return (cntOk , cntErr , cntMissed,ltisRefsRunbErrbMiss)  
            
    
    def testCompare(self, srefData, srunData, bVisual=False):
        """
        as in Shahad et al, DAS 2010

        Correct Detections 
        Partial Detections 
        Over-Segmented 
        Under-Segmented 
        Missed        
        False Positive
                
        """
        dicTestByTask = dict()
        if self.bEvalCluster:
#             dicTestByTask['CLUSTER']= self.testCluster(srefData,srunData,bVisual)
            dicTestByTask['CLUSTER100']= self.testCluster(1.0,srefData,srunData,bVisual)
            dicTestByTask['CLUSTER90']= self.testCluster(0.9,srefData,srunData,bVisual)
            dicTestByTask['CLUSTER80']= self.testCluster(0.8,srefData,srunData,bVisual)
#             dicTestByTask['CLUSTER50']= self.testCluster(0.5,srefData,srunData,bVisual)

        else:
            dicTestByTask['T80']= self.testGeometry(0.50,srefData,srunData,bVisual)
#             dicTestByTask['T100']= self.testGeometry(1.0,srefData,srunData,bVisual)

    #         dicTestByTask['FirstName']= self.testFirstNameRecord(srefData, srunData,bVisual)
#         dicTestByTask['Year']= self.testYear(srefData, srunData,bVisual)
    
        return dicTestByTask    
        
    
    def createRefGeo(self,doc):
        """
            create a REF file
            :params doc: xmldoc
            
            returns a doc xml (ref format)
        """
        ODoc = XMLDSDocument()
        ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))        
  
        root=etree.Element("DOCUMENT")
        refdoc=etree.ElementTree(root)
        

        for page in ODoc.getPages():
#             print(page)
            #imageFilename="..\col\30275\S_Freyung_021_0001.jpg" width="977.52" height="780.0">
            pageNode = etree.Element('PAGE')
            pageNode.set("number",page.getAttribute('number'))
            pageNode.set("pagekey",os.path.basename(page.getAttribute('imageFilename')))
            pageNode.set("width",str(page.getAttribute('width')))
            pageNode.set("height",str(page.getAttribute('height')))

            root.append(pageNode)   
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                tableNode = etree.Element('TABLE')
                tableNode.set("x",table.getAttribute('x'))
                tableNode.set("y",table.getAttribute('y'))
                tableNode.set("width",str(table.getAttribute('width')))
                tableNode.set("height",str(table.getAttribute('height')))
                tableNode.set("points",table.getAttribute('points'))
                pageNode.append(tableNode)

                # iterate over rows
                for index,row in enumerate(table.getColumns()):
#                     print (index, list([c for c in row.getCells()]))
#                     rowpolygon = cascaded_union([c.toPolygon() for c in row.getCells()])
#                     if type(rowpolygon) == shapely.geometry.multipolygon.MultiPolygon:
#                         print ([list(p.exterior.coords) for p in rowpolygon.geoms])
#                     x1,y1,x2,y2 = rowpolygon.bounds
                    rowNode= etree.Element("COL")
                    tableNode.append(rowNode)
#                     rowNode.set('y',str(y1))
#                     rowNode.set('height',str(y2 - y1))
#                     rowNode.set('x',str(x1))
#                     rowNode.set('width',str(x2 - x1))
                    rowNode.set('id',str(index))
#                     rowNode.set('points',str(rowpolygon.exterior.coords))
                    for cell in row.getCells():
                        cNode =  etree.Element("CELL")
                        rowNode.append(cNode)
                        cNode.set('points',polygon2points(cell.toPolygon()))
                        
#         print (etree.tostring(refdoc,pretty_print = True))       
        return refdoc
        
    def createRunGeo(self,doc):
        """
            create a RUN file
            :params doc: xmldoc
            
            returns a doc xml (ref format)
        """
        self.ODoc = doc #XMLDSDocument()
#         self.ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))        
  
  
        root=etree.Element("DOCUMENT")
        refdoc=etree.ElementTree(root)
        

        for page in self.ODoc.getPages():
            pageNode = etree.Element('PAGE')
            pageNode.set("number",page.getAttribute('number'))
            pageNode.set("pagekey",os.path.basename(page.getAttribute('imageFilename')))
            pageNode.set("width",str(page.getAttribute('width')))
            pageNode.set("height",str(page.getAttribute('height')))

            root.append(pageNode)   
            tableNode = etree.Element('TABLE')
            tableNode.set("x","0")
            tableNode.set("y","0")
            tableNode.set("width","0")
            tableNode.set("height","0")
            pageNode.append(tableNode)
                
            lCols  = page.getAllNamedObjects(XMLDSTABLECOLUMNClass)
            for col in lCols:
                cNode= etree.Element("COL")
                tableNode.append(cNode)
                if col.getAttribute('points'):
                    cNode.set('points',col.getAttribute('points'))
                        
        return refdoc        
        
    def createRefCluster(self,doc):
        """
            Ref: a row = set of textlines
            :param doc: dox xml
            returns a doc (ref format): each column contains a set of ids (textlines ids)
        """            
        self.ODoc = XMLDSDocument()
        self.ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))        
  
  
        root=etree.Element("DOCUMENT")
        refdoc=etree.ElementTree(root)
        

        for page in self.ODoc.getPages():
            pageNode = etree.Element('PAGE')
            pageNode.set("number",page.getAttribute('number'))
            pageNode.set("pagekey",os.path.basename(page.getAttribute('imageFilename')))
            pageNode.set("width",str(page.getAttribute('width')))
            pageNode.set("height",str(page.getAttribute('height')))

            root.append(pageNode)   
            lTables = page.getAllNamedObjects(XMLDSTABLEClass)
            for table in lTables:
                ## fake table for separators
                if table.getWidth() < 40: continue
                dCols={}
                tableNode = etree.Element('TABLE')
                tableNode.set("x",table.getAttribute('x'))
                tableNode.set("y",table.getAttribute('y'))
                tableNode.set("width",str(table.getAttribute('width')))
                tableNode.set("height",str(table.getAttribute('height')))
                pageNode.append(tableNode)
                for cell in table.getAllNamedObjects(XMLDSTABLECELLClass):
                    try:dCols[int(cell.getAttribute("col"))].extend(cell.getObjects())
                    except KeyError:dCols[int(cell.getAttribute("col"))] = cell.getObjects()
        
                for rowid in sorted(dCols.keys()):
                    cNode= etree.Element("COL")
                    cNode.text =""
                    tableNode.append(cNode)
                    for elt in dCols[rowid]:
                        if elt.getY() >200:
                            txtNode = etree.Element("TEXT")
                            txtNode.set('id',elt.getAttribute('id'))
                            cNode.append(txtNode)
#                             cNode.text += str(elt.getContent())
                        
        return refdoc
                

    def createRunCluster(self,doc):
        """
            Ref: a row = set of textlines
            :param doc: dox xml
            returns a doc (ref format): each column contains a set of ids (textlines ids)
        """            
        self.ODoc = doc #XMLDSDocument()
#         self.ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))        
  
  
        root=etree.Element("DOCUMENT")
        refdoc=etree.ElementTree(root)
        

        for page in self.ODoc.getPages():
            pageNode = etree.Element('PAGE')
            pageNode.set("number",page.getAttribute('number'))
            pageNode.set("pagekey",os.path.basename(page.getAttribute('imageFilename')))
            pageNode.set("width",str(page.getAttribute('width')))
            pageNode.set("height",str(page.getAttribute('height')))

            root.append(pageNode)   
            tableNode = etree.Element('TABLE')
            tableNode.set("x","0")
            tableNode.set("y","0")
            tableNode.set("width","0")
            tableNode.set("height","0")
            pageNode.append(tableNode)
                
            lCols  = page.getAllNamedObjects(XMLDSTABLECOLUMNClass)
            for col in lCols:
                cNode= etree.Element("COL")
                cNode.text=""
                tableNode.append(cNode)
                for elt in col.getObjects():
                    if elt.getY() >200:
                        txtNode= etree.Element("TEXT")
                        txtNode.set('id',elt.getAttribute('id'))
                        cNode.append(txtNode)
                        cNode.text += str(elt.getContent())
#         print (etree.tostring(refdoc,pretty_print = True))       
                
        return refdoc
                
                
    
    #--- RUN ---------------------------------------------------------------------------------------------------------------
    
    def loadDSDoc(self,doc):
        """
        
        """
        self.doc= doc
        self.ODoc = XMLDSDocument()
        
        chronoOn()
        self.ODoc.loadFromDom(self.doc,listPages=range(self.firstPage,self.lastPage+1))        
        self.lPages= self.ODoc.getPages()
#         for p in self.lPages:
#             if len(p.getObjects())> 20: print (p,len(p.getObjects()))
#         ss
        print('chronoloading:', chronoOff())
        sys.stdout.flush()


    def testRun(self, filename, outFile=None):
        """
        evaluate using ABP new table dataset with tablecell
        """
        self.evalData = None
        doc = self.loadDom(filename)
        doc = self.run(doc)
        if self.bEvalCluster:
            self.evalData = self.createRunCluster( self.ODoc)
        else:
            self.evalData = self.createRunGeo(self.ODoc)
        if outFile: self.writeDom(doc)
        return etree.tostring(self.evalData,encoding='unicode',pretty_print=True)
    def cleanupPages(self,lPages):
        """
            delete existing tables
        """
        [p.getNode().remove(t.getNode()) for p in  lPages for t in  p.getAllNamedObjects(XMLDSTABLEClass)] 
    
    def run(self,doc):
        """
            for a set of pages, associate each page with several vertical zones  aka column-like elements
            Populate the vertical zones with page elements (text)

            indicate if bigram page template (mirrored pages)
             
        """
        if self.bCreateRef:
            if self.do2DS:
                dsconv = primaAnalysis()
                doc = dsconv.convert2DS(doc,self.docid)
            
            refdoc = self.createRefGeo(doc)
            return refdoc
        
        elif self.bCreateRefCluster:
            if self.do2DS:
                dsconv = primaAnalysis()
                doc = dsconv.convert2DS(doc,self.docid)
            
            refdoc = self.createRefCluster(doc)            
            return refdoc
        
        elif self.bManual:
#             self.tagWithTemplate(self.manualPattern,self.lPages)
            self.THNUMERICAL = 30
#             level=XMLDSTEXTClass
            self.minePageVerticalFeature(self.lPages, ['x','x2'],level=self.sTag)
            self.processWithTemplate(self.manualPattern,self.lPages)
            return self.doc 

        else:
            if self.do2DS:
                dsconv = primaAnalysis()
                doc = dsconv.convert2DS(doc,self.docid)
            chronoOn()
            self.ODoc = XMLDSDocument()
            self.ODoc.loadFromDom(doc,listPages = range(self.firstPage,self.lastPage+1))
            # first mine page size!!
            ## if width is not the 'same' , then  initial values are not comparable (x-end-ofpage)
            self.lPages = self.ODoc.getPages() 
            lSubPagesList = self.highLevelSegmentation(self.lPages)
            lTemplates = self.iterativeProcessVSegmentation(lSubPagesList)
#             print( 'chronoprocessing: ', chronoOff())
            self.addTagProcessToMetadata(doc)
#             doc = self.createRefGeo(doc)
#             doc = self.createRefCluster(doc)
            
            return doc #, lTemplates, self.lPages
        

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
    
    
    docM = tableTemplateSep()

    #prepare for the parsing of the command line
    docM.createCommandLineParser()
    docM.add_option("-f", "--first", dest="first", action="store", type="int", help="first page number", metavar="NN")
    docM.add_option("-l", "--last", dest="last", action="store", type="int", help="last page number", metavar="NN")
    docM.add_option("-t", "--tag", dest="tag", action="store", type="string", help="tag level", metavar="S")
    docM.add_option("--pattern", dest="pattern", action="store", type="string", help="pattern to be applied", metavar="[]")
    docM.add_option("--TH", dest="THNUM", action="store", type="int", help="TH as eq delta", metavar="NN")
    docM.add_option("--KTH", dest="KLEENETH", action="store", type="float", help="TH for sequentiality", metavar="NN")
    docM.add_option("--createref", dest="createref", action="store_true", default=False, help="create REF file (GEO) for component")
    docM.add_option("--createrefC", dest="createrefCluster", action="store_true", default=False, help="create REF file for component (cluster of textlines)")
    docM.add_option("--evalC", dest="evalCluster", action="store_true", default=False, help="evaluation using clusters (of textlines)")
    docM.add_option("--dsconv", dest="dsconv", action="store_true", default=False, help="convert page format to DS")
    docM.add_option("--glength", dest="glength", action="store",type='int', default=100, help="minimal length of a graphical element")
        
    #parse the command line
    dParams, args = docM.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    docM.setParams(dParams)
    
    doc = docM.loadDom()
    docM.loadDSDoc(doc)
#     docM.bDebug = True
    doc = docM.run(doc)
    
    if doc and docM.getOutputFileName() != "-":
        docM.writeDom(doc, True)

        
