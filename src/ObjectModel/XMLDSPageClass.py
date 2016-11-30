# -*- coding: latin-1 -*-
"""

    XML page object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for page object from a XMLDocument

"""

from XMLDSObjectClass import XMLDSObjectClass
from config import ds_xml_def as ds_xml
from XMLDSLINEClass import XMLDSLINEClass
from XMLDSTEXTClass import XMLDSTEXTClass
from XMLDSBASELINEClass import XMLDSBASELINEClass

class  XMLDSPageClass(XMLDSObjectClass):
    """
        PAGE class
        
        or image class ?????
        
    """
    orderID = 0
    name = ds_xml.sPAGE
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
        self._number = None
        
        self._BB = None
        
        
        ### attributes for building vertical zones in the page
        self._VX1Info = []
        self._VX2Info = []
        self._VXCInfo = []
        self._VWInfo = []
        
        self._X1X2 = []
        self._fXcuts = []

        
        
        ## list of vertical page zones Templates
        # by default: one column Template
        self._verticalZoneTemplates = []
        
        self.lVSeparator = {}
        # INIT: emptyfeatures?
        self.lVSeparator['INIT']=[]
                  
        # index them by verticalTemplates?? 
        ## or add attribtute to the region -> region.myTemplate=..
        self.lVerticalObjects={}
        self.lVerticalObjects['INIT'] = [self]
        
        
    
    def __repr__(self):
        return "%s %s %d" % (self.getName(),self.getNumber(), len(self.getObjects()))
    
    def getNumber(self): return  self._number
    def setNumber(self,n): self._number = n
    
    def getBB(self): return self._BB
    def setBB(self,b): self._BB = b
    
    
    def getX(self): return 0
    def getY(self): return 0
    def getX2(self):
        return self.getWidth()
    def getY2(self): return self.getHeight()
    
    def fromDom(self,domNode,lEltNames):
        """
            if lEltName empty = take all children.
            
            if several objects? see old pageClass?
        """
        
        # must be PAGE        
        self._name = domNode.name
        self.setNode(domNode)
        # get properties
        prop = domNode.properties
        while prop:
            self.addAttribute(prop.name,prop.getContent())
            # add attributes
            prop = prop.next
            
        ctxt = domNode.doc.xpathNewContext()
        ctxt.setContextNode(domNode)
        ldomElts = ctxt.xpathEval('./*')
        ctxt.xpathFreeContext()
        for elt in ldomElts:
            if elt.name in lEltNames:
                if elt.name  == XMLDSLINEClass.name:
                    myObject= XMLDSLINEClass(elt)
                    # per type?
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)
                elif elt.name  == XMLDSTEXTClass.name:
                    myObject= XMLDSTEXTClass(elt)
                    # per type?
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)
#                     print myObject,myObject.getObjects()
                elif elt.name == XMLDSBASELINEClass.name:
                    myObject= XMLDSBASELINEClass(elt)
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)                    
                else:
                    myObject= XMLDSObjectClass()
                    myObject.setNode(elt)
                    # per type?
                    self.addObject(myObject)
                    myObject.setPage(self)
                    myObject.fromDom(elt)




    def addVerticalObject(self,Template,o): 
        try:self.lVerticalObjects[Template]
        except KeyError: self.lVerticalObjects[Template]=[]
        
        if o not in self.lVerticalObjects[Template]:
            self.lVerticalObjects[Template].append(o)
        # store in self.getObjects()?????
        
    def getVerticalObjects(self,Template):
        """
            for a given template , get the objects of the page
        """
        return self.lVerticalObjects[Template]
    
    
    def setVerticalTemplates(self,lvm):
        self._verticalZoneTemplates = lvm
        
    def addVerticalTemplate(self,vm): 
        if vm not in self._verticalZoneTemplates:
            self._verticalZoneTemplates.append(vm)
        self.lVSeparator[str(vm)]=[]
        
    def getVerticalTemplates(self): return self._verticalZoneTemplates
    
    def getVX1Info(self): return self._VX1Info
    def setVX1Info(self,lInfo):
        """
            list of X1 features for the H structure of the page
                corresponds to information to segment the page vertically
        """
        self._VX1Info = lInfo

    def getVX2Info(self): return self._VX2Info
    def setVX2Info(self,lInfo):
        """
            list of X1 features for the H structure of the page
                corresponds to information to segment the page vertically
        """
        self._VX2Info = lInfo

    def getX1X2(self): return self._X1X2
    def setX1X2(self,x): self._X1X2.append(x) 

    def getVWInfo(self): return self._VWInfo
    def setVWInfo(self,lInfo):
        self._VWInfo = lInfo

    def getVXCInfo(self): return self._VXCInfo
    def setVXCInfo(self,lInfo):
        self._VXCInfo = lInfo        


    def addBoundingBox(self,lElts=None):
        """
            XMLDSObjectClass level?
        """
        if self.getBB() is not None:
            return self.getBB()
        
        minbx = 9e9
        minby = 9e9
        maxbx = 0
        maxby = 0
        
        if lElts is None:
            lElts= self.getAllNamedObjects(XMLDSTEXTClass)
        
        for elt in lElts:
#            if len(elt.getContent()) > 0 and elt.getHeight()>4 and elt.getWidth()>4:
                if elt.getX()>=0 and elt.getX() < minbx: minbx = elt.getX()
                if elt.getY()>=0 and elt.getY() < minby: minby = elt.getY()
                if elt.getX() + elt.getWidth() > maxbx: maxbx = elt.getX() + elt.getWidth()
                if elt.getY() + elt.getHeight()  > maxby: maxby = elt.getY() + elt.getHeight()
        
        self._BB = [minbx,minby,maxby-minby,maxbx-minbx]
        
        return minbx,minby,maxby-minby,maxbx-minbx         
         
           
         
    #### SEGMENTATION
    
    def createVerticalZones(self,Template):
        """
           Create 'columns' according to X cut
               Object
        """
        #reinit lVerticalObjects
        self.lVerticalObjects[Template]=[]
        for lseg in self.lVSeparator[Template]:
            prevCut=0
            for xcut in lseg: 
                region=XMLDSObjectClass()
                region.addAttribute('x', 0)
                region.addAttribute('y', 0)
                region.addAttribute('height', self.getAttribute('height'))
                region.addAttribute('width', str(xcut.getValue()-prevCut))
                prevcut=xcut.getValue()
    #             print        region.getX(),region.getY(),region.getWidth(),region.getHeight()
    #             print page.getAttributes(), page.getX2()
                lObjects = self.clipMe(region,self.getObjects())
                print lObjects
                region.setObjectsList(lObjects)
                self.addVerticalObject(Template,region)               
    
    
    
    ############# FEATURING #######    

    def getSetOfMigratedFeatures(self,TH,lInitialFeatures,myObject):
        """
            lInitialFeatures is produced at a different levels: TEXT
                -> need to 'migrate' fi into sef level :
            
        """
        from feature import featureObject,sequenceOfFeatures, emptyFeatureObject
     
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR     
   
        else:
            pass
        
        lFeatures = []
        for oldfi in lInitialFeatures:
            fi = featureObject()
#             fi.setName(oldfi.getName())
            fi.setName('x')
            fi.setValue(oldfi.getValue())
            fi.setTH(oldfi.getTH())
            fi.addNode(self)
            fi.setType(oldfi.getType())
            fi.setObjectName(self)
            if fi not in lFeatures:
                lFeatures.append(fi)

        seqOfF = sequenceOfFeatures()
        for f in lFeatures:
            seqOfF.addFeature(f)
        self._lBasicFeatures=seqOfF
        return seqOfF 
        
    def getSetOfMutliValuedFeatures(self,TH,lMyFeatures,myObject):
        """
            define a multivalued features 
        """
        from feature import multiValueFeatureObject,sequenceOfFeatures, emptyFeatureObject
     
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR     
   
        else:

            lFeatures=[]
            mv =multiValueFeatureObject()
            name= "multi" #'|'.join(i.getName() for i in lMyFeatures)
            mv.setName(name)
            mv.addNode(self)
            mv.setObjectName(self)
            mv.setTH(TH)
            mv.setObjectName(self)
            mv.setValue(map(lambda x:x,lMyFeatures))
            mv.setType(multiValueFeatureObject.COMPLEX)
            lFeatures.append(mv)

            
        if lFeatures == []:
            feature = multiValueFeatureObject()
            feature.setName('EMPTY')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(True)
            feature.setType(verticalZonefeatureObject.BOOLEAN)
            lFeatures.append(feature)            

        seqOfF = sequenceOfFeatures()
        for f in lFeatures:
            seqOfF.addFeature(f)
        self._lBasicFeatures=seqOfF
        return seqOfF                  
    
    def getSetOfVInfoFeatures(self,TH,lAttributes,myObject):
        """
            
        """
        from feature import featureObject,sequenceOfFeatures, emptyFeatureObject
     
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR     
   
        else:

            lFeatures = []
            for attr in lAttributes:
#                 print attr
                name= attr[0].getName()
                value = attr[0].getValue()
#                 if value > 2:
                feature = featureObject()
                feature.setName(name)
                feature.setTH(TH)
                feature.addNode(self)
                feature.setObjectName(self)
                feature.setValue(value)
                feature.setType(feature.NUMERICAL)
                if feature not in lFeatures:
                    lFeatures.append(feature)
      
            
        if lFeatures == []:
            feature = featureObject()
            feature.setName('EMPTY')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(True)
            feature.setType(featureObject.BOOLEAN)
            lFeatures.append(feature)            

        seqOfF = sequenceOfFeatures()
        for f in lFeatures:
            seqOfF.addFeature(f)
        self._lBasicFeatures=seqOfF
        return seqOfF          
        
    def getSetOfFeaturesPageSize(self,TH,lAttributes,myObject):
        """
            features: BB X Y H W
            
        """
        from feature import featureObject,sequenceOfFeatures, emptyFeatureObject
     
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR     
   
        else:


            lFeatures = []
            feature = featureObject()
            feature.setName('h')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self.getName())
            feature.setValue(round(float(self.getAttribute('height'))))
            feature.setType(feature.NUMERICAL)
            lFeatures.append(feature)

            feature = featureObject()
            feature.setName('w')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self.getName())
            feature.setValue(round(float(self.getAttribute('width'))))
            feature.setType(feature.NUMERICAL)
            lFeatures.append(feature)
      
            
        if lFeatures == []:
            feature = featureObject()
            feature.setName('EMPTY')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self.getName())
            feature.setValue(True)
            feature.setType(featureObject.BOOLEAN)
            lFeatures.append(feature)            

        seqOfF = sequenceOfFeatures()
        for f in lFeatures:
            seqOfF.addFeature(f)
        self._lBasicFeatures=seqOfF
        return seqOfF        
                 
        

    def getSetOfFeaturesBB(self,TH,lAttributes,myObject):
        """
            features: BB X Y H W
            
        """
        from feature import featureObject,sequenceOfFeatures, emptyFeatureObject
     
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR     
   
        else:

            #build BB 
            if self.getBB() is None:
                self.addBoundingBox()
            x,y,h,w = self.getBB()

            lFeatures = []

                
            feature = featureObject()
            feature.setName('lm')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self.getName())
            feature.setValue(round(x))
            feature.setType(feature.NUMERICAL)
            lFeatures.append(feature)

            feature = featureObject()
            feature.setName('rm')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self.getName())
            feature.setValue(round(x+w))
            feature.setType(feature.NUMERICAL)
            lFeatures.append(feature)
      
            
        if lFeatures == []:
            feature = featureObject()
            feature.setName('EMPTY')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self.getName())
            feature.setValue(True)
            feature.setType(featureObject.BOOLEAN)
            lFeatures.append(feature)            

        seqOfF = sequenceOfFeatures()
        for f in lFeatures:
            seqOfF.addFeature(f)
        self._lBasicFeatures=seqOfF
        return seqOfF        
        
        
    def getSetOfListedAttributes(self,TH,lAttributes,myObject):
        """
            Generate a set of features: X start of the lines
            
            
        """
        from feature import featureObject,sequenceOfFeatures, emptyFeatureObject
     
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR     
   
        else:
            lHisto={}
            for elt in self.getAllNamedObjects(myObject):
                if float(elt.getAttribute('width')) > 00:
                    for attr in lAttributes:
                        try:lHisto[attr]
                        except KeyError:lHisto[attr] = {}
                        if elt.hasAttribute(attr):
                            try:lHisto[attr][round(float(elt.getAttribute(attr)))].append(elt)
                            except: lHisto[attr][round(float(elt.getAttribute(attr)))] = [elt]
            
            lFeatures = []
            for attr in lAttributes:
                for value in lHisto[attr]:
                    if  len(lHisto[attr][value]) > 0.1:
                        ftype= featureObject.NUMERICAL
                        feature = featureObject()
                        feature.setName(attr)
                        feature.setTH(TH)
                        feature.addNode(self)
                        feature.setObjectName(self)
                        feature.setValue(value)
                        feature.setType(ftype)
                        lFeatures.append(feature)
        
        if lFeatures == []:
            feature = featureObject()
            feature.setName('EMPTY')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(True)
            feature.setType(featureObject.BOOLEAN)
            lFeatures.append(feature)            

        seqOfF = sequenceOfFeatures()
        for f in lFeatures:
            seqOfF.addFeature(f)
        self._lBasicFeatures=seqOfF
        return seqOfF