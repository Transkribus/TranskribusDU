# -*- coding: latin-1 -*-
"""

    XML object class 
    
    Hervé Déjean
    cpy Xerox 2009
    
    a class for object from a XMLDocument

"""

from XMLDSObjectClass import XMLDSObjectClass
from config import ds_xml_def as ds_xml
from XMLDSTOKENClass import XMLDSTOKENClass


class  XMLDSTEXTClass(XMLDSObjectClass):
    """
        TEXT (chunk) class
    """
    name = ds_xml.sTEXT
    def __init__(self,domNode = None):
        XMLDSObjectClass.__init__(self)
        XMLDSObjectClass.id += 1
        self._domNode = domNode
        
        self.Obaseline=None
    
#     def getX(self): return float(self.getAttribute('x'))
#     def getY(self): return float(self.getAttribute('y'))
    def getX2(self):
        return float(self.getAttribute('x'))+self.getWidth()
    def getY2(self): 
        return float(self.getAttribute('y'))+self.getHeight()        
#     def getHeight(self): return float(self.getAttribute('height'))
#     def getWidth(self): return float(self.getAttribute('width'))
        
    def fromDom(self,domNode):
        """
            only contains TEXT?
            
            attributes x y  id height width  (all!)
            
        """
        # must be PAGE        
        self._name = domNode.name
        self.setNode(domNode)
        # get properties
        prop = domNode.properties
        while prop:
            self.addAttribute(prop.name,prop.getContent())
            prop = prop.next
        try: 
            self._id = self.getAttribute('id')
        except:pass
        self.addAttribute('x2', float(self.getAttribute('x'))+self.getWidth())
        self.addAttribute('y2',float(self.getAttribute('y'))+self.getHeight() )

        if self.hasAttribute('blpoints'): 
            from ObjectModel.XMLDSBASELINEClass import XMLDSBASELINEClass
            b= XMLDSBASELINEClass()
            b.fromDom(domNode)
            b.setParent(self.getParent())
            self.setBaseline(b)
#             print self, self.getBaseline()
        
        ctxt = domNode.doc.xpathNewContext()
        ctxt.setContextNode(domNode)        
        ldomElts = ctxt.xpathEval('./%s'%(ds_xml.sTOKEN))
        for elt in ldomElts:
            try:
                myObject= XMLDSTOKENClass(elt)
                self.addObject(myObject)
                myObject.setPage(self)
                myObject.fromDom(elt)
            except: pass #print 'issue with token'
        
        ctxt.xpathFreeContext()        
    
    def setBaseline(self,ob): self.Obaseline = ob
    def getBaseline(self):
        return self.Obaseline
    
    def computeBaseline(self):
        
        if self.getBaseline() is not None:
            return self.getBaseline()
            
#         lHisto={}
        lY=[]
        lX=[]
        
        # test if TOKEN has position (not in GT)!
        for token in self.getAllNamedObjects(XMLDSTOKENClass):
            try:
                lX.append(token.getX())
                lX.append(token.getX2())
                lY.append(token.getY2())
                lY.append(token.getY2())
            except  TypeError:
                pass            

        import numpy as np
        import libxml2
        
        if len(lX) > 0:
            a,b = np.polyfit(lX, lY, 1)
            
#             lPoints = ','.join(map(lambda (x,y):x,y,zip(lX,lY)))
            lPoints = ','.join(["%d,%d"%(xa,ya) for xa,ya  in zip(lX, lY)])
#             print lPoints
#             print '\t' , a,b
#             import math
#             print 'ANLGE:',math.degrees(math.atan(a))
            ymax = a*self.getWidth()+b        
#             self.baseline[1]= a
            from ObjectModel.XMLDSBASELINEClass import XMLDSBASELINEClass
            b= XMLDSBASELINEClass()
            b.setNode(self)
#             b.addAttribute("points",lPoints)
            b.setAngle(a)
            b.setPoints(lPoints)
            b.setParent(self.getParent())
            self.setBaseline(b)
            b.computePoints()
            
#             verticalSep  = libxml2.newNode('BASELINE')
#             verticalSep.setProp('points', '%f,%f,%f,%f'%(self.getX(),b,self.getX2(),ymax))
#             print self, self.baseline, lX, lY
#             self.getNode().addChild(verticalSep)
            
    
    def getSetOfFeaturesXPos(self,TH,lAttr,myObject):

        from feature import featureObject,sequenceOfFeatures, emptyFeatureObject
     
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR     
   
        else:
            
            lFeatures = []

            ftype= featureObject.NUMERICAL
            feature = featureObject()
            feature.setName('x')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(round(self.getX()))
            feature.setType(ftype)
            lFeatures.append(feature)
            
            ftype= featureObject.NUMERICAL
            feature = featureObject()
            feature.setName('x2')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(round(self.getX()+self.getWidth()))
            feature.setType(ftype)
            lFeatures.append(feature)  
                      
            ftype= featureObject.NUMERICAL
            feature = featureObject()
            feature.setName('xc')
            feature.setTH(TH)
            feature.addNode(self)
            feature.setObjectName(self)
            feature.setValue(round(self.getX()+self.getWidth()/2))
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
            
            lHisto = {}
            for elt in self.getAllNamedObjects(myObject):
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
        
        
            if 'text' in lAttributes:
                if len(self.getContent()):
                    ftype= featureObject.EDITDISTANCE
                    feature = featureObject()
                    feature.setName('content')
                    feature.setTH(90)
                    feature.addNode(self)
                    feature.setObjectName(self)
                    feature.setValue(self.getContent())
                    feature.setType(ftype)
                    lFeatures.append(feature)            
            
            if 'xc' in lAttributes:
                ftype= featureObject.NUMERICAL
                feature = featureObject()
                feature.setName('xc')
                feature.setTH(TH)
                feature.addNode(self)
                feature.setObjectName(self)
                feature.setValue(round(self.getX()+self.getWidth()/2))
                feature.setType(ftype)
                lFeatures.append(feature)    
#           
#             if 'x2' in lAttributes:
#                 ftype= featureObject.NUMERICAL
#                 feature = featureObject()
#                 feature.setName('x2')
#                 feature.setTH(TH)
#                 feature.addNode(self)
#                 feature.setObjectName(self)
#                 feature.setValue(round(self.getX()+self.getWidth()))
#                 feature.setType(ftype)
#                 lFeatures.append(feature)            
        
            if 'bl' in lAttributes:
                for next in self.next:
#                     print 'c\t',self, next
                    ftype= featureObject.NUMERICAL
                    feature = featureObject()
                    baseline = self.getBaseline()
                    nbl=next.getBaseline()
                    if baseline and nbl:
                        feature.setName('bl')
                        feature.setTH(TH)
                        feature.addNode(self)
                        feature.setObjectName(self)
#                         print nbl, baseline
                        feature.setValue(round(nbl.getY2()-baseline.getY2()))
    #                     feature.setValue(round(next.getY2()-self.getY2()))
                        feature.setType(ftype)
                        lFeatures.append(feature)                
                
            if 'linegrid' in lAttributes:
                #lgridlist.append((ystart,rowH, y1,yoverlap))
                for ystart,rowh,y1,yoverlpa  in self.lgridlist:
                    ftype= featureObject.BOOLEAN
                    feature = featureObject()
                    feature.setName('linegrid%s'%rowh)
                    feature.setTH(TH)
                    feature.addNode(self)
                    feature.setObjectName(self)
                    feature.setValue(ystart)
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
    
         


