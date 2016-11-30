

class sequenceAPI():
    
    def __init__(self):
        self._featureFunction = None
        self._lBasicFeatures = None
        self._featureFunctionTH = None
        
        self.objectClass = None
        self._lFeatureList= None
        
    def __hash__(self):
        return hash(self.getSetofFeatures())
    
    def __repr__(self):
#         return str(len(self._node.getContent())) + " " + self._node.getContent().encode('utf-8').strip()[:20]
        try:
            return len(self._node.getContent()) , " [" + self._node.name+self._node.getContent()[:20]+']'
        except AttributeError:
            # no node: use features?
            return str(self.getSetofFeatures())

    def resetFeatures(self):
        ## assume structure define elsewhere
        self.setStructures([])
        self._lBasicFeatures = None           
    
        
        
    def setFeatureFunction(self,foo,TH=5,lFeatureList = None,myLevel=None):
        """
            select featureFunction that have to be used
            
        """
        self._featureFunction = foo
        self._featureFunctionTH=TH    
        self._lFeatureList=lFeatureList
        self._subObjects = myLevel

    def computeSetofFeatures(self,TH=90):
        """
        
            for fuzzy matching: getSetofDegradedFeatures() ??
        """
        
        from feature import sequenceOfFeatures
        try:
            self._lBasicFeatures
        except AttributeError:
            self._lBasicFeatures=None
            
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR

        self._featureFunction(self._featureFunctionTH,self._lFeatureList,self._subObjects)
      
      
      
    def getSetofFeatures(self,bAll=False):
        """
            skeleton
        """
        from feature import sequenceOfFeatures
        
        if self._lBasicFeatures and len(self._lBasicFeatures.getSequences()) > 0:
            lR=sequenceOfFeatures()
            for f in self._lBasicFeatures.getSequences():
                if f.isAvailable():
                    lR.addFeature(f)
            return lR
        x= sequenceOfFeatures()
        
        
        return x                