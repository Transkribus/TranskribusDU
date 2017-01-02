# -*- coding: utf-8 -*-
""" 
    H. Déjean
    copyright Xerox 2016
    
    READ project 
    
"""

class sequenceOfFeatures(object):
    
    id = 0
    def __init__(self):
        
        self._lFeatures = []
        self._id = sequenceOfFeatures.id
        sequenceOfFeatures.id += 1
    
        # list of list of nodes
        self._lnodes = []
        self._flatListofNodes = []
        
        self._hash = None
        
        self._contiguousSeq = 0
        self._freq = 0
        
#    def __ne__(self,other):
#        if other == None:
#            return False
#        if self.__eq__(other):
#            return False
#        return True
        
    def __eq__(self,other):
        if other:
            if len(other.getSequences()) != len(self.getSequences()):
                return False
            return other.getSequences() == self.getSequences()
        return False
        
#        if len(other.getSequences()) != len(self.getSequences()):
#            return False
#        for i,e in enumerate(other.getSequences()):
#            if e != self.getSequences()[i]:
#                return False
#        return  True
    
    def __hash__(self):
        if self._hash:
            return self._hash
        else:
            s = 1
            for f in self.getSequences():
                s += hash(f)
            self._hash = s
        return self._hash
    
    
    def __repr__(self):
        s="{"
        for x in self.getSequences():
            s += str(x)
        return s+"}"
        
    def addFeature(self,f):
        self._lFeatures.append(f)
        s = 1
        for f in self.getSequences():
            s += hash(f)
        self._hash = s        
        
    def getNodes(self): return self._lnodes
    def getFlatNodes(self): return self._flatListofNodes
    def addSeqofNodes(self,s):
        for x in s: 
            if x not in self._flatListofNodes:
                self._flatListofNodes.append(x) 
        self._lnodes.append(s)
        
    def getTwiceSequence(self):
        twice = sequenceOfFeatures()
        for f in self.getSequences()+self.getSequences():
            twice.addFeature(f)
        return twice
            
    def getLen(self): return len(self.getSequences())        
    
    def setUniSequences(self,s):
        self._lFeatures = [s]
    def getSequences(self):
        return self._lFeatures
    
    def deleteFeature(self,f):
        f.setNonAvailable()
        try: self.getSequences().remove(f)
        except:pass
        
    def updateFeature(self,f):
        
        for i,myf in enumerate(self.getSequences()):
            if f == myf:
                myf.storeOldValue(myf.getValue())
                myf.setValue(f.getValue())
                
    
    def isEmptyFeatureSequence(self):
#        print filter(lambda x:x.getClassName()=='emptyFeatureObject',self.getSequences())
        return len(filter(lambda x:x.getClassName()=='emptyFeatureObject',self.getSequences())) == len(self.getSequences())
       
    def isMostlyEmptyFeatureSequence(self):
        return len(filter(lambda x:x.getClassName()=='emptyFeatureObject',self.getSequences())) >0.5* len(self.getSequences())

       
    # used?
    def incrementContiguousSequence(self):
        self._contiguousSeq += 1
        
    def getContiguous(self): return self._contiguousSeq
    def getFrequency(self): return self._freq
    def incrementFreq(self):
        self._freq += 1
        
        
        
    def isSingleton(self,th=2):
        """
            unigram: how to detect elements what are not part of bigrams, unigrams?
        """
    def isBigram(self,th=2):
        """
            bigram: typical example: 2-column galley
        """
        return self._contiguousSeq >= th  and (self._contiguousSeq >= 0.750 * (self._freq - self._contiguousSeq) )


    def isKleenePlus(self,th=2,th2=0.75):
        ## size of the longest covered sequence >= TH  TH=3
        # th2 for proportion?
        # replace freq by len(self._flatListofNodes)???   =>depend on type of features?
#        return  self._contiguousSeq >= th
        
        return self._contiguousSeq >= th  and (self._contiguousSeq >= th2 * (self._freq - self._contiguousSeq) )
     
    def getScore(self):
        """
            to rank seq to start with the "best" ones.
             freq 
             freq * nbelt?   => then for unigrams, the 2,3,4 grams are overscored (n-grams overlap each over) 
             try to find the longest covered sequence
             
        """
        
        return len(self._flatListofNodes)
        ### ---------------
        if len(set(self.getSequences()))== 1:
            return self.getFrequency() #* self.getLen()
        else:
            return self.getFrequency() #* self.getLen()


    def getLongestSequence(self):
        """
            is the information enough at feature level?
        """
    
    def matchButOne(self,s,f):
        """
            Does s match self minus f
            assumption: f in self
        """
        if self == s:
            return -1
        if len(s.getSequences()) != len(self.getSequences()):
            return -1
        pos = -1
        for i,f2 in enumerate(s.getSequences()):
            if f2 == self.getSequences()[i]:
                pass
#                print "\t\t = ",f2,self.getSequences()[i],self.getSequences()[i] != f
            elif self.getSequences()[i] != f:
                    return -1
            elif self.getSequences()[i] == f:
#                print "\t\t pos =",i
                pos = i
            else:
                return -1
        return pos
            
        
    def generateCModel(self):
        """
            
        """

    
class  featureObject(object):
    
    """
        a feature created from an element (elt terminal) 
        a feature created from a signature (Node())
            corresponds to the signature
        
        
        a feature has a type
        the type defines the way the feature is compared
        as well as the fuzzy matching
        
        TYPE1 : comp1, fuzzydistance1
        TYPE2: comp2, fuzzydistance1
        
        
        
        
    """
    NUMERICAL       = 0 # use 
    BOOLEAN         = 1       #   use instead a numerical value ?? 
    EDITDISTANCE    = 2 # textual value
    COMPLEX         = 3       # signature
    PATTERN         = 4      # for content
    DISTANCE        = 5    # two dimension feature [x,y] 
    OBJECT          = 6    # programmatic object with __eq__  (and __hash__)
    ##GRAPHICAL LINES
    ## IMAGES   : ideally, pixel level
    #font: ? fuzzy with bold, italic, fontname, color ...???  -> use traditional font familty tree (serif, sans serif)
    
    
    id  = 0
    
    def __init__(self):
        import math
        ##  feature._value can be a list 
        self._objectName = None ### associate to a document object 
        self._type = None  # numerical 
        
        self._featureType = None   # content, zone, typo, color, graphic, images
        
        self._modelObject = None   # points to its model (at _featureType level)
        
        
        self._name = None
        self._value = None
        self._oldValue= None
        self._id = featureObject.id
        featureObject.id += 1
        self._TH = 0.5
        
        ##12/08/2016
        self._weight = 1.0
        
        
        # take care: refers to the lnde of the canonical feature (most frequent nearest feature)
        # for canonical feature: list of nodes of all nearest features
        self._lnodes=[]
        
        self._bAvailable = True
        self._element = None # for fuzzy matching

        self.hash = None
    
        self._canonical  = self
    
        
        # sequence ??? YES point to its sequence (n-grams)
        self._seqOFFeat = None
    
    
    def __hash__(self):
        return hash((self.getName(),self.getStringValue()))
            
    def __repr__(self):
        return  "'%s=%s'" % (self.getName(),self.getStringValue())
    
    def getID(self): return self._id
    def getClassName(self): return self.__class__.__name__
    def isAvailable(self):
        return  self._bAvailable
    def setAvailable(self): self._bAvailable = True
    def setNonAvailable(self): self._bAvailable = False

    def getWeight(self): return self._weight
    def setWeight(self,w): self._weight = w
    

    def matchLCS(self,perc, t1, t2):
        (s1, n1) = t1
        (s2, n2) = t2
    
        nmax = max(n1, n2)
        nmin = min(n1, n2)
        
        if nmax <=0: return False,0
        #cases that obviously fail
        if nmin < (nmax * perc / 100.0 ): return False, 0
    
        #LCS
        n = self.lcs(s1, s2)
        val = round(100.0 * n / nmax)
        return ((val >= perc), val)
    
        
    def setSequence(self,s):
        self._seqOFFeat = s
    def getSequence(self): return self._seqOFFeat
    
    #--------- LCS code
    # Return the length of the longest common string of a and b.
    def lcs(self,a, b):
    
        na, nb = len(a), len(b)
    
        #switch a and b if b is shorter
        if nb < na:
            a, na, b, nb = b, nb, a, na
    
        curRow = [0]*(na+1)
    
        for i in range(nb):
            prevRow, curRow = curRow, [0]*(na+1)
            for j in range(na):
                if b[i] == a[j]:
                    curLcs = max(1+prevRow[j], prevRow[j+1], curRow[j])
                else:
                    curLcs = max(prevRow[j+1], curRow[j])
                curRow[j+1] = curLcs
        return curRow[na] 

    def __ne__(self,other):
        return not (self == other) 
   
    def __eq__(self,other):
        """
            + _objectName?
        """
        try:other.getClassName()
        except:return False
        
        if self.getClassName() != other.getClassName():
            return False
        if self.getName() == other.getName():
#             print self.getName() , other.getName()
            if self.getType() == other.getType():
                distance = 9e9
                if self._type == featureObject.NUMERICAL:
                    try:
                        distance = abs(float(self.getValue()) - float(other.getValue()))
                    except:
                        pass
#                         print self, self._type
#                        print other,other.getType()
#                     print self.getName() , other.getName(), self.getValue(), other.getValue(),distance, self.getTH(),distance < self.getTH()
                    return distance <= self.getTH()
                    # must be <= when distance == 0 ! abd TH=0
                elif self._type == featureObject.EDITDISTANCE:
                    # EDITDISTANCE
                    if self.getValue() == other.getValue():
                        return True
                    else:
#                        try:
                            bNear,val = self.matchLCS(self.getTH(), (self.getValue(),len(self.getValue())), (other.getValue(),len(other.getValue())))
                            return bNear #self.getValue() == other.getValue()
#                        except:
#                            print self.getValue(),other.getValue()
#                            ddd
                elif self._type == featureObject.COMPLEX:
                    return self.getValue() == other.getValue()
                elif self._type == featureObject.BOOLEAN:
                    return self.getValue() == other.getValue()
                return False
                
        
        return False
       
       
     
#     def __repr__(self):  return  "(%s=%s %s)" % (self.getName(),self.getValue(),self.getType())
    
    
    def getName(self): return self._name
    def setName(self,n): self._name  = n
 
    def getType(self): return self._type
    def setType(self,t): self._type  = t
    
    
    def setCanonical(self,f): self._canonical = f
    def getCanonical(self): return self._canonical
    
    def getValue(self): return self._value
    def getStringValue(self): 
        try:
            float(self._value)
            return str(self.getValue())
        except TypeError,ValueError:
            try:
                return str(self.getValue().encode('utf-8'))
            except AttributeError:
                return str(self.getValue())

    
    def storeOldValue(self,v):
        self._oldValue= v
    def getOldValue(self): return self._oldValue
    
    def setValue(self,v):
        self._value  = v
        if self._oldValue is None:
            self._oldValue = v
        return
    
    def setTH(self,th): self._TH = th
    def getTH(self): return self._TH
           
           
    # specific object from which the feature was generated
    # can be different from nodes
    def setObjectName(self,o): self._objectName = o
    def getObjectName(self): return self._objectName
    
    def addNode(self,n):
        #for parsing
        self._lnodes.append(n)
#         if n not in self._lnodes:
#             self._lnodes.append(n)
 
    def getNodes(self): return self._lnodes
    
        

class multiValueFeatureObject(featureObject):
    """
    
        multivalue featureObject: define as a set of unitary features (of the same type: yes)
        TH: % of common values  
    """
    def __init__(self):
        
        featureObject.__init__(self)
        self._value = []
        self._TH= 1.0
            
    def getStringValue(self):
        if self.getName() == 'EMPTY':
            return 'EMPTY'
        if self.getType() == 1:
            return self.getName()
        return "|".join(map(lambda x:x.getStringValue(),self.getValue()))
            
    def __eq__(self,other):
        try: other.getClassName()
        except AttributeError:return False
        if self.getClassName() == other.getClassName():
            if self.getName() == other.getName():
                nbCommon = 0
                for x in self.getValue():
                    if x in other.getValue(): nbCommon +=1
#                 print self, other, self.getTH() , nbCommon,self.getTH() * len(self.getValue()),nbCommon >= ( self.getTH() * len(self.getValue()))
                return nbCommon >= ( self.getTH() * len(self.getValue())) #and self.getTH() * len(other.getValue()))
        return False
class emptyFeatureObject(featureObject):
    
    def __init__(self):
        featureObject.__init__(self)
    
    def __hash__(self):
        sH ="" 
        for s in "EMPTYFEATURE":
            sH += str(ord(s))
        return int(sH)        
     
    def __eq__(self,other):
        if other:
            return other.__class__.__name__ == "emptyFeatureObject"
        return False
    def getName(self): return "EMPTY"
    def __repr__(self):  return  "(EMPTY)"     
    
    

