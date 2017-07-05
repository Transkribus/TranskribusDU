# -*- coding: utf-8 -*-
"""

    Record Class
    Hervé Déjean
    cpy Xerox 2017

    READ project 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.    

"""

import cPickle
import regex


class recordClass(object):
    """
        a record
        
        should be able to perform inference
        should be linked to an ontology + rules
    """
    
    def __init__(self,name=None):
        self._name =name
        # list of fields
        self._lFields = []

        # link to the document : a (list of) documentObject where the record occurs
        self._location = None        
        
        ## link to extaction: list of candidates
        self._lcandidates = []
    
    
    def __eq__(self,o):
        """
            same fields and same values
        """
        return self.getName() == o.getName() and map(lambda x:x.getName(),self.getFields()) == map(lambda x:x.getName(),o.getFields()) 
        
    def getName(self): return self._name
    def setName(self,s): self._name =s
    def getFields(self): return self._lFields
    def addField(self,field, mandatory=False):
        if field not in self.getFields():
            self.getFields().append(field)
    
    def getFieldByName(self,name):
        for f in self.getFields():
            if f.getName() == name: return f
        return None
            
    def setDocumentElement(self,docElt): self._location = docElt
    def getDocumentElement(self): return self._location
    
    #specifc example for table
    def propagate(self,table):
        """
            field/value propagation
            Assuming: field tagging done:
            
        """
        #for any element in docobject (recursively)
        
        for curCell in table.getCells():
            for field in curCell.getFields():
                #propagate to the right for all cells of the row
                for cell in table.getRows()[curCell.getIndex()[0]].getCells()[curCell.getIndex()[1]:]:
                    cell.addField(field,field.getValue)
                
        table.displayPerRow()
            
    def genericPropagate(self,docObject):
        """
            for a given la object:
                for each of its fields:
                    propagate to the objects of the laobject
                    #issue: for a cell the propagation has to be done at the row/col level
                    OR table subobject: rows and cols
                    
                    
            How to del with ABP dates  in centered row : table split?
            How to generate several alternatives? use of different scopes? how to store: duplicate docobject? 
            
        """
    def addCandidate(self,cnd):
        if cnd not in self.getCandidates():
            self._lcandidates.append(cnd)
    
    def getCandidates(self):return self._lcandidates
        
            
    def scoreCandidat(self,cnd):
        """
            sum of the valued fields
            negative score if mantadary fields are not present 
            
            if a candidate has several values for a field: use LA distance, score for find the right one? 
            
        """
        docElt, lFields = cnd
        score=0
        for field in self.getFields():
            if field in lFields:
                score+=1
                if field.isMandatory():
                    score+=1
            elif field.isMandatory():
                score -=1
        
        return score
                
                
    def rankCandidates(self):
        """
            score each candidate and scor them
        """
        
    
    
class fieldClass(object):
    """
         a record field
    """
    def __init__(self,name=None):
        self._name = name
        # allow for muti-value, range,...
        self._value = None
        
        # backref to record
        self._myrecord = None
        
        # how to extract textual representation
        self._lTaggers = []
        
        # C side
        self._object        = None      # reference to the document object of the field (list?)
        self._lCndValues    = None      # list of candidate values
        self._offsets       = None      # (xstart, xend) of the string in the parsed content
        
        # link to the document : a (list of) documentObject
        # where the record occurs
        self._location = None


    def __eq__(self,o): return str(self) == str(o)
    
    def __str__(self):return "%s-%s"%(self.getName(),self.getValue())        
    def __repr__(self):  return "%s-%s"%(self.getName(),self.getValue())        
    
    def getName(self): return self._name
    def setName(self,s): self._name =s        

    def getValue(self): return self._value
    def setValue(self,v): self._value = v   
    
    def addTagger(self,t): 
        self._lTaggers.append(t)
    def getTaggers(self):return self._lTaggers
    
    def applyTaggers(self,o):
        lres=[]
        for t in self.getTaggers():
            res= t.runMe(o)
            if res:
                lres.append([self.getName()]+res)
        return lres
    
    def cloneMe(self):
        clone = self.__class__()
        clone.setName(self.getName())
        clone.setValue(self.getValue())
        clone._lTaggers = self._lTaggers
        
        return clone 

class taggerClass(object):
    """
        a tagger: tag a string with a field  (one field?)
        
        
        a mapping function is required when the tagger does more than needed (more ne than the one to be extracted)
    """
    FSTTYPE =   0               # requires path/myressource.fst 
    MLCRFTYPE  =   1            # requires path to modeldirectory     
    EXTERNALTAGGERTYPE = 2      # path to exe ; assume text as input
    
    def __init__(self,name=None):
        self._name = name
        self._lFields = None
        
        # tagger type: how to apply the tagger
        self._typeOfTagger = None
        
        
        ## tagger type paramater: needed resources
        self._path = None
        
        self._externalTagger = None
    def getName(self): return self._name
    def setName(self,s): self._name =s    
    
        
    def runMe(self,documentObject):
        """
            tag s
        """ 
        #exceution
        # get output and map into the fields list
        if self._typeOfTagger == taggerClass.FSTTYPE:
            return self.runRE(documentObject)
        elif self._typeOfTagger == taggerClass.MLCRFTYPE:
            return self.runCRF(documentObject)
        else: 
            raise "SOFTWARE ERROR: your component must define a taggerType"
        
    def runFST(self,documentObject):
        """
         apply fst to s
        """
        
    def runCRF(self,documentObject):
        """
            apply CRF
        """
    
    def runExternalTagger(self,documentObject):
        """
            apply external tagger
        """    
        
    
    
    
    
    
class RETaggerClass(taggerClass):
    
    def __init__(self,name='tagger'):
        taggerClass.__init__(self, name)
        self._lressources = None
        
    def loadRessources(self,lfilenames):
        """
            Open and read ressoruce files
            
        """
        
        self._lressources=[]
        for filename in lfilenames:
            lre,ln=cPickle.load(open(filename,'r'))
            self._lressources.append((filename,lre,ln))
        
        return self._lressources
    
    def runMe(self,documentObject):
        """
            return what? tag and offset in documentObject.getContent()?
        """  
        
        # contains: token + score
        lParsingRes = []
        
        txt = documentObject.getContent()
        ltokens = txt.split()
        for _, lreNames,lnames in self._lressources:
            for token in ltokens:
                for i,myre in enumerate(lreNames):
                    xx = myre.fullmatch(token)
                    if xx:
                        if len(token) >1 and sum(xx.fuzzy_counts) <3:
                            lParsingRes.append((token,xx.fuzzy_counts,lnames[i][0],lnames[i][1]))
#                             print token,_,xx.fuzzy_counts,lnames[i]      
                            
        return lParsingRes  
    
    
class dateRETaggerClass(RETaggerClass):
    """
        class for recognizing date 
    """
    def __init__(self,name='datetagger'):
        RETaggerClass.__init__(self, name)
        lmonth=['januar', 'februar', 'mars' ,'april','mai','juni','juli','august','october','november','december']
        lReNames=[]
        lNames=[]
        for m in lmonth:
            lReNames.append(regex.compile(r"(%s){e<=3}"%regex.escape(m), regex.IGNORECASE))
            lNames.append((m,1))
        self._lressources = [('REDate',lReNames, lNames)]
        
    
