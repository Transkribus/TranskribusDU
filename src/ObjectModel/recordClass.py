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

class recordClass(object):
    """
        a record
        
        should be able to perform inference
        should be linked to an ontology + rules
    """
    
    def __init__(self,name=None):
        self._name =name
        # list of fields
        self._lFields = None

        # link to the document : a (list of) documentObject where the record occurs
        self._location = None        
        
    def getName(self): return self._name
    def setName(self,s): self._name =s
    def getFields(self): return self._lFields()
    def addField(self,field):
        if field not in self.getFields():
            self.getFields().append(field)
            
            
            
            
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
        self._lTaggers = None
        
        # C side
        self._object        = None      # reference to the document object of the field (list?)
        self._lCndValues    = None      # list of candidat values
        self._offsets       = None      # (xstart,xend) of the string in the parsed content
        
        # link to the document : a (list of) documentObject
        # where the record occurs
        self._location = None
        
    def getName(self): return self._name
    def setName(self,s): self._name =s        

    def getValue(self): return self._value
    def setValue(self,v): self._value = v   
    
    def addTagger(self,t): self._lTaggers.append(t)
    
    
class taggerClass(object):
    """
        a tagger: tag a string with a field  (one field?)
    """
    FSTTYPE =   0               # requires path/myressource.fst + fieldmapping
    MLCRFTYPE  =   1            # requires path to modeldirectory + fieldmapping    
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
    
    def fieldMapping(self,lM):
        """
            map the tagger tags and the field names 
        """
        
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
        
    
    def getFields(self): return self._lFields()
    def addField(self,field):
        if field not in self.getFields():
            self.getFields().append(field)    
    
class RETaggerClass(taggerClass):
    
    def __init__(self):
        taggerClass.__init__(self, 'RETagger')
        self._lressources = None
        
    def loadRessources(self,lfilenames):
        """
            Open and read ressoruce files
            
        """
        
        lRes=[]
        for filename in lfilenames:
            lre,ln=cPickle.load(open(filename))
            lRes.append((filename,lre,ln))

        return lRes  
    
    def runMe(self,documentObject):
        """
            return what? tag and offset in documentObject.getContent()?
            return recordField with name, value and position in content 
        """  
        txt = documentObject.getContent()
        for rname, lreNames,lnames in self._lressources:
            for i,myre in enumerate(lreNames):
                xx = myre.fullmatch(txt)
                if xx:
                    if len(txt) > 4 and sum(xx.fuzzy_counts) <3:
                        print txt,rname,xx.fuzzy_counts,lnames[i]      
                            
                            