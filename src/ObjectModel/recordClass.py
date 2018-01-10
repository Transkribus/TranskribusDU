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
from __future__ import unicode_literals

import cPickle
# import regex
import gzip
#from keras.models import load_model
from contentProcessing.taggerTrainKeras2 import DeepTagger


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
        # mandatory fields
        self._lmandatoryFields = []
        
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
    def addField(self,field):
        if field not in self.getFields():
            field.setRecord(self)
            self.getFields().append(field)
            if field.isMandatory():
                self._lmandatoryFields.append(field)
    
    def getFieldByName(self,name):
        for f in self.getFields():
            if f.getName() == name: return f
        return None
            
    def setDocumentElement(self,docElt): self._location = docElt
    def getDocumentElement(self): return self._location
    
    def isComplete(self,cnd):
        """
            cnd has all mandatory fields with value
        """
        
        cpt=0
        for field in self.getFields():
            if field.isMandatory():
                if cnd.getFieldByName(field.getName()) is not None and cnd.getFieldByName(field.getName()).getValue() is not None:
                    cpt+=1
                
        return cpt == len(self._lmandatoryFields) 
        
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
            negative score if mandatory fields are not present 
            
            if a candidate has several values for a field: use LA distance, score for find the right one? 
            
            
            a candidate is a LAobject
        """
        score=0
        for field in self.getFields():
            # lookup cnd.objects as well!
            lObjects = [cnd]
            lObjects.extend(cnd.getObjects())
            for obj in lObjects:
                if  obj.getFieldByName(field.getName()) is not None and obj.getFieldByName(field.getName()).getValue() is not None:
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
            
        self.getCandidates().sort(key=lambda x:self.scoreCandidat(x),reverse=True)
#         for cand in self.getCandidates():
#             print cand, self.scoreCandidat(cand)        
    
    def display(self):
        """
            list candidates (and location)
            
            a candidate must be a LAobject 
        """
        for cand in self.getCandidates():
            # currently cell
            if len(cand.getContent()) > 5:
                print  cand.getContent().encode('utf-8')
                for field in cand.getFields():
                    if field.getBestValue() is not None:
                        print "f:",field.getName().encode('utf-8'), field.getBestValue().encode('utf-8'), field.isMandatory()
            
        
    def generateOutput(self):
        """
            serialization for output format 
        """
        raise "must be instanciated"
    
class fieldClass(object):
    """
         a record field
    """
    def __init__(self,name=None):
        self._name = name
        # allow for muti-value, range,...
        self._value = None
        
        # backref to record
        self._record = None
        
        self._bMandatory = False
        # how to extract textual representation
        self._lTaggers = []
        
        # label from taggers corresponding to this field
        self._lMapping = []
        
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
    
    def setRecord(self,r): self._record = r
    def getRecord(self):return self._record
    
    def setLabelMapping(self,l): self._lMapping = l
    
    def extractLabel(self,lres):
        """
            extract only lables in lMapping
            [ 
            (  [u'Theresia', u'irainger', u'Eder.', u'Baur.', u'im'], 
            [(u'firstNameGenerator', 0.99970764), (u'lastNameGenerator', 0.99180144), (u'locationGenerator', 0.99952209), (u'locationGenerator', 0.99951875), (u'locationGenerator', 0.60138053)])]
            
            
            cand= filter(lambda (tok,(label,score)):label.split('_')[1] in self._lMapping,zip(res[0],res[1]))
             
            
            new verson with BIES

            [((0, 1), u'Anna Maria', u'firstNameGenerator',[0.222,0.222]), ((2, 2), u'Stadler', u'lastNameGenerator',[0.22]), ((3, 3), 'MAx', u'firstNameGenerator'), ((4, 4), 'Str', u'lastNameGenerator',[0.22])]

            list of : (toffset,string,label,score)
            
            
        """
        return filter(lambda (offset,value,label,score):label in self._lMapping,lres)
    
    def getBestValue(self):
        # old (u'List', (2, 0, 0), u'Ritt', 987)
        # now [(u'Theresia',  0.9978103), (u'Sebald',0.71877468)]
        if self.getValue() is not None:
            # score = list! take max
            self.getValue().sort(key = lambda x:max(x[1]),reverse=True)
            ## onlt content, not score
            return self.getValue()[0][0]
            
    
    def isMandatory(self): return self._bMandatory
    def setMandatory(self): self._bMandatory = True
    
    def addTagger(self,t): 
        self._lTaggers.append(t)
    def getTaggers(self):return self._lTaggers
    
    def applyTaggers(self,o):
        lres=[]
        for t in self.getTaggers():
            res= t.runMe(o)
            ## assume one sample!  (.proedict assume  a  list of content)
            if res:
                lres.extend(res[0])
        return lres
    
    def cloneMe(self):
        clone = self.__class__()
        clone.setName(self.getName())
        clone.setValue(self.getValue())
        clone._lTaggers = self._lTaggers
        clone._lMapping = self._lMapping
        clone._record  = self._record
        clone._bMandatory = self.isMandatory()
        return clone 

class taggerClass(object):
    """
        a tagger: tag a string with a field  (one field?)
        
        
        a mapping function is required when the tagger does more than needed (more ne than the one to be extracted)
    """
    FSTTYPE             = 0               # requires path/myresource.fst 
    MLCRFTYPE           = 1            # requires path to modeldirectory
    DEEP                = 2     
    EXTERNALTAGGERTYPE  = 3      # path to exe ; assume text as input
    
    def __init__(self,name=None):
        self._name = name
        self._lFields = None
        
        # tagger type: how to apply the tagger
        self._typeOfTagger = None
        
        
        ## tagger type paramater: needed resources
        self._path = None
        self._lresources = None
        
        self._externalTagger = None
    def getName(self): return self._name
    def setName(self,s): self._name =s    
    

    def getResources(self): return self._lresources        
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
        
 
class KerasTagger(taggerClass):
    """
        see taggerTrainKeras
            -> use directly DeepTagger?
    """ 
    def __init__(self,name):
        taggerClass.__init__(self, name)
        self.myTagger = DeepTagger()
        self.myTagger.bPredict = True
#         self.myTagger.sModelName = None
#         self.myTagger.dirName = 'IE
#         self.myTagger.loadModels()
    
    def loadResources(self,sModelName,dirName):
        # location from sModeName, dirName
        self.myTagger.sModelName = sModelName
        self.myTagger.dirName = dirName        
        self.myTagger.loadModels()
    
    def runMe(self,documentObject):
        '''
            delete '.' because of location in GT
        '''
#         res = self.myTagger.predict([documentObject.getContent()])
#         return res
    
        if self.myTagger.bMultiType:
            res = self.myTagger.predict_multiptype([documentObject.getContent()])
        else:
            res = self.myTagger.predict([documentObject.getContent().replace('.','')])
        return res
    
class CRFTagger(taggerClass):
    """
        
    """    
    
    
    def defineLabelMapping(self,dMapping):
        # mapping between label number and label name
        self.dMapping = dMapping
        
    def loadResources(self,lfilenames):
        """
            load all need files
        """
        
        model, trans= lfilenames
        self.model = cPickle.load(gzip.open(model, 'rb'))
        self.trans = cPickle.load(gzip.open(trans, 'rb'))
       
    def runMe(self,documentObject): 
        txt = documentObject.getContent()
        allwords= self.trans.transform(txt.split())
        lY_pred = self.model.predict(allwords)
        # mapp output !
#         print lY_pred    
    
class RETaggerClass(taggerClass):
    
    def __init__(self,name='tagger'):
        taggerClass.__init__(self, name)
        self._lresources = None
        
    def loadResources(self,lfilenames):
        """
            Open and read resource files
            
        """
        
        self._lresources=[]
        for filename in lfilenames:
            lre,ln=cPickle.load(open(filename,'r'))
            self._lresources.append((filename,lre,ln))
        
        return self._lresources
    
    def runMe(self,documentObject):
        """
            return what? tag and offset in documentObject.getContent()?
            
            value : u'Anna', (2, 0, 0), u'Lina', 607)
                    string,  edit distance (useful?), dictentry, weight
        """  
        
        # contains: token + score
        lParsingRes = []
        
        txt = documentObject.getContent()
        ltokens = txt.split()
        for _, lreNames,lnames in self._lresources:
            for token in ltokens:
                for i,myre in enumerate(lreNames):
                    xx = myre.fullmatch(token)
#                     print token.encode('utf-8'), myre, xx
                    if xx:
                        if len(token) >1 and sum(xx.fuzzy_counts) <1:
                            lParsingRes.append((token,xx.fuzzy_counts,lnames[i][0],lnames[i][1]))
#                             print token,_,xx.fuzzy_counts,lnames[i]      
                            
        return lParsingRes  
    
    
# class dateRETaggerClass(RETaggerClass):
#     """
#         class for recognizing date 
#     """
#     def __init__(self,name='datetagger'):
#         RETaggerClass.__init__(self, name)
#         lmonth=['januar', 'februar', 'mars' ,'april','mai','juni','juli','august','october','november','december']
#         lReNames=[]
#         lNames=[]
#         for m in lmonth:
#             lReNames.append(regex.compile(r"(%s){e<=3}"%regex.escape(m), regex.IGNORECASE))
#             lNames.append((m,1))
#         self._lresources = [('REDate',lReNames, lNames)]
        
    
