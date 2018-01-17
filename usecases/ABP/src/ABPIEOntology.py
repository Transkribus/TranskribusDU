# -*- coding: utf-8 -*-
"""

    ABP Death records IEOntology
    Hervé Déjean
    cpy Xerox 2017, NLE 2017
    
    death record

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
from __future__ import print_function

import sys, os.path
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'src')
from ObjectModel.recordClass import recordClass,fieldClass,KerasTagger, RETaggerClass #, dateRETaggerClass
from ObjectModel.documentClass import documentObject

import libxml2

class deathRecord(recordClass):
    sName = 'deathrecord' 
    def __init__(self,sModelName,sModelDir):
        recordClass.__init__(self,deathRecord.sName)
        
        myTagger = ABPTagger()
        myTagger.loadResources(sModelName ,sModelDir )  
        
        fnField = firstNameField()
        fnField.setLabelMapping( ['firstNameGenerator'])
        fnField.addTagger(myTagger)
        fnField.setMandatory()
        self.addField(fnField)
    
        nfield = lastNameField()
        nfield.addTagger(myTagger)
        nfield.setLabelMapping(['lastNameGenerator'])
        nfield.setMandatory()
        self.addField(nfield)

        
        rfield=religionField()
        rfield.addTagger(myTagger)
        rfield.setLabelMapping(['religionGenerator'])
        self.addField(rfield)
        
        lfield= locationField()
        lfield.addTagger(myTagger)
        lfield.setLabelMapping(['locationGenerator'])
        self.addField(lfield)
        
        ofield= occupationField()
        ofield.addTagger(myTagger)
        ofield.setLabelMapping(['professionGenerator'])
        self.addField(ofield)
#         
        sfield= situationField() 
        sfield.addTagger(myTagger)
        sfield.setLabelMapping(['familyStatus'])
        self.addField(sfield)
#         
        dDate= deathDate()
        dDate.addTagger(myTagger)
#         dDate.setLabelMapping(['weekDayDateGenerator','MonthDayDateGenerator','MonthDateGenerator'])         
        dDate.setLabelMapping(['MonthDateGenerator'])         
        self.addField(dDate)
        
        bDate= burialDate()
        bDate.addTagger(myTagger)
#         bDate.setLabelMapping(['weekDayDateGenerator','MonthDayDateGenerator','MonthDateGenerator'])         
        bDate.setLabelMapping(['MonthDateGenerator'])         
        self.addField(bDate)         


        agefield=age()
        agefield.addTagger(myTagger)
        agefield.setLabelMapping(['ageValueGenerator'])
        self.addField(agefield)        

        blfield= burialLocation()
        blfield.addTagger(myTagger)
        blfield.setLabelMapping(['locationGenerator'])
        self.addField(blfield)

        reasonField = deathreasonField()
        reasonField.addTagger(myTagger)
        reasonField.setLabelMapping(['deathreasonGenerator'])
        self.addField(reasonField)        

        drField = doktorField()
        drField.addTagger(myTagger)
        drField.setLabelMapping(['lastNameGenerator'])  #lastNameGenerator
        self.addField(drField)  
    
    def generateOutput(self,outDom):
        """
            generateOutput
            
?xml version="1.0" encoding="UTF-8"?>
<DOCUMENT>
  <PAGE number="1" pagenum="0" nbrecords="4" years="1883-1873">
    <RECORD lastname="Thaler" firstname="Johann" role="Verstorbener" location="Hinterschmiding" occupation="" year="1873" month="5" day="22"/>
    <RECORD lastname="Thurner" firstname="Juliana" role="Verstorbener" location="Freyung" occupation="" year="1883" month="1" day="11"/>
    <RECORD lastname="Raab" firstname="Georg" role="Verstorbener" location="Hinterschmiding" occupation="" year="1871" month="5" day="9"/>
    <RECORD lastname="Maurus" firstname="Anna" role="Verstorbener" location="Freyung" occupation="" year="1871" month="1" day="5"/>
  </PAGE>            
        """
        if outDom is None:
            outDom = libxml2.newDoc('1.0')
            root = libxml2.newNode('DOCUMENT')
            outDom.setRootElement(root)
        else:
            root = outDom.getRootElement()
        ## group cand by page
        ## store all with score; evaluation uses scoresTH
        lPages={}
        for cand in self.getCandidates():
#             print cand, cand.getPage(), cand.getAllFields()
            try:lPages[cand.getPage()].append(cand)
            except:lPages[cand.getPage()]=[cand]

        for page in sorted(lPages):
            # page node
            domp=libxml2.newNode('PAGE')
            domp.setProp('number',str(page.getNumber()))
            #in ref :Seebach_006_03_0030
            key=os.path.basename(page.getAttribute('imageFilename'))[:-4]
            key=key.replace('-','_')
            key=key[2:]
            domp.setProp('pagenum',key)

            domp.setProp('years','NA')
            root.addChild(domp)         
            sortedRows = lPages[page]
            sortedRows.sort(key=lambda x:int(x.getIndex()))   
            for cand in sortedRows:
                #record
                record = libxml2.newNode('RECORD')
                # record fields
                nbRecords = 0
                for field in cand.getAllFields():
                    if field.getName() is not None and field.getBestValue() is not None:
                        record.setProp(field.getName(),field.getBestValue().encode('utf-8'))
                        nbRecords+=1
                if nbRecords > 0:
                    domp.addChild(record)
                domp.setProp('nbrecords',str(nbRecords))
        return outDom    


class deathreasonField(fieldClass):
    sName='deathreason'
    def __init__(self):
        fieldClass.__init__(self, deathreasonField.sName)
        
class locationField(fieldClass):
    sName='location'
    def __init__(self):
        fieldClass.__init__(self, locationField.sName)
           
class deathDate(fieldClass):
    sName='deathDate'
    def __init__(self):
        fieldClass.__init__(self, deathDate.sName)
        
class burialDate(fieldClass):
    sName='burialDate'
    def __init__(self):
        fieldClass.__init__(self, burialDate.sName)
          
class burialLocation(fieldClass):
    sName='burialLocation'
    def __init__(self):
        fieldClass.__init__(self, burialLocation.sName)
                    
class age(fieldClass):
    sName='age'
    def __init__(self):
        fieldClass.__init__(self, age.sName)
        
class firstNameField(fieldClass):
    sName = 'firstname'
    def __init__(self):
        fieldClass.__init__(self, firstNameField.sName)

class lastNameField(fieldClass):
    sName = 'lastname'
    def __init__(self):
        fieldClass.__init__(self, lastNameField.sName)


class situationField(fieldClass):
    sName='situation'
    def __init__(self):
        fieldClass.__init__(self, situationField.sName)    

class doktorField(fieldClass):
    sName='doktor'
    def __init__(self):
        fieldClass.__init__(self, doktorField.sName) 

class religionField(fieldClass):
    sName='religion'
    def __init__(self):
        fieldClass.__init__(self, religionField.sName) 

class occupationField(fieldClass):
    sName='occupation'
    def __init__(self):
        fieldClass.__init__(self, occupationField.sName)

from sklearn.base import BaseEstimator, TransformerMixin  
class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        
    def fit(self, l, y=None):
        return self

    def transform(self, l):
        assert False, "Specialize this method!"
        
class SparseToDense(Transformer):
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, o):
        return o.toarray()    

class NodeTransformerTextEnclosed(Transformer):
    """
    we will get a list of block and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings  
    """
    def transform(self, lw):
        return map(lambda x: x, lw) 

class ABPTagger(KerasTagger):
    sName='abprecord'
    def __init__(self):
        KerasTagger.__init__(self, ABPTagger.sName)
        self._typeOfTagger = self.DEEP
#         self.loadResources( ( 'models/%s.hd5'%('fnln'), 'models/%s.aux.pkl' %('abprecord') ) )    
#         self.loadResources('model1_h64_nbf2000_epoch3', 'IEdata/model' )    
#         self.loadResources('model1_h32_nbf5000_epoch3', 'IEdata/model' )    
    
    
class fnameTagger(RETaggerClass):
    sName= 'firstNameTagger'
    def __init__(self):
        RETaggerClass.__init__(self, fnameTagger.sName)
        self._typeOfTagger = RETaggerClass.FSTTYPE
        self._lpath=[os.path.abspath('./resources/firstnames.1000.pkl')]
#         self._lpath=[os.path.abspath('./resources/deathreason.pkl')]
        self.loadResources(self._lpath)


class lnameTagger(RETaggerClass):
    sName= 'lastNameTagger'
    def __init__(self):
        RETaggerClass.__init__(self, lnameTagger.sName)
        self._typeOfTagger = RETaggerClass.FSTTYPE
        self._lpath=[os.path.abspath('./resources/lastnames.1000.pkl')]
        self.loadResources(self._lpath)
      
      
# class dateTagger(dateRETaggerClass):    
#     sName="German Date"
#     def __init__(self):
#         RETaggerClass.__init__(self, dateTagger.sName)
        

#def test_ieo():  
if __name__ == "__main__":
        
    dr =deathRecord()
    mydocO= documentObject()
    mydocO.setContent('Veronika Schmid')
    for field in dr.getFields():
        lParsingRes = field.applyTaggers(mydocO)
        foo = field.extractLabel(lParsingRes)
        print(field.getName(),lParsingRes)
        print(foo)

