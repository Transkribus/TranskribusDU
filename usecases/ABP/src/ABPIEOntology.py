# -*- coding: utf-8 -*-
"""

    ABP  records IEOntology
    
    Hervé Déjean
    cpy Xerox 2017, NLE 2017
    
    death record

    READ project 


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import sys, os.path
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'Transkribus')
from ObjectModel.recordClass import recordClass,fieldClass,KerasTagger, RETaggerClass #, dateRETaggerClass
from ObjectModel.documentClass import documentObject

from lxml import etree



class BaptismRecord(recordClass):
    """
        name (firstname only possible)  +death date sometimes
        hebamme , was the birth easy , date?
        vater + info
        mutter + info
        location  ; instead date of the uxor!
        birth date
        baptism date
        priester + info
    """

class weddingRecord(recordClass):
    sName = 'weddingrecord'
    def __init__(self,sModelName,sModelDir):
        recordClass.__init__(self,deathRecord.sName)
        
        myTagger = ABPTagger()
        myTagger.loadResources(sModelName ,sModelDir )  
        self.tagger=myTagger
        #bride
        bfnField = firstNameField()
        bfnField.setLabelMapping( ['firstNameGenerator'])
        bfnField.addTagger(myTagger)
        bfnField.setMandatory()
        self.addField(bfnField)
    
        gnfield = lastNameField()
        gnfield.addTagger(myTagger)
        gnfield.setLabelMapping(['lastNameGenerator'])
        gnfield.setMandatory()
        self.addField(gnfield)
        
	# need a specific class for bride and groom and witnesses...
        blfield= locationField()
        blfield.addTagger(myTagger)
        blfield.setLabelMapping(['locationGenerator'])
        self.addField(blfield)

        wDate= weddingDate()
        wDate.addTagger(myTagger)
        wDate.setLabelMapping(['yearGenerator'])         
        self.addField(wDate)
        
        wmDate= weddingDateMonth()
        wmDate.addTagger(myTagger)
        wmDate.setLabelMapping(['MonthDateGenerator'])         
        self.addField(wmDate)
        
        wdDate= weddingDateDay()
        wdDate.addTagger(myTagger)
        wdDate.setLabelMapping(['MonthDayDateGenerator'])         
        self.addField(wdDate)
        

        bDate= brideDate()
        bDate.addTagger(myTagger)
        bDate.setLabelMapping(['yearGenerator'])         
        self.addField(bDate)
        
        bmDate= brideDateMonth()
        bmDate.addTagger(myTagger)
        bmDate.setLabelMapping(['MonthDateGenerator'])         
        self.addField(bmDate)
        
        bdDate= brideDateDay()
        bdDate.addTagger(myTagger)
        bdDate.setLabelMapping(['MonthDayDateGenerator'])         
        self.addField(bdDate)

        bfield= occupationField()
        bfield.addTagger(myTagger)
        bfield.setLabelMapping(['professionGenerator'])
        self.addField(bfield)

        #groom
        gfnField = firstNameField()
        gfnField.setLabelMapping( ['firstNameGenerator'])
        gfnField.addTagger(myTagger)
        gfnField.setMandatory()
        self.addField(gfnField)
    
        gnfield = lastNameField()
        gnfield.addTagger(myTagger)
        gnfield.setLabelMapping(['lastNameGenerator'])
        gnfield.setMandatory()
        self.addField(gnfield)
        
        lfield= locationField()
        lfield.addTagger(myTagger)
        lfield.setLabelMapping(['locationGenerator'])
        self.addField(lfield)
        
        gDate= groomDate()
        gDate.addTagger(myTagger)
        gDate.setLabelMapping(['yearGenerator'])         
        self.addField(gDate)
        
        gmDate= groomDateMonth()
        gmDate.addTagger(myTagger)
        gmDate.setLabelMapping(['MonthDateGenerator'])         
        self.addField(gmDate)
        
        gdDate= groomDateDay()
        gdDate.addTagger(myTagger)
        gdDate.setLabelMapping(['MonthDayDateGenerator'])         
        self.addField(gdDate)

        
        gfield= occupationField()
        gfield.addTagger(myTagger)
        gfield.setLabelMapping(['professionGenerator'])
        self.addField(gfield)

	#parent bride
        bpfnField = firstNameField()
        bpfnField.setLabelMapping( ['firstNameGenerator'])
        bpfnField.addTagger(myTagger)
        self.addField(bpfnField)
    
        bpnfield = lastNameField()
        bpnfield.addTagger(myTagger)
        bpnfield.setLabelMapping(['lastNameGenerator'])
        self.addField(bpnfield)
        
        bpofield= occupationField()
        bpofield.addTagger(myTagger)
        bpofield.setLabelMapping(['professionGenerator'])
        self.addField(bpofield)
    
        bmfnField = firstNameField()
        bmfnField.setLabelMapping( ['firstNameGenerator'])
        bmfnField.addTagger(myTagger)
        self.addField(bmfnField)
    
        bmnfield = lastNameField()
        bmnfield.addTagger(myTagger)
        bmnfield.setLabelMapping(['lastNameGenerator'])
        self.addField(bmnfield)
        
        bmofield= occupationField()
        bmofield.addTagger(myTagger)
        bmofield.setLabelMapping(['professionGenerator'])
        self.addField(bmofield)

	# parent groom
        gpfnField = firstNameField()
        gpfnField.setLabelMapping( ['firstNameGenerator'])
        gpfnField.addTagger(myTagger)
        self.addField(gpfnField)
    
        gpnfield = lastNameField()
        gpnfield.addTagger(myTagger)
        gpnfield.setLabelMapping(['lastNameGenerator'])
        self.addField(gpnfield)
        
    
        gpofield= occupationField()
        gpofield.addTagger(myTagger)
        gpofield.setLabelMapping(['professionGenerator'])
        self.addField(gpofield)

        gmfnField = firstNameField()
        gmfnField.setLabelMapping( ['firstNameGenerator'])
        gmfnField.addTagger(myTagger)
        self.addField(gmfnField)
    
        gmnfield = lastNameField()
        gmnfield.addTagger(myTagger)
        gmnfield.setLabelMapping(['lastNameGenerator'])
        self.addField(gmnfield)


        gmofield= occupationField()
        gmofield.addTagger(myTagger)
        gmofield.setLabelMapping(['professionGenerator'])
        self.addField(gmofield)

	# witnesses

        wpfnField = firstNameField()
        wpfnField.setLabelMapping( ['firstNameGenerator'])
        wpfnField.addTagger(myTagger)
        self.addField(wpfnField)
    
        wpnfield = lastNameField()
        wpnfield.addTagger(myTagger)
        wpnfield.setLabelMapping(['lastNameGenerator'])
        self.addField(wpnfield)
        
    
        wpofield= occupationField()
        wpofield.addTagger(myTagger)
        wpofield.setLabelMapping(['professionGenerator'])
        self.addField(wpofield)

        wfield= locationField()
        wfield.addTagger(myTagger)
        wfield.setLabelMapping(['locationGenerator'])
        self.addField(wfield)

        wmfnField = firstNameField()
        wmfnField.setLabelMapping( ['firstNameGenerator'])
        wmfnField.addTagger(myTagger)
        self.addField(wmfnField)
    
        wmnfield = lastNameField()
        wmnfield.addTagger(myTagger)
        wmnfield.setLabelMapping(['lastNameGenerator'])
        self.addField(wmnfield)

        wfield= locationField()
        wfield.addTagger(myTagger)
        wfield.setLabelMapping(['locationGenerator'])
        self.addField(wfield)

        wmofield= occupationField()
        wmofield.addTagger(myTagger)
        wmofield.setLabelMapping(['professionGenerator'])
        self.addField(wmofield)


class deathRecord(recordClass):
    sName = 'deathrecord' 
    def __init__(self,sModelName,sModelDir):
        recordClass.__init__(self,deathRecord.sName)
        
        myTagger = ABPTagger()
        myTagger.loadResources(sModelName ,sModelDir )  
        self.tagger=myTagger
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
        lfield.setLabelMapping(['location2Generator'])
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

        # specific tagger for dates ?
        dDate= deathDate()
        dDate.addTagger(myTagger)
#         dDate.setLabelMapping(['weekDayDateGenerator','MonthDayDateGenerator','MonthDateGenerator'])         
        dDate.setLabelMapping(['MonthDateGenerator'])         
        self.addField(dDate)
        
        ddDate= deathDateDay()
        ddDate.addTagger(myTagger)
#         dDate.setLabelMapping(['weekDayDateGenerator','MonthDayDateGenerator','MonthDateGenerator'])         
        ddDate.setLabelMapping(['MonthDayDateGenerator'])         
        self.addField(ddDate)
        
        bDate= burialDate()
        bDate.addTagger(myTagger)
#         bDate.setLabelMapping(['weekDayDateGenerator','MonthDayDateGenerator','MonthDateGenerator'])         
        bDate.setLabelMapping(['MonthDateGenerator'])         
        self.addField(bDate)         

        year=deathYear()
        year.addTagger(myTagger)
        year.setLabelMapping(['yearGenerator'])         
        self.addField(year)

        agefield=age()
        agefield.addTagger(myTagger)
        agefield.setLabelMapping(['ageValueGenerator'])
        self.addField(agefield)        
        
        ageUnitfield=ageUnit()
        ageUnitfield.addTagger(myTagger)
        ageUnitfield.setLabelMapping(['AgeUnitGenerator'])
        self.addField(ageUnitfield)           

        blfield= burialLocation()
        blfield.addTagger(myTagger)
        blfield.setLabelMapping(['location2Generator'])
        self.addField(blfield)

        reasonField = deathreasonField()
        reasonField.addTagger(myTagger)
        reasonField.setLabelMapping(['deathreasonGenerator'])
        self.addField(reasonField)        

        drField = doktorField()
        drField.addTagger(myTagger)
        drField.setLabelMapping(['lastNameGenerator'])  #lastNameGenerator
        self.addField(drField)  
    
        drField = PriestField()
        drField.addTagger(myTagger)
        drField.setLabelMapping(['lastNameGenerator'])
        drField.setMandatory()
        self.addField(drField)    
    
    
#     def decoratePageXml(self):
#         """
#             ONGOING....
#             add in @custom the field name
#                <TextLine id="xrce_p1_p1_TableCell_1511814954659_67l4" 
#                custom="readingOrder {index:0;} person {offset:0; length:11;} firstname {offset:0; length:6;} lastname {offset:7; length:4;}">
#                
#                
#             currenlty 
#         """
#         lPages={}
#         for cand in self.getCandidates():
#             try:lPages[cand.getPage()].append(cand)
#             except:lPages[cand.getPage()]=[cand]
# 
#         for page in sorted(lPages):
#             sortedRows = lPages[page]
#             sortedRows.sort(key=lambda x:int(x.getIndex()))   
#             for cand in sortedRows:
#                 for field in cand.getAllFields():
#                     if field.getName() is not None and field.getBestValue() is not None:
#                         print (field, field.getOffset()

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
            root = etree.Element('DOCUMENT')
            outDom= etree.ElementTree(root)
        else:
            root = outDom.getroot()
        ## group cand by page
        ## store all with score; evaluation uses scoresTH
        lPages={}
        for cand in self.getCandidates():
#             print cand, cand.getPage(), cand.getAllFields()
            try:lPages[cand.getPage()].append(cand)
            except:lPages[cand.getPage()]=[cand]

        for page in sorted(lPages):
            # page node
            domp=etree.Element('PAGE')
            domp.set('number',str(page.getNumber()))
            #in ref :Seebach_006_03_0030
            key=os.path.basename(page.getAttribute('imageFilename'))[:-4]
            key=key.replace('-','_')
            key=key[2:]
            domp.set('pagenum',key)

            ## -> page has now a year attribute (X-X)
            if page.getAttribute('computedyear') is None:
                page.addAttribute('computedyear','')
            domp.set('years',str(page.getAttribute('computedyear')))
            root.append(domp)         
            sortedRows = lPages[page]
            sortedRows.sort(key=lambda x:int(x.getIndex()))   
            for cand in sortedRows:
                #record
                record = etree.Element('RECORD')
                # record fields
                nbRecords = 0
                lSeenField=[]
                for field in cand.getAllFields():
                    # take the best one 
                    if field.getName() is not None and field.getBestValue() is not None:
                        record.set(field.getName().lower(),field.getBestValue())
                        lSeenField.append(field.getName().lower())
                        nbRecords=1
                    elif field.getName().lower() not in lSeenField:record.set(field.getName().lower(),"") 
                if nbRecords > 0:
                    domp.append(record)
            domp.set('nbrecords',str(len(domp)))
        return outDom    


class deathreasonField(fieldClass):
    sName='deathreason'
    def __init__(self):
        fieldClass.__init__(self, deathreasonField.sName)
        
class locationField(fieldClass):
    sName='location'
    def __init__(self):
        fieldClass.__init__(self, locationField.sName)


class weddingDate(fieldClass):
    sName='weddingDate'
    def __init__(self):
        fieldClass.__init__(self, weddingDate.sName)
    
class weddingDateMonth(fieldClass):
    sName='weddingDateMonth'
    def __init__(self):
        fieldClass.__init__(self, weddingDateMonth.sName)

class weddingDateDay(fieldClass):
    sName='weddingDateDay'
    def __init__(self):
        fieldClass.__init__(self, weddingDateDay.sName)

class brideDate(fieldClass):
    sName='brideDate'
    def __init__(self):
        fieldClass.__init__(self, brideDate.sName)
    
class brideDateMonth(fieldClass):
    sName='brideDateMonth'
    def __init__(self):
        fieldClass.__init__(self, brideDateMonth.sName)

class brideDateDay(fieldClass):
    sName='brideDateDay'
    def __init__(self):
        fieldClass.__init__(self, brideDateDay.sName)

class groomDate(fieldClass):
    sName='groomDate'
    def __init__(self):
        fieldClass.__init__(self, groomDate.sName)
    
class groomDateMonth(fieldClass):
    sName='groomDateMonth'
    def __init__(self):
        fieldClass.__init__(self, groomDateMonth.sName)

class groomDateDay(fieldClass):
    sName='groomDateDay'
    def __init__(self):
        fieldClass.__init__(self, groomDateDay.sName)

class deathYear(fieldClass):
    sName='deathYear'
    def __init__(self):
        fieldClass.__init__(self, deathYear.sName)          
class deathDate(fieldClass):
    sName='deathDate'
    def __init__(self):
        fieldClass.__init__(self, deathDate.sName)

class deathDateDay(fieldClass):
    sName='MonthDayDateGenerator'
    def __init__(self):
        fieldClass.__init__(self, deathDateDay.sName)  
              
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

class ageUnit(fieldClass):
    sName='ageUnit'
    def __init__(self):
        fieldClass.__init__(self, ageUnit.sName)

        
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

class PriestField(fieldClass):
    sName='priest'
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

