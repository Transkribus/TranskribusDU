# -*- coding: utf-8 -*-
"""


    contentGenerator.py

    create annotated textual data 
     H. Déjean
    

    copyright NLE 2017
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
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import sys, os.path
import random
import datetime
from optparse import OptionParser
import pickle,gzip
import platform

sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'TranskribusDU')


from dataGenerator.generator import Generator
from dataGenerator.textGenerator import textGenerator 
from dataGenerator.numericalGenerator import integerGenerator, positiveIntegerGenerator
from dataGenerator.textRandomGenerator import textRandomGenerator
from dataGenerator.textRandomGenerator import textletterRandomGenerator


class NumberedItemSepGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self.loadResourcesFromList( [[('.',50),(')',30),('/',20)]])   
        
class numberedItems(textGenerator):
    def __init__(self,mean,std):
        textGenerator.__init__(self,lang=None)        
        self._structure = [ ( (positiveIntegerGenerator(mean,std),1,100),(NumberedItemSepGenerator(),1,50),100)
            ]
        self._separator = ""
    def generate(self):
        return Generator.generate(self)  

    
class AgeUnitGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self.loadResourcesFromList( [[('Jahre',50),('Ja',10),('Monate',10),('M.',10),('W',5),('Wochen',5),('T',3),('Tag',6),('Stunde',10)]])        
        
class ageValueGenerator(integerGenerator):
    """ 
       similar to  integerGenerator bit class name different
    """
    def __init__(self,mean, sd):
        integerGenerator.__init__(self,mean, sd)
        
class AgeGenerator(textGenerator):
    """
        need to cover 1 Jahr 6 Monate 3 tage  or any subsequence
    """
    def __init__(self):

#         startDate =  datetime.datetime(1900, 01, 01)
#         randdays = random.uniform(1,364*100)
#         randhours = random.uniform(1,24)
#         step = datetime.timedelta(days=randdays,hours=randhours)
#     
#         d = startDate + step
#         delta=  relativedelta(d+step,d)
#         self.years  = delta.months
#         self.months = delta.months        
#         self.weeks  = delta.days // 7 
#         self.days   = delta.days % 7
        
        textGenerator.__init__(self,lang=None)
        self.measure = ageValueGenerator(50,10)
        self.unit = AgeUnitGenerator()
                
        self._structure = [
                ( (self.measure,1,100), (self.unit,1,100),100)
#                 ( (self.measure,1,100), (self.unit,1,100),(self.measure,1,100), (self.unit,1,100),100),
#                 ( (self.measure,1,100), (self.unit,1,100),(self.measure,1,100), (self.unit,1,100),(self.measure,1,100), (self.unit,1,100),100)
             ]
    def generate(self):
        return Generator.generate(self)    
    
class legitimGenerator(textGenerator):
    """ 
        ID      name          namelabel    kuerzel
        1_1    legitim        leg.           l
        1_2    illegitim       ill.          i
        1_3    adoptiert       adopt.        a
        1_4    durch nachfolge    p.m.s.l.    vor

    """
    def __init__(self):
        textGenerator.__init__(self,lang=None)
#         self._value = ['leg','legitim','illeg','illegitim']
        self.loadResourcesFromList( [[('leg',60),('legitim',20),('illeg',10),('illegitim',20)]])        
            
class religionGenerator(textGenerator):
    """
    2_1    katholisch    kath.    rk
2_2    evangelisch    ev.    ev
2_3    orthodox    orth.    or
2_4    sonstige    sonst.    ss
2_5    altkatholisch    altkath.    alt
2_6    christlich    christlich    ch
2_7    Konvertit    Konvertit    kon
2_8    protestantisch    prot.    pr

    """
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self.loadResourcesFromList( [[('K',30),('kath',40),('katholic',5),('katho',5),('K. R.',5),("evangelist",5),('evang.',5),("evg.",5)]])  
#         self._value = ['k','kath','katholic','katho','k. R.','evangelist','evang.','evg.']
    
class familyStatus(textGenerator):
    """
        3_1    ledig    ledig    ld
3_2    verheiratet    verh.    vh
3_3    verwitwet    verw.    vw

    children not  covered
    """
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self.loadResourcesFromList( [[('knabe',5),('mädchen',5),('kind',30),('Säugling',5),('ledig',20), ('verehelichet.',10),('erehelicht',10),('wittwe',20),('wittwer',10),('verwitwet',5),('verw.',5),('verheirathet',10),('verhei',10)]])
                     
class deathReasonColumnGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        self.deathreson = deathreasonGenerator()
        self.doktor = doktorGenerator(lang)
        self.location = location2Generator()
        self._structure = [
                ( (self.deathreson,1,100), (self.doktor,1,30),(self.location,1,10),100)
             ]
    def generate(self):
        return Generator.generate(self) 
    
class deathreasonGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'deathreason'
        self._lpath=[os.path.abspath('../resources/old/deathreason.pkl')]
        self._value = list(map(lambda x:x[0],self.loadResources(self._lpath)))
        
        self._lenRes= len(self._lresources)
    

class locationPrepositionGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._value = ['in','bei','zu','zum','A','W','H','von','beÿ']
    
"""
    find with Hausnummer
    two locations 
"""
        
class NGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self.loadResourcesFromList( [[('N',50),('Num',10)]])         
        
class HausnumberGenerator(textGenerator):
    def __init__(self,mean,std):
        textGenerator.__init__(self,lang=None)        
        self._structure = [ ( (NGenerator(),1,80) ,(positiveIntegerGenerator(mean,std),1,100),100 )  ]
    def generate(self):
        return Generator.generate(self)  
            
class location2Generator(textGenerator):
    """
        missing Rothsmansdorf Nr̳ 12
    """
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'location'
        self.location = locationGenerator()
        self.location2 = locationGenerator()
        self.prep = locationPrepositionGenerator()    
        self._structure = [
                ( (self.location2,1,20),(self.prep,1,10), (self.location,1,100),(HausnumberGenerator(50,10),1,20),(legitimGenerator(),1,10),100)
             ]
    def generate(self):
        return Generator.generate(self)        
        
class locationGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'location'
        self._lpath=[os.path.abspath('../resources/old/location.pkl')]
        self._value = list(map(lambda x:x[0],self.loadResources(self._lpath)))
        self._lenRes= len(self._lresources)
    

"""
    occupation :
    profession
    
    [leg,..][knabe, tocher,sohn,kind] deSR  person, profession,[zu] location
    unehelicher Knabe der Maria Friedl = Schödermaier
    ehl. Kind des Uhr¬ machers Martin Grammel
    Kind der Creszenz Gigl, Bauerstochter von Haid.
    
    
    also add location von ....
"""
class professionGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'profession'
        self._lpath=[os.path.abspath('../resources/old/profession.pkl')]
        self._value = list(map(lambda x:x[0],self.loadResources(self._lpath)))
        self._lenRes= len(self._lresources)
    
class professionGenerator2(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._structure = [
                 ( (professionGenerator(),1,100), (locationPrepositionGenerator(),1,25),(locationGenerator(),1,25),100 )
             ]
    def generate(self):
        return Generator.generate(self)      

class firstNameGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'firstName'
        self._lpath=[os.path.abspath('../resources/old/firstname.pkl')]
        self._value = list(map(lambda x:x[0],self.loadResources(self._lpath)))
        self._lenRes= len(self._lresources)
    

class particuleGenerator(textGenerator):
    def __init__(self,lang=None):
        textGenerator.__init__(self,lang)
        self._value = ['v.', 'von']   

  
class lastNameGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'lastName'
        self._lpath=[os.path.abspath('../resources/old/lastname.pkl')]
        self._value = list(map(lambda x:x[0],self.loadResources(self._lpath)))
        
        self._lenRes= len(self._lresources)
    
    
class lastNameGenerator2(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._structure = [
                 ( (particuleGenerator(lang=None),1,5), (lastNameGenerator(),1,100),100 )
             ]
    def generate(self):
        return Generator.generate(self)         
  

class CUMSACRGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        self._value = ['cum sacramentum', 'cum sacr']      

class GEBGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        self._value = ['geb.', 'geb','geboren']                
      

class undGenerator(textGenerator):
        def __init__(self,lang=None):
            textGenerator.__init__(self,lang=None)
            self._value = ['ünd.', 'ü','und']       
    
class zeugenNameGenerator(textGenerator):
    """
        first last prof von location  und same
    """
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        self._structure = [
                 (   (firstNameGenerator(),1,100)
                    ,(lastNameGenerator(),1,100)
                    ,(professionGenerator(),1,100)
                    ,(locationPrepositionGenerator(),1,75)
                    ,(locationGenerator(),1,75 )
                    ,100)
             ]
    def generate(self):
        return Generator.generate(self)     

class zeugenName2Generator(textGenerator):
    """
        zeugenName  UND zeugenName
    """
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        self._structure = [
                 (   (zeugenNameGenerator(lang=None),1,100)
                    ,(undGenerator(),1,50),(zeugenNameGenerator(lang=None),1,50)  ,100)
             ]
    def generate(self):
        return Generator.generate(self)       
    
class MutterStandGenerator(textGenerator):
    """
        dessen , uxor,... see flair embeddings!!
    """
    def __init__(self,lang=None):
        textGenerator.__init__(self,lang=None)
        self._value = ['ehefrau', 'uxor','dessen','ehe']         

class MutterStand2Generator(textGenerator):
    """
        MutterStand + profession
    """
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        
        self._structure = [
                ( (MutterStandGenerator(),1,100),50)
                ,( (professionGenerator(),1,100),50)

             ]
    def generate(self):
        return Generator.generate(self) 
    
class ParentName(textGenerator):
    """
    for wedding 
        Josef UND Maria LASTNAME  geb lastname
        josef Lastname und maria geb lastname
    """
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        
        self._structure = [
                ( (firstNameGenerator(),1,100), (undGenerator(),1,100) ,(firstNameGenerator(),1,100),(lastNameGenerator2(),1,100),(GEBGenerator(lang),1,50), (lastNameGenerator2(),1,100),(religionGenerator(),1,20),50)
                ,( (firstNameGenerator(),1,100), (lastNameGenerator2(),1,100), (undGenerator(),1,100) (firstNameGenerator(),1,100), (lastNameGenerator2(),1,100),50)

                #, ( (lastNameGenerator2(),1,00), (firstNameGenerator(),1,100), (GEBGenerator(lang),1,50),(religionGenerator(),1,20),100)

             ]
    def generate(self):
        return Generator.generate(self)             
    

class birthName(textGenerator):    
    """
        pattern: Firstname  ( lastname  ))   (death date )
    """
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        #deathdate = ABPGermanDateGenerator()
        #deathdate.defineRange(1700, 2000)
        self._structure = [
                 ( (firstNameGenerator(),1,100) ,(legitimGenerator(),1,10),100) #(deathdate,1,10),80)
                ,( (firstNameGenerator(),1,100), (lastNameGenerator2(),1,100),(legitimGenerator(),1,10),20)
             ]
    def generate(self):
        return Generator.generate(self) 
    
class BrautNameGenerator(textGenerator):    
    """
        pattern: Firstname  [geb] lastname
    """
    
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        
        self._structure = [
                ( (firstNameGenerator(),1,100), (GEBGenerator(lang),1,50), (lastNameGenerator2(),1,100),(religionGenerator(),1,20),50)
                ,( (firstNameGenerator(),1,100), (lastNameGenerator2(),1,100), (religionGenerator(),1,20),50)

                #, ( (lastNameGenerator2(),1,00), (firstNameGenerator(),1,100), (GEBGenerator(lang),1,50),(religionGenerator(),1,20),100)

             ]
    def generate(self):
        return Generator.generate(self)     

class PersonName2(textGenerator):
    """
    """
    def __init__(self,lang=None):
        textGenerator.__init__(self,lang=None)
        
        self._structure = [
                 ( (firstNameGenerator(),1,100), (lastNameGenerator2(),1,100),(CUMSACRGenerator(lang),1,10), (religionGenerator(),1,20),(legitimGenerator(),1,10),100)
                , ( (lastNameGenerator2(),1,100), (firstNameGenerator(),1,100), (CUMSACRGenerator(lang),1,10),(religionGenerator(),1,20),(legitimGenerator(),1,10),100)
                #noisy ones ?
#                 ,(    (firstNameGenerator(),1,100), (CUMSACRGenerator(lang),1,25),100)
#                 ,(    (lastNameGenerator(),1,100), (CUMSACRGenerator(lang),1,25),100)

             ]
    
    def generate(self):
        return Generator.generate(self)            
        
class PersonName(textGenerator):
    """
    """
    def __init__(self,lang=None):
        textGenerator.__init__(self,lang=None)
        
        self._structure = [
                ( (firstNameGenerator(),1,100), (lastNameGenerator(),1,100),(CUMSACRGenerator(lang),1,25),100)
                ,( (lastNameGenerator(),1,100), (firstNameGenerator(),1,100),(CUMSACRGenerator(lang),1,25),100)
                # noisy one 
#                 ,( (lastNameGenerator(),1,50), (firstNameGenerator(),1,50),(CUMSACRGenerator(lang),1,25),100)

             ]
    
    def generate(self):
        return Generator.generate(self)    


class ohneArtzGenerator(textGenerator):
     def __init__(self,lang=None):
        textGenerator.__init__(self,lang=None)
        self._value = ['ohne Arzt','O. A.','Ohne A.']
          
class doktorTitleGenerator(textGenerator):
    
    """ 
        need to weight
    """
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        self._value = ['Arzt','Dr', 'Chirurg' ,'Landarzt','doktor', 'hebamme','Hebam̄e','Frau','Sch.', 'Schwester']    
        
class doktorGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        self._structure = [
                ( (doktorTitleGenerator(lang),1,100), (lastNameGenerator(),1,100),50),
                ( (ohneArtzGenerator(),1,100),50)
             ]
    def generate(self):
        return Generator.generate(self)  
    
class hebammeGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        self._structure = [
                ( (doktorTitleGenerator(lang),1,10),(firstNameGenerator(),1,100), (lastNameGenerator(),1,100),90),
                ( (ohneArtzGenerator(),1,100),10)
             ]
    def generate(self):
        return Generator.generate(self) 
    
class MonthDateGenerator(textGenerator):
    def __init__(self,lang,value=None):
        textGenerator.__init__(self,lang)
        self._value = [value]   
#         self.realization = ['b','B','m']   # m = 01 02 ...
        self.realization = ['b','B']
        
                   
    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.month]
    
    def generate(self):
        # P3 or P2
        try:self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))
        except UnicodeDecodeError: self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')


        return self

class MonthDayDateGenerator(textGenerator):
    """
    '16. Nov'   -> [((0, 0), '16.', 'numberedItems', [0.9996762]), ((1, 1), 'Nov', 'MonthDateGenerator', [0.9997758])]  
    
    
    add . after number ?
    """
    def __init__(self,lang,value=None):
        textGenerator.__init__(self,lang)
        self._value = [value]     
        self.realization=  ['d']

    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.day]

    def generate(self):
        day= int(self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)))
        day= "%d"%day
        try: self._generation = day
        except UnicodeDecodeError: self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')
        return self

class weekDayDateGenerator(textGenerator):
    def __init__(self,lang,value=None):
        self._fulldate = None
        textGenerator.__init__(self,lang)
        self._value = [value]     
#         self.realization=['a','A','w']
        self.realization=['a','A']
         
    def __repr__(self): 
        try:
            return self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))
        except AttributeError:return "tobeInst"

    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.weekday()]
        
    def generate(self):
        try: self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))
        except UnicodeDecodeError :self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')
        return self
        
class HourDateGenerator(textGenerator):
    def __init__(self,lang,value=None):
        self._fulldate = None
        textGenerator.__init__(self,lang)
        #self._value = [value]
        self.realization=['H','I']      
    
    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.hour]
    
    def generate(self):
        try:self._generation = u""+str(int(self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))))
        except UnicodeDecodeError: self._generation = u""+self._fulldate.strftime('%'+ '%d'%self.getRandomElt(self.realization)).decode('latin-1')
        return self        
        
class yearGenerator(textGenerator):
    def __init__(self,lang,value=None):
        self._fulldate = None
        textGenerator.__init__(self,lang)
        self._value = [value]     
        self.realization=['Y']
        self.offset=0
         
    def __repr__(self): 
        try:
            return self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))
        except AttributeError:return "tobeInst"

    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.year]
        
    def generate(self):
        try:self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))
        except UnicodeDecodeError :self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')
        return self   
              
class DayPartsGenerator(textGenerator):
    def __init__(self,lang,value=None):
        textGenerator.__init__(self,lang)
        self._value=['abends','morgens','vormittags','nachmittags','mittags','nacht','fruh','früh']
        
         
class FullHourDateGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        self.hour=HourDateGenerator(lang)
        self._structure =  [
                ( (UMGenerator(lang),1,50),(self.hour,1,100),(UHRGenerator(lang),1,100),(DayPartsGenerator(lang),1,25),100 )
                ] 
    
    def setValue(self,d): 
        self.hour._fulldate= d
        self.hour.setValue(d)
                     
    def generate(self):
        Generator.generate(self)
    
class DateGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        
        self.yearoffset = 0
        self.monthGen = MonthDateGenerator(lang)
        self.monthdayGen = MonthDayDateGenerator(lang)
        self.weekdayGen= weekDayDateGenerator(lang)
        self.hourGen = FullHourDateGenerator(lang) 
        self.yearGen = yearGenerator(lang)
        self._structure = [ 
                           ((self.yearGen,1,90),(self.weekdayGen,1,90),(self.monthdayGen,1,90),(self.monthGen,1,90),(self.hourGen,1,100), 100)
                           ]
    def setValue(self,v):
        """
        """
        for subgen in [self.yearGen,self.monthGen,self.monthdayGen,self.weekdayGen,self.hourGen]:
            subgen.setValue(v)
    
    def setYearOffset(self,o): 
        self.yearoffset=o
        self.yearGen.offset=o
    def getYearOffset(self): return self.yearoffset
    def defineRange(self,firstDate,lastDate):
        self.year1 = firstDate
        self.year2 = lastDate
        self.startDate =  datetime.datetime(self.year1, 1, 1)
    
    def getdtTime(self):
        randdays = random.uniform(1,364*100)
        randhours = random.uniform(1,24)
        step = datetime.timedelta(days=randdays,hours=randhours)
        
        return self.startDate + step    
    

class DENGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
#         self._structure = ['DEN']
        self._value = ['am','den', 'Den','der']      
          
    
class UMGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        self._value = ['um']      
          
class UHRGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        self._value = ['uhr']             


    
class ABPGermanDateGenerator(DateGenerator):
    def __init__(self):
          
        if platform.system() == 'Windows':
            self.lang= 'deu_deu.1252'
        else:
            self.lang='de_DE'   
        DateGenerator.__init__(self,self.lang)


        self._structure = [ 
                              ( (self.monthdayGen,1,90),(self.monthGen,1,100), 100)
                            , ( (self.weekdayGen,1,90),(self.monthdayGen,1,90),(self.monthGen,1,90),(self.yearGen,1,40),(self.hourGen,1,100), 100)
                            , ( (DENGenerator(self.lang),1,100),(self.monthdayGen,1,100),(self.monthGen,1,90), (self.hourGen,1,10) ,100)
                            # ??
                            ,( (self.yearGen,1,100),50)
                           
                           ]
        
    def generate(self):
        lStringForDate = []
        sStringForDate = ''

        self._generation = []
        ## getValue
        objvalue = self.getdtTime()
        self.setValue(objvalue)
        
        self._generation  = []
        for obj in self._instance:
            obj.generate()
            self._generation.append(obj)
        return self
     
    def getDayParts():
        return self.lDayParts[random.randint(0,len(lDayParts)-1)]
    
class ABPRecordGenerator(textGenerator):
    """
        Generator for death records
        
        missing: 
            location:  street names for passau?
            priester
            bemerkung ??
    """  
    if platform.system() == 'Windows':
        lang= 'deu_deu.1252'
    else:
        lang='de_DE.utf8'    
        
    # per type as wel!!
    lClassesToBeLearnt = [[],[]]
    lClassesToBeLearnt[1] = [
         'deathreasonGenerator'
        ,'doktorGenerator'
        ,'legitemGenerator'
        ,'doktorTitleGenerator'
        ,'lastNameGenerator'
        ,'firstNameGenerator'
        ,'professionGenerator'
        ,'religionGenerator'
        ,'familyStatus'
        ,'textletterRandomGenerator'
        ,'numberedItems'
        ,'location2Generator'
        ,'ageValueGenerator' 
        ,'AgeUnitGenerator'
        ,'DENGeneratornum'
        ,'MonthDayDateGenerator'
        ,'weekDayDateGenerator'
        ,'MonthDateGenerator'
        ,'UMGenerator'
        ,'HourDateGenerator'
        ,'UHRGenerator'
        ,'yearGenerator'
        ,'numericalGenerator'
        ,'textRandomGenerator'
        ,'integerGenerator'
        ,'textletterRandomGenerator'
        ,'legitimGenerator'
        ]
    
    lClassesToBeLearnt[0]= [
        'deathreasonGenerator'
        ,'doktorGenerator'
        ,'PersonName2'
        ,'AgeGenerator'
        ,'ABPGermanDateGenerator'
        ]
        
    # method level otherwise loadresources for each sample!!
    person= PersonName2(lang)
    date= ABPGermanDateGenerator()
    date.defineRange(1700, 2000)
    deathreasons = deathReasonColumnGenerator(lang)
    doktor= doktorGenerator(lang)
    location= location2Generator()
    profession= professionGenerator2()
    status = familyStatus()
    
    leg= legitimGenerator()
    religion=  religionGenerator()
    
    age = AgeGenerator() 
    numItems = numberedItems(15,10)     
    misc = integerGenerator(1000, 1000)
    noise = textRandomGenerator(10,8)
    noise2 = textletterRandomGenerator(10,5)


    """
        Create sequence order wise??? what is the sequence easiest to learn/tag??  column is more regular
    """
    
    def __init__(self):
        textGenerator.__init__(self,self.lang)


        myList = [self.religion,self.leg,self.numItems,
                  #self.noise,
                  self.noise2,self.person, 
                  self.date,self.deathreasons,self.doktor,self.location,self.profession,self.status, self.age, self.misc]

#         myList=[self.person]
        for g in myList: g.setClassesToBeLearnt(self.lClassesToBeLearnt)
        self._structure = []
        
        
        
#         for nbfields in range(1,len(myList)+1):
        for nbfields in range(1,4):

            # x structures per length
            # n!/(k!(n-k)!) = 364  pour 14/3
            nbx=0
            while nbx < 400:
                lseq=[]
                sCoveredFields = set()
                while len(sCoveredFields) < nbfields:
                    curfield = myList[random.randint(0,len(myList)-1)]
                    if curfield in sCoveredFields:
                        pass
                    else:
                        sCoveredFields.add(curfield)
                        lseq.append((curfield,1,100))
                        nbx += 1
                lseq.append(100)
                if tuple(lseq) not in  self._structure:
                    self._structure.append(tuple(lseq))
        

    def generate(self):
        return Generator.generate(self)
    
    
    def test(self):
        nbX = 1
        for i in range(nbX):
            for gen in [AgeGenerator, legitimGenerator,ABPRecordGenerator]:
                g= gen()
                g.generate()
                print(g._generation)
                print(g.exportAnnotatedData())
                print(g.formatAnnotatedData(g.exportAnnotatedData()))


class ABPBRecordGenerator(textGenerator):
    """
        Generator for birth records
        
        missing: 
            location:  street names for passau?
            priester
            bemerkung ??
    """  
    if platform.system() == 'Windows':
        lang= 'deu_deu.1252'
    else:
        lang='de_DE'    
        
    # per type as wel!!
    lClassesToBeLearnt = [[],[]]
    lClassesToBeLearnt[1] = [
         'birthName'
        ,'hebammeGenerator'
        ,'legitemGenerator'
#         ,'doktorTitleGenerator'
        ,'lastNameGenerator'
        ,'firstNameGenerator'
        ,'professionGenerator'
        ,'religionGenerator'
        ,'familyStatus'
        ,'textletterRandomGenerator'
        ,'numberedItems'
        ,'location2Generator'
#         ,'ageValueGenerator' 
#         ,'AgeUnitGenerator'
        ,'DENGeneratornum'
        ,'MonthDayDateGenerator'
        ,'weekDayDateGenerator'
        ,'MonthDateGenerator'
        ,'UMGenerator'
        ,'HourDateGenerator'
        ,'UHRGenerator'
        ,'yearGenerator'
        ,'numericalGenerator'
        ,'textRandomGenerator'
        ,'integerGenerator'
        ,'textletterRandomGenerator'
        ,'legitimGenerator'
        ,'zeugenName2Generator'
        ,'MutterStand2Generator'
        ,'particuleGenerator'
        ,'locationPrepositionGenerator'
        ]
    
    lClassesToBeLearnt[0]= [
#         'deathreasonGenerator'
#         ,'doktorGenerator'
#         ,'PersonName2'
#         ,'AgeGenerator'
#         ,'ABPGermanDateGenerator'
        ]
        
    # method level otherwise loadresources for each sample!!
    person= birthName(lang)
    date= ABPGermanDateGenerator()
    date.defineRange(1700, 2000)
    location= location2Generator()
    profession= professionGenerator2()
    status = familyStatus()
    
    hebammeGenerator=hebammeGenerator(lang)
    zeugen= zeugenName2Generator(lang)
    mutterStand=MutterStand2Generator(lang)
    leg= legitimGenerator()
    religion=  religionGenerator()
    
    numItems = numberedItems(15,10)     
    misc = integerGenerator(1000, 1000)
    noise = textRandomGenerator(10,8)
    noise2 = textletterRandomGenerator(10,5)


    """
        Create sequence order wise??? what is the sequence easiest to learn/tag??  column is more regular
    """
    
    def __init__(self):
        textGenerator.__init__(self,self.lang)


        myList = [ self.person
                  ,self.leg
                  ,self.numItems
                  ,self.noise2
                  ,self.hebammeGenerator
                  ,self.date
                  ,self.zeugen
                  ,self.mutterStand
                  ,self.location
                  ,self.profession
                  ,self.status
                  ,self.misc]

        #myList=[self.person,self.zeugen,self.mutterStand]
        for g in myList: g.setClassesToBeLearnt(self.lClassesToBeLearnt)
        self._structure = []
        
        
        
#         for nbfields in range(1,len(myList)+1):
        for nbfields in range(1,2):

            # x structures per length
            # n!/(k!(n-k)!) = 364  pour 14/3
            nbx=0
            while nbx < 400:
                lseq=[]
                sCoveredFields = set()
                while len(sCoveredFields) < nbfields:
                    curfield = myList[random.randint(0,len(myList)-1)]
                    if curfield in sCoveredFields:
                        pass
                    else:
                        sCoveredFields.add(curfield)
                        lseq.append((curfield,1,100))
                        nbx += 1
                lseq.append(100)
                if tuple(lseq) not in  self._structure:
                    self._structure.append(tuple(lseq))
        

    def generate(self):
        return Generator.generate(self)
    
    
class ABPWRecordGenerator(textGenerator):
    """
        Generator for wedding records
        
        missing: 
            priester
            bemerkung ??
    """  
    if platform.system() == 'Windows':
        lang= 'deu_deu.1252'
    else:
        lang='de_DE'    
        
    # per type as wel!!
    lClassesToBeLearnt = [[],[]]
    lClassesToBeLearnt[1] = [
         'lastNameGenerator'
        ,'firstNameGenerator'
        ,'professionGenerator'
        ,'religionGenerator'
        ,'familyStatus'
        ,'textletterRandomGenerator'
        ,'numberedItems'
		,'locationGenerator'
		,'undGenerator'
		,'CUMSACRGenerator'
		,'GEBGenerator'
        ,'location2Generator'
        ,'ageValueGenerator' 
        ,'AgeUnitGenerator'
        ,'DENGeneratornum'
        ,'MonthDayDateGenerator'
        ,'weekDayDateGenerator'
        ,'MonthDateGenerator'
        ,'UMGenerator'
        ,'HourDateGenerator'
        ,'UHRGenerator'
        ,'yearGenerator'
        ,'numericalGenerator'
        ,'textRandomGenerator'
        ,'integerGenerator'
        ,'textletterRandomGenerator'
        ,'legitimGenerator'
        #,'zeugenName2Generator'
        ,'MutterStand2Generator'
        ,'particuleGenerator'
        ,'locationPrepositionGenerator'
        ]
    
    lClassesToBeLearnt[0]= [
#         'deathreasonGenerator'
#         ,'doktorGenerator'
#         ,'PersonName2'
#         ,'AgeGenerator'
#         ,'ABPGermanDateGenerator'
        ]
        
    # method level otherwise loadresources for each sample!!
    brautigam = PersonName(lang)
    braut= BrautNameGenerator(lang)
    date= ABPGermanDateGenerator()
    date.defineRange(1700, 2000)
    location= location2Generator()
    profession= professionGenerator2()
    status = familyStatus()
    parent1= PersonName(lang)
    parent2= BrautNameGenerator(lang)
    zeugen= zeugenName2Generator(lang)
    hebamme= hebammeGenerator(lang)
    mutterStand=MutterStand2Generator(lang)
    leg= legitimGenerator()
    religion=  religionGenerator()
    
    numItems = numberedItems(15,10)     
    misc = integerGenerator(1000, 1000)
    noise = textRandomGenerator(10,8)
    noise2 = textletterRandomGenerator(10,5)


    """
        Create sequence order wise??? what is the sequence easiest to learn/tag??  column is more regular
    """
    
    def __init__(self):
        textGenerator.__init__(self,self.lang)


        myList = [ self.brautigam
                  ,self.braut 
                  ,self.leg
                  ,self.numItems
                  ,self.noise2
                  ,self.hebamme
                  ,self.date
                  ,self.zeugen
                  ,self.mutterStand
                  ,self.location
                  ,self.profession
                  ,self.status
                  ,self.misc]

        #myList=[self.person,self.zeugen,self.mutterStand]
        for g in myList: g.setClassesToBeLearnt(self.lClassesToBeLearnt)
        self._structure = []
        
        
        
#         for nbfields in range(1,len(myList)+1):
        for nbfields in range(1,2):

            # x structures per length
            # n!/(k!(n-k)!) = 364  pour 14/3
            nbx=0
            while nbx < 400:
                lseq=[]
                sCoveredFields = set()
                while len(sCoveredFields) < nbfields:
                    curfield = myList[random.randint(0,len(myList)-1)]
                    if curfield in sCoveredFields:
                        pass
                    else:
                        sCoveredFields.add(curfield)
                        lseq.append((curfield,1,100))
                        nbx += 1
                lseq.append(100)
                if tuple(lseq) not in  self._structure:
                    self._structure.append(tuple(lseq))
        

    def generate(self):
        return Generator.generate(self)
class ABPRecordGeneratorTOK(textGenerator):
    """
        generator for string/tok normalisation
    """  
    if platform.system() == 'Windows':
        lang= 'deu_deu.1252'
    else:
        lang='de_DE'    
        
    # method level otherwise loadresources for each sample!!
    person= PersonName2(lang)
    date= ABPGermanDateGenerator()
    date.defineRange(1700, 1900)
    deathreasons = deathReasonColumnGenerator(lang)
    doktor= doktorGenerator(lang)
    location= location2Generator()
    profession= professionGenerator()
    status = familyStatus()
    
    
    def __init__(self):
        textGenerator.__init__(self,self.lang)

        myList = [self.person, self.date,self.deathreasons,self.doktor,self.location,self.profession,self.status]
        
        self._structure = []
        
        for nbfields in range(1,4): #3
            nbx=0
            while nbx < 100: # 30 
#             for nbx in range(0,30):
                lseq=[]
                sCoveredFields = set()
                while len(sCoveredFields) < nbfields:
                    curfield = myList[random.randint(0,len(myList)-1)]
                    if curfield in sCoveredFields:
                        pass
                    else:
                        sCoveredFields.add(curfield)
                        lseq.append((curfield,1,100))
                        nbx += 1
                lseq.append(100)
#                 print "seq:", lseq
                if tuple(lseq) not in  self._structure:
                    self._structure.append(tuple(lseq))
        

    def generate(self):
        return Generator.generate(self)
    
         

def ABP(options,args):
    """
    """
    
    if options.bFairseq:
        #                   args[2]
        iosource = open(sys.argv[2],'w',encoding='utf-8')
        iotarget = open(sys.argv[3],'w',encoding='utf-8')
    
    
    if options.bTok:
        g = ABPRecordGeneratorTOK()
        for i in range(nbX):
            g.instantiate()
            g.generate()
            g.GTForTokenization()
    else:
        if options.bLoad:
            with gzip.open(os.path.join(options.dirname,options.name+".pkl"), "rb") as fd:
                g = pickle.load(fd)        
                print('generator loaded:%s'%(os.path.join(options.dirname,options.name+".pkl")))
                print (g.__class__.__name__)
                print (g.getNoiseLevel())
        else:     
            if options.recordtype == 'D':
                g = ABPRecordGenerator()
            elif options.recordtype == 'B':
                g = ABPBRecordGenerator()
            elif options.recordtype == 'W':
                g = ABPWRecordGenerator()
            else:
                print ('record type not covered: ', options.recordtype)
                sys.exit(0)
            g.setNoiseType(options.noiseType)
            g.setNoiseLevel(options.noiseLevel)
        
        if options.bconll:
            lReport={}
        
            lvlrange =  [0,10]
            lfd=[None for i in range(len(lvlrange))]
            for i,lvl in enumerate(lvlrange):
                lfd[i] = open(os.path.join(options.dirname,options.name+"_%s_%s.txt"%(lvl,g.getNoiseType())), "w",encoding='utf-8')
            
        for i in range(options.nbX):
            g.instantiate()
            # store the history?
            g.generate()
            try:lReport[tuple(g._instance)] +=1
            except KeyError: lReport[tuple(g._instance)] = 1
            
            if  options.bFairseq:
                sS,sT =g.formatFairSeqWord(g.exportAnnotatedData([]))
                if len(sS.strip()) > 0:
                    iosource.write("%s\n"%sS)
                    iotarget.write("%s\n"%sT)
            
            elif options.bconll:
                for i,lvl in enumerate(lvlrange):
                    g.setNoiseLevel(lvl)
                    sGen = g.formatAnnotatedData(g.exportAnnotatedData([ "None","None" ,"None"]),mode=2)
                    lfd[i].write(sGen)
        
        if options.bconll:
            [lfd[i].write("# %s %s\n"%(lReport[inst],inst)) for i in range(len(lvlrange)) for inst in lReport] 
            [fd.close() for fd in lfd]
        
        if options.bFairseq:
            iosource.close()
            iotarget.close()
        
#         elif options.bconll:
#             if g is not None and  not options.bLoad:
#                 with gzip.open(os.path.join(options.dirname,options.name+".pkl"), "wb") as fd:
#                     pickle.dump(g, fd, protocol=2)
    
if __name__ == "__main__":

    if platform.system() == 'Windows':
        lang= 'deu_deu.1252'
    else:
        lang='de_DE.UTF-8'        

    parser = OptionParser(usage="", version="0.1")
    parser.description = "text Generator"
    parser.add_option("--model", dest="name",  action="store", type="string",default="test", help="model name")
    parser.add_option("--dir", dest="dirname",  action="store", type="string", default=".",help="directory to store model")
    parser.add_option("--noise", dest="noiseType",  action="store", type=int, default=0, help="add noise of type N")
    parser.add_option("--noiselvl", dest="noiseLevel",  action="store", type=int, default=10, help="noise level (percentage) NN")

    parser.add_option("--record", dest="recordtype",  action="store", type="string", default='B', help="B W or D")


    parser.add_option("--load", dest="bLoad",  action="store_true", default=False, help="load model")
    parser.add_option("--number", dest="nbX",  action="store", type=int, default=10,help="number of samples")
    parser.add_option("--tok", dest="bTok",  action="store", type=int,default=False, help="correct tokenisation GT")
    parser.add_option("--fairseq", dest="bFairseq",  action="store", type=int, default=False,help="seq2seq GT")
    parser.add_option("--conll", dest="bconll",  action="store_true", default=True,help="conll like GT")
    

    (options, args) = parser.parse_args()    

    ABP(options,args)
#     g = PersonName2(lang)
    
#     g= ABPGermanDateGenerator()
#     g.defineRange(1900, 2000)
#     g = location2Generator()
#     g= deathReasonColumnGenerator('deu_deu.1252')
#     g= numberedItems(20,10)
    
