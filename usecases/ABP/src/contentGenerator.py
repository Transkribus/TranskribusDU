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
# from dateutil.relativedelta import *

import platform

sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'TranskribusDU')


from dataGenerator.generator import Generator
from dataGenerator.textGenerator import textGenerator 
from dataGenerator.numericalGenerator import integerGenerator, positiveIntegerGenerator
from dataGenerator.textRandomGenerator import textRandomGenerator
from dataGenerator.textRandomGenerator import textletterstRandomGenerator


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
        self.loadResourcesFromList( [[('Jahre',60),('Monate',20),('Wochen',10),('Tag',10),('Stunde',10)]])        
        
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
                ( (self.measure,1,100), (self.unit,1,100),100),
                ( (self.measure,1,100), (self.unit,1,100),(self.measure,1,100), (self.unit,1,100),100),
                ( (self.measure,1,100), (self.unit,1,100),(self.measure,1,100), (self.unit,1,100),(self.measure,1,100), (self.unit,1,100),100)
             ]
    def generate(self):
        return Generator.generate(self)    
    
class legitimGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
#         self._value = ['leg','legitim','illeg','illegitim']
        self.loadResourcesFromList( [[('leg',60),('legitim',20),('illeg',10),('illegitim',20)]])        
            
class religionGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self.loadResourcesFromList( [[('K',30),('kath',40),('katholic',5),('katho',5),('K. R.',5),("evangelist",5),('evang.',5),("evg.",5)]])  
#         self._value = ['k','kath','katholic','katho','k. R.','evangelist','evang.','evg.']
    
class familyStatus(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self.loadResourcesFromList( [[('kind',30),('Säugling',5),('ledig',20), ('verehelichet.',10),('erehelicht',10),('witwe',20),('witwer',10),('verwitwet',5),('verw.',5),('verheirathet',10),('verhei',10)]])
                     
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
        self._lpath=[os.path.abspath('../resources/deathreason.pkl')]
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
        
class location2Generator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'location'
        self.location = locationGenerator()
        self.location2 = locationGenerator()
        self.prep = locationPrepositionGenerator()    
        self._structure = [
                ( (self.location2,1,20),(self.prep,1,10), (self.location,1,100),(legitimGenerator(),1,10),100)
             ]
    def generate(self):
        return Generator.generate(self)        
        
class locationGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'location'
        self._lpath=[os.path.abspath('../resources/location.pkl')]
        self._value = list(map(lambda x:x[0],self.loadResources(self._lpath)))
        self._lenRes= len(self._lresources)
    

"""
    occupation :
    profession
    
    [leg,..][knabe, tocher,sohn,kind] deSR  person, profession,[zu] location
    unehelicher Knabe der Maria Friedl = Schödermaier
    ehl. Kind des Uhr¬ machers Martin Grammel
    Kind der Creszenz Gigl, Bauerstochter von Haid.
"""
class professionGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'profession'
        self._lpath=[os.path.abspath('../resources/profession.pkl')]
        self._value = list(map(lambda x:x[0],self.loadResources(self._lpath)))
        self._lenRes= len(self._lresources)
    

class firstNameGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'firstName'
        self._lpath=[os.path.abspath('../resources/firstname.pkl')]
        self._value = list(map(lambda x:x[0],self.loadResources(self._lpath)))
        self._lenRes= len(self._lresources)
    

class lastNameGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'firstName'
        self._lpath=[os.path.abspath('../resources/lastname.pkl')]
        self._value = list(map(lambda x:x[0],self.loadResources(self._lpath)))
        
        self._lenRes= len(self._lresources)
    

class CUMSACRGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        self._value = ['cum sacramentum', 'cum sacr']      
          
      
      
        
class PersonName2(textGenerator):
    """
        TODO: add the profile for the noisegenerator: merge 10%, split ? 
        
        add a point at the end of the person :  Gotthard Wurstbauer.
    """
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        
        self._structure = [
                 ( (firstNameGenerator(),1,100), (lastNameGenerator(),1,100), (religionGenerator(),1,20),(legitimGenerator(),1,10),100)
                ,( (lastNameGenerator(),1,100),  (firstNameGenerator(),1,100),(religionGenerator(),1,20),(legitimGenerator(),1,10),100)
                #noisy ones ?
                ,(    (firstNameGenerator(),1,100), (CUMSACRGenerator(lang),1,25),100)
                ,(    (lastNameGenerator(),1,100), (CUMSACRGenerator(lang),1,25),100)

             ]
    
    def generate(self):
        return Generator.generate(self)            
        
class PersonName(textGenerator):
    """
        TODO: add the profile for the noisegenerator: merge 10%, split ? 
    """
    def __init__(self,lang):
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
        self._value = ['Arzt','Dr', 'Landarzt','doktor', 'hebamme','Frau','Sch.' 'Schwester']    
        
class doktorGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        self._structure = [
                ( (doktorTitleGenerator(lang),1,100), (lastNameGenerator(),1,100),50),
                ( (ohneArtzGenerator(),1,100),50)
             ]
    def generate(self):
        return Generator.generate(self)  
    
class MonthDateGenerator(textGenerator):
    def __init__(self,lang,value=None):
        textGenerator.__init__(self,lang)
        self._value = [value]   
        self.realization = ['b','B','m']
        
                   
    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.month]
    
    def generate(self):
        # P3 or P2
        try:self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))
        except UnicodeDecodeError: self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')

        return self

class MonthDayDateGenerator(textGenerator):
    def __init__(self,lang,value=None):
        textGenerator.__init__(self,lang)
        self._value = [value]     
        self.realization=  ['d']

    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.day]

    def generate(self):
        try: self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))
        except UnicodeDecodeError: self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')
        return self

class weekDayDateGenerator(textGenerator):
    def __init__(self,lang,value=None):
        self._fulldate = None
        textGenerator.__init__(self,lang)
        self._value = [value]     
        self.realization=['a','A','w']
         
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
        self._value = [value]
        self.realization=['H','I']      
    
    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.hour]
    
    def generate(self):
        try:self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))
        except UnicodeDecodeError: self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')
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
        self._value=['abends','morgens','nachmittags','mittags','nacht','fruh']
        
         
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
                           ((self.yearGen,1,90),(self.weekdayGen,1,90),(self.monthdayGen,1,90),(self.monthGen,1,90),(self.hourGen,1,100), 75)
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
            self.lang='de-DE'   
        DateGenerator.__init__(self,self.lang)


        self._structure = [ 
                             ( (self.weekdayGen,1,90),(self.monthdayGen,1,90),(self.monthGen,1,90),(self.yearGen,1,40),(self.hourGen,1,100), 100)
                            ,( (DENGenerator(self.lang),1,100),(self.monthdayGen,1,100),(self.monthGen,1,90), (self.hourGen,1,10) ,100)
                           
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

        return self
     
    def getDayParts():
        return self.lDayParts[random.randint(0,len(lDayParts)-1)]
    
class ABPRecordGenerator(textGenerator):
    """
        Generator composed of:
            firstname
            date
    """  
    if platform.system() == 'Windows':
        lang= 'deu_deu.1252'
    else:
        lang='de-DE'    
        
    # method level otherwise loadresources for each sample!!
    person= PersonName2(lang)
    date= ABPGermanDateGenerator()
    date.defineRange(1900, 2000)
    deathreasons = deathReasonColumnGenerator(lang)
    doktor= doktorGenerator(lang)
    location= location2Generator()
    profession= professionGenerator()
    status = familyStatus()
    
    leg= legitimGenerator()
    religion=  religionGenerator()
    
    age = AgeGenerator() 
    numItems = numberedItems(15,10)     
    misc = integerGenerator(1000, 1000)
    noise = textRandomGenerator(10,8)
    noise2 = textletterstRandomGenerator(10,5)

    #missing: legitimGenerator, religionGenerator

    
    def __init__(self):
#         if platform.system() == 'Windows':
#             self.lang= 'deu_deu.1252'
#         else:
#             self.lang='de-DE'  
        textGenerator.__init__(self,self.lang)


#         myList = [self.person,self.person2]
        myList = [self.religion,self.leg,self.numItems,self.noise,self.noise2,self.person, 
                  self.date,self.deathreasons,self.doktor,self.location,self.profession,self.status, self.age, self.misc]
#         myList = [self.numItems,self.noise,self.noise2,firstNameGenerator(), lastNameGenerator(),self.person, self.date,self.deathreasons,self.doktor,self.location,self.profession,self.status, self.age, self.misc]
#         myList = [self.age,self.date]

#         myList = [firstNameGenerator(), lastNameGenerator()]
        
        self._structure = []
        
        
        ## need to collect histogram of the generated structure 
        
#         for nbfields in range(1,len(myList)+1):
        for nbfields in range(1,3):

            # x structures per length
            nbx=0
            while nbx < 30:
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
        
#         for s in self._structure:
#             print "# ",len(s),s

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

class ABPRecordGeneratorTOK(textGenerator):
    """
        generator for string/tok normalisation
    """  
    if platform.system() == 'Windows':
        lang= 'deu_deu.1252'
    else:
        lang='de-DE'    
        
    # method level otherwise loadresources for each sample!!
    person= PersonName2(lang)
    date= ABPGermanDateGenerator()
    date.defineRange(1900, 2000)
    deathreasons = deathReasonColumnGenerator(lang)
    doktor= doktorGenerator(lang)
    location= location2Generator()
    profession= professionGenerator()
    status = familyStatus()
    
    
    def __init__(self):
        textGenerator.__init__(self,self.lang)

        myList = [self.person, self.date,self.deathreasons,self.doktor,self.location,self.profession,self.status]
        
        self._structure = []
        
        for nbfields in range(1,3):
            nbx=0
            while nbx < 30:
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
    
         
if __name__ == "__main__":

    if platform.system() == 'Windows':
        lang= 'deu_deu.1252'
    else:
        lang='de-DE'        
    try:
        nbX = int(sys.argv[1])
    except:nbX = 10
#     g = PersonName2(lang)
    
#     g= ABPGermanDateGenerator()
#     g.defineRange(1900, 2000)
#     g = location2Generator()
#     g= deathReasonColumnGenerator('deu_deu.1252')
#     g= numberedItems(20,10)
    
    bGTTOK=False
    
    if bGTTOK:
        g = ABPRecordGeneratorTOK()
        for i in range(nbX):
            g.instantiate()
            g.generate()
            g.GTForTokenization()
    else:
        g = ABPRecordGenerator()
#         g = ABPGermanDateGenerator()
#         g.defineRange(1900, 2000)
        lReport={}
        for i in range(nbX):
            g.instantiate()
            g.generate()
            try:lReport[tuple(g._instance)] +=1
            except KeyError: lReport[tuple(g._instance)] = 1
            uString= g.formatAnnotatedData(g.exportAnnotatedData([]),mode=2)
    
        for inst in lReport:
            print("# ", lReport[inst],inst) 
            
        