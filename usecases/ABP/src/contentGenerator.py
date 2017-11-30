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
from __future__ import unicode_literals

import sys, os.path
import random
import datetime
from dateutil.relativedelta import *

import platform
import cPickle

sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'src')

from dataGenerator.generator import Generator
from dataGenerator.textGenerator import textGenerator 
from dataGenerator.numericalGenerator import integerGenerator
from dataGenerator.textRandomGenerator import textRandomGenerator



class numberingGenerator(textGenerator):
    def __init__(self,mean, sd):
        integerGenerator.__init__(self,mean, sd)
            
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
                ( (self.measure,1,100), (self.unit,1,100),100)
             ]
    def generate(self):
        return Generator.generate(self)    
    
class legitimGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._value = ['leg','legitim','illeg','illegitim']
            
class religionGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._value = ['k','kath','katholic','katho','k. R.','evangelist','evang.']
    
class familyStatus(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._value = ['kind','Säugling','ledig', 'erehelicht','witwe','witwer','verheirathet','verhei']
                     
class deathreasonGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'deathreason'
        self._lpath=[os.path.abspath('../resources/deathreason.pkl')]
        self._value = map(lambda x:x[0],self.loadResources(self._lpath))
        
        self._lenRes= len(self._lresources)
    

class locationGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'location'
        self._lpath=[os.path.abspath('../resources/location.pkl')]
        self._value = map(lambda x:x[0],self.loadResources(self._lpath))
        self._lenRes= len(self._lresources)
    
        
class professionGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'profession'
        self._lpath=[os.path.abspath('../resources/profession.pkl')]
        self._value = map(lambda x:x[0],self.loadResources(self._lpath))
        self._lenRes= len(self._lresources)
    

class firstNameGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'firstName'
        self._lpath=[os.path.abspath('../resources/firstname.pkl')]
#         self._value = map(lambda x:x[0],self.loadResources(self._lpath))
#         self._value = self.loadResources(self._lpath)
        self._value = map(lambda x:x[0],self.loadResources(self._lpath))
        self._lenRes= len(self._lresources)
    

class lastNameGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'firstName'
        self._lpath=[os.path.abspath('../resources/lastname.pkl')]
        self._value = map(lambda x:x[0],self.loadResources(self._lpath))
        
        self._lenRes= len(self._lresources)
    

class CUMSACRGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        self._value = ['cum sacramentum', 'cum sacr']      
          
          
        
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


class doktorTitleGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        self._value = ['artz','Ohne Artz','Dr', 'docktor', 'hebamme','Frau']    
        
class doktorGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        self._structure = [
                ( (doktorTitleGenerator(lang),1,100), (lastNameGenerator(),1,100),100)
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
        self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')
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
        self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')
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
        self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')
        return self
        
class HourDateGenerator(textGenerator):
    def __init__(self,lang,value=None):
        textGenerator.__init__(self,lang)
        self._value = [value]      
    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.hour]
        
class DayPartsGenerator(textGenerator):
    def __init__(self,lang,value=None):
        textGenerator.__init__(self,lang)
        self._value=['abends','morgens','nachmittags','mittags','nacht','fruh']
        
         
class FullHourDateGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        self.hour=HourDateGenerator(lang)
        self._structure =  [
                ( (UMGenerator(lang),1,50),(self.hour,1,100),(UHRGenerator(lang),1,25),(DayPartsGenerator(lang),1,25),100 )
                ] 
    
    def setValue(self,d): 
        self.hour._fulldate= d
        self.hour.setValue(d)
                     
    def generate(self):
        Generator.generate(self)
    
class DateGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        
        self.monthGen = MonthDateGenerator(lang)
        self.monthdayGen = MonthDayDateGenerator(lang)
        self.weekdayGen= weekDayDateGenerator(lang)
        self.hourGen = FullHourDateGenerator(lang) 
#         self.hourGen = HourDateGenerator(lang) #HourDateGenerator(lang)

        self._structure = [ 
                           ((self.weekdayGen,1,90),(self.monthdayGen,1,90),(self.monthGen,1,90),(self.hourGen,1,100), 75)
                           ]
    def setValue(self,v):
        """
        """
        for subgen in [self.monthGen,self.monthdayGen,self.weekdayGen,self.hourGen]:
            subgen.setValue(v)
    
    def defineRange(self,firstDate,lastDate):
        self.year1 = firstDate
        self.year2 = lastDate
        self.startDate =  datetime.datetime(self.year1, 01, 01)
    
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
                             ( (self.weekdayGen,1,90),(self.monthdayGen,1,90),(self.monthGen,1,90),(self.hourGen,1,100), 100)
                            ,( (DENGenerator(self.lang),1,100),(self.monthdayGen,1,100),(self.monthGen,1,90), (self.hourGen,1,10) ,100)
                           
                           ]
        
    def generate(self):
        lStringForDate = []
        sStringForDate = ''

        self._generation = []
        ## getValue
        objvalue = self.getdtTime()
        self.setValue(objvalue)
        
#         # then build serialisation
#         ## getAnnotation
#         structproba = self.getRandomElt(self._structure)
# #         print structproba
#         struct, proba = structproba[:-1], structproba[-1]
#         for obj, number,proba in struct: #self._structure:
#                 generateProb = random.randint(1,100)
#                 if generateProb < proba:
#                     if isinstance(obj,Generator):
#                         self._generation.append(obj.generate())

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
    person= PersonName(lang)
    person2= PersonName(lang)
    date= ABPGermanDateGenerator()
    date.defineRange(1900, 2000)
    deathreasons = deathreasonGenerator()
    doktor= doktorGenerator(lang)
    location= locationGenerator()
    profession= professionGenerator()
    status = familyStatus()
    
    age = AgeGenerator()      
    misc = integerGenerator(1000, 1000)
    noise = textRandomGenerator(10,8)
    
    def __init__(self):
#         if platform.system() == 'Windows':
#             self.lang= 'deu_deu.1252'
#         else:
#             self.lang='de-DE'  
        textGenerator.__init__(self,self.lang)


#         myList = [self.person,self.person2]
        myList = [self.noise,firstNameGenerator(), lastNameGenerator(),self.person, self.date,self.deathreasons,self.location,self.profession,self.status, self.age, self.misc]
#         myList = [self.age,self.date]

#         myList = [firstNameGenerator(), lastNameGenerator()]
        
        self._structure = []
        
        
        ## need to collect histogram of the generated structure 
        
        for nbfields in range(1,len(myList)+1):
            # x structures per length
            for nbx in range(0,5):
                lseq=[]
                sCoveredFields = set()
                while len(sCoveredFields) < nbfields:
                    curfield= myList[random.randint(0,len(myList)-1)]
                    if curfield in sCoveredFields:
                        continue
                    else:
                        sCoveredFields.add(curfield)
                        lseq.append((curfield,1,100))
                lseq.append(100)
#                 print "seq:", lseq
                self._structure.append(tuple(lseq))
        

    def generate(self):
        return Generator.generate(self)
    
    
    def test(self):
        nbX = 1
        for i in range(nbX):
            for gen in [AgeGenerator, legitimGenerator,ABPRecordGenerator]:
                g= gen()
                g.generate()
                print g._generation
                print g.exportAnnotatedData()
                print g.formatAnnotatedData(g.exportAnnotatedData())
                
if __name__ == "__main__":

  
    try:
        nbX = int(sys.argv[1])
    except:nbX = 10
    #recordGen = ABPRecordGenerator()
    g = ABPRecordGenerator()
    for i in range(nbX):
        g.instantiate()
        g.generate()
        g.generateNoise()
#         print g._generation
        print g.exportAnnotatedData([])
        
#         uString= g.formatAnnotatedData(g.exportAnnotatedData([]),mode=2)
#         if uString is not None:
#             print uString

        