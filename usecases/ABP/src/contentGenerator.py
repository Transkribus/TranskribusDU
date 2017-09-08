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
import platform
import cPickle

sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'src')

import common.Component as Component
from dataGenerator.generator import Generator
from dataGenerator.textGenerator import textGenerator 


from  ABPIEOntology import *


class coordGen(textGenerator):
    """
        und ;  -   
    """
    
class deathreasonGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'deathreason'
        self._lpath=[os.path.abspath('./resources/deathreason.pkl')]
#         self._value = self.loadResources(self._lpath)
        self._value = map(lambda x:x[0],self.loadResources(self._lpath))
        
        self._lenRes= len(self._lresources)
    

class locationGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'location'
        self._lpath=[os.path.abspath('./resources/location.pkl')]
        self._value = map(lambda x:x[0],self.loadResources(self._lpath))
        self._lenRes= len(self._lresources)
    
    
class professionGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'profession'
        self._lpath=[os.path.abspath('./resources/profession.pkl')]
#         self._value = self.loadResources(self._lpath)
        self._value = map(lambda x:x[0],self.loadResources(self._lpath))
        self._lenRes= len(self._lresources)
    

class firstNameGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'firstName'
        self._lpath=[os.path.abspath('./resources/firstname.pkl')]
#         self._value = self.loadResources(self._lpath)
        self._value = map(lambda x:x[0],self.loadResources(self._lpath))

        self._lenRes= len(self._lresources)
    

class lastNameGenerator(textGenerator):
    def __init__(self):
        textGenerator.__init__(self,lang=None)
        self._name  = 'firstName'
        self._lpath=[os.path.abspath('./resources/lastname.pkl')]
#         self._value = self.loadResources(self._lpath)
        self._value = map(lambda x:x[0],self.loadResources(self._lpath))
        
        self._lenRes= len(self._lresources)
    

class CUMSACRGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
#         self._structure = ['DEN']
        self._value = ['cum sacramentum', 'cum sacr']      
          
          
class PersonName(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        
        self._structure = [
                ( (firstNameGenerator(),1,100), (lastNameGenerator(),1,100),(CUMSACRGenerator(lang),1,25),100)
             ]
    
    def generate(self):
        return Generator.generate(self)    

class MonthDateGenerator(textGenerator):
    def __init__(self,lang,value=None):
        textGenerator.__init__(self,lang)
        self._value = [value]   
        self.realization = ['b','B','m']
#     def __repr__(self): 
#         return self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))
        
                   
    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.month]
    
    def generate(self):
        self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')
#         self._generation = self.getRandomElt(self._value)
        return self

class MonthDayDateGenerator(textGenerator):
    def __init__(self,lang,value=None):
        textGenerator.__init__(self,lang)
        self._value = [value]     
        self.realization=  ['d']
         
#     def __repr__(self): 
#         return self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))
    def setValue(self,d): 
        self._fulldate= d
        self._value = [d.day]

    def generate(self):
        self._generation = u""+self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization)).decode('latin-1')
        return self

class weekDayDateGenerator(textGenerator):
    def __init__(self,lang,value=None):
        textGenerator.__init__(self,lang)
        self._value = [value]     
        self.realization=['a','A','w']
         
    def __repr__(self): 
        return self._fulldate.strftime('%'+ '%s'%self.getRandomElt(self.realization))

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
        self._value=['abends','morgens','nachmittags','mittags']
        
         
class FullHourDateGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        self.hour=HourDateGenerator(lang)
        self._structure =  ((UMGenerator(lang),1,50),(self.hour,1,100),(DayPartsGenerator(lang),1,25)) 
    
    def setValue(self,d): 
        self.hour._fulldate= d
        self.hour.setValue(d)
                     
    def generate(self):
        lList= []
        for subgen, number,proba in self._structure:
            lList.append(subgen.generate())
        self._generation = lList
        return self
    
class DateGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
        
        self.monthGen = MonthDateGenerator(lang)
        self.monthdayGen = MonthDayDateGenerator(lang)
        self.weekdayGen= weekDayDateGenerator(lang)
        self.hourGen = FullHourDateGenerator(lang) #HourDateGenerator(lang)
#         self.hourGen = HourDateGenerator(lang) #HourDateGenerator(lang)

#         self.year= ['y','Y']
#         self.month= ['b','B','m']
#         self.weekday = ['a','A','w']
#         self.monthday= ['d']
#         self.hour = ['H', 'I']
        
#         self._structure = [ 
#                            ((self.weekday,90),(self.month,90),(self.hour,10) ,0.75),
#                            ((self.month,90),(self.weekday,90),(self.hour,10),0.5)
#                            ]
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
        self._value = ['den', 'Den']      
          
    
class UMGenerator(textGenerator):
    def __init__(self,lang):
        textGenerator.__init__(self,lang)
#         self._structure = ['']
        self._value = ['um']      
          

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
#         self._structure = [ 
#                              ( (self.monthdayGen,1,90),(self.monthGen,1,90),(self.hourGen,1,10) ,75)
#                             ,( (self.weekdayGen,1,90),(self.monthdayGen,1,90),(self.monthGen,1,90), 75)
#                             ,( (self.weekdayGen,1,100),(DENGenerator(self.lang),1,100),(self.monthdayGen,1,100),(self.monthGen,1,90), (self.hourGen,1,10) ,75)
#                             ,( (DENGenerator(self.lang),1,100),(self.monthdayGen,1,100),(self.monthGen,1,90), (self.hourGen,1,10) ,75)
# 
#                            ]
        
    def generate(self):
        lStringForDate = []
        sStringForDate = ''

        self._generation = []
        ## getValue
        objvalue = self.getdtTime()
        self.setValue(objvalue)
        
        # then build serialisation
        ## getAnnotation
        structproba = self.getRandomElt(self._structure)
#         print structproba
        struct, proba = structproba[:-1], structproba[-1]
        for obj, number,proba in struct: #self._structure:
                generateProb = random.randint(1,100)
                if generateProb < proba:
                    if isinstance(obj,Generator):
                        self._generation.append(obj.generate())
        return self
     
    def getDayParts():
        return self.lDayParts[random.randint(0,len(lDayParts)-1)]
    
class ABPRecordGenerator(textGenerator):
    """
        Generator composed of:
            firstname
            date
    """    
    def __init__(self):
        if platform.system() == 'Windows':
            self.lang= 'deu_deu.1252'
        else:
            self.lang='de-DE'  
        textGenerator.__init__(self,self.lang)

        self.person= PersonName(self.lang)
        self.date= ABPGermanDateGenerator()
        self.date.defineRange(1900, 2000)
        self.deathreasons = deathreasonGenerator()
        self.location= locationGenerator()

        
        self._structure = [
                ( ( self.person,1,100), (self.date,1,100),(self.deathreasons,1,100),100 )
                ,( ( self.person,1,100), (self.deathreasons,1,100),100 )
                ,( ( self.person,1,100), (self.location,1,100),(self.deathreasons,1,100),100 )

                ]

    def generate(self):
        return Generator.generate(self)
    
if __name__ == "__main__":

#     dateGen = ABPGermanDateGenerator()
#     dateGen.defineRange(1900, 1900)
#     for i in range(10):
#         dgen, lvalues = dateGen.generate()
#         print dgen, map(lambda x:x,lvalues)
    
    recordGen = ABPRecordGenerator()
    for i in range(10):
        recgen = recordGen.generate()
        print recgen.serialize().encode('utf-8')
#         print recgen.exportAnnotatedData()
    
        