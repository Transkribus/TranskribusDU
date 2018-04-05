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
        self.loadResourcesFromList( [[('Jahre',50),('Ja',10),('Monate',20),('Wochen',10),('Tag',10),('Stunde',10)]])        
        
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
    """
    def __init__(self,lang):
        textGenerator.__init__(self,lang=None)
        
        self._structure = [
                 ( (firstNameGenerator(),1,100), (lastNameGenerator(),1,100),(CUMSACRGenerator(lang),1,10), (religionGenerator(),1,20),(legitimGenerator(),1,10),100)
                , ( (lastNameGenerator(),1,100), (firstNameGenerator(),1,100), (CUMSACRGenerator(lang),1,10),(religionGenerator(),1,20),(legitimGenerator(),1,10),100)
                #noisy ones ?
#                 ,(    (firstNameGenerator(),1,100), (CUMSACRGenerator(lang),1,25),100)
#                 ,(    (lastNameGenerator(),1,100), (CUMSACRGenerator(lang),1,25),100)

             ]
    
    def generate(self):
        return Generator.generate(self)            
        
class PersonName(textGenerator):
    """
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
        self._value = ['Arzt','Dr', 'Landarzt','doktor', 'hebamme','Frau','Sch.', 'Schwester']    
        
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
    date.defineRange(1700, 2000)
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
    noise2 = textletterRandomGenerator(10,5)


    # per type as wel!!
    lClassesToBeLearnt =['deathreasonGenerator'
                              ,'doktorGenerator'
                                ,'doktorTitleGenerator'
                              ,'PersonName2'
                                ,'lastNameGenerator'
                                ,'firstNameGenerator'
                              ,'professionGenerator'
                              ,'religionGenerator'
                              ,'familyStatus'
                              ,'textletterRandomGenerator'
                              ,'locationGenerator'
                              ,'AgeGenerator'
                                ,'ageValueGenerator'
                                ,'AgeUnitGenerator'
                              ,'ABPGermanDateGenerator'
                                ,'DENGeneratornum'
                                ,'MonthDayDateGenerator'
                                ,'weekDayDateGenerator'
                                ,'MonthDateGenerator'
                                ,'UMGenerator'
                                ,'HourDateGenerator'
                                ,'UHRGenerator'
                                ,'yearGenerator'
                              ]
    
    def __init__(self):
        textGenerator.__init__(self,self.lang)


        myList = [self.religion,self.leg,self.numItems,
                  #self.noise,
                  self.noise2,self.person, 
                  self.date,self.deathreasons,self.doktor,self.location,self.profession,self.status, self.age, self.misc]

        
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
    date.defineRange(1700, 2000)
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
            pass
        
        g = ABPRecordGenerator()
        g.setNoiseType(options.noiseType)
        lReport={}
        fd= open(os.path.join(options.dirname,options.name+".txt"), "w",encoding='utf-8')
        for i in range(options.nbX):
            g.instantiate()
            g.generate()
            try:lReport[tuple(g._instance)] +=1
            except KeyError: lReport[tuple(g._instance)] = 1
            if  options.bFairseq:
                sS,sT =g.formatFairSeqWord(g.exportAnnotatedData([]))
                if len(sS.strip()) > 0:
                    iosource.write("%s\n"%sS)
                    iotarget.write("%s\n"%sT)
            else:
                sGen = g.formatAnnotatedData(g.exportAnnotatedData([]),mode=2)
                fd.write(sGen)
        for inst in lReport:
            fd.write("# %s %s\n"%(lReport[inst],inst)) 
        fd.close()
        
        if options.bFairseq:
            iosource.close()
            iotarget.close()
        
        elif options.bconll:
            if g is not None:
                with gzip.open(os.path.join(options.dirname,options.name+".pkl"), "wb") as fd:
                    pickle.dump(g, fd, protocol=2)
    
if __name__ == "__main__":

    if platform.system() == 'Windows':
        lang= 'deu_deu.1252'
    else:
        lang='de-DE'        

    parser = OptionParser(usage="", version="0.1")
    parser.description = "text Generator"
    parser.add_option("--model", dest="name",  action="store", type="string",default="test.pkl", help="model name")
    parser.add_option("--dir", dest="dirname",  action="store", type="string", default=".",help="directory to store model")
    parser.add_option("--noise", dest="noiseType",  action="store", type=int, default=0, help="add noise of type N")
    parser.add_option("--load", dest="bLoad",  action="store_true", default=False, help="model name")
    parser.add_option("--number", dest="nbX",  action="store", type=int, default=10,help="number of samples")
    parser.add_option("--tok", dest="bTok",  action="store", type=int,default=False, help="correct tokination GT")
    parser.add_option("--fairseq", dest="bFairseq",  action="store", type=int, default=False,help="seq2seq GT")
    parser.add_option("--conll", dest="bconll",  action="store", type=int, default=True,help="conll like GT")
    

    (options, args) = parser.parse_args()    

    ABP(options,args)
#     g = PersonName2(lang)
    
#     g= ABPGermanDateGenerator()
#     g.defineRange(1900, 2000)
#     g = location2Generator()
#     g= deathReasonColumnGenerator('deu_deu.1252')
#     g= numberedItems(20,10)
    
