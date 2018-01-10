# -*- coding: utf-8 -*-
"""


    textGenerator.py

    create (generate) textual annotated data 
     H. Déjean
    

    copyright Xerox 2017
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

import sys
import cPickle
import gzip
import numpy as np
import locale

import random

from generator import Generator 

class textGenerator(Generator):
    
    """
        see faker
        https://faker.readthedocs.io/en/master/index.html
        
        Korean!
        https://faker.readthedocs.io/en/master/locales/ko_KR.html
        
        
        what is first generated (from lexicon,...) is a normalised form. What is needed is to apply some transformation for simulating
        real forms: hyphenation, abbreviation, typo?
    
    """
    
    def __init__(self,lang):
        self.lang = lang
        locale.setlocale(locale.LC_TIME, self.lang)        
        Generator.__init__(self)
        # list of content
        self._value = None
        # reference to the list of content (stored)        
        self._lresources = None
        self.isWeighted = False


        self._surface=None
        self._hyphenated=False
        self._merged=False #with next one
        self._abbreviated=False
        self._separator=" "
    def getSeparator(self):
        """
            separator between token
            should be a generator as well!!
        """
        return self._separator
    
    def loadResourcesFromList(self,lLists):
        """
            Open and read resource files
            take just (Value,freq)
        """
        self._lresources =[]
        for mylist in lLists:
            self._lresources.extend(mylist)
        if self.totalW is None:
            self.totalW = 1.0 * sum(map(lambda (_,y):y, self._lresources))
        if self.totalW != len(self._lresources):
            self.isWeighted = True
        if self._prob is None:
            self._prob = map(lambda (_,y):y / self.totalW,self._lresources)           
        if self._flatlr is None:
            self._flatlr = map(lambda (x,_):x,self._lresources)
        # generate many (100000) at one ! otherwise too slow
        self._lweightedIndex  = list(np.random.choice(self._flatlr,100000,p=self._prob))

        return self._lresources        

    def loadResources(self,lfilenames):
        """
            Open and read resource files
            
            take just (Value,freq)
        """
        self._lresources =[]
        for filename in lfilenames:
            res = cPickle.load(gzip.open(filename,'r'))
            self._lresources.extend(res)
        if self.totalW is None:
            self.totalW = 1.0 * sum(map(lambda (_,y):y, self._lresources))
        if self.totalW != len(self._lresources):
            self.isWeighted = True
        if self._prob is None:
            self._prob = map(lambda (_,y):y / self.totalW,self._lresources)           
        if self._flatlr is None:
            self._flatlr = map(lambda (x,_):x,self._lresources)
        # generate many (100000) at one ! otherwise too slow
        self._lweightedIndex  = list(np.random.choice(self._flatlr,100000,p=self._prob))

        return self._lresources
              
    def getValue(self):
        """
            return value 
        """
        return self._value
    
    
#     def instantiate(self):
#         """
#             for terminal elements (from a list): nothing to do
#         """
#         return []
        
    def generate(self):
        """
            need to take into account element frequency! done in getRandomElt
            this generate is for terminal elements (otherwise use Generator.generate() )
        """
        # 11/14/2017: 
        self._generation = self.getRandomElt(self._value)
        while len(unicode(self._generation).strip()) == 0:
            self._generation = self.getRandomElt(self._value)
        
        # create the noiseGenerrator?
        return self
       
       
    
    def generateNoise(self):
        """
             add noise to pureGen
        """
        #use textnoiseGen to determine if noise will be generated?
#         if self.getNoiseGen() is not None:
            
            
       
    def noiseSplit(self,lGTTokens):
        """
            S -> BM,E
            B -> BM,I
            I -> IM, I
            E -> IM,E
            
            Simply split in to 2 elements. Is there a need for N elements 
            
            noiseSplit as NoiseGenerator: a NoiseGenerator which generates the noise (n elements) 
        """
        
        #randomly split words 
        # th value in the profile
        probaTH = 100
        # how many splits
        # when to split 
        for token, labels in lGTTokens:
            generateProb = 1.0 * random.uniform(1,100)
            print (generateProb)
            if generateProb < probaTH:
                uLabels  = '\t'.join(labels)        
                print (token, uLabels)
        
        return lGTTokens

    def hierarchicalBIES(self,lList):
        """
            add BIES to labels
            input: 
            ("659" ,    [u'ABPRecordGenerator_36', u'integerGenerator_34']),
            ("71 ",     [u'ABPRecordGenerator_36', u'AgeGenerator_31', u'ageValueGenerator_32']),
            ("Wochen",  [u'ABPRecordGenerator_36', u'AgeGenerator_31', u'AgeUnitGenerator_33']),
            
            Output:
            ("659" ,    [u'B_ABPRecordGenerator_36', u'S_integerGenerator_34']),
            ("71 ",     [u'I_ABPRecordGenerator_36', u'B_AgeGenerator_31', u'S_ageValueGenerator_32']),
            ("Wochen",  [u'I_ABPRecordGenerator_36', u'E_AgeGenerator_31', u'S_AgeUnitGenerator_33']),
            
            
            10/18/2017: add M(erge with the next?) and S(split but where?:keep information:S_Pos(ition)): at a different level:
                S: need to be sent to another system to know where to split?
                M: add to BIES: merge to the next one
            
            Split :if different structure
                Prei    B_ABPRecordGenerator_36    S_lastNameGenerator_38
                (GXdqU    I_ABPRecordGenerator_36    S_textNoiseGenerator_35
                Johanna    E_ABPRecordGenerator_36    S_firstNameGenerator_37            
                
                ->
                Prei    B_ABPRecordGenerator_36    S_lastNameGenerator_38                
                (GXdqUJohanna (IE->E)E_ABPRecordGenerator_36 S_textNoiseGenerator_35+S_firstNameGenerator_37 
            
                2-step process or seq2seq?
            Merge: the elements are of the same structure
            Prei      B_ABPRecordGenerator_36        S_lastNameGenerator_38
            (GXdqU    I_ABPRecordGenerator_36        S_textNoiseGenerator_35
            Joha      (E->I)I_ABPRecordGenerator_36  BM_firstNameGenerator_37
            nna       E_ABPRecordGenerator_36        E_firstNameGenerator_37
            
            S -> BM,E
            B -> BM,I
            I -> IM, I
            E -> IM,E
              
        """
        maxY = max(map(lambda (x,y):len(y),lList))
        lTranspose =  [[0 for x in range(len(lList))] for y in range(maxY)] 
        for pos in range(len(lList)):
            _, ltags= lList[pos]
            for itag,tag in enumerate(ltags):
                lTranspose[itag][pos]=tag.split('_')[-1]
                
        for k,row in enumerate(lTranspose):
            pos = 0
            while pos < len(row):
                isAsPrev = False
                isAsNext = False
                bies="??"
                if  pos > 0:
                    isAsPrev = row[pos-1] == row[pos]
                if  pos < len(row)-1:
                    isAsNext = row[pos+1] == row[pos]
                if isAsPrev and isAsNext:
                    bies= 'I_'
                elif not isAsPrev and not isAsNext:
                    bies= 'S_'
                elif isAsPrev and not isAsNext:
                    bies= 'E_'
                elif not isAsPrev and isAsNext:
                    bies='B_'
                else:
                    pass
                try:
                    lList[pos][1][k]="%s%s"%(bies,lList[pos][1][k])
                except:pass #zero/empty slot
                pos += 1           
        return lList
         
    
    def tokenize(self,token):
        if type(token) == unicode:
            return  token.split(" ")
        elif type(token) in [float,int ]:
            return[token]
        return None
        
    def tokenizerString(self,s):
        """
            Use a generator?
            How to represent an hyphentated token? attribute SpaceAfter=No ; or hyphen=Yes
            
            need for than one level of annotation 
                BIES: entity level; 
                BIES token level:   
                otter17, p 92: FronzXlavsberingr
                FronzXlavsberingr      ????            One solution is also to learn how to normalize the sequence (seq2seq)
                oter17, p 75: Eva Schwingen LF schlögl
                Eva        S_firstname
                Schwingen  Bh_lastname  h=hyphenated
                schlögl    E_lastname
                
                  
        """
    
    def getHyphentatedToken(self,token):
        """
            Schwingen  Bh_lastname  h=hyphenated
            schlögl    E_lastname
            
            SS B-l
            SQ 
        
        for one token, return two lines    
        """
        if len(token) %2 == 0:
            ltokens=[token[:-3],token[-3:]] 
        else:
            ltokens = token
        return ltokens
    
    def formatAnnotatedData(self,gtdata,mode=2):
        """
            format with bIES hierarchically
            
            need to tokenize the strings from dictionaries (Franz Xaver as entry for firstname)
            
            mode 1: flat stuff : return last label w/o BIES
        """
        
        lnewGT=[]
        # duplicate labels for multitoken
        for token,label in gtdata:
            ltoken = self.tokenize(token)
            if type(token) == unicode:
                ltoken= token.split(" ")
            elif type(token) in [float,int ]:
                ltoken= [token]
#             #hyphe :
#             llast =  self.getHyphentatedToken(ltoken[-1])
#             print token, llast,label
#             if type(llast) == list:
#                 ltoken.pop(-1)
#                 ltoken.extend(llast)
#                 label[-1]='h='+label[-1]
            if len(ltoken) == 1:
                lnewGT.append((token,label))
            else:
                for tok in ltoken:
                    lnewGT.append((tok,label[:]))
    
        if lnewGT == []:return 
        # compute BIES
        lnewGT = self.hierarchicalBIES(lnewGT)
        
        # noise  here?
#         lnewGT = self.noiseSplit(lnewGT)
        
        if lnewGT != []:
            #output for GT
            for token, labels in lnewGT:
                uLabels  = '\t'.join(labels)
                uString = "%s\t%s" % (token,uLabels)
                print (uString.encode('utf-8'))
            print ("EOS".encode('utf-8'))
        
    def exportAnnotatedData(self,lLabels):
        """
            here surface form generation?
                tokenize the sub elements (space, spe, no sep/space)
                hyphenation
        """
        self._GT  = []

        lLabels.append(self.getName())
        
        if type(self._generation) == unicode:
            self._GT.append((self._generation,lLabels[:]))
        elif type(self._generation) == int:
            self._GT.append((self._generation,lLabels[:]))
        else:
            for _,obj  in enumerate(self._generation):
                if type(obj) == unicode:
                    self._GT.append((obj._generation,lLabels[:]))
                elif type(obj) == int:
                    self._GT.append((obj._generation,lLabels[:]))      
                else:
                    self._GT.extend(obj.exportAnnotatedData(lLabels[:]))
        
        return self._GT 
            
    def GTForTokenization(self):
        """
            GT to learn how to correct a surface string into a properly tokenized string
            
            split tolken  (hyphenation)
            merge token  (noise)
            abbrevaition : mm -> m̄   A Mar.  Anna Maria   Joan̄ B. Johann Baptist Sailer   
            
            
            not better to have xcqdqsd xwxcc wcxc   -> keep  merge keep   -> merge= merge with next token
        """
        gt= self.serialize()
        gttok=self.GTtokenize()
#         print gttok, gt
        # merge some token
        ltokens = gt.split(' ')
        i=10
        while i < 3 and len(ltokens)>3:
            lenLto=len(ltokens)
            pos= random.randint(1,lenLto-2)
            mtoken =ltokens.pop(pos+1)
            ltokens[pos]=ltokens[pos]+mtoken
            i+=1
        
        
        # split some tokens
        i=0
        lpos=[]
        while i < 2 and len(ltokens) >  1:
            lenLto=len(ltokens)
            pos= random.randint(1,lenLto-1)
            if pos not in lpos:
                lpos.append(pos)
                poscut= random.randint(2,6)
                stoken =ltokens[pos]
                if len(stoken) > 7:
                    ltokens[pos]=stoken[:-poscut]
                    ##  ¬   -   add hyph signs randomyl
                    if random.randint(1,100) >= 80:
                        if random.randint(1,100) >= 50:
                            ltokens[pos] += '¬'
                        else:
                            ltokens[pos] += '-'
                    ltokens.insert(pos+1,stoken[-poscut:])

            i+=1
        
        print (" ".join(ltokens).encode('utf-8'),'\t',gttok.encode('utf-8'))
         
    def GTtokenize(self):
        self._tokens  = ""
        
        #terminal
        if type(self._generation) == unicode:
            self._tokens =  self._generation
        elif type(self._generation) == int:
            self._tokens = "%d" % (self._generation)            
        else:
            for i,obj  in enumerate(self._generation):
                if type(obj) == unicode:
                    self._tokens +=  " "+obj
                elif type(obj) == int:
                    self._tokens +=  " %d" % (self._generation)    
                elif type(obj) == float:
                    self._tokens +=  " %f" % (self._generation)                             
                else:
                    self._tokens += " " + obj.GTtokenize()                        

        return self._tokens.strip()
                 
    def serialize(self):
        self._serialization  = ""
        
        #terminal
        if type(self._generation) == unicode:
            self._serialization +=  self._generation
        elif type(self._generation) == int:
            self._serialization += "%d" % (self._generation)            
        else:
            for i,obj  in enumerate(self._generation):
                if type(obj) == unicode:
                    self._serialization +=  obj
                elif type(obj) == int:
                    self._serialization +=  "%d" % (self._generation)    
                elif type(obj) == float:
                    self._serialization +=  "%f" % (self._generation)                             
                else:
                    if i == 0:
                        try:self._serialization +=  obj.serialize()
                        except: 
                            print (obj.serialize() , type(obj.serialize()))
                    else:
                        self._serialization += self.getSeparator() + obj.serialize()                        
        return self._serialization    
    

if __name__ == '__main__':
    cmp = textGenerator()
    cmp.createCommandLineParser()
    dParams, args = cmp.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    cmp.setParams(dParams)
    #This component is quite special since it does not take one XML as input but rather a series of files.
    #doc = cmp.loadDom()
    doc = cmp.run()    