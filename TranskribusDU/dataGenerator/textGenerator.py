# -*- coding: utf-8 -*-
"""


    textGenerator.py

    create (generate) textual annotated data 
     H. Déjean
    

    copyright Xerox 2017
    READ project 


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

import pickle

from builtins import str
import gzip
import numpy as np
import locale

import random

from dataGenerator.generator import Generator 
import token

class textGenerator(Generator):
    
    """
        see faker
        https://faker.readthedocs.io/en/master/index.html
        
        Korean!
        https://faker.readthedocs.io/en/master/locales/ko_KR.html
    """
    
    def __init__(self,lang):
        self.lang = lang
        locale.setlocale(locale.LC_TIME, self.lang)        
        Generator.__init__(self,{})
        # list of content
        self._value = None
        # reference to the list of content (stored)        
        self._lresources = None
        self.isWeighted = False

    
    def getSeparator(self):
        """
            separator between token
        """
        return " "
    
    
    
       
    def generate(self):
        """
            need to take into account element frequency! done in getRandomElt
            this generate is for terminal elements (otherwise use Generator.generate() )
        """
        # 11/14/2017: 
        
        self._generation = self.getRandomElt(self._value)
        while len(self._generation.strip()) == 0:
            self._generation = self.getRandomElt(self._value)
        
        return self
       
       
    
    def delCharacter(self,s,th):
        """
             delete characters 
        """
            
        ns=""
        for i in range(len(s)):
            generateProb = random.uniform(0,100)
            if generateProb >= th:
                ns+=s[i]
        
        # at least one char
        if ns=="":ns=s[0]
        return ns
       
    def replaceCharacter(self,s,th):
        """
             add noise  (replace char) to pureGen
        """
            
        ns=""
        for i in range(len(s)):
            generateProb = random.uniform(0,100)
            if generateProb < th:
                ns+=chr(int(random.uniform(65,240)))
            else: ns+=s[i]
        return ns
       
    def noiseSplit(self,token,label):
        """
            Simply split in to 2 elements. Is there a need for N elements? 
    
            How to indicate the split?
        """
        
        #randomly split word 
        # th value in the profile
        probaTH = 10
        lTokens=[]
        lLabels=[]
        # how many splits
        #select a position where to cut
        if len(token)>5:
            poscut= random.randint(1,len(token)-1)
            generateProb = 1.0 * random.uniform(1,100)
            if generateProb < probaTH:
                tok2=token[:-poscut]
                if random.randint(1,100) >= 66:
                    tok2 += '¬'
                elif random.randint(1,100) >= 33:
                    tok2 += '-'
                else:pass
                lTokens.append(tok2)
                lTokens.append(token[-poscut:])
                #        add a third column for label about tokenisation?
                lLabels = [label[:],label]
                lLabels[0][-1] = "merge"

            else:
                lTokens = [token]
                lLabels = [label]
        else:
            lTokens = [token]
            lLabels = [label]                
        return lTokens,lLabels


    def TypedBIES(self,lList):
        """
            fixed length of types 
            
            ABPRecordGenerator Maria ['PersonName2', 'firstNameGenerator']
            ABPRecordGenerator Pfeiffer ['PersonName2', 'lastNameGenerator']
            ABPRecordGenerator Forster [None, 'professionGenerator']

        """
        lNewList = []
        for pos,(token,llabels) in enumerate(lList):
            # need to copy while we update llabels but need to keep the original version for the prev/next test
            lNewList.append((token,llabels[:]))
            for type in range(len(self.lClassesToBeLearnt)):
                isAsPrev = False
                isAsNext = False
                bies="??"
                if pos > 0:
                    isAsPrev = llabels[type] ==lList[pos-1][1][type]
#                     print (llabels[type],lList[pos-1][1][type],llabels[type] == lList[pos-1][1][type])
                if pos < len(lList) -1 :
                    isAsNext = llabels[type] ==lList[pos+1][1][type]            
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
                #update     
                if lNewList[-1][1][type] != None:
                    lNewList[-1][1][type]=  bies+  llabels[type]
        return lNewList
    
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
        maxY = max(list(map(lambda xy:len(xy[1]),lList)))
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
         
    
        
    def formatFairSeqWord(self,gtdata):
        """
            FairSeq Format at character level
            C C C C   \t BIESO
        """
        lnewGT=[]
        # duplicate labels for multitoken
        for token,label in gtdata:
            # should be replace by self.tokenizer(token)
            if isinstance(token, str) : #type(token) == unicode:
                ltoken = token.split(" ")
            elif type(token) in [float,int ]:
                ltoken = [token]
            
            if len(ltoken) == 1:
                lnewGT.append((token,label))
            else:
                for tok in ltoken:
                    lnewGT.append((tok,label[:]))
    
        # compute BIES
        assert lnewGT != []
        lnewGT = self.hierarchicalBIES(lnewGT)
        
        #output for GT
        sSource  = ""
        sTarget  = ""
        for token, labels in lnewGT:
            sTarget  += labels[-1] + " "
            sSource  += str(token) + " "
        return sSource, sTarget
        
        
    def formatAnnotatedData(self,gtdata,mode=2):
        """
            format with bIES hierarchically
            
            need to tokenize the strings from dictionaries (Franz Xaver as entry for firstname)
            
            mode 1: flat stuff : return last label w/o BIES
        """
        
        lnewGT=[]
        # duplicate labels for multitoken
        for token,label in gtdata:
            # should be replace by self.tokenizer(token)
            if isinstance(token, str) : #type(token) == unicode:
                ltoken= token.split(" ")
            elif type(token) in [float,int ]:
                ltoken= [str(token)]
            
            if len(ltoken) == 1:
                # token is a str hereafter
                lnewGT.append((str(token),label))
            else:
                for tok in ltoken:
                    lnewGT.append((tok,label[:]))
    
        # compute BIES
        assert lnewGT != []
        lnewGT = self.TypedBIES(lnewGT)
        
        # noise  here?
#         lnewGT = self.noiseSplit(lnewGT)
        
        #output for GT
        sReturn = ""
        for token, labels in lnewGT:
            assert type(token) != int
            if len(str(token)) > 0:
                lTokens = [token]
                if self.getNoiseType() in [1]:
                    token = self.delCharacter(token,self.getNoiseLevel())
                # if split: add a last label: splitted!
#                 elif self.getNoiseType() in [2]:
                lTokens,luLabels= self.noiseSplit(token,labels)
                for token,label in zip(lTokens,luLabels):    
                    uString = "%s\t%s" % (token,'\t'.join(label))
                    sReturn +=uString+'\n'
        sReturn+="EOS\n"
        return sReturn
            
    def exportAnnotatedData(self,lLabels):
        # export (generated value, label) for terminal 
        self._GT  = []

        # here test if the label has to be in the classes to be learned
        for i,ltype  in enumerate(self.lClassesToBeLearnt):
            if self.getName() in ltype:
                lLabels[i]=self.getName()
        if isinstance(self._generation, str) : #type(self._generation) == unicode:
            self._GT.append((self._generation,lLabels[:]))
        elif type(self._generation) == int:
            self._GT.append((self._generation,lLabels[:]))
        else:
            for _,obj  in enumerate(self._generation):
                if isinstance(obj,str) : #type(obj) == unicode:
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
        if isinstance(self._generation, str) :
            self._tokens =  self._generation
        elif type(self._generation) == int:
            self._tokens = "%d" % (self._generation)            
        else:
            for i,obj  in enumerate(self._generation):
                if isinstance(obj,str):
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
        if isinstance(self._generation, str) : #type(self._generation) == unicode:
            self._serialization +=  self._generation
        elif type(self._generation) == int:
            self._serialization += "%d" % (self._generation)            
        else:
            for i,obj  in enumerate(self._generation):
                if isinstance(obj,str) :#type(obj) == unicode:
                    self._serialization +=  obj
                elif type(obj) == int:
                    self._serialization +=  "%d" % (self._generation)                         
                else:
                    if i == 0:
                        self._serialization +=  obj.serialize()
                    else:
                        self._serialization += self.getSeparator() + obj.serialize()                        
    #             self._serialization += self.getSeparator() + obj.serialize()
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
