# -*- coding: utf-8 -*-
"""


    ABPCSV2XML.py
    
    convert CVS data to xml (ref format)
    need a list of ABP image keys to extract the correct line (S_Passau_Hals_008_0084)
     H. Déjean

    copyright NLE  2017
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
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from optparse import OptionParser

#import sys, os.path
from lxml import etree

import csv


class CSV2REF(object):
    #DEFINE the version, usage and description of this particular component
    version = "v1.0"
    description = "load CSV file and extract records corresponding to image key. Generate a ref "
    name="CSV2REF"
    usage= ""       
    
    def __init__(self):
        
        self.inputFileName = None
        self.outputFileName  = None
        self.sDelimiter = ',' #';'
        self.keylist= None
    ## Add a parameter to the componenet
    ## Syntax is siilar to optparse.OptionParser.add_option (the Python module optparse, class OptionParser, method add_option)
    #@param *args    (passing by position)
    #@param **kwargs (passing by name)
    def add_option(self, *args, **kwargs):
        """add a new command line option to the parser"""
        self.parser.add_option(*args, **kwargs)
        
    def createCommandLineParser(self):
        self.parser = OptionParser(usage=self.usage, version=self.version)
        self.parser.description = self.description
        self.add_option("-i", "--input", dest="input", default="-", action="store", type="string", help="input DB file", metavar="<file>")
        self.add_option("-o", "--output", dest="output", default="-", action="store", type="string", help="output REF file", metavar="<file>")
        self.add_option("-d", "--delimiter", dest="delimiter", default=";", action="store", type="string", help="delimiter used in csv", metavar="S")
#         self.add_option("--test", dest="test", default=False, action="store_true", help="test")

    def parseCommandLine(self):
        (options, args) = self.parser.parse_args()
        
        dOptions = {}
        for k,v in options.__dict__.items():
            if v != None: dOptions[k] = v
        return dOptions, args
    
    def setParams(self, dParams):
        """
        Here, we set our internal attribute according to a possibly specified value (otherwise it stays at its default value)
        """
        #if some input or output was defined, they take precedence to the config
        bInput = "input" in dParams.keys()  
        if bInput: self.inputFileName  = dParams["input"]
        
        bOutput = "output" in dParams.keys()
        if bOutput: self.outputFileName = dParams["output"]        
        
        if "delimiter" in dParams.keys(): self.sDelimiter=dParams['delimiter']
        if "keylist" in  dParams.keys(): self.keylist=dParams['keylist']
    
    
    def loadDB(self,dbfilename,keylist):
        """
            Open and read csv file
            
            'Aicha-an-der-Donau','4','3','5','Erber','Barbara','1','','','1','Täufling','1847','','','','','','','','','','','5356405','5392401','springer_2016-07-26'
            
            simply filter with parishes first
        """
        lParishes = list(map(lambda x:x[0],keylist))
        db=[]
        with open(dbfilename,'r',encoding='utf-8') as dbfilename:
            dbreader= csv.reader(dbfilename,delimiter=self.sDelimiter,quotechar="'")
            for lFields in dbreader:
                if lFields[0] in lParishes:
                    db.append(lFields)
        return db

    def convertIntoRef(self,db,keylist):
        """
        'qapfarrei','qaband','teilband','qaseite','name','vorname','folge','ort','beruf','fkt','funktion','jahr','monat','tag','alter_jahre','alter_monate','alter_wochen','alter_tage','alter_stunden','kommentar','zusatz','num_zusatz','mkz','pdkz','benutzer'
        
        'Aicha-an-der-Donau','4','3','5','Erber','Barbara','1','','','1','Täufling','1847','','','','','','','','','','','5356405','5392401','springer_2016-07-26'

        """
        
        # group per page
        # or per date ? 
        dPages= {}
        for key in keylist:
            keyfound=None
            for field in db:
                # 4: verst
                if field[9] != '4':
                    continue
                savedKey=key[:]
                bandteil = key[1].split('-')
                if len(bandteil) == 1:
                    bandteil.append('0')
#                 print(key, bandteil, field)
                if field[2] =='': field[2]='0'
                if field[3] =='': field[3]='0'                
                if field[0] == key[0] and int(field[1]) == int(bandteil[0]) and int(field[2]) == int(bandteil[1]) and int(field[3]) == int(key[2]):
                    if field[4][-1] == ' ' or field[5][-1] == ' ':continue
                    keyfound=savedKey
                    pagefield=field[3]
                    skey =field[0].replace('-','_')
                    if field[2] =='0':
                        abpkey = "%s_%03d_%04d"%(skey,int(field[1]),int(field[3]))
                    else:
                        abpkey = "%s_%03d_%02d_%04d"%(skey,int(field[1]),int(field[2]),int(field[3]))
                    try: 
                        dPages[abpkey].append(field)
                    except KeyError: dPages[abpkey]=[field]
                    continue
                
            
        rootNode= etree.Element("DOCUMENT")
        refdoc = etree.ElementTree(element=rootNode)

        for i,pagenum in enumerate(sorted(dPages)):
            domp=etree.Element("PAGE")
            # some pages may be missing
            domp.set('number',str(i+1))
#             domp.set('key',"%s_%s_%s_%s"%(field[0],field[1],field[2],field[3]))
            domp.set('pagenum',str(pagenum))
            domp.set('nbrecords',str(len( dPages[pagenum])))
            #year(s)
            lyears = set(map(lambda x:x[11],dPages[pagenum]))
            if len(lyears)==1:
                domp.set('years',list(lyears)[0])
            else:
                domp.set('years',"%s-%s"%(list(lyears)[0],list(lyears)[1]))
#                 print pagenum, domp.prop('years')
                
            rootNode.append(domp)         
            for lfields in  dPages[pagenum]:
#                 print lfields
            #'qapfarrei','qaband','teilband','qaseite','name','vorname','folge','ort','beruf','fkt','funktion','jahr','monat','tag','alter_jahre','alter_monate','alter_wochen','alter_tage','alter_stunden','kommentar','zusatz','num_zusatz','mkz','pdkz','benutzer'
            #   0           1        2           3        4      5         6      7
                record = etree.Element("RECORD")
                domp.append(record)
                record.set('lastname',lfields[4])
                record.set('firstname',lfields[5])
                record.set('role',lfields[10])
                record.set('location',lfields[7])                
                record.set('occupation',lfields[8])
                record.set('family',lfields[19])
                record.set('year',lfields[11])
                record.set('month',lfields[12])
                record.set('day',lfields[13])
                record.set('age-year',lfields[14])
                record.set('age-month',lfields[15])
                record.set('age-week',lfields[16])                                              
                record.set('age-day',lfields[17])   
                record.set('age-hour',lfields[18])                   
        if self.outputFileName == '-':
            self.outputFileName = self.inputFileName[:-3]+'xml'                                              
#         rootNode.saveFormatFileEnc(self.outputFileName, "UTF-8",True)
        refdoc.write(self.outputFileName, xml_declaration=True, encoding='UTF-8',pretty_print=True)
        
    def processKey(self,lKeys):
        """
            key can have _ in the name part! S_Bayerbach_008-01_0083
        """
        lFinalKey=[]
        for key in lKeys:
            lk = key.split('_')
            # two last elt : bandteil, (-NN), page (NNNN)
            # skip the frist one: role =S, T,..
            newkey= ["-".join(lk[1:-2])]
            newkey.extend(lk[-2:])
            lFinalKey.append(newkey) 
#             if len(lk) == 5:
#                 lFinalKey.append(['%s-%s'%(lk[1],lk[2]),lk[3],lk[4]])
#             elif len(lk) == 4: 
#                 lFinalKey.append([lk[1],lk[2],lk[3]])
        return lFinalKey
                               
    def run(self):
        
        keyfile= open(self.keylist)
        lKeys= list(map(lambda x:x[:-5].strip(),keyfile.readlines()))
        lKeys = self.processKey(lKeys)
        db = self.loadDB(self.inputFileName,lKeys)
        
        refxml = self.convertIntoRef(db,lKeys)

if __name__ == "__main__":
    cmp = CSV2REF()

    #prepare for the parsing of the command line
    cmp.createCommandLineParser()
    cmp.add_option("-k", '--key',dest="keylist", action="store", type="string", help="file containing the image keys")

    dParams, args = cmp.parseCommandLine()

    cmp.setParams(dParams)
    doc = cmp.run()
    print ("conversion done")