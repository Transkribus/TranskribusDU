# -*- coding: utf-8 -*-
"""


    ABPCSV2XML.py
    
    convert CVS data to xml (ref format)
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
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from optparse import OptionParser

#import sys, os.path
import libxml2

import csv

def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        print line 
        print line.decode()
        yield line.encode('utf-8')

class CSV2REF(object):
    #DEFINE the version, usage and description of this particular component
    version = "v1.0"
    description = "load CSV file and convert them into ref file"
    name="CSV2XML"
    usage= ""       
    
    def __init__(self):
        
        self.inputFileName = None
        self.outputFileName  = None
        self.sDelimiter = ';'
        
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
        bInput = dParams.has_key("input")  
        if bInput: self.inputFileName  = dParams["input"]
        
        bOutput = dParams.has_key("output")
        if bOutput: self.outputFileName = dParams["output"]        
        
        if  dParams.has_key("delimiter"): self.sDelimiter=dParams['delimiter']
    
    def loadDB(self,dbfilename):
        """
            Open and read csv file
        """
        
        db=[]
        with open(dbfilename,'rb') as dbfilename:
            dbreader= csv.reader(dbfilename,delimiter=self.sDelimiter )
            for lFields in dbreader:
                db.append(lFields)

        return db

    def convertIntoRef(self,db):
        """
MKZ    PDKZ    Pfarrei    Band    Teilband    Seite    Teilseite    Nummer    Nummer Zusatz    Recto / Verso    Rolle    Nachname    Vornamen    Ort    Beruf    Jahr    Monat    Tag    Geschätzt?    Zusatz    Alter    
0        1        2        3        4         5        6                7        8                9              10        11            12        13    14        15      16      17        18         19        20
            
        """
        
        # group per page
        # or per date ? 
        dPages= {}
        for field in db:
            pagefield=field[5]
            try : 
                pnum= int(pagefield)
                try: 
                    dPages[pnum].append(field[10:18])
                except KeyError: dPages[pnum]=[field[10:18]]
            except ValueError:
                pass
            
        refdoc= libxml2.newDoc("1.0")
        rootNode= libxml2.newNode("DOCUMENT")
        refdoc.setRootElement(rootNode)
        for i,pagenum in enumerate(sorted(dPages)):
            domp=libxml2.newNode('PAGE')
            # some pages may be missing
            domp.setProp('number',str(i+1))
            domp.setProp('pagenum',str(pagenum))
            domp.setProp('nbrecords',str(len( dPages[pagenum])))
            #year(s)
            lyears = set(map(lambda x:x[5],dPages[pagenum]))
            if len(lyears)==1:
                domp.setProp('years',list(lyears)[0].decode('ISO-8859-1').encode('UTF-8'))
            else:
                domp.setProp('years',"%s-%s"%(list(lyears)[0].decode('ISO-8859-1').encode('UTF-8'),list(lyears)[1].decode('ISO-8859-1').encode('UTF-8')))
#                 print pagenum, domp.prop('years')
                
            rootNode.addChild(domp)         
            for lfields in  dPages[pagenum]:
                record = libxml2.newNode('RECORD')
                domp.addChild(record)
                record.setProp('firstname',lfields[1].decode('ISO-8859-1').encode('UTF-8'))
                record.setProp('lastname',lfields[2].decode('ISO-8859-1').encode('UTF-8'))
                record.setProp('role',lfields[0].decode('ISO-8859-1').encode('UTF-8'))
                record.setProp('location',lfields[3].decode('ISO-8859-1').encode('UTF-8'))                
                record.setProp('occupation',lfields[4].decode('ISO-8859-1').encode('UTF-8'))
                record.setProp('year',lfields[5].decode('ISO-8859-1').encode('UTF-8'))
                record.setProp('month',lfields[6].decode('ISO-8859-1').encode('UTF-8'))
                record.setProp('day',lfields[7].decode('ISO-8859-1').encode('UTF-8'))
                                              
        if self.outputFileName == '-':
            self.outputFileName = self.inputFileName[:-3]+'xml'                                              
        refdoc.saveFormatFileEnc(self.outputFileName, "UTF-8",True)
                                
    def run(self):
        
        db = self.loadDB(self.inputFileName)
        
        refxml = self.convertIntoRef(db)

if __name__ == "__main__":
    cmp = CSV2REF()

    #prepare for the parsing of the command line
    cmp.createCommandLineParser()

    dParams, args = cmp.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    cmp.setParams(dParams)
    doc = cmp.run()
    print "conversion done"