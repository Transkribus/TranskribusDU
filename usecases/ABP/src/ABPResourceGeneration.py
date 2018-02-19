# -*- coding: utf-8 -*-
"""


    ABPResourceGeneration.py

    from ABP csv files generates pickle files for generator 
    

    copyright Naverlabs 2017
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
from __future__ import print_function
from __future__ import unicode_literals

import os
from optparse import OptionParser
from io import open
import csv
import pickle, gzip


class ResourceGen(object):
    #DEFINE the version, usage and description of this particular component
    version = "v1.0"
    description = "convert ABP csv to pickle"
    name="RessourceGen"
    usage= ""       

    def __init__(self):
        
        self.outputDir  = "."
        self.lFileRes = []
        self.lOutNames = []   
        self.lDelimitor=[] 
        self.bFreqOne = False
        self.bFreqNorm = False
        self.bHeader = True
        self.bTok =False
        self.normalizeFreq = None
        self.firstN  = -1
        
        
    ## Add a parameter to the component
    ## Syntax is similar to optparse.OptionParser.add_option (the Python module optparse, class OptionParser, method add_option)
    #@param *args    (passing by position)
    #@param **kwargs (passing by name)
    def add_option(self, *args, **kwargs):
        """add a new command line option to the parser"""
        self.parser.add_option(*args, **kwargs)
        
    def createCommandLineParser(self):
        self.parser = OptionParser(usage=self.usage, version=self.version)
        self.parser.description = self.description
        self.add_option("--freqone", dest="freqone",  action="store_true",  help="assume freq=1 for all entries")
        self.add_option("--freqnorm", dest="freqnorm",  action="store_true",  help="normalse (<1) for all entries")
        self.add_option("--normalize", dest="freanormalize",  action="store",type='int',default=None, help="normalize max weight to N")
        self.add_option("--firstN", dest="firstN",  action="store",type='int',default=None, help="take the N most frequent")
        self.add_option("--outdir", dest="outputDir", default="-", action="store", type="string", help="output folder", metavar="<dir>")
        self.add_option("--res", dest="lres",  action="append", type="string", help="resources for tagger/genrator/HTR")
        self.add_option("--name", dest="lnames",  action="append", type="string", help="output file names")
        self.add_option("--delimitor", dest="ldelimits",  action="append", type="string", help="CSV delimitors")
        self.add_option("--tokenize", dest="bTok",  action="store_true",default=False, help="perform tokenization")


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
        if dParams.has_key("outputDir"): self.outputDir = dParams["outputDir"]        
        
        if  dParams.has_key("lres"): self.lFileRes=dParams['lres']
        if  dParams.has_key("lnames"): self.lOutNames=dParams['lnames']
        if  dParams.has_key("ldelimits"): self.lDelimitor=dParams['ldelimits']
        if  dParams.has_key("freqone"): self.bFreqOne=True
        if  dParams.has_key("bTok"): self.bTok=True
        if  dParams.has_key("freqnorm"): self.bFreqNorm=True
        if  dParams.has_key("freanormalize"): self.normalizeFreq=dParams['freanormalize']
        if  dParams.has_key("firstN"): self.firstN=dParams['firstN']

    def createResource(self,destDir,dbfilename,outname,sDelimiter=' ', nbEntries=-1):
        """
            create resource files (pickel)
        """
        import operator
        
        if sDelimiter == 'tab':sDelimiter=str('\t')
        
        db={}
        fSum = 0.0
        fMax = 0.0 
        with open(dbfilename,'rb') as dbfilename:
            dbreader= csv.reader(dbfilename,delimiter=sDelimiter )
            for lFields in dbreader:
                if lFields == []:
                    continue
                try: 
                    int(lFields[1])
                    if self.bFreqOne:lFields[1] = 1
                    
                    if  len(lFields[0].strip()) > 0:
                        if self.bTok:
                            lTok = lFields[0].strip().split(' ')
                        else:lTok=[lFields[0]]
                        for tok in lTok:    
                            try:
                                db[tok.decode('utf-8').strip()] += int(lFields[1])
                            except KeyError: db[tok.decode('utf-8').strip()] = int(lFields[1])
                            fSum += int(lFields[1])
                            fMax=max(fMax,db[tok.decode('utf-8').strip()])
                except IndexError:
                    #just one column with the string; no frequency
                    if  len(lFields[0].strip()) > 0:
                        db[lFields[0].decode('utf-8').strip()] = 1
                except ValueError:
                    continue
        if self.bFreqNorm:
            for item in db:
                db[item] = db[item]/fSum
        elif self.normalizeFreq is not None:
            for item in db:
                db[item] =  1.0*self.normalizeFreq * (1.0*db[item]/fMax)
                db[item] =  max(1.0,db[item])              
                
        sorted_db = sorted(db.items(), key=operator.itemgetter(1),reverse=True)
        
        sorted_db = sorted_db[:self.firstN]
        # where to store them
        outFile=gzip.open(os.path.join(destDir,outname+'.pkl'),'w')
        print(os.path.join(destDir,outname+".pkl"))
        pickle.dump(sorted_db, outFile)
        outFile.close()        
        
        #readable version !
        outFile=open(os.path.join(destDir,outname+'.txt'),encoding='utf-8',mode='w')
        print( os.path.join(destDir,outname+".txt"))
        for x,y in sorted_db:
            outFile.write("%s\t%s\n"%(x,y))
        outFile.close()                 
        
        return dict(sorted_db[:nbEntries])
        
    def loadResources(self,filename):
        """
            Open and read resource files
            take just (Value,freq)
        """
        self._lresources =[]
        res = pickle.load(gzip.open(filename,'r'))
        return res
        
    def run(self):
        print (self.lFileRes,self.lOutNames,self.lDelimitor)
        for filename,outname ,sep in zip(self.lFileRes,self.lOutNames,self.lDelimitor):
            print (filename,outname,sep)
            mydict = self.createResource(self.outputDir, filename,outname,sDelimiter=sep,nbEntries=-1)
#             print self.loadResources(os.path.join(self.outputDir,outname+'.pkl'))

if __name__ == "__main__":
    cmp = ResourceGen()

    #prepare for the parsing of the command line
    cmp.createCommandLineParser()

    dParams, args = cmp.parseCommandLine()
    
    #Now we are back to the normal programmatic mode, we set the componenet parameters
    cmp.setParams(dParams)
    cmp.run()
