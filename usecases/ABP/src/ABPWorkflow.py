# -*- coding: utf-8 -*-
"""


    ABPWorkflow.py

    prep
    Process a full collection with table analysis and IE extraction
        
    H. Déjean
    
    copyright Naver LAbs Europe 2017
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
import sys, os

from optparse import OptionParser

import csv
import glob

sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'src')

from tasks.performCVLLA import LAProcessor
class ABPWorkflow:

    cCVLProfileTabReg ="""
[%%General]  
FileList="%s"
OutputDirPath="%s"
FileNamePattern=<c:0>.<old>
SaveInfo\Compression=-1
SaveInfo\Mode=2
SaveInfo\DeleteOriginal=false
SaveInfo\InputDirIsOutputDir=true
PluginBatch\pluginList=Forms Analysis | Apply Template (Single)
PluginBatch\FormAnalysis\FormFeatures\\formTemplate="%s"
PluginBatch\FormAnalysis\FormFeatures\distThreshold=200
PluginBatch\FormAnalysis\FormFeatures\colinearityThreshold=20
PluginBatch\FormAnalysis\FormFeatures\\variationThresholdLower=0.2
PluginBatch\FormAnalysis\FormFeatures\\variationThresholdUpper=0.3
PluginBatch\FormAnalysis\FormFeatures\saveChilds=false
     """
     
    if sys.platform == 'win32':
        cNomacs = '"C:\\Program Files\\READFramework\\bin\\nomacs.exe"'
        #cNomacsold = '"C:\\Program Files\\READFramework\\bin\\nomacs.exe"'
        cNomacsold = '"C:\\Program Files\\READFramework2\\nomacs-x64\\nomacs.exe"'

    else:
        cNomacs = "/opt/Tools/src/tuwien-2017/nomacs/nomacs"
        cNomacsold = "/opt/Tools/src/tuwien-2017/nomacs/nomacs"
        
    
    def __init__(self):
        
        self.templatelocations= None
        self.laproc = LAProcessor()
    
        self.templateDir ='C:\localdata\READ\ABP\ABP_DR_template\trnskrbs_6285\col\20404'
        self.DRDir = 'C:\localdata\READ\ABP\DEATHRECORDS'
        self.dtemplateIndex = {
            '1':'ABP_S_1847-1878_01-01'
            ,'2':'ABP_S_1847-1878_02-01'
            ,'3':'ABP_S_1847-1878_03-01'
            ,'4':'ABP_S_1847-1878_04-01'
            ,'5':'ABP_S_1847-1878_05-01'
            ,'6':'ABP_S_1847-1878_06-01'
            ,'7':'ABP_S_1847-1878_07-01'
            ,'8':'ABP_S_1847-1878_08-01'
            ,'9':'ABP_S_1847-1878_09-01'
            ,'10':'ABP_S_1847-1878_10-01'
            ,'11':'ABP_S_1847-1878_11-01'
            ,'hand':'None'

            }
    def openTemplateDocFile(self,dbfilename):
        """
        Adldorf    M    S    004_03    1803    1850    241    364    2: Laufzeitbeginn 13. Februar 1803; 6: Bericht über Unwetter 1844 auf dem Vorsatzblatt, nicht zuzuordnende Notizen am Schluß; 7: Sterbefälle Februar 1803 siehe auch Bd. 003_03    S_Adldorf_004-03    1803    1850    349    363    no    8    349        hand    d    1    1    10            x            letzte Seite falsch; nur ungerade Seiten    363 statt 364    S_Adldorf_004-03    S_Adldorf_004-03_0349.jpg    Adldorf_004_03                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

{'': '', 'Seite von': '0020', 'Band-Nr.': '005_05', 'Pfarrei': 'Zimmern', 'Anmerkung': '', 'drawn?': '', 'relevant from page': '20', 
'OLD Template Subcategory': '46', 'link to local folder': 'S_Zimmern_005-05', 'Kirchen-buchart': 'M', 'end': '1876', 'volume on matricula': 'Zimmern_005_05', 
'pageno. of first template occurrence ': '20', 'count of different categories per volume': '1', 'Seite bis': '0020', 'changes on page no.': '', 
'Laufzeit von': '1876', 'local path to first template image': 'S_Zimmern_005-05_0020.jpg', 'begin': '1876', 'No_Cols': '10', 'found records in database': 'yes', 
'two_tables': 'x', 'Date_3cols': '', 'count of images': '1', 'Laufzeit bis': '1876', 'Sakra-mente / Zusam-menstel-lungen': 'S',
 'count of different templates per volume': '1', 'Bemerkung': '7: enth\xef\xbf\xbdlt auch Sterbef\xef\xbf\xbdlle 1878', 'header_printed_kurrent': 'x', 
 'Template Category': '1', 'Korrekturhinweis': '', 'Remarks_1st_col': 'x', 'relevant to page': '20', 'name of volume': 'S_Zimmern_005-05'}

       
        """    
        sDelimiter=str(',')
        
        db=[]
        with open(dbfilename,'rb') as dbfilename:
            dbreader= csv.DictReader(dbfilename,delimiter=sDelimiter )
            for row in dbreader:
                #               M,S, T,..
                db.append((row['Kirchen-buchart'],row['Template Category'], row['name of volume']))
        
        return db


    def collectDocuments(self,coldir):
        """
            collect documents to be processed
        """
        ldir={}
        lsdir = glob.iglob(os.path.join(self.coldir, "*"))
        for doc in lsdir:
            lfiles = glob.iglob(os.path.join(doc, "*.jpg"))
            ldir[os.path.basename(doc)] = (doc,lfiles)
        return ldir
            
            
    def createRegistrationProfile(self,coldir,docname,temDir,sTemplatefile):
        # get all images files
        localpath =  os.path.abspath(os.path.join(coldir,docname))
        l =      glob.glob(os.path.join(localpath, "*.jpg"))
        l.sort()
        listfile = ";".join(l)
        listfile  = listfile.replace(os.sep,"/")
#         print coldir, temDir, sTemplatefile
        txt=  ABPWorkflow.cCVLProfileTabReg % (listfile,localpath.replace(os.sep,"/"),os.path.abspath(os.path.join(temDir,sTemplatefile)+'.pxml').replace(os.sep,"/"))

        # wb mandatory for crlf in windows
        prnfilename = os.path.join(coldir,docname)+'.prn'
        f=open(prnfilename,'wb')
        f.write(txt)
#         print prnfilename  #, txt.encode('utf-8') 
        return prnfilename
    def generateProfile(self,mycollec,templateDir,db):
        """
            for each doc: directory name + its template file
        """
        
        for dir in mycollec:
            dbentry = filter(lambda (x,y,z):dir==z,db)
            if dbentry != []:
                try:sTemplatefile= self.dtemplateIndex[dbentry[0][1]]
                except KeyError:
                    # 1;9
                    sTemplatefile= self.dtemplateIndex[dbentry[0][1].split(';')[0]]
                print dir, dbentry, sTemplatefile   
                self.createRegistrationProfile(self.coldir, dir,templateDir,sTemplatefile) 
                
    def run(self):
        """
        workflow with data reading
        """
        self.templatelocations = self.options.templateDir.decode('utf-8')
        self.temdoc = self.options.templatedocument.decode('utf-8')
        self.coldir = self.options.coldir
        print "template dir and template/doc file: ",self.templatelocations, self.temdoc
        mydb = self.openTemplateDocFile(self.temdoc)
        mycollec = self.collectDocuments(self.coldir)
        self.generateProfile(mycollec, self.templatelocations,mydb)
        
if __name__ == "__main__":

    parser = OptionParser(usage="", version="")
    parser.add_option("--coldir"  , dest='coldir'   , action="store", type="string", help="collection dir")

    parser.add_option("--template"  , dest='templateDir'   , action="store", type="string", help="template directory")
    parser.add_option("--temdoc"  , dest='templatedocument'   , action="store", type="string", help="template/doc file")

    (options, args) = parser.parse_args()
    
    abp = ABPWorkflow()
    abp.options = options
    abp.run()
