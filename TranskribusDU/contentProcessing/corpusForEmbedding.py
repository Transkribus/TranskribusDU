# -*- coding: utf-8 -*-
"""
generate textual corpus from PageXml  (for embeddings)
create a training structure with test and valid files

Dec. 2019
copyright Naver labs Europe 2018

@author:  Hervé Déjean
"""
import sys, os
import glob
from optparse import OptionParser
from collections import defaultdict
import lxml.etree as etree
from random import shuffle

from common.trace import traceln

class XML_to_Text:
    """
    convert a file or a collection to .txt files
    """
    NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    iTextLine  = 0
    iCell      = 1
    iRow       = 2
    iColumn    = 3

    def __init__(self,options):
        """
        Constructor
        sLayout: cell, row, column
        """
        self.sDocPattern = options.ext
        self.sDocIgnorePattern = "_duxxx.mpxml"
        self.sLayout    = options.layout
        self.iTextLine  = 0
        self.iCell      = 1
        self.iRow       = 2
        self.iColumn    = 3
        
    def extractText(self,doc):
        """
            Generate a sequence of characters using slayout
            For table: TextLine id="p1_r4l27" custom="readingOrder {index:26;}" DU_cluster="3" row="3" rowSpan="1" col="3" colSpan="1">
                !! Assume correct 'order' in  cells
            
        """
        lAllLines=[]
        #per page!
        for page in doc.getroot().xpath(".//pc:Page", namespaces={"pc":self.NS_PAGE_XML}):
            dS= defaultdict(list)
            for tl in page.xpath(".//pc:TextLine", namespaces={"pc":self.NS_PAGE_XML}):
                if self.sLayout == self.iCell:
                    key = "%s_%s"%(tl.get('row'), tl.get('col'))
                elif self.sLayout == self.iRow:
                    key = tl.get('row')
                elif self.sLayout == self.iColumn:
                    key = tl.get('col')
                elif self.sLayout == self.iTextLine:
                    key = tl.get('id')                
                else:
                    print('specify layout structure!')
                ltext = tl.findall('./pc:TextEquiv/pc:Unicode', namespaces={"pc":self.NS_PAGE_XML})
                if ltext!=[]:
                    txt = ' '.join([x.text for x in ltext if x.text is not None])
                    dS[key].append(txt.strip())
            
            [lAllLines.append(l) for l in dS.values()]
            
#                 fd.write(" ".join(lText))
#                 fd.write("\n")
        return lAllLines
    
    
    def createTrainingFolder(self,destDir,lAllLines,prcValid,prcTest):
        """
            train/split...
            test.txt
            valid.txt
            
            provide percentage for test and valid
        """
        shuffle(lAllLines)
        lenTr=len(lAllLines)
        print (lenTr, int(lenTr*prcTest),int(lenTr*prcValid))
        
        lTestLines  = lAllLines[:int(lenTr*prcTest)]
        lValidLines = lAllLines[int(lenTr*prcTest):int(lenTr*prcValid)+int(lenTr*prcTest)]
        lTrainingLines = lAllLines[int(lenTr*prcTest)+int(lenTr*prcValid):]
        
        print (f'{lenTr} --> Train:{len(lTrainingLines)}\tTest:{len(lTestLines)}\tValid:{len(lValidLines)}')
        
        #create folder
        trainDir = os.path.join(destDir, "train")
        #Creating folder structure
        if os.path.exists(trainDir):
            print(f'deleting folder {trainDir}')
            os.rmdir(trainDir)
        
        print(f'- creating folder: {trainDir}')
        os.mkdir(trainDir)

        nbTokenTrn=0
        nbTokenTst=0
        nbTokenVld=0

        with open( os.path.join(destDir, "train.txt"), "w", encoding="utf-8") as fd:
            for lText in lTrainingLines:
                nbTokenTrn+=len(lText)
                fd.write(" ".join(lText))
                fd.write("\n")
        with open( os.path.join(destDir, "test.txt"), "w", encoding="utf-8") as fd:
            for lText in lTestLines:
                nbTokenTst+=len(lText)
                fd.write(" ".join(lText))
                fd.write("\n")
        with open( os.path.join(destDir, "valid.txt"), "w", encoding="utf-8") as fd:
            for lText in lValidLines:
                nbTokenVld+=len(lText)
                fd.write(" ".join(lText))
                fd.write("\n")
        
        print (f'Nb Words: Train: {nbTokenTrn}\tTest:{nbTokenTst}\tValid {nbTokenVld}')
        
        return 
    
    def run(self, lsDir, sOutput):
        traceln("%d folders: %s" % (len(lsDir), lsDir))
        nDoc = 0
#         with open(sOutput, "w", encoding="utf-8") as fd:
        lFullTrainLines, lFullValidLines, lFullTestLines=  [], [],[]
        for sDir in lsDir:
            traceln("--- FOLDER ", sDir, self.sDocPattern)
            lDocFile = [filename for filename in glob.iglob(os.path.join(sDir, self.sDocPattern), recursive=True) 
                                    if os.path.isfile(filename) and not(filename.endswith(self.sDocIgnorePattern))]
            lDocFile.sort()
            traceln("  - %d Documents" % (len(lDocFile)))
            nDoc += len(lDocFile)
            for i, docFile in enumerate(lDocFile):
                traceln( "%d - Document %s" % (1+i, docFile))
                doc = etree.parse(docFile)
                lFullTrainLines.extend( self.extractText(doc) )        
                       
                del doc
        self.createTrainingFolder(sOutput, lFullTrainLines, 0.1, 0.1)
                
            
if __name__ == "__main__":
    sUsage="usage: %s <outputdir> <col-dir>" % sys.argv[0]

    parser = OptionParser(usage=sUsage)
    
    parser.add_option('--ext', dest='ext', action="store", type="string", help='file extension to consider')
    parser.add_option("--layout", dest='layout',  action="store", type="int"
                      , default=3
                      , help="Layout Structured used :cell(1), row(2),column(3), textline(0)")   
    (options, args) = parser.parse_args()
    
    print (options)
    try:
        sOutput     = args[0]
        lsDir       = args[1:]
        assert len(lsDir)
    except:
        traceln(sUsage % sys.argv[0])
        exit(1)
        
    doer = XML_to_Text(options)
    #doer = XML_to_Text(XML_to_Text.iRow)

    doer.run(lsDir, sOutput)
    print ('>> Done')
    
