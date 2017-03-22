# -*- coding: utf-8 -*-

"""
    Utility to compute statistics regarding a PageXml collection.
    
    How many document? pages? objects? labels?
    
    The raw result is stored as a pikle file in a CSV file.  (in the future version!!!) 
    The statistics are reported on stdout.
    
    Copyright Xerox(C) 2017 JL. Meunier

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
import sys, os, collections, pickle, glob, libxml2
from optparse import OptionParser


# ===============================================================================================================
#DEFINING THE CLASS OF GRAPH WE USE

# ===============================================================================================================

class DoubleHistogram:
    """
    Double keyed histogram
    """
    def __init__(self, name):
        self.name = name
        self.dCnt = collections.defaultdict(lambda : collections.defaultdict(int) )
        
    def seenK1K2(self, k1, k2):
        self.dCnt[k1][k2] += 1
    
    #--- First Key
    def addFirstKeys(self, lk1):
        """
        Make sure those key are present in the histogram, possibly with count of zero
        """
        for k1 in lk1: self.dCnt[k1]

    def getFirstKeyList(self): 
        """
        return the sorted list of first key
        """
        l = self.dCnt.keys(); l.sort()
        return l
    
    #--- Second Key
    def getAllSecondKeys(self):
        setK = set()
        for k in self.getFirstKeyList():
            setK = setK.union( self.getSecondKeyList(k) )
        return list(setK)
        
    def getSecondKeyList(self, k): 
        """
        return the sorted list of observed labels for this tag
        """
        l = self.dCnt[k].keys(); l.sort()
        return l
    
    def getSecondKeyCountList(self, k):
        """
        return the count of observed second keys, in same order as the second key list, for that first key
        """
        return [self.dCnt[k][v] for v in self.getSecondKeyList(k)]
    
    def getCount(self, k1, k2): return self.dCnt[k1][k2]

    #--- Sum
    def getSumByFirstKey(self, k1):
        """
        return the sum of counts of observed second keys, for that first key
        """
        return sum( self.dCnt[k1][v] for v in self.getSecondKeyList(k1) )
    
    def getSumBySecondKey(self, k2):
        """
        return the sum of counts of observed first keys, for that second key
        """
        cnt = 0
        for k1 in self.getFirstKeyList():
            if k2 in self.getSecondKeyList(k1): cnt += self.getCount(k1, k2)
        return cnt

class CollectionAnalyzer:
    def __init__(self, lTag):
        self.start()
        self.lTag = lTag    #all tag names
        
    def start(self):
        """
        reset any accumulated data
        """
        self.hPageCountPerDoc = DoubleHistogram("Page count stat")
        self.hTagCountPerDoc  = DoubleHistogram("Tag stat per document")
        self.hLblCountPerTag  = DoubleHistogram("Label stat per tag")
        
        self.lDoc           = None    #all doc names
        self.lNbPage        = None
        
    def run(self, sDir):
        """
        process one folder
        """
        assert False, "Method must be specialized"
    
    def end(self):
        """
        Consolidate the gathered data
        """
        self.lDoc = self.hPageCountPerDoc.getFirstKeyList()  #all doc are listed here
        self.hTagCountPerDoc.addFirstKeys(self.lDoc)         #to make sure we have all of them listed, even those without tags of interest
        #self.lTag = self.hTagCountPerDoc.getAllSecondKeys()  #all tag of interest observed in dataset
        
        self.lNbPage = list()
        for doc in self.lDoc:
            lNb = self.hPageCountPerDoc.getSecondKeyList(doc)
            assert len(lNb) == 1
            self.lNbPage.append(lNb[0])
        #label list per tag: self.hLblCountPerTag.getSecondKeyList(tag)
        
    def save(self, filename):
        t = (self.hPageCountPerDoc, self.hTagCountPerDoc, self.hLblCountPerTag)
        with open(filename, "wb") as fd: pickle.dump(t, fd)

    def load(self, filename):
        with open(filename, "rb")as fd: 
            self.hPageCountPerDoc, self.hTagCountPerDoc, self.hLblCountPerTag = pickle.load(fd)
        
    def report(self):
        """
        report on accumulated data so far
        """
        print "-"*60
        
        print " ----- %d documents, %d pages" %(len(self.lDoc), sum(self.lNbPage))
        for doc, nb in zip(self.lDoc, self.lNbPage): 
            print "\t---- %40s  %6d pages"%(doc, nb)
        
        print
        print " ----- %d objects of interest: %s"%(len(self.lTag), self.lTag)
        for doc in self.lDoc:
            print "\t---- %s  %6d occurences"%(doc, self.hTagCountPerDoc.getSumByFirstKey(doc))
            for tag in self.lTag: 
                print "\t\t--%20s  %6d occurences" %(tag, self.hTagCountPerDoc.getCount(doc, tag))
        print
        for tag in self.lTag: 
            print "\t-- %s  %6d occurences" %(tag, self.hTagCountPerDoc.getSumBySecondKey(tag))
            for doc in self.lDoc:
                print "\t\t---- %40s  %6d occurences"%(doc, self.hTagCountPerDoc.getCount(doc, tag))

        print
        for tag in self.lTag: 
            print "\t-- %s  %6d occurences  %-d labelled" %(tag, self.hTagCountPerDoc.getSumBySecondKey(tag)
                                                                , self.hLblCountPerTag.getSumByFirstKey(tag))
            for lbl in self.hLblCountPerTag.getSecondKeyList(tag):
                print "\t\t- %20s  %6d occurences"%(lbl, self.hLblCountPerTag.getCount(tag, lbl))
            
        print "-"*60
        return ""
    
    def seenDocPageCount(self, doc, pagecnt):
        self.hPageCountPerDoc.seenK1K2(doc, pagecnt)    #strange way to indicate the page count of a doc....
    def seenDocTag(self, doc, tag):
        self.hTagCountPerDoc.seenK1K2(doc, tag)
    def seenTagLabel(self, tag, lbl):
        self.hLblCountPerTag.seenK1K2(tag, lbl)

class PageXmlCollectionAnalyzer(CollectionAnalyzer):
    """
    Annalyse a collection of PageXmlDocuments
    """
    def __init__(self, sDocPattern, sPagePattern, ltTagAttr):
        """
        sRootDir is the root directory of the collection
        sDocPattern is the pattern followed by folders, assuming one folder contains one document
        sPagePattern is the pattern followed by each PageXml file , assuming one file contains one PageXml XML
        ltTagAttr is the list of pair of tag of interest and label attribute
        """
        lTag, _ = zip(*ltTagAttr)
        CollectionAnalyzer.__init__(self, lTag)
        self.sDocPattern    = sDocPattern
        self.sPagePattern   = sPagePattern
        self.ltTagAttr      = ltTagAttr

    def run(self, sRootDir):
        lFolder = [os.path.basename(folder) for folder in glob.iglob(os.path.join(sRootDir, self.sDocPattern)) 
                                if os.path.isdir(os.path.join(sRootDir, folder))]
        lFolder.sort()
        print "Documents: ", lFolder
        
        for docdir in lFolder:
            print "Document ", docdir
            lPageFile = [os.path.basename(name) for name in glob.iglob(os.path.join(sRootDir, docdir, self.sPagePattern)) 
                                if os.path.isfile(os.path.join(sRootDir, docdir, name))]
            lPageFile.sort()
            self.seenDocPageCount(docdir, len(lPageFile))
            for sPageFile in lPageFile: 
                print ".",
                self.parsePage(sRootDir, docdir, sPageFile)
            print
            sys.stdout.flush()
            
    def parsePage(self, sRootDir, sDocDir, sPageFile):
        doc = libxml2.parseFile(os.path.join(sRootDir, sDocDir, sPageFile))
        ctxt = doc.xpathNewContext()
        ctxt.xpathRegisterNs("pg", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")
        for tag, attr in self.ltTagAttr:
            lNdTag = ctxt.xpathEval("//pg:%s"%tag)
            for nd in lNdTag:
                self.seenDocTag(sDocDir, tag)
                lbl = nd.prop(attr)
                if lbl: self.seenTagLabel(tag, lbl)
        ctxt.xpathFreeContext()
        doc.freeDoc()
        
        
if __name__ == "__main__":

    #prepare for the parsing of the command line
    parser = OptionParser()
    
#     parser.add_option("--dir", dest='lTrn',  action="store", type="string"
#                       , help="Train or continue previous training session using the given annotated collection.")    
#     parser.add_option("--tst", dest='lTst',  action="store", type="string"
#                       , help="Test a model using the given annotated collection.")    
#     parser.add_option("--run", dest='lRun',  action="store", type="string"
#                       , help="Run a model on the given non-annotated collection.")    
#     parser.add_option("-w", "--warm", dest='warm',  action="store_true"
#                       , help="Attempt to warm-start the training")   
#     parser.add_option("--rm", dest='rm',  action="store_true"
#                       , help="Remove all model files")   

    # --- 
    print sys.argv
    
    #parse the command line
    (options, args) = parser.parse_args()
    # --- 
    try:
        sRootDir, sDocPattern, sPagePattern  = args[0:3]
        ls = args[3:]
        ltTagAttr = zip(ls[slice(0, len(ls), 2)], ls[slice(1, len(ls), 2)])
        print sRootDir, sDocPattern, sPagePattern, ltTagAttr
    except:
        print "Usage: %s sRootDir sDocPattern sPagePattern [Tag Attr]+"%(sys.argv[0] )
        exit(1)

    doer = PageXmlCollectionAnalyzer(sDocPattern, sPagePattern, ltTagAttr)
    doer.start()
    doer.run(sRootDir)
    doer.end()
    sReport = doer.report()
    
    print sReport
    
