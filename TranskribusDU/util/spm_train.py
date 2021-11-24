"""
train a sentencepiece model from a collection

Created on 29 nov. 2019

@author: meunier
"""
import sys, os
import glob
from optparse import OptionParser

import lxml.etree as etree
import sentencepiece as spm

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln

class XML_to_Text:
    """
    convert a file or a collection to .txt files
    """
    NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

    def __init__(self, sDocExtension=None,lowercase=False):
        """
        Constructor
        """
        self.lowercase=lowercase
        self.sDocExtension       =         sDocExtension
        self.sDocIgnoreExtension = "_du" + sDocExtension
        
    def run(self, lsDir, sOutput):
        traceln("%d folders: %s" % (len(lsDir), lsDir))
        nDoc, nText = 0, 0
        with open(sOutput, "w", encoding="utf-8") as fd:
            for sDir in lsDir:
                traceln("--- FOLDER ", sDir)
                lDocFile = [filename for filename in glob.iglob(os.path.join(sDir, "**", "*"+self.sDocExtension), recursive=True) 
                                        if os.path.isfile(filename) and not(filename.endswith(self.sDocIgnoreExtension))]
                lDocFile.sort()
                traceln("  - %d Documents" % (len(lDocFile)))
                nDoc += len(lDocFile)
                for i, docFile in enumerate(lDocFile):
                    traceln( "%d - Document %s" % (1+i, docFile))
                    doc = etree.parse(docFile)
                    for txt in doc.getroot().xpath(".//pc:TextLine//pc:Unicode/text()"
                                        , namespaces={"pc":self.NS_PAGE_XML}):
                        if self.lowercase:txt=txt.lower()
                        fd.write(txt)
                        fd.write("\n")
                        nText += 1
                    del doc
        return nDoc, nText
            
if __name__ == "__main__":
    sUsage="usage: %s <model-name> <col-dir>+" % sys.argv[0]

    parser = OptionParser(usage=sUsage)
    
    parser.add_option("--ext", dest='sExt',  action="store", type="string"
                      , help="Expected extension of the data files, e.g. '.mpxml'"
                      , default=".mpxml")    
    parser.add_option("--vocab_size", "--vocab-size", dest='iVocabSize',  action="store", type="int"
                      , default=1000
                      , help="Vocabulary size")
    parser.add_option( "--lowercase", dest='blowercase',  action="store_true"
                      , default=False
                      , help="Lower vocabulary")         
    (options, args) = parser.parse_args()

    try:
        sModelName  = args[0]
        lsDir       = args[1:]
        assert len(lsDir)
    except:
        traceln(sUsage % sys.argv[0])
        exit(1)
        
    sAllTextFile = sModelName + "." + "alltext"
    
    doer = XML_to_Text(options.sExt,options.blowercase)
    nDoc, nText = doer.run(lsDir, sAllTextFile)
    traceln(" --> %d doc ---> %d texts" % (nDoc, nText))
    
    sCmdArg = '--input=%s --model_prefix=%s --vocab_size=%d' % (sAllTextFile, sModelName, options.iVocabSize)
    traceln("Training sentencepiece: ")
    spm.SentencePieceTrainer.Train(sCmdArg)