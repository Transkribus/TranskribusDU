# -*- coding: utf-8 -*-

"""
    Keep doc with more than given ratio of empty TextLine
    
    Copyright Naver Labs Europe(C) 2018 JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os
from optparse import OptionParser
import shutil

from lxml import etree


try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from xml_formats.PageXml import PageXml
from tasks import _exit


def isTexted(sFilename, fRatio):
    parser = etree.XMLParser(remove_blank_text=True)
    doc = etree.parse(sFilename, parser)
    
    cntTxt, cnt = PageXml.countTextLineWithText(doc)
    
    fDocRatio = float(cntTxt) / cnt
    
    del doc
    
    if fDocRatio > fRatio:
        return True
    elif fDocRatio > 0:
            traceln("Warning: %d texted out of %d  (%.2f) %s" % (cntTxt, cnt, fDocRatio, sFilename))
    
    return False
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    usage = """<from-dir> <to-dir>"""
    version = "v.01"
    parser = OptionParser(usage=usage, version="0.1")
    parser.add_option("--ratio", dest='fRatio', action="store"
                      , type=float
                      , help="Keep doc with more than given ratio of empty TextLine"
                      , default=0.75) 
            
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()

    traceln(options)
    
    if len(args) == 2 and os.path.isdir(args[0]) and os.path.isdir(args[1]):
        # ok, let's work differently...
        sFromDir,sToDir = args
        for s in os.listdir(sFromDir):
            if not s.endswith("pxml"): pass
            sFilename = sFromDir + "/" + s
            if isTexted(sFilename, options.fRatio):
                traceln(sFilename,"  -->  ", sToDir)
                shutil.copy(sFilename, sToDir)
            else:
                traceln(" skipping: ", sFilename)
    else:
        for sFilename in args:
            if isTexted(sFilename, options.fRatio):
                traceln("texted : %s"%sFilename)
            else:
                traceln("no text: %s"%sFilename)
