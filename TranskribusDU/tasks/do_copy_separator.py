# -*- coding: utf-8 -*-

"""
    Copy the separators from one collection to another, creating a new collection
    
    Copyright Naver Labs Europe(C) 2020 JL Meunier
"""

import sys, os

from lxml import etree


try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln
from xml_formats.PageXml import PageXml
import pathlib

sepDir, docDir, outDir = sys.argv[1:4]
traceln("Getting separators from ", sys.argv[1])
traceln("Getting documents from  ", sys.argv[2])
traceln("Generating documents with separators in ", sys.argv[3])

assert os.path.isdir(sepDir)
assert os.path.isdir(docDir)
pathlib.Path(outDir).mkdir(parents=True, exist_ok=True)

# files containing separators
dSepFileByStem = {}
for s in pathlib.Path(sepDir).glob("**/*.pxml"):
    # s is like: S_Reut_009_0250.pxml
    sStem = pathlib.PurePath(s).stem  # e.g. S_Reut_009_0250
    v = dSepFileByStem.get(sStem, None)
    assert v == None, "several files with same stem?? %s   and  %s" % (dSepFileByStem[sStem], s)
    dSepFileByStem[sStem] = s
traceln("%d files with separators listed" % len(dSepFileByStem))

# check doc files
dDocFileByPath = {}
for s in pathlib.Path(docDir).glob("**/*.xml"):
    # s is like: Fischer_Michael_S__S_Reut_009_0244.xml
    sHand, sName = str(s).split("__")
    sStem = pathlib.PurePath(sName).stem  # e.g. S_Reut_009_0250
    assert dSepFileByStem[sStem], "File %s has no corresponding file with separator" % s
    dDocFileByPath[s] = dSepFileByStem[sStem]
traceln("%d files listed with proper separator file association" % len(dSepFileByStem))

# merge!!!
for sDocFile, sSepFile in dDocFileByPath.items():
    print(pathlib.Path(sDocFile.name), "(%s)"%sDocFile.resolve())
    sOutFile = pathlib.PurePath.joinpath(pathlib.Path(outDir), sDocFile.stem + "_sep" + sDocFile.suffix)
    traceln("- merging %s and %s   into  %s"%(sSepFile,sDocFile, sOutFile))
    
    docSep = etree.parse(str(sSepFile.resolve()))
    docDoc = etree.parse(str(sDocFile.resolve()))
    
    for ( ndPageSep
        , ndPageDoc)in zip(PageXml.xpath(docSep.getroot(), ".//pc:Page")
                         , PageXml.xpath(docDoc.getroot(), ".//pc:Page")):  
        lNdSep = PageXml.xpath(ndPageSep, ".//pc:SeparatorRegion")
        for ndSep in lNdSep:
            ndPageDoc.append(ndSep)
    
    docDoc.write(str(sOutFile.resolve()),
                    xml_declaration=True,
                    encoding="utf-8",
                    pretty_print=True
                    #compression=0,  #0 to 9
                  )
    
    del docSep
    del docDoc

    
print("Done")
