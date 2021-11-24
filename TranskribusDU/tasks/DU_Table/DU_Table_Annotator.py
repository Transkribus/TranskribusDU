# -*- coding: utf-8 -*-

"""
    USAGE:  DU_Table_Annotator.py input-folder
    
    You must run this on your GT collection to create a training collection.
    
    If you pass a folder, you get a new folder with name postfixed  by a_
    
    Does 2 things:
    
    - 1 - 
    Annotate textlines  for Table understanding (finding rows and columns)
    
    It tags the TextLine, to indicate:
    - the table header, vs data, vs other stuff:
        @DU_header = 'CH'  |  'D'  |  'O'
        
    - the vertical rank in the table cell:
        @DU_row = 'B' | 'I' | 'E' | 'S' | 'O'
    
    - something regarding the number of text in a cell??
        # NO SURE THIS WORKS...
        @DU_col = 'M' | 'S' | 'O'
    
        
    - 2 - 
    Aggregate the borders of the cells by linear regression to reflect them
    as a line, which is stored as a SeparatorRegion element. 
    

    Copyright Naver Labs Europe 2017, 2018 
    H. Déjean
    JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
import sys, os

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
import tasks.DU_Table.DU_ABPTableRCAnnotation


if __name__ == "__main__":
    try:
        if sys.argv[1] == '--nosep':
            bSep = False
            sys.argv = sys.argv[0:1] + sys.argv[2:]  # moche...
        else:
            bSep = True
        #we expect a folder 
        sInputDir = sys.argv[1]
        if not os.path.isdir(sInputDir): raise Exception()
    except IndexError:
        traceln("Usage: %s <folder>" % sys.argv[0])
        exit(1)

    sOutputDir = "a_"+sInputDir
    traceln(" - Output will be in ", sOutputDir)
    try:
        os.mkdir(sOutputDir)
        os.mkdir(os.path.join(sOutputDir, "col"))
    except:
        pass
    
    lsFilename = [s for s in os.listdir(os.path.join(sInputDir, "col")) if s.endswith(".mpxml") ]
    lsFilename.sort()
    lsOutFilename = [os.path.join(sOutputDir, "col", "a_"+s) for s in lsFilename]
    if not lsFilename:
        lsFilename = [s for s in os.listdir(os.path.join(sInputDir, "col")) if s.endswith(".pxml") ]
        lsFilename.sort()
        lsOutFilename = [os.path.join(sOutputDir, "col", "a_"+s[:-5]+".mpxml") for s in lsFilename]

    lsInFilename  = [os.path.join(sInputDir , "col", s)      for s in lsFilename]

    traceln(lsFilename)
    traceln("%d files to be processed" % len(lsFilename))

    tasks.DU_Table.DU_ABPTableRCAnnotation.main(lsInFilename, lsOutFilename, bSep)
