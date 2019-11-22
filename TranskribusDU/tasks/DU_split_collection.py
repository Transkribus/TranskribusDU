# -*- coding: utf-8 -*-

"""
    DU task: split a collection in N equal parts, at random
    
    Copyright Xerox(C)  2019  Jean-Luc Meunier
    

    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
 
import sys, os, random
from shutil import copyfile
from optparse import OptionParser
import math

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
TranskribusDU_version

from common.trace import traceln
from util.CollectionSplitter import getSplitIndexList


if __name__ == "__main__":
    #     import better_exceptions
    #     better_exceptions.MAX_LENGTH = None
    sUsage = """
USAGE: %s  DIR  ( N | p1,p2(,p)+ )
Split in N folders
  or
Split in folders following the proportions p1, ... pN

The folders are named after the DIR folder by adding suffix_part_<1 to N>

(Expecting to find a 'col' subfolder in DIR)""" % sys.argv[0]
  
    parser = OptionParser(usage=sUsage)
    (options, args) = parser.parse_args()

    try:
        sDir, sN = args
    except:
        print(sUsage)
        exit(1)
    
    sColDir= os.path.join(sDir, "col")
    assert os.path.isdir(sColDir), "%s is not a folder"%sColDir
    
    print("- looking at ", sColDir)
    lsFile = []
    for _fn in os.listdir(sColDir):
        _fnl = _fn.lower()
        if _fnl.endswith("_du.mpxml") or _fnl.endswith("_du.pxml"):
            continue
        if not(_fnl.endswith(".mpxml") or _fnl.endswith(".pxml")):
            continue
        lsFile.append(_fn)

    nbFile = len(lsFile)
    
    try:
        lP = [int(_s) for _s in sN.split(',')]
        if len(lP) < 2: raise ValueError("want to run the except code")
        lP = [p / sum(lP) for p in lP]
        traceln(" %d files to split in %d parts with proportions %s" % (
            nbFile
            , len(lP)
            , ",".join("%.2f"%_p for _p in lP)))
        lP.sort()
        ld = []
        for i, p in enumerate(lP):
            ld += [i] * math.ceil(p * nbFile)
        ld = ld[:nbFile]
        while len(ld) < nbFile: ld.append(len(lP)-1)
        random.shuffle(ld)
    except ValueError:
        # Split in N parts
        traceln(" %d files to split in %d parts" % (nbFile, int(sN)))    
        n = int(sN)
    
        ld = getSplitIndexList(nbFile, n, traceln)
        assert len(ld) == nbFile

        # *** SHUFFLING!! ***
        random.shuffle(ld)
    
    # ld [I]  gives the folder index where to put the Ith file

    def get_sToColDir(sDir, d, bExistIsOk=False):
        """
        make "<sDir>" and "sDir>/col" folders
        if bExist is False, raise an exception if any folder exists already
        return "<sDir>/col"
        """
        sToDir      = "%s_part_%d" % (sDir, d)
        sToColDir   = os.path.join(sToDir , "col")
        if bExistIsOk:
            try:                    os.mkdir(sToDir)
            except FileExistsError: pass
            try:                    os.mkdir(sToColDir)
            except FileExistsError: pass
        else: 
            try:
                os.mkdir(sToDir)
                os.mkdir(sToColDir)
            except:
                raise Exception("First remove the destination folders: ", (sToDir, sToColDir))
        return sToColDir
    
    assert len(ld) == len(lsFile)
    
    # make sure the folder are not already containing some stuff (from previous runs...)
    for _d in set(ld): 
        get_sToColDir(sDir, _d+1, bExistIsOk=False)
    
    ld = [1+d for d in ld]  # convenience
    for d, sFilename in zip(ld, lsFile):
        sToDir      = "%s_part_%d" % (sDir, d)
        sToColDir   = os.path.join(sToDir , "col")
        try:
            os.mkdir(sToDir)
            os.mkdir(sToColDir)
        except FileExistsError: pass
        sToColDir = get_sToColDir(sDir, d, bExistIsOk=True)
        sInFile = os.path.join(sColDir, sFilename)
        copyfile(sInFile, os.path.join(sToColDir, sFilename))
        print(sInFile, "  --> ", sToColDir)
        
