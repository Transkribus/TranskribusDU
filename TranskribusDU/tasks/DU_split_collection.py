# -*- coding: utf-8 -*-

"""
    DU task: split a collection in N equal parts, at random
    
    Copyright Xerox(C)  2019  Jean-Luc Meunier
    
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
 
import sys, os, random
from shutil import copyfile

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

    try:
        sDir = sys.argv[1]
        n = int(sys.argv[2])
    except:
        print("USAGE: %s DIR N"%sys.argv[0])
        exit(1)
    
    sColDir= os.path.join(sDir, "col")
    
    print("- looking at ", sColDir)
    lsFile = []
    for _fn in os.listdir(sColDir):
        _fnl = _fn.lower()
        if _fnl.endswith("_du.mpxml") or _fnl.endswith("_du.pxml"):
            continue
        if not(_fnl.endswith(".mpxml") or _fnl.endswith(".pxml")):
            continue
        lsFile.append(_fn)
    traceln(" %d files to split in %d parts" % (len(lsFile), n))    
    
    
    N = len(lsFile)
    ld = getSplitIndexList(N, n, traceln)
    assert len(ld) == N

    # *** SHUFFLING!! ***
    random.shuffle(ld)
    

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
    
    
    # make sure the folder are not already containing some stuff (from previous runs...)
    for _d in range(1, n+1): 
        get_sToColDir(sDir, _d, bExistIsOk=False)
    
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
        
