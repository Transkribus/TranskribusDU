# -*- coding: utf-8 -*-

"""
    Keep doc with more than given ratio of empty TextLine
    
    Copyright Naver Labs Europe(C) 2018 JL Meunier
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.

    Copyright Naver Labs Europe(C) 2020 JL Meunier
"""
import os

from optparse import OptionParser

from graph.GraphModel import GraphModel

def load_X_Y(sPklFilename):
    o = GraphModel.gzip_cPickle_load(sPklFilename)
    try:
        (lX, lY) = o
        lY[0]
    except ValueError:
        lX = o
        lY = None
    return lX, lY

def do_shorten(lsName, lsFile, iShorten):
    for _sName, sFilename in zip(lsName, lsFile):
        # try:
        lX, lY = load_X_Y(sFilename)
        N = len(lX)
        assert N >= iShorten, "Cannot shorten to %d : only %d graphs" %(iShorten, N)
        sOutFilename = "%s.%d" % (sFilename, iShorten)
        print("   -->  ", sOutFilename,)
        if lY is None:
            GraphModel.gzip_cPickle_dump(sOutFilename, lX[0:iShorten])
        else:
            GraphModel.gzip_cPickle_dump(sOutFilename, (  lX[0:iShorten]
                                                        , lY[0:iShorten]))
        print("\t\t done")

def do_show(lsName, lsFile):
    sPreFmt = "%%%ds" % max(len(s) for s in lsName)
    sFmtI = sPreFmt + " %6d/%d"
    sFmt  = sFmtI   + "  %15s:NF  %15s:E  %15s:EF"
    sFmtY = sFmt    + "  %9s:Y (%d-%d labels)"
    sFmtO = sFmtI   + "  %d objects like: %s and %s"
    for sName, sFilename in zip(lsName, lsFile):
        lX, lY = load_X_Y(sFilename)
        N = len(lX)
        if type(lX) == tuple and len(lX) == 3:
            lX, lY = [lX], [lY]  # single graph pickle!
            N = 1
        else:
            assert lY is None or len(lY) == N
        if not options.bList and N > 2:
            # shorten the data to be displayed
            lI = [0, N-1]
            lX = [lX[0], lX[-1]]
            lY = None if lY is None else [lY[0], lY[-1]]
        else:
            lI = list(range(N))
        if lY is None:
            for (i, X) in zip(lI, lX):
                try:
                    NF, E, EF = X
                    print(sFmt  % (sName if i == 0 else "", i+1, N, NF.shape, E.shape, EF.shape))
                except ValueError:
                    print(sFmtO % (sName if i == 0 else "", i+1, N, len(X), X[0], X[-1]))
        else:
            for (i, X, Y) in zip(lI, lX, lY):
                NF, E, EF = X
                print(sFmtY % (sName if i == 0 else "", i+1, N, NF.shape, E.shape, EF.shape
                               , Y.shape, Y.min(), Y.max()))
        
#               except Exception as e:
#                 print(sName, sFilename, e)
#                 pass
    
if __name__ == "__main__":
        usage = """do_info_xy_pkl.py [-l] <model-folder> <model-name>
do_info_xy_pkl.py [-l] <file>+
do_info_xy_pkl.py --shorten N <file>+  shorten the file, to keep N graphs
        """

        #prepare for the parsing of the command line
        parser = OptionParser(usage=usage, version="0.1")
        parser.add_option("-l", "--list", dest='bList',  action="store_true"
                          , help="Detailed information per graph")   
        parser.add_option("-s", "--shorten", dest='iShorten',  action="store", type="int"
                          , help="Shorten the files")   
        (options, args) = parser.parse_args()
        if len(args) == 2 and os.path.isdir(args[0]):
            sDir, sName = args
            gm = GraphModel(sName, sDir)
            gm.sSurname = 'crf'
            lsName, lsFile = [], []
            for name in ['trn', 'vld', 'tst']:
                s = gm.getTrainDataFilename(name)
                if os.path.exists(s):
                    lsName.append("%s %s"%(sName, name))
                    lsFile.append(s)
        elif len(args) >= 1:
            lsName = args
            lsFile = args
        else:
            print(usage)
            exit(1)
        
        if options.iShorten:
            assert options.iShorten > 0
            do_shorten(lsName, lsFile, options.iShorten)
        else:
            do_show(lsName, lsFile)
                

    
    
