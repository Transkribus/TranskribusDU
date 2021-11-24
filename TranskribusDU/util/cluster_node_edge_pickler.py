# coding: utf8

"""
Analyse edge break/continue accuracy as a fonction of edge characteristics

JL Meunier
Dec. 2020

Copyright NAVER France
"""

import sys, os
import glob
import re
import math 
import ast
from optparse import OptionParser

from lxml import etree
import numpy as np
# from scipy.stats.stats import pearsonr 

from common.trace           import traceln
from xml_formats.PageXml    import PageXml as PageXml
from util.gzip_pkl          import gzip_pickle_dump
from util.Polygon           import Polygon

crePXml = re.compile(".*_du\..*xml")


# let's memorize what is displayed to store it in the pickle, if any
lsLog = []

def log(*s):
    global lsLog
    lsLog.extend(map(str,s))
    traceln(*s)

def getLog():
    global lsLog
    return "\n".join(lsLog)

# -----------------------------------------------------------------------------
class HierarchicalEdgeAnalizer:
    
    dIndex_by_EdgeType = { sClass : i for i,sClass in enumerate([  "HorizontalEdge"
                                                                  , "VerticalEdge"
                                                                  , "CelticEdge"
                                                                  , "ReifiedEdge"])}
    def __init__(self, sWhat, funGetGT, depth):
        self.sWhat      = sWhat         # e.g. TextLine
        self.funGetGT   = funGetGT      # return a list (in case of hierarchy)
        self.D          = depth         # should match length of above list
        self.begin()


    def begin(self):
        self.ltEdgeData         = []                # pythonic data: edge : n_edge x 6
        self.ltEdgeProbaDist    = []                # proba distribution: n_edge  x  depth+1
        self.lltEdgeAB          = []                # pythonic data: (edge.A, edge.B)  : n_page  x  n_edge  x  2
        self.lltNodeClustersIdx = []                # pythonic data: (cl_idx, cl_idx, ...)    : (n_page  x  n_node  x  depth)
        self.lltNodeBB          = []                # pythonic data: [(x1, y1, x2, y2), ...]  : (n_page  x  n_node  x  4)
        self.lsFilename         = []
        
        self._a         = None              # numpy array
        self._a_chk     = len(self.ltEdgeData)  # check
        self._aDistr    = None              # proba distribution
        
        self._n         = 0                 # number of edges
        self._m         = None              # number of classes in GT
        
    def addFile(self, sFile, bBB=True):
        """
        include in the data all edges from this file
        return the number of seen edges 
        """
        xp = PageXml.xpath
    
        n = 0
        doc = etree.parse(sFile)
        for i, ndPage in enumerate(xp(doc.getroot(), "//pc:Page")):
            #log("\tPage %d  %s" % (i+1, ndPage.get("imageFilename")))
            ltEdgeAB = list()
            # dict: node_id --> [GT label , .... ]
            dlGT = {}
            # dict: node_id --> node index
            dNodeIdx = {}
            # list per node [cluster_index, ...]
            ltNodeClustersIdx = []
            # list per node [(x1, y1, x2, y2) ...]
            ltNodeBB = []

            dIdx = dict()                  # [lvl][GT_sid] --> GT_index
            for i in range(self.D):
                dIdx[i] = dict()

            for i, _nd in enumerate(xp(ndPage, self.sWhat)):
                sid = _nd.get("id")
                lGT_sid = self.funGetGT(_nd)
                assert len(lGT_sid) == self.D, "invalid GT labels : expected %d : got %s" %(self.D, lGT_sid)

                dNodeIdx[sid] = i               # index of node
                dlGT    [sid] = lGT_sid         # list of GT clusters per level of this node
                
                lClusterIdx = list()
                for d,sid in enumerate(lGT_sid):
                    try:
                        GT_idx = dIdx[d][sid]
                    except KeyError:
                        GT_idx = len(dIdx[d])
                        dIdx[d][sid] = GT_idx
                    lClusterIdx.append(GT_idx)  # list of cluster index per level for this node
                ltNodeClustersIdx.append(lClusterIdx)   # list of cluster index per level, per node
                
                if bBB: # Bounding box
                    lXY = PageXml.getPointList(_nd)  #the polygon
                    assert bool(lXY)
                    
                    plg = Polygon(lXY)
                    try:
                        x1,y1, x2,y2 = plg.fitRectangle(bPreserveWidth=True)
                    except:
                        x1,y1,x2,y2 = plg.getBoundingBox()       
                    ltNodeBB.append( (x1,y1,x2,y2) )             
            # dlGT = { _nd.get("id") : self.funGetGT(_nd) for _nd in xp(ndPage, ".//pc:"+self.sWhat) }
            
            
            
            for ndEdge in xp(ndPage, ".//pc:Edge"):
                
                # computing GT edge label
                src_sid = ndEdge.get("src")
                tgt_sid = ndEdge.get("tgt")  
                src_lGT = dlGT[src_sid]
                tgt_lGT = dlGT[tgt_sid]
                if src_lGT == tgt_lGT:
                    cls_GT = 0  # continue
                else:
                    i = 0
                    while src_lGT[i] == tgt_lGT[i]: i += 1
                    # i+1 indicates at which level there is a break
                    cls_GT = i + 1

                cls = int(ndEdge.get("label_cls"))

#                 print("A ", src_lGT)
#                 print("B ", tgt_lGT)
#                 print("                                                                  GT ", cls_GT, "    pred", cls)
                
                # edge attributes
                iEdgeType = self.dIndex_by_EdgeType[ndEdge.get("type")]
                o = ndEdge.get("proba")
                fProba    = float(o) if o else -1.0
                (x1, y1, x2, y2) = [float(_sF)  for _sT2 in ndEdge.get("points").split(" ")  for _sF in _sT2.split(",")]
                fLength   = math.sqrt( (x1-x2)**2 + (y1-y2)**2 )
                #assert x1==x2 or y1==y2
            
                # proba distribution if stored
                o = ndEdge.get("distr")
                if o is None:
                    tEdgeProbaDist = [fProba, 1.0-fProba] if cls == 0 else [1.0-fProba, fProba]
                else:
                    tEdgeProbaDist = ast.literal_eval(o)
                #            0   1       2    3           4      5
                tPageData = (0, cls_GT, cls, iEdgeType, fProba, fLength)
                self.ltEdgeData.append(tPageData)
                
                self.ltEdgeProbaDist.append(tEdgeProbaDist)

                ltEdgeAB.append((dNodeIdx[src_sid], dNodeIdx[tgt_sid]))
                
                n += 1
                
            self.lltEdgeAB         .append(ltEdgeAB)
            self.lltNodeClustersIdx.append(ltNodeClustersIdx)
            self.lltNodeBB         .append(ltNodeBB)
            self.lsFilename        .append(sFile)
        return n

    def end(self):
        """
        idem-potent: array made if not done before, or ltData changed afterward
        """
        if self._a is None or len(self.ltEdgeData) != self._a_chk: 
            # the edge data matrix
            self._a     = np.array(self.ltEdgeData)
            self._a_chk = len(self.ltEdgeData)
            
            self._aDistr = np.array(self.ltEdgeProbaDist)
            if self.ltEdgeData:
                assert self._aDistr.shape[1] == (self.D+1), "ERROR: probably the @distr attribute was missing from the XML"
            
            # check
            if self.ltEdgeData:
                assert self._a.shape[1] == 6, "Internal error: some new column appeared??"
            
                # fill the bOk column
                self._a[:                         , 0 ] = 1
                self._a[self._a[:,1]==self._a[:,2], 0 ] = 0
                
                # number of edges
                self._n = self._a.shape[0] 
                # number of classes
                self._m = 1 + int(max(self._a[:,1]))
                
                # the nodes' clusters per page
                # print(self.lltNodeClustersIdx)
    
                # the edge A.idx B.idx per page
                # print(self.lltEdgeAB) 
        
                assert self.D == (self._m-1)
        
        return self._n, self._m, self._a, self._aDistr
    
    def analyse_1(self):
        """
        simple analysis, to compare with test report (should be same!!)
        """
        self.end()
        
        eps=1e-8
        
        assert self._a.shape[0] == self._n
        m = self._m     # nb labels
        
        # let's analyse per edge type as well!
        for a, sTitle in [  (np.compress(self._a[:,3]==0, self._a, axis=0), "Horizontal edges only")
                          , (np.compress(self._a[:,3]==1, self._a, axis=0), "Vertical edges only")
                          , (self._a, "All edges")
                          ]:
            n = a.shape[0]
            # count and accuracy by label
            log("==========  %s  ===========================================" % (sTitle))
            lN_byLbl    = [ sum(a[:,1]==i) for i in range(m) ]
            aOk = a[ a[:,1]==a[:,2] ]
            lOk_byLbl   = [ sum(aOk[:,1]==i) for i in range(m) ] # ok by label
            lOE_byLbl   = [ sum(  a[:,2]==i) for i in range(m) ] # ok + err by label
            lfAcc_byLbl = [ (100.0 * lOk_byLbl[i]) / lN_byLbl[i] for i in range(m)] # acc by label
            for i in range(m):
                _ok, _oe, _n = lOk_byLbl[i], lOE_byLbl[i], lN_byLbl[i]
                
                _p   = _ok / (eps+_oe)
                _r   = _ok / (eps+_n)
                _f1  = 2*_p*_r/(eps+_p+_r)            
                log(" label %10d  : ok=%8d  count=%8d  P = %.3f  R = %.3f  F1 = %.3f    accuracy = %6.2f%%" % (  i
                                                                            , _ok, lN_byLbl[i]
                                                                            , _p, _r, _f1
                                                                            , lfAcc_byLbl[i]))
            # accuracy
            ok = np.sum(a[:,1]==a[:,2])
            fAcc = (100.0*ok) / n
            log("       Globally      ok=%8d  count=%8d   accuracy = %6.2f%%" % (ok, n, fAcc))

            # correlation ok vs length    
            log("")
            a = np.compress([True, True, True, False, True, True], a, axis=1) # removing edge type
            #log("pearsonr    = ", pearsonr   (a[:,0], a[:,5]))
            np.set_printoptions(precision=2)
            log("   ok    GTy   y     prob  Len     (Correlation: np.corrcoef)") 
            log(np.corrcoef(a, rowvar=False))     
            log("")
               
        return lN_byLbl, lOk_byLbl, lOE_byLbl
        
                
# -----------------------------------------------------------------------------
class MultiEdgeAnalizer(HierarchicalEdgeAnalizer):
    """
    useful when the GT is given by a several XML attributes
    """

    def __init__(self, sWhat, lsGTAttr):
        log("XML attributes: ", lsGTAttr)
        
        funGT = lambda nd: [nd.get(_s) for _s in lsGTAttr]
        
        HierarchicalEdgeAnalizer.__init__(self, sWhat, funGT, len(lsGTAttr))


# -----------------------------------------------------------------------------
class SimpleEdgeAnalizer(MultiEdgeAnalizer):
    """
    useful when the GT is given by a single XML attribute
    """

    def __init__(self, sWhat, sGTAttr):
        log("Single XML attribute: ", sGTAttr)
        MultiEdgeAnalizer.__init__(self, sWhat, [sGTAttr])


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = OptionParser(usage="[FILE|DIR]+", version=0.1)
    
    parser.add_option("--pkl"       , dest='sPkl'       ,  action="store", type="string"
                      , help="Store accumulated data in a pickle")    
    parser.add_option("-q", "--quiet"       , dest='bQuiet',  action="store_true"
                      , help="No analysis, just pickling")    
    parser.add_option("--bb"        , dest='bBB',  action="store_true"
                      , help="Store bounding box of nodes")    
    
    parser.add_option("--row"       , dest='bRow'       ,  action="store_true"
                      , help="When the @row XML attribute gives the node GT value")    
    parser.add_option("--cell"       , dest='bCell'       ,  action="store_true"
                      , help="When the @cell XML attribute gives the node GT value")    

    parser.add_option("--section"    , dest='bSection'    ,  action="store_true"
                      , help="Look at the @section attribute to find GT value")
    parser.add_option("--item"       , dest='bItem'       ,  action="store_true"
                      , help="Look at the @menu_item attribute to find GT value")
    parser.add_option("--line"       , dest='bLine'       ,  action="store_true"
                      , help="Look at the @menu_line attribute to find GT value")
    
    (options, args) = parser.parse_args()
    
    
    if options.bSection and options.bItem and options.bLine:
        doer = MultiEdgeAnalizer(".//pc:Word", ["section", "menu_item", "menu_line"])
    elif                      options.bItem and options.bLine:
        doer = MultiEdgeAnalizer(".//pc:Word", [           "menu_item", "menu_line"])
    elif options.bSection and options.bItem                  :
        doer = MultiEdgeAnalizer(".//pc:Word", ["section", "menu_item"             ])
    elif options.bSection                   and options.bLine:
        doer = MultiEdgeAnalizer(".//pc:Word", ["section"             , "menu_line"])
    elif options.bSection:
        doer = SimpleEdgeAnalizer(".//pc:Word", "section"   )
    elif options.bItem:
        doer = SimpleEdgeAnalizer(".//pc:Word", "menu_item" )
    elif  options.bLine:
        doer = SimpleEdgeAnalizer(".//pc:Word", "menu_line" )
    
    elif options.bRow and options.bCell:
        doer = MultiEdgeAnalizer(".//pc:TableCell//pc:TextLine", ["row", "cell"])
    elif options.bRow:
        doer = SimpleEdgeAnalizer(".//pc:TableCell//pc:TextLine", "row")
    elif options.bCell:
        doer = SimpleEdgeAnalizer(".//pc:TableCell//pc:TextLine", "cell")
    elif False:    
        doer = HierarchicalEdgeAnalizer("TextLine")
    else:
        print("Bad options")
        print(parser.print_help())
        sys.exit(1)
    
    lsFile = []
    ltData = []
    for sArg in args:
        if os.path.isfile(sArg):
            log("+ file   ", sArg)
            lsFile = [sArg]
            n = doer.addFile(sArg)
            log("\t\t + %d edges"%n)
        elif os.path.isdir(sArg):
            log("+ folder ", sArg)
            n, m = 0, 0
            for sFile in glob.iglob(sArg+"/**", recursive=True):
                if crePXml.match(sFile.lower()) and os.path.isfile(sFile):
                    lsFile.append(sFile)
                    n += doer.addFile(sFile)
                    m += 1
            log("\t\t + %d files    + %d edges" % (m,n))
        else:
            log("Ignored: ", sArg)

    log("--- %d files" % len(lsFile))
    
    (nb_edge, nb_label, A, ADistr) = doer.end()
    
    if nb_edge > 0:
        if not options.bQuiet: doer.analyse_1()
        
        log("--- %d nodes" % sum(len(_l) for _l in doer.lltNodeClustersIdx))
        if options.sPkl:
            log("==== gzip-pickle file  --> ", options.sPkl)
            if not options.bQuiet: 
                if options.bBB:
                    log("""
    ( nb_edge                   # global: nb edges
    , nb_label                  # global: nb edge labels (e.g. 3 for cont, brk_0, brk_1)
    , A                         # global: array: A is the edge data : bOk, Y_GT, Y, iEdgeType, fProba, fLength
    , ADistr                    # global: array: probability distribution per edge
    , doer.D                    # global: depth (same as nb_label-1, e.g. 2
    , doer.lltEdgeAB            # per page: (edge.A.idx, dge.B.idx)
    , doer.lltNodeClustersIdx   # per page, per node: [cluster_idx, ...]   of length D
    , doer.lltNodeBB            # par page, per node: (x1, y1, x2, y2)
    , lsFilename                # filename
    , getLog()                  # global: computation log
    )
    """)
                else:
                    log("""
    ( nb_edge                   # global: nb edges
    , nb_label                  # global: nb edge labels (e.g. 3 for cont, brk_0, brk_1)
    , A                         # global: array: A is the edge data : bOk, Y_GT, Y, iEdgeType, fProba, fLength
    , ADistr                    # global: array: probability distribution per edge
    , doer.D                    # global: depth (same as nb_label-1, e.g. 2
    , doer.lltEdgeAB            # per page: (edge.A.idx, dge.B.idx)
    , doer.lltNodeClustersIdx   # per page, per node: [cluster_idx, ...]   of length D
    , lsFilename                # filename
    , getLog()                  # global: computation log
    )
    """)
            if options.bBB:
                gzip_pickle_dump(options.sPkl, ( nb_edge                    # global: nb edges
                                                , nb_label                  # global: nb edge labels (e.g. 3 for cont, brk_0, brk_1)
                                                , A                         # global: A is the edge data : bOk, Y_GT, Y, iEdgeType, fProba, fLength
                                                , ADistr                    # global: array: probability distribution per edge
                                                , doer.D                    # global: depth (same as nb_label-1, e.g. 2
                                                , doer.lltEdgeAB            # per page: (edge.A.idx, dge.B.idx)
                                                , doer.lltNodeClustersIdx   # per page, per node: [cluster_idx, ...]   of length D
                                                , doer.lltNodeBB            # par page, per node: (x1, y1, x2, y2)
                                                , doer.lsFilename
                                                , getLog()))                # global: computation log
            else:
                gzip_pickle_dump(options.sPkl, ( nb_edge                    # global: nb edges
                                                , nb_label                  # global: nb edge labels (e.g. 3 for cont, brk_0, brk_1)
                                                , A                         # global: A is the edge data : bOk, Y_GT, Y, iEdgeType, fProba, fLength
                                                , ADistr                    # global: array: probability distribution per edge
                                                , doer.D                    # global: depth (same as nb_label-1, e.g. 2
                                                , doer.lltEdgeAB            # per page: (edge.A.idx, dge.B.idx)
                                                , doer.lltNodeClustersIdx   # per page, per node: [cluster_idx, ...]   of length D
                                                , doer.lsFilename
                                                , getLog()))                # global: computation log
              
        
    
    
"""        
sbatch -p gpu-mono --mem 30000 ./do_runx_jl_dev_features.sh SgmTripleWord_Line_MenuItem_Section.py 1976104 "--detail --eval_cluster_level     --graph --mlflow_expe=jl_menus_202011 --mlflow_run=1976104.run   --ecn_config ecn_10lay_6conv_stack_256.json --BB2 --g1o"
"""