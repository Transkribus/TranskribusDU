# -*- coding: utf-8 -*-

"""
    Example DU task for ABP Table
    
    Copyright Xerox(C) 2017 H. Déjean

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
import sys, os

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from common.trace import traceln
from tasks import _checkFindColDir, _exit

from xml_formats.PageXml import MultiPageXml 
from crf.Graph_MultiPageXml import Graph_MultiPageXml
from crf.NodeType_PageXml   import NodeType_PageXml_type_woText
from DU_CRF_Task import DU_CRF_Task
from crf.FeatureDefinition_PageXml_std_noText import FeatureDefinition_PageXml_StandardOnes_noText

# ===============================================================================================================

lLabels = ['RB', 'RI', 'RE', 'RS','RO','SI', 'SO']

lIgnoredLabels = None

nbClass = len(lLabels)

"""
if you play with a toy collection, which does not have all expected classes, you can reduce those.
"""

lActuallySeen = None
if lActuallySeen:
    print "REDUCING THE CLASSES TO THOSE SEEN IN TRAINING"
    lIgnoredLabels  = [lLabels[i] for i in range(len(lLabels)) if i not in lActuallySeen]
    lLabels         = [lLabels[i] for i in lActuallySeen ]
    print len(lLabels)          , lLabels
    print len(lIgnoredLabels)   , lIgnoredLabels
    nbClass = len(lLabels) + 1  #because the ignored labels will become OTHER

#DEFINING THE CLASS OF GRAPH WE USE
DU_GRAPH = Graph_MultiPageXml
nt = NodeType_PageXml_type_woText("abp"                   #some short prefix because labels below are prefixed with it
                      , lLabels
                      , lIgnoredLabels
                      , False    #no label means OTHER
                      )
ntA = NodeType_PageXml_type_woText("abp"                   #some short prefix because labels below are prefixed with it
                      , lLabels
                      , lIgnoredLabels
                      , False    #no label means OTHER
                      )

nt.setXpathExpr( (".//pc:TextLine"        #how to find the nodes
                  , "./pc:TextEquiv")       #how to get their text
               )

ntA.setXpathExpr( (".//pc:TextLine | .//pc:TextRegion | .//pc:SeparatorRegion"        #how to find the nodes
                  , "./pc:TextEquiv")       #how to get their text
                )




# ===============================================================================================================

 
class DU_ABPTableAnnotator(DU_CRF_Task):
    """
    We will do a CRF model for a DU task
    , with the below labels 
    """
    sXmlFilenamePattern = "*.mpxml"
    
    sLabeledXmlFilenamePattern = "*.a_mpxml"

    sLabeledXmlFilenameEXT = ".a_mpxml"


    #=== CONFIGURATION ====================================================================
    def __init__(self, sModelName, sModelDir, sComment=None, C=None, tol=None, njobs=None, max_iter=None, inference_cache=None): 
        
        DU_CRF_Task.__init__(self
                     , sModelName, sModelDir
                     , DU_GRAPH
                     , dFeatureConfig = { }
                     , dLearnerConfig = {
                            'C'                : .1 
                         , 'njobs'            : 2
                         , 'inference_cache'  : 50
                        , 'tol'              : .1
                        , 'save_every'       : 51     #save every 50 iterations,for warm start
                         , 'max_iter'         : 100
                         }
                     , sComment=sComment
                     ,cFeatureDefinition=FeatureDefinition_PageXml_StandardOnes_noText
                     )
        
        
        self.bsln_mdl = self.addBaseline_LogisticRegression()    #use a LR model trained by GridSearch as baseline
    
    #=== END OF CONFIGURATION =============================================================

  
    
    def annotateCollection(self,lsTrnColDir):
        """
            KapelMeister
            
            input full table:
            
            what we need: table with columns only
            
            if table:
                
        """
        ## load
    
    def unLinkBadBaselines(self,ltextLines):
        """
            assume horizontal lines only
            TextLine/Baseline/@points="1518,337 1553,257"/>
        """
        lNewList=[]
        for t in ltextLines:
            w,h = t.getWidthHeight()
            if w < h:
                t.node.unlinkNode()
                t.node.freeNode()
            else:
                lNewList.append(t)
        return lNewList
    
    def annotateDocument(self,lsTrnColDir):
        """
            from cell elements: annotate textLine elements 
        """
        
        #load graphs
        _, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, self.sXmlFilenamePattern)

        DU_GRAPH.addNodeType(ntA)
        lGraph_trn = DU_GRAPH.loadGraphs( lFilename_trn, bNeighbourhood=True, bLabelled=False,iVerbose=True)


        # get cells
        for i,graph in enumerate(lGraph_trn):
            
            lCells = filter(lambda x:x.node.name=='TextRegion',graph.lNode)
            lTextLine = filter(lambda x:x.node.name=='TextLine',graph.lNode)
            lTextLine = self.unLinkBadBaselines(lTextLine)
            lSeparator = filter(lambda x:x.node.name=='SeparatorRegion',graph.lNode)
            lPTextLine={}
            #sort textLine per page
            for tl in lTextLine:
                try: lPTextLine[tl.page].append(tl)
                except KeyError : lPTextLine[tl.page]=[tl]

            lPSep={}
            for sep in lSeparator:
                if (sep.x2 - sep.x1)  > (sep.y2-sep.y1):
                    try: lPSep[sep.page].append(sep)
                    except KeyError : lPSep[sep.page]=[sep]
                
            # need to test if elements on the same page!!
            for cell in lCells:
                ## SEP
                try :
                    lPSep[cell.page]
                    for sep in lPSep[cell.page]:
                        # extend height 
                        sep.y2=sep.y2+30
                        dh,dv=  sep.getXYOverlap(cell)
#                         print sep.node, dh,dv
                        # at least  
                        if dh > 0 and dv > 0:
                            sep.node.setProp(sep.type.sLabelAttr,lLabels[5])
                except KeyError :pass
                
                #SEP  
                lText=[]
                for line in lPTextLine[cell.page]:
                    dh,dv=  line.getXYOverlap(cell)
                    # at least  
                    if dh > (line.x2 - line.x1)*0.1 and dv > 0:
                        line.node.unlinkNode()
                        cell.node.addChild(line.node)
                        lText.append(line)
                if len(lText) == 1:
                    lText[0].node.setProp(lText[0].type.sLabelAttr,lLabels[3])
                elif len(lText) >1:
                    lText.sort(key=lambda x:x.y1)
                    lText[0].node.setProp(lText[0].type.sLabelAttr,lLabels[0])
                    lText[-1].node.setProp(lText[-1].type.sLabelAttr,lLabels[2])
                    [x.node.setProp(x.type.sLabelAttr,lLabels[1]) for x in lText[1:-1]]

                
            # check if all labelled
            # if no labelled: add other
            ## TEXT
            for tl in lTextLine:
                sLabel = tl.type.parseDomNodeLabel(tl.node)
                try:
                    cls = DU_GRAPH._dClsByLabel[sLabel]  #Here, if a node is not labelled, and no default label is set, then KeyError!!!
                except KeyError:              
                    tl.node.setProp(tl.type.sLabelAttr,lLabels[4])
            ## SEP
            for sep in lSeparator:
                sLabel = tl.type.parseDomNodeLabel(tl.node)
                try:
                    cls = DU_GRAPH._dClsByLabel[sLabel]  #Here, if a node is not labelled, and no default label is set, then KeyError!!!
                except KeyError:              
                    tl.node.setProp(lText[0].type.sLabelAttr,lLabels[6])
            doc =graph.doc    
            MultiPageXml.setMetadata(doc, None, self.sMetadata_Creator, self.sMetadata_Comments)
            # save mpxml as mpxml.orig
            sDUFilename = lFilename_trn[i][:-len(MultiPageXml.sEXT)]+  self.sLabeledXmlFilenameEXT
            print sDUFilename
            doc.saveFormatFileEnc(sDUFilename, "utf-8", True)  #True to indent the XML
            doc.freeDoc()
                    
    
        
if __name__ == "__main__":

    version = "v.01"
    usage, description, parser = DU_CRF_Task.getBasicTrnTstRunOptionParser(sys.argv[0], version)

    
    # --- 
    #parse the command line
    (options, args) = parser.parse_args()
    
    # --- 
    try:
        sModelDir, sModelName = args
    except Exception as e:
        traceln("Specify a model folder and a model name!")
        _exit(usage, 1, e)
        
    doer = DU_ABPTableAnnotator(sModelName, sModelDir,
                      C                 = options.crf_C,
                      tol               = options.crf_tol,
                      njobs             = options.crf_njobs,
                      max_iter          = options.crf_max_iter,
                      inference_cache   = options.crf_inference_cache)
    
    
    
    if options.rm:
        doer.rm()
        sys.exit(0)

    lTrn, lTst, lRun, lFold = [_checkFindColDir(lsDir) for lsDir in [options.lTrn, options.lTst, options.lRun, options.lFold]] 
    doer.annotateDocument(lTrn)
    traceln('annotation done')    
    
