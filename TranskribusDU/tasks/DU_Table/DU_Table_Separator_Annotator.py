# -*- coding: utf-8 -*-

"""
    USAGE:  DU_Table_Separator_Annotator.py input-folder
    
    You must run this on your GT collection to create a training collection.
    
    If you pass a folder, you get a new folder with name postfixed  by a_
    
    It annotate the separator found in XML file as:
    - S = a line separating consistently the items of a table
    - I = all other lines
    
    For near-horizontal (i.e. more horizontal than vertical) lines, the maximum
    @row+@row_span-1 of the items above the line must be strictly lesser than 
    the minimum @row of the item below the line. 

    Copyright Naver Labs Europe 2019 
    JL Meunier
"""

import sys, os

from lxml import etree
import shapely.geometry as geom

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version
    TranskribusDU_version

from common.trace           import traceln
from xml_formats.PageXml    import MultiPageXml 
from util.Shape             import ShapeLoader, ShapePartition

DEBUG=False

def isBaselineHorizontal(ndText):
    lNdBaseline = MultiPageXml.getChildByName(ndText ,'Baseline')
    if lNdBaseline:
        try:
            o = ShapeLoader.node_to_LineString(lNdBaseline[0])
        except:
            return True
        (minx, miny, maxx, maxy) = o.bounds
        return bool(maxx-minx >= maxy-miny)
    return True            
                            
def main(lsFilename, lsOutFilename):
    #for the pretty printer to format better...
    parser = etree.XMLParser(remove_blank_text=True)
    cnt, cntS = 0, 0
    for sFilename, sOutFilename in zip(lsFilename, lsOutFilename):
        cntDoc, cntDocS = 0, 0

        doc = etree.parse(sFilename, parser)
        root = doc.getroot()
        
        # Separators are not under tableRegion... :-/
        lNdSep = MultiPageXml.getChildByName(root ,'SeparatorRegion')
        loSep = [ShapeLoader.node_to_LineString(ndSep) for ndSep in lNdSep]
        for _o in loSep: _o._bConsistent = True
        
        if not lNdSep:
            traceln("Warning: no separator in %s"%sFilename)
        else:
            traceln("%25s  %d separators" % (sFilename, len(lNdSep)))
            lNdTR = MultiPageXml.getChildByName(root ,'TableRegion')
            for ndTR in lNdTR:
                lNdCells= MultiPageXml.getChildByName(ndTR ,'TableCell')
                if not lNdCells:
                    continue
                
                nbRows = max(int(x.get('row')) for x in lNdCells)
        
                # build a list of Shapely objects augmented with our own table attributes
                loText = [] #
                for ndCell in lNdCells:
                    minRow = int(ndCell.get('row'))
                    minCol = int(ndCell.get('col'))
                    maxRow = minRow + int(ndCell.get('rowSpan')) - 1
                    maxCol = minCol + int(ndCell.get('colSpan')) - 1
#                     # ignore cell spanning the whole table height
#                     if maxRow >= nbRows:
#                         continue
                    for ndText in MultiPageXml.getChildByName(ndCell ,'TextLine'):
                        try:
                            oText = ShapeLoader.node_to_Polygon(ndText)
                        except:
                            traceln("WARNING: SKIPPING 1 TExtLine: cannot make a polygon from: %s" % etree.tostring(ndText))
                            continue
                        # reflecting the textbox as a single point
                        (minx, miny, maxx, maxy) = oText.bounds
                        
                        # is the baseline horizontal or vertical??
                        fDelta = min((maxx-minx) / 2.0, (maxy-miny) / 2.0) 
                        if isBaselineHorizontal(ndText):
                            # supposed Horizontal text
                            oText = geom.Point(minx + fDelta  , (maxy + miny)/2.0)
                            ndText.set("Horizontal", "TRUE")

                        else:
                            ndText.set("Horizontal", "nope")
                            oText = geom.Point((minx + maxx)/2.0  , miny + fDelta)
                            
                        # considering it as a point, using its centroid
                        # does not work well due to loooong texts oText = oText.centroid
                        oText._minRow, oText._minCol = minRow, minCol
                        oText._maxRow, oText._maxCol = maxRow, maxCol
                        if DEBUG: oText._domnd = ndText
                        loText.append(oText)
                
                traceln("    TableRegion  %d texts" % (len(loText)))
                
                if loText:
                    # checking in tun each separator for table-consistency
                    sp = ShapePartition(loText)
                    
                    for oSep in loSep:
                        (minx, miny, maxx, maxy) = oSep.bounds
                        if maxx - minx >= maxy - miny:
                            # supposed Horizontal
                            l = sp.getObjectAboveLine(oSep)
                            if l:
                                maxRowBefore = max(_o._maxRow for _o in l)
                                l = sp.getObjectBelowLine(oSep)
                                if l:
                                    minRowAfter  = min(_o._minRow for _o in l)
                                    if maxRowBefore >= minRowAfter: oSep._bConsistent = False
                        else:
                            l1 = sp.getObjectOnLeftOfLine(oSep)
                            if l1:
                                maxColBefore = max(_o._maxCol for _o in l1)
                                l2 = sp.getObjectOnRightOfLine(oSep)
                                if l2:
                                    minColAfter  = min(_o._minCol for _o in l2)
                                    if maxColBefore >= minColAfter: 
                                        oSep._bConsistent = False
                                        if DEBUG:
                                            # DEBUG
                                            for o in l1:
                                                if o._maxCol >= minColAfter: print("too much on right", etree.tostring(o._domnd))
                                            for o in l2:
                                                if o._minCol <= maxColBefore: print("too much on left", etree.tostring(o._domnd))
                # end of TableRegion
            # end of document
            for ndSep, oSep in zip(lNdSep, loSep): 
                if oSep._bConsistent:
                    ndSep.set("DU_Sep", "S")
                    cntDocS += 1
                else:
                    ndSep.set("DU_Sep", "I")
                cntDoc += 1
            
        doc.write(sOutFilename, encoding='utf-8',pretty_print=True,xml_declaration=True)
        traceln('%.2f%% consistent separators - annotation done for %s  --> %s' % (100*float(cntDocS)/(cntDoc+0.000001), sFilename, sOutFilename))
        
        del doc
        cnt, cntS = cnt+cntDoc, cntS+cntDocS
    traceln('%.2f%% consistent separators - annotation done for %d files' % (100*float(cntS)/(cnt+0.000001), cnt))
    
        
if __name__ == "__main__":
    try:
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

    main(lsInFilename, lsOutFilename)
