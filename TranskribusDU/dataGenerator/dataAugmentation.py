# -*- coding: utf-8 -*-
"""


    Data Augmentation: 
    
    generate Layout annotated data 
    
    copyright Naver Labs 2019
    READ project 

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
    
    @author: H. Déjean
"""

import os,sys, optparse
import numpy as np
from lxml import etree
from shapely.geometry import LineString
from shapely.affinity import translate
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0])))

def skewing(lCoordDom,angle):
    """
        skewing  (0,0) top left corner
        
        need a skewing for double page, skewing from the binding??
        
        need also to adapt the page size!
    """        

    theta= np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))  
    
    lSkewed= []
    for coords in lCoordDom:
        xx,yy = np.dot(R,[coords[0],coords[1]])
        lSkewed.append((int(round(xx)),int(round(yy))))
    return lSkewed
    
    
def rotateV180(lCoordDom,pageWidth):
    """
        rotate the page 180 vertically
    """ 
    vaxis =  pageWidth * 0.5
    for x,y in lCoordDom:
        d = abs(vaxis - x )
        if x > vaxis:
            x = vaxis - d 
        else: x = vaxis + d
    return lCoordDom


def linePolygon(baseline,param):
    lList= [(x) for xy  in baseline.attrib['points'].split(' ') for x in xy.split(',')]
    lcoord =[ (float(x),float(y)) for x,y  in zip(lList[0::2],lList[1::2]) ]
    try: line=LineString(lcoord)
    except ValueError: return  # LineStrings must have at least 2 coordinate tuples
    topline=translate(line,yoff=param)
    spoints = ' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in line.coords)
    lp=list(topline.coords)
    lp.reverse()
    spoints =spoints+ ' ' +' '.join("%s,%s"%(int(x[0]),int(x[1])) for x in lp) 
    baseline.getparent()[0].set('points',spoints)

def loadFile(file):
    return etree.parse(file)
     
def processPageXml(xmlfile,foo,param,bNorm=True):
    xpath  = "//*[@%s]" % ("points")
    lNodes= xmlfile.xpath(xpath)
    for node in lNodes:
        spoints=node.attrib['points']
        lList= [(x) for xy  in spoints.split(' ') for x in xy.split(',')]
        lp =[ (float(x),float(y)) for x,y  in zip(lList[0::2],lList[1::2]) ]
        lsk= foo(lp,param)
        node.attrib['points']=" ".join("%s,%s"%(x,y) for x,y in lsk)
    

def textlineNormalisation(xmlfile,param):    
    NS_PAGE_XML         = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"    
    xpath  = "//a:TextLine/a:Baseline"
    lNodes= xmlfile.xpath(xpath,namespaces ={'a':NS_PAGE_XML})
    for node in lNodes:        
        linePolygon(node,param)

if __name__ == "__main__":

    parser = optparse.OptionParser(usage="")
    
    parser.add_option("--informat", dest='inputFormat',  action="store", type='string'
                      , help="input format: PageXml (default) or DSXML")    
    parser.add_option("--skew", dest='bSkew',  action="store"
                      , help="perform skewing deformation")    
    parser.add_option("--perpective", dest='extension',  action="store", default=''
                      , help="perform perspective deformation")  
    (options, args) = parser.parse_args()

    try:
        lsDir = args
        lsDir[0]
    except:
        parser.print_help()
        parser.exit(1, "")
        
    try: 
        inputFile = sys.argv[1]
        outFile = sys.argv[2]
    except IndexError as e:
        print("usage: dataAugmentation.py INPUTFILE OUTFILE")
        sys.exit() 
    xml = loadFile(inputFile)
    for a in [-2,-1.5,-1,1,1.5,2]:
        processPageXml(xml, skewing,a)
        for h in [20,25,30]:
            textlineNormalisation(xml, h)
            xml.write("%s_%s_%s"%(h,a,outFile))
    for h in [20,25,30]:
        processPageXml(xml,rotateV180,8400)
        textlineNormalisation(xml, h)
        xml.write("%s_%s_%s"%(h,180,outFile))
    
        