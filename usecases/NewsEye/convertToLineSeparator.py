"""
    Newseye
    convert 
    <TextRegion id="r_11_separator" custom="readingOrder {index:17;} structure {type:separator;}">
            <Coords points="126,574 1282,574 1282,592 126,592"/>
        </TextRegion>
        
    into
    <Separator id="r_11_separator" >
      <Coords points="126,574 1282,574 1282,592 126,592"/>
    </Separator>
    
"""

from glob import glob
from optparse import OptionParser
import sys,os
from lxml import etree

from xml_formats.PageXml import PageXml, PageXmlException
from util.Polygon import Polygon


def convertTR2Sep(filename):
    """
    """
    print (filename)
    tagname='TextRegion'
    xml = etree.parse(filename)
    ltextsep = xml.getroot().findall(f".//pc:{tagname}[@custom]", {"pc":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"})
    
    for x in ltextsep:
        if "separator" in x.get('custom'):
            x.tag = 'SeparatorRegion'
            
            #now we need to convert that object to a line
            lXY = PageXml.getPointList(x)  #the polygon
            assert lXY, "Separator without Coord??"
            
            plg = Polygon(lXY)
            try:
                x1,y1, x2,y2 = plg.fitRectangle()
            except ValueError:
                print("Warning: Coords might be bad, taking bounding box: ", lXY)
                x1,y1,x2,y2 = plg.getBoundingBox()
#             try:
#                 x1,y1, x2,y2 = plg.fitRectangle()
#             except ZeroDivisionError:
#                 x1,y1,x2,y2 = plg.getBoundingBox()
#             except ValueError:
#                 x1,y1,x2,y2 = plg.getBoundingBox()            
            if abs(x2-x1) > abs(y2-y1): # horizontal
                y1 = (y1+y2)/2
                y2 = y1
            else:
                x1 = (x1+x2)/2
                x2=x1
            
            ndCoord = x.xpath(".//pc:Coords", namespaces={"pc":PageXml.NS_PAGE_XML})[0]
            PageXml.setPoints(ndCoord, [(x1,y1), (x2,y2)])
                
    return xml
    
    
def convertFiles(lfilename,outdir):
    
    for filename in lfilename:
        
        doc = convertTR2Sep(filename)
        newfilename = outdir + os.path.sep + f'a_{os.path.basename(filename)}'
        with open(f'{newfilename}', 'wb') as f:
            doc.write(f, encoding="utf-8", xml_declaration=True, pretty_print=True)
if __name__ == '__main__':
    
    version = "v.01"
    sUsage="""
    Usage: %s <InputDir> <OuputDir>   
    
    """ % (sys.argv[0])

    parser = OptionParser(usage=sUsage)
    parser.add_option("--ext", dest='extension',  type='string',action="store"
                        , help="file extension")
    (options, args) = parser.parse_args()
    
    
    try:
        folderIn = args[0]
        folderOut = args[1]
    except ValueError:
        sys.stderr.write(sUsage)
        sys.exit(1)
        
    
    lsFile = sorted([s for s in glob(os.path.join(folderIn, '*')) if s.endswith(options.extension)])
    convertFiles(lsFile,folderOut)
    
    print (f'done for {len(lsFile)} files.')