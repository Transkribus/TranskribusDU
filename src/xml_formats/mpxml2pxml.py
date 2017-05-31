# -*- coding: utf-8 -*-
"""
    mpxml to pxml convertor
    
    @author: H Déjean
    
    READ project
    31/05/2017
""" 

import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))

import libxml2
import xml_formats.PageXml as PageXml
    
if __name__ == "__main__":
    
    import sys, glob, optparse
    usage = """
%s dirname+
Utility to create a set of pageXml XML files from a mpxml file.
""" % sys.argv[0]

    parser = optparse.OptionParser(usage=usage)
    
    parser.add_option("--format", dest='bIndent',  action="store_true" , help="reformat/reindent the input")    
    parser.add_option("--dir", dest='destdir',  action="store", default='pxml' , help="directory ouptut")  
    (options, args) = parser.parse_args()

    try:
        dir  = args[0]
        docid= args[1]
    except:
        parser.print_help()
        parser.exit(1, "")
    
    sDocFilename = "%s%scol%s%s.mpxml" % (dir,os.sep,os.sep,docid)        
        
    doc = libxml2.parseFile(sDocFilename)

    for pnum, pageDoc in PageXml.MultiPageXml._iter_splitMultiPageXml(doc, bInPlace=True):
        outfilename = "%s%s%s%s%s_%03d.pxml" % (dir,os.sep,options.destdir,os.sep,docid,pnum)
        print outfilename        
        pageDoc.saveFormatFileEnc(outfilename, "utf-8", bool(options.bIndent))
    doc.freeDoc()
    print "DONE"    