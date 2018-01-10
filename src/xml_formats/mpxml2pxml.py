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
    
    import optparse
    usage = """
%s dir filename
Utility to create a set of pageXml XML files from a mpxml file.
""" % sys.argv[0]

    parser = optparse.OptionParser(usage=usage)
    parser.add_option("--docid", dest='docid',  action="store" ,default=None, help="output name named with docid")    
    parser.add_option("--format", dest='bIndent',  action="store_true" , help="reformat/reindent the input")    
    parser.add_option("--dir", dest='destdir',  action="store", default='pxml' , help="directory ouptut")  
    (options, args) = parser.parse_args()

    try:
        dir  = args[0]
        mpxml  = args[1]
    except:
        parser.print_help()
        parser.exit(1, "")
    
    if options.docid is not None:
        sDocFilename = "%s%scol%s%s.mpxml" % (dir,os.sep,os.sep,options.docid)
    else:
        sDocFilename = "%s%scol%s%s" % (dir,os.sep,os.sep,mpxml)
    
    if sDocFilename is None:
        print 'File not found: %s' % sDocFilename
        sys.exit(1)
    doc = libxml2.parseFile(sDocFilename)

    i=0
    print sDocFilename
    for pnum, pageDoc in PageXml.MultiPageXml._iter_splitMultiPageXml(doc, bInPlace=False):
        if options.docid is not None:
            filename= options.docid
            outfilename = "%s%s%s%s%s_%03d.pxml" % (dir,os.sep,options.destdir,os.sep,filename,pnum)
        else:
            try:
                pageNd = PageXml.MultiPageXml.getChildByName(doc.getRootElement(),'Page')[0]
                assert  pageNd.prop('imageFilename') is not None
                filename = pageNd.prop('imageFilename')[:-3] + 'pxml'
                outfilename = "%s%s%s%s%s" % (dir,os.sep,options.destdir,os.sep,filename)
            except IndexError: outfilename = None
        print outfilename        
        if outfilename is not None:pageDoc.saveFormatFileEnc(outfilename, "utf-8", bool(options.bIndent))
        i+=1
    doc.freeDoc()
    print "DONE for %d pages" % i    