

import sys, os
from optparse import OptionParser
import glob

import lxml.etree as etree

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version


from common.trace import traceln
from common.TestReport import TestReport


dNS = {"pc":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


def fail_safe_xpath(nd, sXp):
    """
    convenience to read XML node attributes, for instance
    """
    try:
        val = nd.xpath(sXp, namespaces=dNS)[0]        
    except IndexError:
        val = None
    return str(val)

def main(sDir, sPattern, sXp, sXpGT, sXpPred, iVerbose=0):
    lsFilename = sorted(glob.iglob(os.path.join(sDir, sPattern)))

    lGTTag, lPredTag = [], []
    for sFilename in lsFilename:
        if iVerbose: traceln("\t%s" % sFilename)
        doc = etree.parse(sFilename)

        lNd = doc.xpath(sXp, namespaces=dNS)

        _lGT   = [fail_safe_xpath(nd, sXpGT  ) for nd in lNd]
        _lPred = [fail_safe_xpath(nd, sXpPred) for nd in lNd]

        assert len(_lGT) == len(_lPred), "File %s : different number of GT and predicted tags?? %d versus %d" % (sFilename, len(_lGT), len(_lPred)) 
        lGTTag  .append(_lGT)
        lPredTag.append(_lPred)

    lAllTag = sorted(set(o for l in lGTTag for o in l).union(o for l in lPredTag for o in l ))
    dTag2I = {s:i for i,s in enumerate(lAllTag)}
    lYGT   = [[dTag2I[s] for s in l] for l in lGTTag]
    lYPred = [[dTag2I[s] for s in l] for l in lPredTag]
    tr = TestReport("Classification", lYPred, lYGT, lAllTag)

    sRpt = tr.toString(bShowBaseline=False)

    print(sRpt)


if __name__ == "__main__":
    usage = """%s FOLDER PATTERN XPATH GT_attr_xpath Pred_attr_xpath
        e.g. foo/col .mpxml //pc:TextLine ./@type_gt ./@type
        with pc="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        Compare the GT and predicted tag extracted from the XML stored in the folder.
        Compute classifier evaluation
    """ %sys.argv[0]
    parser = OptionParser(usage=usage)

    # ---
    #parse the command line
    (options, args) = parser.parse_args()

    try:
        sDir, sExt, sXP, sXpGT, sXpPred = args
    except:
        traceln(usage)
        exit(1)

    main(sDir, "*"+sExt, sXP, sXpGT, sXpPred)