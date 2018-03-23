# -*- coding: utf-8 -*-

#REMOVE THIS annoying warning saying:
#  /usr/lib/python2.7/site-packages/requests-2.12.1-py2.7.egg/requests/packages/urllib3/connectionpool.py:843: InsecureRequestWarning: Unverified HTTPS request is being made. 
#  Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings  InsecureRequestWarning)

import sys, os

DEBUG=0

sCOL = "col"

def _exit(usage, status, exc=None):
    if usage: sys.stderr.write("ERROR: usage : %s\n"%usage)
    if exc != None: sys.stderr.write(str(exc))  #any exception?
    sys.exit(status)    

def _checkFindColDir(lsDir, sColName=sCOL, bAbsolute=True):
    """
    For each directory in the input list, check if it is a "col" directory, or look for a 'col' sub-directory
    If a string is given instead of a list, make of it a list
    If None is given, just return an empty list
    return the list of "col" directory absolute path
    or raise an exception
    """
    if lsDir == None: return list()
    if type(lsDir) != list: lsDir = [lsDir]
    lsColDir = list()
    for sDir in lsDir:  
        if not(sDir.endswith(sColName) or sDir.endswith(sColName+os.path.sep)): 
            sColDir = os.path.join(sDir, sColName)
        else:
            sColDir = sDir
        if bAbsolute:
            sColDir = os.path.abspath(sColDir)
        if not( os.path.exists(sColDir) and os.path.isdir(sColDir) ):
            raise ValueError("Non-existing folder: %s"%sColDir)
        lsColDir.append(sColDir)
    return lsColDir
