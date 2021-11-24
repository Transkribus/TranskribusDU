'''
Created on 23 nov. 2020

@author: meunier
'''

import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle


def gzip_pickle_dump(sFilename, dat):
    with gzip.open(sFilename, "wb") as zfd:
            pickle.dump( dat, zfd, protocol=2)

def gzip_pickle_load(sFilename):
    with gzip.open(sFilename, "rb") as zfd:
            return pickle.load(zfd)        
