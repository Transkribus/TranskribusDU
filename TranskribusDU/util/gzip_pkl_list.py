'''
A list of pickled data

Behave as a list of data, initialized with filenames, and load the required data from the disk

(threaded version)

Created on 26 March, 2021

@author: JL Meunier

Copyright NAVER 2021
'''
import sys
import threading
import queue
import time

from util.gzip_pkl import gzip_pickle_load


# us e a thread to fill in a queue of data
bQUEUE = True


traceln_Lock = threading.Lock()
def traceln(*msg):
    try: 
        traceln_Lock.acquire(True)
        for i in msg:
            try:
                sys.stderr.write(str(i))
            except UnicodeEncodeError:
                sys.stderr.write(i.encode("utf-8"))
        sys.stderr.write("\n")
        sys.stderr.flush()
    finally:
        traceln_Lock.release()


#DEBUG = 2

def QueueLoaderThread(lsFilename, q):
    """
    Queue loading function
    put None when finished
    """
    sumTimeLoad = 0 #to compute average waiting time
    cntTimeLoad = 0
    for sFilename in lsFilename:
        # if DEBUG > 1: traceln("\t Thread is reading ", sFilename)
        t0 = time.time()
        
        o = gzip_pickle_load(sFilename)
        
        sumTimeLoad += (time.time() - t0)
        cntTimeLoad += 1
        
        # if DEBUG > 1: traceln("\t Thread has read ", sFilename)
        
        q.put( o )
        
        #if DEBUG > 1: traceln("\t Thread has put data in queue, from ", sFilename)
    q.put(None)
    q.put( (sumTimeLoad, cntTimeLoad) )
    
    
class PklList(list):
    
    def __init__(self
                 , n        # queue size
                 , fun      # how to process the (X,Y, index)
                 , *args, **kwargs):
        """
        initialize an empty list, with a cache of n elements in RAM
        with a user fonction to be applied on each item when iterating over the list
        """
        assert n > 0
        super(PklList, self).__init__(*args, **kwargs)
        self.n   = n
        self.fun = fun
        # threading stuff
        self.sumTimeWait = 0.0  # total waiting time from queue
        self.cntTimeWait = 0
        self.sumTimeLoad = 0.0  # total loading time from disk
        self.cntTimeLoad = 0
        self._t = None  # thread
        self._q = None  # queue
        
    def aslist(self):
        """
        return the real list of values (not the data on disk)
        """
        return [self[i] for i in range(len(self))]
    
    def averageLoadWaitTime(self):   
        """
        average loading and waiting time on queue read operations
        """
        s = """
            """
        try:
            return (  self.sumTimeLoad / self.cntTimeLoad
                    , self.sumTimeWait / self.cntTimeWait )
        except ZeroDivisionError:
            return (-1, -1)

    def averageTimeReport(self):   
        """
        average loading and waiting time on queue read operations as a report string
        """
        try:
            s = """  #load: %d   avg load time: %.4fs
  #wait: %d   avg wait time: %.4fs
                """ % (  self.cntTimeLoad, self.sumTimeLoad / self.cntTimeLoad
                       , self.cntTimeWait, self.sumTimeWait / self.cntTimeWait )
        except ZeroDivisionError:
            s = """  #load: %d   avg load time: N/A
  #wait: %d   avg wait time: N/A
                """ % (  self.cntTimeLoad
                       , self.cntTimeWait )
        return s

    def __iter__(self):
        """
        iterate over th edata stored in the files
        """
        #traceln("__iter__")
        self._i = 0
    
        if bQUEUE:
            #a queue and a worker that feed it
            assert self._t == None
            self._q = queue.Queue(maxsize=self.n)
            self._t = threading.Thread(target=QueueLoaderThread, name="pklloader"
                                 , args=(self.aslist(), self._q))
            self._t.setDaemon(True)
            self._t.start()
 
        return self
    
    def __next__(self):
        #traceln("__next__")
        t0 = time.time()
        if bQUEUE:
            # queue mode
            qo = self._q.get()
            if self._i == len(self):
                assert qo is None
#             if qo is None:
#                 assert self._i == len(self)
                (_stl, _ctl) = self._q.get()
                self.sumTimeLoad += _stl
                self.cntTimeLoad += _ctl
                del self._t, self._q
                self._t = None
                raise StopIteration()
            else:
                X, Y = qo
        else:
            try:
                fn = self[self._i]
            except IndexError:
                raise StopIteration()
            X, Y = gzip_pickle_load(fn)
        
        if self.fun is None:
            o = (X, Y)
        else:
            o = self.fun(X, Y, self._i)
        # traceln("--> data")
        self._i += 1
        
        self.sumTimeWait += (time.time() - t0)
        self.cntTimeWait += 1
        
        return o
        
    def sDebug(self):
        return "n=%d l=%s" %(self.n, self)


def test_folder(sDir):
    import os, glob
    def myfun(X, Y, i): 
        return ("processed", X, Y, i)
    
    l = PklList(2
                , myfun
                , glob.iglob(os.path.join(sDir, "[0-9]*[0-9].pkl")))
    print("============")
    print(len(l), l)
    for i, o in enumerate(l):
        print(type(o))
        print(o)
    print(l.averageLoadWaitTime())
    print(l.averageTimeReport())
                         
# if __name__ == "__main__":
#      
#     test_folder("C:/tmp_MENUS/all_202011/models_pkls/tw_4NN_ter._.ecn.tst_tXY")

