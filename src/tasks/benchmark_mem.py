
try:
    import psutil

except:
    print 'Could not load PSutil'

import resource
#import threading
from threading import Thread
import os
import time
import numpy as np
import pickle
import sys



def monitor_mem(pid):
    s = psutil.Process()
    while True:
        #with s.oneshot():
        s = psutil.Process()
        print s.name(), s.cpu_times(),s.memory_info()
        time.sleep(0.01)

'''
function bytesToSize(bytes) {
   var sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
   if (bytes == 0) return '0 Byte';
   var i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
   return Math.round(bytes / Math.pow(1024, i), 2) + ' ' + sizes[i];
};

'''

import Dodge_Tasks

import math

def bytesToSize(bytes):
    sizes=['Bytes', 'KB', 'MB', 'GB', 'TB']
    if (bytes == 0):
        return '0 Byte'
    else:
        i = int(math.floor(math.log(bytes) / math.log(1024)))
        x = float(bytes)/math.pow(1024, i)
        return str(x)+' ' + sizes[i]


class MonitorThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.process = None
        self.mem_values=[]
        self.vms=[]
        self.rss=[]
        self.uss=[]
        self.is_ok=True


    def run(self):
        while self.is_ok:
            #with s.oneshot():
            s = psutil.Process()
            #print s.name(), s.cpu_times(),s.memory_info()
            tp=s.memory_info() #todo get full memory info on update of psutil
            self.vms.append(tp.vms)
            self.rss.append(tp.rss)
            #self.uss.append(tp.uss)
            #self.mem_values.append(s.memory_info)
            time.sleep(0.1)

    def stop(self):
        print "Trying to stop thread "
        self.is_ok=False
        print(self.vms)
        print('Maximum Memory Usage')
        max_vms =max(self.vms)
        max_rss =max(self.rss)
        print('VMS',max_vms,bytesToSize(max_vms))
        print('RSS',max_rss,bytesToSize(max_rss))



class RMemThread(MonitorThread):

    def __init__(self):
        Thread.__init__(self)
        self.process = None
        self.mem_values=[]
        self.rss=0
        self.is_ok=True


    def run(self):
        while self.is_ok:
            #with s.oneshot():
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            #Seem to be in KyloBytes
            self.rss =max(self.rss,mem)
            print self.rss
            time.sleep(0.5)


    def stop(self):
        print "Trying to stop thread "
        self.is_ok=False
        print(self.rss)
        print('Maximum Memory Usage')
        max_rss =self.rss
        print('RSS',1024*max_rss,bytesToSize(1024*max_rss))





def test_main_thread(i):
    print(i)
    time.sleep(5)
    X=np.random.rand(1000,1000)
    Xinv=np.linalg.inv(X)

    print('Main Finished')
    return Xinv


def dodge_test(task_index):

    Z=pickle.load(open('dodge_test_plan.pickle'))
    model_src,feat_select,nb_feat,mid,test_collection=Z[task_index]
    model_name=Dodge_Tasks.get_model_name(model_src,feat_select,nb_feat,mid)
    model_dir='DODGE_TRAIN'
    print (model_name)
    res=Dodge_Tasks.test_model(model_name,model_dir,test_collection)
    print(res)




if __name__=='__main__':
    pid=os.getpid()


    task_index=int(sys.argv[1])

    #monitor_thread =Thread(target=monitor_mem,args=(pid,))
    #monitor_thread = MonitorThread()
    #monitor_thread.start()

    monitor_thread = RMemThread()
    monitor_thread.start()


    #main_thread = Thread(target = test_main_thread, args = (10,))
    main_thread = Thread(target = dodge_test, args = (task_index,))

    #dodge_test(task_index)

    main_thread.start()
    main_thread.join()

    monitor_thread.stop()



    exit()



