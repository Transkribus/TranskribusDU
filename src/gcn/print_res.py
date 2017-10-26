import pickle
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


def read_res(fname):
    f=open(fname,'rb')
    R=pickle.load(f)

    val = R['val_acc']
    test = R['test_acc']
    best_idx=np.argmax(val)
    ftest = test[best_idx]

    return ftest

def read_max_res(fname):
    f = open(fname, 'rb')
    R = pickle.load(f)

    val = R['val_acc']
    test = R['test_acc']


    return np.max(test)



def print_res(Fs,Cs,P):

    num_c = len(Cs)

    for c in Cs:
        line_res = ''
        avg = 0.0
        valid_fold = 0
        # print('c',c)
        for f in Fs:
            # print('F ',f)
            k = str(f) + ':' + str(c)
            print('K', k)
            if k in P:
                line_res += '%.4f' % P[k] + ' &'
                avg += P[k]
                valid_fold += 1
            else:
                line_res += '    &'

        if valid_fold > 0:
            print('A', avg, valid_fold)
            a = avg / float(valid_fold)
            line_res += '%.4f' % a + "\\\\ \n"
        else:
            line_res += "   \\\\\n"
        print('C' + str(c), line_res)

def plot_progress(folder_resname,configid=5):

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    #ax1.plot(x, y)
    #ax1.set_title('Sharing x per column, y per row')
    #ax2.scatter(x, y)
    #ax3.scatter(x, 2 * y ** 2 - 1, color='r')
    #ax4.plot(x, 2 * y ** 2 - 1, color='r')

    for fold_id,ax in zip([1,2,3,4],[ax1,ax2,ax3,ax4]):
        ax.set_ylim([0.6, 1.05])
        fname=os.path.join(folder_resname,'table_F'+str(fold_id)+'_C'+str(configid)+'.pickle')
        f = open(fname, 'rb')
        R = pickle.load(f)

        train=np.array(R['train_acc'])
        val = np.array(R['val_acc'])
        test = np.array(R['test_acc'])
        x=range(train.shape[0])
        #ax.plot(x,test,'--',x,val,'+',x,train,'.')
        test_l,= ax.plot(x, test)
        test_l.set_label('test')
        val_l, = ax.plot(x, val)
        val_l.set_label('val')
        train_l,= ax.plot(x, train)
        train_l.set_label('train')
        ax.set_xlabel('epoch % 10')
        ax.set_ylabel('accuracy')
        ax.set_title('Fold: '+str(fold_id))
        ax.legend()


    plt.show()






def get_res(folder_resname):
    L=os.listdir(folder_resname)
    F=[]
    C=[]

    P={}
    M={}
    for fname in L:
        s=fname.replace('.pickle','')
        tok=s.split('_')
        foldid=int(tok[1][1:])
        configid=int(tok[2][1:])
        F.append(foldid)
        C.append(configid)

        fn=os.path.join(folder_resname,fname)
        perf = read_res(fn)
        max_perf =read_max_res(fn)

        print(foldid, configid,perf)

        P[str(foldid)+':'+str(configid)]=perf
        M[str(foldid)+':'+str(configid)]=max_perf

    Fs = sorted(np.unique(F))
    Cs = sorted(np.unique(C))

    print_res(Fs,Cs,P)
    #print_res(Fs, Cs, M)
    #Strange that on Fold1, I get not so good results a I used to have better results
    #Did i change something, use a smaller validation set ? 10% instead of 20





def plot_We(fn = 'table_plot_wE/table_F1_C5.pickle'):


    f = open(fn, 'rb')
    R = pickle.load(f)
    f.close()
    R
    R.keys()
    We = R['W_edge']
    len(We)




    W = np.vstack(We)
    W.shape

    plt.figure()
    ax = plt.gca()
    im=ax.imshow(W)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show()


if __name__ == '__main__':
    folder_resname=sys.argv[1]

    get_res(folder_resname)


