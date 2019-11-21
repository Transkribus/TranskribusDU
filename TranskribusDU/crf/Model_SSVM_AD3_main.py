# -*- coding: utf-8 -*-

"""
    Train, test, predict steps for a CRF model
    - CRF model is EdgeFeatureGraphCRF  (unary and pairwise potentials)
    - Train using SSM
    - Predict using AD3

    Copyright Xerox(C) 2016 JL. Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Please install matplotlib to get graphical display of the model convergence")
    raise e

from pystruct.utils import SaveLogger

try: #to ease the use without proper Python installation
    from common.trace import traceln
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    from common.trace import traceln

from crf.Model_SSVM_AD3 import Model_SSVM_AD3


# --- MAIN: DISPLAY STORED MODEL INFO ------------------------------------------------------------------

if __name__ == "__main__":
    try:
        sModelDir, sModelName = sys.argv[1:3]
    except:
        print("Usage: %s <model-dir> <model-name>"%sys.argv[0])
        print("Display some info regarding the stored model")
        exit(1)
        
    mdl = Model_SSVM_AD3(sModelName, sModelDir)
    print("Loading %s"%mdl.getModelFilename())
    if False:
        mdl.load()  #loads all sub-models!!
    else:
        mdl.ssvm = mdl._loadIfFresh(mdl.getModelFilename(), None, lambda x: SaveLogger(x).load())

    print(mdl.getModelInfo())
    
    fig = plt.figure()
    fig.canvas.set_window_title("dir=%s  model=%s  "%(sModelDir, sModelName))
    plt.plot(mdl.ssvm.loss_curve_)
    plt.xlabel("Iteration / 10")
    plt.ylabel("Loss")
    plt.show()

    
    
