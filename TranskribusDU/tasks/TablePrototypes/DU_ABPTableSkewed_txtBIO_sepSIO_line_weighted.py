# -*- coding: utf-8 -*-

"""
    *** Same as its parent apart that text baselines are reflected as a LineString (instead of its centroid)
    
    DU task for ABP Table: 
        doing jointly row BIO and near horizontal cuts SIO
    
    block2line edges do not cross another block.
    
    The cut are based on baselines of text blocks, with some positive or negative inclination.

    - the labels of cuts are SIO 
    
    Copyright Naver Labs Europe(C) 2018 JL Meunier


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys, os

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )
    import TranskribusDU_version

from crf.Model_SSVM_AD3_Multitype import Model_SSVM_AD3_Multitype

from tasks.DU_ABPTableSkewed_txtBIO_sepSIO_line import DU_ABPTableSkewedRowCutLine, main_command_line


class Model_SSVM_AD3_Multitype_MYWEIGHTS(Model_SSVM_AD3_Multitype):
    
    def computeClassWeight(self, _lY):
        return [[0.1/1.5,  # row_B   5173
                 0.1/1.5,  # row_I   4070
                 0.1/1.5]  # row_O    317
                 , [
                 1.0/1.5,  #sepH_S    524
                 0.1/1.5,  #sepH_I   6294
                 0.1/1.5]   #sepH_O    280
            ]


class DU_ABPTableSkewedRowCutLineWeighted(DU_ABPTableSkewedRowCutLine):
    """
    Weighted version 
    """

    def __init__(self, sModelName, sModelDir, 
                 iBlockVisibility = None,
                 iLineVisibility = None,
                 fCutHeight = None,
                 bCutAbove = None,
                 lRadAngle = None,
                 sComment = None,
                 C=None, tol=None, njobs=None, max_iter=None,
                 inference_cache=None): 
        DU_ABPTableSkewedRowCutLine.__init__(self, sModelName, sModelDir, 
                 iBlockVisibility,
                 iLineVisibility,
                 fCutHeight,
                 bCutAbove,
                 lRadAngle,
                 sComment,
                 C, tol, njobs, max_iter,
                 inference_cache)
        
        self.setModelClass(Model_SSVM_AD3_Multitype_MYWEIGHTS)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main_command_line(DU_ABPTableSkewedRowCutLineWeighted)
