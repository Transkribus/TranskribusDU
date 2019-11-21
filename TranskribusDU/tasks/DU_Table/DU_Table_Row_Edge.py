# -*- coding: utf-8 -*-

"""
    DU task for segmenting text in cols using the conjugate graph after the SW
    re-engineering by JLM
    
    Copyright NAVER(C)  2019  Jean-Luc Meunier
    

    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
 
import sys, os

try: #to ease the use without proper Python installation
    import TranskribusDU_version
except ImportError:
    sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) ))) )
    import TranskribusDU_version
TranskribusDU_version

from tasks.DU_Table.DU_Table_Cell_Edge import main
    

if __name__ == "__main__":
    #     import better_exceptions
    #     better_exceptions.MAX_LENGTH = None

    main(sys.argv[0], "row")