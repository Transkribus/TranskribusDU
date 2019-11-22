# -*- coding: utf-8 -*-
"""


    noiseGenerator.py

    add noise for  a Generator 
     H. Déjean
    

    copyright Naver labs Europe 2017
    READ project 


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""




from .generator import Generator

class noiseGenerator(Generator):
    def __init__(self):
        Generator.__init__(self)
    
        