# -*- coding: utf-8 -*-
"""
    typoGenerator.py

    create (generate) textual annotated data 
     H. Déjean

    copyright Xerox 2017
    READ project 

    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from __future__ import absolute_import
from __future__ import  print_function
from __future__ import unicode_literals

from dataGenerator.generator import Generator 

class typoGenerator(Generator):
    """
        generator for typographical position: 
    """
    lnames=[]
    
    def __init__(self, lw=[1,0,0]):
        """
            config: provides a weight for each typo positions
        """
        assert len(lw) == 3
        self.lw=lw
        Generator.__init__(self,{})
        mylist=[]
        for i,w in enumerate(lw):
            mylist.append((i,w))

        self.loadResourcesFromList([mylist],iMax=10)
    
    def generate(self):
        """
            need to take into account element frequency! done in getRandomElt
            this generate is for terminal elements (otherwise use Generator.generate() )
        """
        self._generation = self.getRandomElt(None)
        return self    
    
class verticalTypoGenerator(typoGenerator):
    TYPO_LEFT       = 0
    TYPO_VCENTER    = 1
    TYPO_RIGHT      = 2
    lnames = ['TYPO_LEFT','TYPO_RIGHT','TYPO_HCENTER']
  
class horizontalTypoGenerator(typoGenerator):
    TYPO_TOP       = 0
    TYPO_BOTTOM      = 2
    TYPO_HCENTER    = 1
    lnames = ['TYPO_TOP','TYPO_BOTTOM','TYPO_VCENTER']
        
if __name__ == '__main__':
    typo=horizontalTypoGenerator([0.5,0.25,0.25])
    typo.instantiate()
    typo.generate()
    print (typo._generation)
    typo=verticalTypoGenerator([0,1.,0])
    typo.instantiate()
    typo.generate()
    print (typo._generation)    