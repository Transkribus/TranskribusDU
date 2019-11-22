# -*- coding: utf-8 -*-
"""


    textNoiseGenerator.py

    add noise for  a text Generator: noisy char and split tokens 
     H. Déjean
    

    copyright Naver labs Europe 2017
    READ project 


    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""




from .noiseGenerator import noiseGenerator

class textNoiseGenerator(noiseGenerator):
    def __init__(self,tuplesplit, tupleNsplit,tuplechar):
        noiseGenerator.__init__(self)
        self.THSplitM,self.THSplitSD  = tuplesplit
        self.NSplitM,self.NSplitSD  = tupleNsplit
        self.THcharNoiseM,self.THcharNoiseSD  = tuplechar
        
        
        self._structure = [()]
        
        
    def generate(self,gtdata):
        """
            apply the noiseGenerator over text
            text: token, label
        """
        lnoisyGT=[]
        # duplicate labels for mutitoken
        
        ## merge noise: select elements which are merged
        ####  need to be 2  consecutive elements  :  need to rebuild the label hierarchy?
        ## split noise: select elements which are split
        ## delete noise: elt is deleted
        ## txt noise: select elements whose content will be noisy
        
        for token,label in gtdata:
            # should be replace by self.tokenizer(token)
            if type(token) == unicode:
                ltoken= token.split(" ")
            elif type(token) in [float,int ]:
                ltoken= [token]
            
            if len(ltoken) == 1:
                lnoisyGT.append((token,label))
            else:
                for tok in ltoken:
                    lnoisyGT.append((tok,label[:]))
            
        
        
        