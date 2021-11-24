# -*- coding: utf-8 -*-

"""
    taggerChrono.py

    task: recognition of a chronological sequence of dates
        
    H. Déjean
    
    copyright Naver labs Europe 2019
"""    
import sys,os
from optparse import OptionParser
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair.embeddings import FlairEmbeddings
from flair.data import Sentence
import torch.nn

import pickle


class trainLM():
    
    def __init__(self):
        # are you training a forward or backward LM?
        self.is_forward_lm = True

    def loadDict(self,dictpath):
        # load the default character dictionary
        #dictionary: Dictionary = Dictionary.load('chars')
        return Dictionary.load_from_file(dictpath)

    def loadCorpus(self,corpusFile, dictionary):
        # get your corpus, process forward and at the character level
        corpus = TextCorpus(corpusFile,
                            dictionary,
                            self.is_forward_lm,
                            character_level=True)
        return corpus

    def trainMe(self,dictionary, corpus,modelfile):
        # instantiate your language model, set hidden size and number of layers
        language_model = LanguageModel(dictionary,
                                       self.is_forward_lm,
                                       hidden_size=128,
                                       nlayers=1)
        print (f'lm created: {language_model}' )

        # train your language model
        trainer = LanguageModelTrainer(language_model, corpus)
        print (f'trainer created: {trainer}' )

        trainer.train(modelfile,
                      sequence_length=100,
                      mini_batch_size=10,
                      max_epochs=10)
        
        return trainer

    def loadEmbedding(self,modelname):
        return   FlairEmbeddings(modelname)
        #char_lm_embeddings.embed(sentence)
        
        
    def testModel(self,femb):
        """
        

        """
        x=Sentence('Ausnahmt gütleir')
        x2=Sentence('Ausnahms¬ Gütler')
        x=Sentence('Jäner')
        x2=Sentence('Juni')        
        femb.embed(x)
        femb.embed(x2)

        b=x.get_token(1).get_embedding()
        b2=x2.get_token(1).embedding
        cos=torch.nn.CosineSimilarity(dim=0)
        print(cos(b,b2))
        print(x.get_token(1))
        print(x2.get_token(1))
        
        
        
    def run(self,corpusFile, dictionaryFile,modelName):
        if True:
            dictionary = self.loadDict(dictionaryFile)
            print (f'dictionary loaded: {dictionary}' )
            corpus = self.loadCorpus(corpusFile, dictionary)
            print (f'corpus loaded: {corpus}' )
            trainer = self.trainMe(dictionary, corpus,modelName)
        lmmodel = FlairEmbeddings(os.path.join(modelName,'best-lm.pt'))
        self.testModel(lmmodel)
        
if __name__ == "__main__":
    sUsage="usage: %s <modelname> <col-dir>" % sys.argv[0]

    parser = OptionParser(usage=sUsage)
    
    parser.add_option("--char", dest='charmap',  action="store", type="string"
                      , help="file containing the char map dictionary")   
    (options, args) = parser.parse_args()

    try:
        sOutput     = args[0]
        lsDir       = args[1]
    except:
        print(sUsage % sys.argv[0])
        exit(1)
        
    doer = trainLM()

    doer.run(lsDir,  options.charmap,sOutput)
    print("Done")
        
