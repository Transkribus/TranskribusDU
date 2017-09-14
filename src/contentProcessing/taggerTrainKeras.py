# -*- coding: utf-8 -*-

"""

    taggerTrainingKeras.py

    train Deep Named entities tagger
        
    H. Déjean
    
    copyright Naverlabs 2017
    READ project 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from __future__ import unicode_literals
  
import sys,os
import codecs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import common.Component as Component
from common.trace import traceln

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import Sequential, load_model
from keras.layers  import Bidirectional, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
import numpy as np

import cPickle
import gzip

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        
    def fit(self, l, y=None):
        return self

    def transform(self, l):
        assert False, "Specialize this method!"
        
class SparseToDense(Transformer):
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, o):
        return o.toarray()    
    
class NodeTransformerTextEnclosed(Transformer):
    """
    we will get a list of block and need to send back what a textual feature extractor (TfidfVectorizer) needs.
    So we return a list of strings  
    """
    def transform(self, lw):
        return map(lambda x: x, lw) 
    

class DeepTagger(Component.Component):
    usage = "" 
    version = "v.01"
    description = "description: keras/bilstm ner"
        
    def __init__(self):
        Component.Component.__init__(self, "keras NER", self.usage, self.version, self.description) 
        
        self.dirName = None
        self.sModelName = None
        self.sAux = "aux.pkl"

        self.nbClasses = None
        self.max_sentence_len = 0
        self.max_features = 500
        self.maxngram = 3
        
        self.nbEpochs = 100
        self.batch_size = 50
        
        self.bTraining, self.bTesting, self.bPredict = False,False,False
        self.lTrain = []
        self.lTest  = []
        self.lPredict= []

        # mapping vector
        self.tag_vector={}


    def setParams(self,dParams):
        """
        
        """
        Component.Component.setParams(self, dParams)

        if dParams.has_key("dirname"): 
            self.dirName = dParams["dirname"]
        
        if dParams.has_key("name"): 
            self.sModelName = dParams["name"]
        
        if dParams.has_key("nbEpochs"): 
            self.nbEpochs = dParams["nbEpochs"]
        
        if dParams.has_key("nbfeatures"): 
            self.max_features = dParams["nbfeatures"]             

        if dParams.has_key("ngram"): 
            self.maxngram = dParams["ngram"]   
                        
        if dParams.has_key("training"): 
            self.lTrain = dParams["training"]      
            self.bTraining=True                  
        
        if dParams.has_key("testing"): 
            self.lTest = dParams["testing"]
            self.bTesting=True 
        
        if dParams.has_key("predict"): 
            self.lPredict = dParams["predict"]
            self.bPredict=True 
        
        
    def initTransformeur(self):
        self.cv= CountVectorizer( max_features = self.max_features
                             , analyzer = 'char' ,ngram_range = (1,self.maxngram)
                             , dtype=np.float64)
        self.node_transformer =  FeatureUnion([
            ("ngrams", Pipeline([
                            ('selector', NodeTransformerTextEnclosed()),
                            ('cv', self.cv),
                           ('todense', SparseToDense())  
                            ])
             )
        ])     
        
         


    def load_data(self,lFName):
        """
            load data as training data (x,y)
            nbClasses must be known!
        """
        
        self.nbClasses = 0
        lClasses=[]
               
        for fname in lFName:
            f=codecs.open(fname,encoding='utf-8')
        
            lTmp=[]
            x=[]
            for l in f:
                l = l.strip()
                if l =='EOS':
                    lTmp.append(x)
                    self.max_sentence_len = max(self.max_sentence_len,len(x))
                    x=[]
                else:
                    try:
                        a,b=l.split('\t')
                    except  ValueError:
                        #print 'cannot find value and label in: %s'%(l)
                        continue
                        
                    assert len(b) != 0 
                    if b not in lClasses:
                        lClasses.append(b)
                    x.append((a,b))
        
            if x != []:
                lTmp.append(x)
            f.close()
            
        self.nbClasses = len(lClasses) + 1
            
        for tag_class_id,b in enumerate(lClasses):
            one_hot_vec = np.zeros(self.nbClasses, dtype=np.int32)
            one_hot_vec[tag_class_id] = 1                
            self.tag_vector[b] = tuple(one_hot_vec)
            self.tag_vector[tuple(one_hot_vec)] = b
                            
        # Add nil class
        if 'NIL' not in self.tag_vector:
            one_hot_vec = np.zeros(self.nbClasses, dtype=np.int32)
            one_hot_vec[self.nbClasses-1] = 1
            self.tag_vector['NIL'] = tuple(one_hot_vec)
            self.tag_vector[tuple(one_hot_vec)] = 'NIL'
        
#         print self.nbClasses
        
        #     shuffle(lTmp)
        lX = []
        lY = []    
        for sample in lTmp:
            lX.append(map(lambda (x,y):x,sample))
            lY.append(map(lambda (x,y):y,sample))
        
        del lTmp
        
        return lX,lY

    

    def storeModel(self,model, aux):
        """
            store model and auxillary data (transformer)
        """
        model.save('%s/%s.hd5'%(self.dirName,self.sModelName))
        traceln('model dumped  in %s/%s.hd5' % (self.dirName,self.sModelName))        
        
        #max_features,max_sentence_len, self.nbClasses,self.tag_vector , node_transformer
        cPickle.dump((self.maxngram,self.max_features,self.max_sentence_len,self.nbClasses,self.tag_vector,self.node_transformer),gzip.open('%s/%s.%s'%(self.dirName,self.sModelName,self.sAux),'wb'))
        traceln('dump done in %s/%s.%s' % (self.dirName,self.sModelName,self.sAux))        
        
    def training(self,traindata):
        """
            training
        """

        train_X,train_Y = traindata #self.load_data(self.lTrain)
        
        fX= [item  for sublist in train_X  for item in sublist ]
        # print fX
        # print map(type,fX)
        self.node_transformer.fit(fX)
    
        #allwords= self.node_transformer.transform(fX)
        
        i=0
        lX = list()
        lY= list()
    #     max_sentence_len=10
        for x,y in zip(train_X,train_Y):
            n=len(x)
            words = self.node_transformer.transform(x)
#             print 'xy',x,y, words
            wordsvec = []
            elem_tags = []
#             for ix,ss in enumerate(allwords[i:i+n]):
            for ix,ss in enumerate(words):
                assert len(ss) == self.max_features, len(ss)
                wordsvec.append(ss)
                elem_tags.append(list(self.tag_vector[y[ix]]))
        
        #     print wordsvec
            nil_X = np.zeros(self.max_features)
            nil_Y = np.array(self.tag_vector['NIL'])
            pad_length = self.max_sentence_len - len(wordsvec)
            lX.append( ((pad_length)*[nil_X]) + wordsvec)
            lY.append( ((pad_length)*[nil_Y]) + elem_tags)        
            
        #     lX.append(wordsvec)
        
            i +=n
        del self.node_transformer
        del fX
        del train_X
        del train_Y
        
        print('Training.\n')        
        lX=np.array(lX)
        lY=np.array(lY)
        print lX.shape
        print lY.shape

        model = Sequential()
        # model.add(Embedding(104, 100))
        #model.add(LSTM(20,input_shape=(3,max_features)))
        
        #model.add(Embedding(input_dim=max_features, input_length=10,output_dim=100,batch_input_shape=(lX.shape[0],max_sentence_len,max_features)))
        #emb= Embedding(input_dim=max_features, input_length=10,output_dim=100,batch_input_shape=(lX.shape[0],max_sentence_len,max_features))
        
        # print model.input_shape
        # print model.output_shape
        
        model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(self.max_sentence_len,self.max_features)))
        model.add(Dropout(0.5))
    #     model.add(Bidirectional(LSTM(50, return_sequences=True)))
    #     model.add(Dropout(0.5))
        #print model.output_shape
        #model.add(Dense(5, activation = 'softmax'))
        # model.add(Dense(2, input_dim=nb_words))
        # model.add(Dense(1, activation='sigmoid'))
    #     model.add(TimeDistributed(Dense(self.nbClasses, activation='softmax')))
        model.add(Dense(self.nbClasses, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        print model.summary()
        
        history = model.fit(lX, lY, epochs = self.nbEpochs,batch_size = self.batch_size, verbose = 1,validation_split = 0.33, shuffle=True)
    #     import matplotlib.pyplot as plt
#         print(history.history.keys())
        
        del lX,lY
        
        auxdata = self.max_features,self.max_sentence_len,self.nbClasses,self.tag_vector,self.node_transformer
        return model, auxdata

    def testModel(self,testdata):
        """
            test model
        """
        #model = load_model('%s/%s.hd5'% (self.dirName,self.sModelName))
        #self.maxngram,self.max_features,self.max_sentence_len, self.nbClasses,self.tag_vector , self.node_transformer = cPickle.load(gzip.open('%s/%s.%s'% (self.dirName,self.sModelName,self.sAux), 'rb'))
        
        test_x,test_y = testdata #self.load_data(lTestFile)
        
#         fX= [item  for sublist in test_x  for item in sublist ]
#         allwords= node_transformer.transform(fX)
        # print allwords
        
        i=0
        lX = list()
        lY= list()
        for x,y in zip(test_x,test_y):
    #         print 'xy',x,y
            n=len(x)
            wordsvec = []
            elem_tags = []
            words = self.node_transformer.transform(x)
            for ix,ss in enumerate(words):
        #         print ss.shape,ss
                wordsvec.append(ss)
                elem_tags.append(list(self.tag_vector[y[ix]]))
        
        #     print wordsvec
            nil_X = np.zeros(self.max_features)
            nil_Y = np.array(self.tag_vector['NIL'])
            pad_length = self.max_sentence_len - len(wordsvec)
            lX.append( ((pad_length)*[nil_X]) + wordsvec)
            lY.append( ((pad_length)*[nil_Y]) + elem_tags)        
            
        #     lX.append(wordsvec)
        
            i +=n
            
        # test_x needed
        del test_y
         
        print('testing.\n')        
        
        lX=np.array(lX)
        lY=np.array(lY)

        scores = self.model.evaluate(lX,lY,verbose=True)
        print zip(self.model.metrics_names,scores)
        
        
        y_pred = self.model.predict(lX)
        for i,x in enumerate(lX): 
            pred_seq = y_pred[i]
            pred_tags = []
            pad_length = self.max_sentence_len - len(test_x[i])
            for class_prs in pred_seq:
                class_vec = np.zeros(self.nbClasses, dtype=np.int32)
                class_vec[ np.argmax(class_prs) ] = 1
                if tuple(class_vec.tolist()) in self.tag_vector:
                    pred_tags.append((self.tag_vector[tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)]))
                    #pred_tags.append(self.tag_vector[tuple(class_vec.tolist())])
            print test_x[i],pred_tags[pad_length:]    
        
        
    def loadModels(self):
        """
            load models and aux data
        """
        self.model = load_model(os.path.join(self.dirName,self.sModelName+'.hd5'))
        self.max_features,self.max_sentence_len, self.nbClasses,self.tag_vector , self.node_transformer = cPickle.load(gzip.open('%s/%s.%s'%(self.dirName,self.sModelName,self.sAux),'r'))
          
        
    
    def predict(self,lsent):
        """
            predict over a set of sentences (unicode)
        """
    
        lRes= []
        for mysent in lsent :
    #         print self.tag_vector
            allwords= self.node_transformer.transform(mysent.split())
#             n=len(mysent.split())
            wordsvec = []
            for w in allwords:
                wordsvec.append(w)
            lX = list()
            nil_X = np.zeros(self.max_features)
            pad_length = self.max_sentence_len - len(wordsvec)
            lX.append( ( pad_length*[nil_X]) + wordsvec)
            lX=np.array(lX)
            assert pad_length*[nil_X] + wordsvec >= self.max_sentence_len
        #     lpred = model.predict(lX)
        #     lpred = model.predict_classes(lX)
        #     print map(lambda x:self.tag_vector[x],lpred[0])
            y_pred = self.model.predict(lX)
            for i,_ in enumerate(lX):
                pred_seq = y_pred[i]
                pred_tags = []
                pad_length = self.max_sentence_len - len(allwords)
                for class_prs in pred_seq:
                    class_vec = np.zeros(self.nbClasses, dtype=np.int32)
                    class_vec[ np.argmax(class_prs) ] = 1
                    if tuple(class_vec.tolist()) in self.tag_vector:
                        pred_tags.append((self.tag_vector[tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)]))
#                 print zip(mysent.encode('utf-8').split(),pred_tags[pad_length:])
                lRes.append((mysent.split(),pred_tags[pad_length:]))   

        return lRes
    
    def run(self):
        """
            
        """
        if self.bTraining:
            lX, lY = self.load_data(self.lTrain)
            print len(lX),len(lY)
            self.initTransformeur()
            model, other = self.training((lX,lY))
            # store
            self.storeModel(model,other)
            del model
        if self.bTesting:
            # frist load model and aux data (for nbClasses!!)
            self.loadModels()
            lX,lY = self.load_data(self.lTest)
            res = self.testModel((lX,lY))
                       
        if self.bPredict:
            # which input  format: [unicode]
            self.loadModels()
            lsent = self.load_dataForPrediction(self.lPredict)
            res = self.predict(lsent)

if __name__ == '__main__':
    
    cmp = DeepTagger()
    cmp.createCommandLineParser()
    cmp.parser.add_option("--name", dest="name",  action="store", type="string", help="model name")
    cmp.parser.add_option("--dir", dest="dirname",  action="store", type="string", help="directory to store model")

    cmp.parser.add_option("--training", dest="training",  action="append", type="string", help="training data")
        
    cmp.parser.add_option("--epochs", dest="nbEpochs",  action="store", type="int", help="nb epochs for training")    
    cmp.parser.add_option("--ngram", dest="ngram",  action="store", type="int", help="ngram size")    
    cmp.parser.add_option("--nbfeatures", dest="nbfeatures",  action="store", type="string", help="nb features")    

    cmp.parser.add_option("--testing", dest="testing",  action="append", type="string", help="test data")    
    cmp.parser.add_option("--run", dest="predict",  action="append", type="string", help="data to be categorized")    

    dParams, args = cmp.parseCommandLine()

    #Now we are back to the normal programmatic mode, we set the component parameters
    cmp.setParams(dParams)
    #This component is quite special since it does not take one XML as input but rather a series of files.
    #doc = cmp.loadDom()
    doc = cmp.run()    
