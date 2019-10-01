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

import sys,os
import codecs
from optparse import OptionParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

# import common.Component as Component
#from common.trace import traceln

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import Sequential, load_model, Model
from keras.layers  import Bidirectional, Dropout, Input
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Masking
from keras.regularizers import L1L2
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
    

class DeepTagger():
    usage = "" 
    version = "v.01"
    description = "description: keras/bilstm ner"
        
    def __init__(self):
        
        self.dirName = None
        self.sModelName = None
        self.sAux = "aux.pkl"

        self.nbClasses = None
        self.max_sentence_len = 0
        self.max_features = 100
        self.maxngram = 3
        
        self.nbEpochs = 10
        self.batch_size = 50
        
        self.hiddenSize= 32
        
        self.bGridSearch  = False
        self.bTraining_multitype,self.bTraining, self.bTesting, self.bPredict = False,False,False, False
        self.lTrain = []
        self.lTest  = []
        self.lPredict= []

        self.bMultiType = False
        # mapping vector
        self.tag_vector={}


    def setParams(self,dParams):
        """
        
        """

        if dParams.dirname:
            self.dirName = dParams.dirname
        
        if dParams.name:
            self.sModelName = dParams.name
        
        if dParams.batchSize:
            self.batch_size = dParams.batchSize
    
        if dParams.nbEpochs:
            self.nbEpochs = dParams.nbEpochs

        if dParams.hidden:
            self.hiddenSize = dParams.hidden
                    
        if dParams.nbfeatures:
            self.max_features = dParams.nbfeatures            

        if dParams.ngram:
            self.maxngram = dParams.ngram  
        
        self.bMultiType = dParams.multitype
            
        if dParams.training:
            self.lTrain = dParams.training
            self.bTraining=True                  
        
        if dParams.testing:
            self.lTest = dParams.testing
            self.bTesting=True 
        
        if dParams.predict:
            self._sent =dParams.predict #.decode('latin-1')
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
        
         
    def load_data_Multitype(self,lFName):
        """
            load data as training data (x,y)
            nbClasses must be known!
        """
        
        self.nbClasses = 0
        
        self.lClasses=[]

        lTmp=[]               
        for fname in lFName:
            f=codecs.open(fname,encoding='utf-8')
            x=[]
            for l in f:
                l = l.strip()
                if l[:3] == '#  ':continue  # comments
                if l =='EOS':
                    lTmp.append(x)
                    self.max_sentence_len = max(self.max_sentence_len,len(x))
                    x=[]
                else:
                    try:
                        la=l.split('\t')
                        b1=la[-1].split('_')[0]
                        b2=la[-1].split('_')[1]
                    except  ValueError:
                        #print 'cannot find value and label in: %s'%(l)
                        continue
                    assert len(la) != 0 
                    if b2 not in self.lClasses:
                        self.lClasses.append(b2)
                    if b1 not in self.lClasses:
                        self.lClasses.append(b1)
                    x.append((la[0],(b1,b2)))
        
            if x != []:
                lTmp.append(x)
            f.close()
            
        self.nbClasses = len(self.lClasses) + 1
            
        for tag_class_id,b in enumerate(self.lClasses):
            one_hot_vec = np.zeros(self.nbClasses, dtype=np.int32)
            one_hot_vec[tag_class_id] = 1                
            self.tag_vector[b] = tuple(one_hot_vec)
            self.tag_vector[tuple(one_hot_vec)] = b
                            
        # Add nil class
        if 'NIL' not in self.tag_vector:
            self.lClasses.append('NIL')
            one_hot_vec = np.zeros(self.nbClasses, dtype=np.int32)
            one_hot_vec[self.nbClasses-1] = 1
            self.tag_vector['NIL'] = tuple(one_hot_vec)
            self.tag_vector[tuple(one_hot_vec)] = 'NIL'
        
#         print self.nbClasses
        
        #     shuffle(lTmp)
        lX = []
        lY = []    
        for sample in lTmp:
            lX.append(list(map(lambda xy:xy[0],sample)))
            lY.append(list(map(lambda xy:xy[1],sample)))
        
        del lTmp
        
        return lX,lY

    def load_data_Multitype2(self,lFName):
        """
            load data as training data (x,y)
            nbClasses must be known!
            
            Y1:  final annotation        S_firstNameGenerator
            Y2: last but one annotation   B_PersonName, S_familyStatus
                if only 2 level: ignore
            
Katharina       B_ABPRecordGenerator    B_PersonName    S_firstNameGenerator
Schönberger     I_ABPRecordGenerator    E_PersonName    S_lastNameGenerator
witwe   E_ABPRecordGenerator    S_familyStatus
EOS
            
            
            
        """
        
        self.nbClasses = 0
        
        self.lClasses=[]

        lTmp=[]               
        for fname in lFName:
            f=codecs.open(fname,encoding='utf-8')
            x=[]
            for l in f:
                l = l.strip()
                if l[:3] == '#  ':continue  # comments
                if l =='EOS':
                    lTmp.append(x)
                    self.max_sentence_len = max(self.max_sentence_len,len(x))
                    x=[]
                else:
                    try:
                        la=l.split('\t')
                        b1=la[-1]
                        if len(la) > 3:
                            b2=la[-2]
                        else:
                            b2='None'
                    except  ValueError:
                        #print 'cannot find value and label in: %s'%(l)
                        continue
                    assert len(la) != 0 
#                     print(la[0].encode('utf-8'),b1,b2)
                    if b2 not in self.lClasses:
                        self.lClasses.append(b2)
                    if b1 not in self.lClasses:
                        self.lClasses.append(b1)
                    x.append((la[0],(b1,b2)))
        
            if x != []:
                lTmp.append(x)
            f.close()
            
        self.nbClasses = len(self.lClasses) + 1
            
        for tag_class_id,b in enumerate(self.lClasses):
            one_hot_vec = np.zeros(self.nbClasses, dtype=np.int32)
            one_hot_vec[tag_class_id] = 1                
            self.tag_vector[b] = tuple(one_hot_vec)
            self.tag_vector[tuple(one_hot_vec)] = b
                            
        # Add nil class
        if 'NIL' not in self.tag_vector:
            self.lClasses.append('NIL')
            one_hot_vec = np.zeros(self.nbClasses, dtype=np.int32)
            one_hot_vec[self.nbClasses-1] = 1
            self.tag_vector['NIL'] = tuple(one_hot_vec)
            self.tag_vector[tuple(one_hot_vec)] = 'NIL'
        
#         print self.nbClasses
        
        #     shuffle(lTmp)
        lX = []
        lY = []    
        for sample in lTmp:
            lX.append(list(map(lambda xy:xy[0],sample)))
            lY.append(list(map(lambda xy:xy[1],sample)))
        
        del lTmp
        
        return lX,lY


    def load_data(self,lFName):
        """
            load data as training data (x,y)
            nbClasses must be known!
        """
        
        self.nbClasses = 0
        
        self.lClasses=[]

        lTmp=[]               
        for fname in lFName:
            f=codecs.open(fname,encoding='utf-8')
            x=[]
            for l in f:
                l = l.strip()
                if l[:3] == '#  ':continue  # comments
                if l =='EOS':
                    lTmp.append(x)
                    self.max_sentence_len = max(self.max_sentence_len,len(x))
                    x=[]
                else:
                    try:
                        la=l.split('\t')
                        b1=la[-1]
                    except  ValueError:
                        print ('cannot find value and label in: %s'%(l))
                        sys.exit()
                    assert len(la) != 0 
                    if b1 not in self.lClasses:
                        self.lClasses.append(b1)
                    x.append((la[0],b1))                    
        
            if x != []:
                lTmp.append(x)
            f.close()
            
        self.nbClasses = len(self.lClasses) + 1
            
        for tag_class_id,b in enumerate(self.lClasses):
            one_hot_vec = np.zeros(self.nbClasses, dtype=np.int32)
            one_hot_vec[tag_class_id] = 1                
            self.tag_vector[b] = tuple(one_hot_vec)
            self.tag_vector[tuple(one_hot_vec)] = b
                            
        # Add nil class
        if 'NIL' not in self.tag_vector:
            self.lClasses.append('NIL')
            one_hot_vec = np.zeros(self.nbClasses, dtype=np.int32)
            one_hot_vec[self.nbClasses-1] = 1
            self.tag_vector['NIL'] = tuple(one_hot_vec)
            self.tag_vector[tuple(one_hot_vec)] = 'NIL'
        
#         print self.nbClasses
        
        #     shuffle(lTmp)
        lX = []
        lY = []    
        for sample in lTmp:
            lX.append(list(map(lambda xy:xy[0],sample)))
            lY.append(list(map(lambda xy:xy[1],sample)))
        del lTmp
        return lX,lY

    
    def load_data_for_testing(self,lFName):
        """
            load data as training data (x,y)
            nbClasses must be known!
            loadModel first!
        """
        
        lTmp=[]
        for fname in lFName:
            f=codecs.open(fname,encoding='utf-8')
            x=[]
            for l in f:
                l = l.strip()
                if l[:3] == '#  ':continue  # comments
                if l =='EOS':
                    if x!=[]:
                        lTmp.append(x)
                    x=[]                    
                else:
                    try:
                        la=l.split('\t')
                        b1=la[-1]
                    except  ValueError:
                        print('cannot find value and label in: %s'%(l))
                        sys.exit(1)
                    x.append((la[0],b1))      
        
            if x != []:
                lTmp.append(x)
            f.close()
            
        lX = []
        lY = []    
        for sample in lTmp:
            lX.append(list(map(lambda xy:xy[0],sample)))
            lY.append(list(map(lambda xy:xy[1],sample)))
        
        del lTmp
        
        return lX,lY

    def load_data_for_testing_Multitype(self,lFName):
        """
            load data as training data (x,y)
            nbClasses must be known!
            loadModel first!
        """
        
        lTmp=[]
        for fname in lFName:
            f=codecs.open(fname,encoding='utf-8')
            x=[]
            for l in f:
                l = l.strip()
                if l[:3] == '#  ':continue  # comments
                if l =='EOS':
                    if x!=[]:
                        lTmp.append(x)
                    x=[]
                else:
                    try:
                        la=l.split('\t')
                        b1=la[-1].split('_')[0]
                        b2=la[-1].split('_')[1]
                    except  ValueError:
                        print('ml:cannot find value and label in: %s'%(l))
                        sys.exit()
                    assert len(la) != 0 
                    x.append((la[0],(b1,b2)))
        
            if x != []:
                lTmp.append(x)
            f.close()
            
        lX = []
        lY = []    
        for sample in lTmp:
            lX.append(list(map(lambda xy:xy[0],sample)))
            lY.append(list(map(lambda xy:xy[1],sample)))            
        
        del lTmp
        
        return lX,lY

    def load_data_for_testing_Multitype2(self,lFName):
        """
            load data as training data (x,y)
            nbClasses must be known!
            loadModel first!
        """
        
        lTmp=[]
        for fname in lFName:
            f=codecs.open(fname,encoding='utf-8')
            x=[]
            for l in f:
                l = l.strip()
                if l[:3] == '#  ':continue  # comments
                if l =='EOS':
                    if x!=[]:
                        lTmp.append(x)
                    x=[]
                else:
                    try:
                        la=l.split('\t')
                        b1=la[-1].split('_')[0]
                        b2=la[-1].split('_')[1]
                    except  ValueError:
                        print('ml:cannot find value and label in: %s'%(l))
                        sys.exit()
                    assert len(la) != 0 
                    x.append((la[0],(b1,b2)))
        
            if x != []:
                lTmp.append(x)
            f.close()
            
        lX = []
        lY = []    
        for sample in lTmp:
            lX.append(list(map(lambda xy:xy[0],sample)))
            lY.append(list(map(lambda xy:xy[1],sample)))            
        
        del lTmp
        
        return lX,lY

    def storeModel(self,model, aux):
        """
            store model and auxillary data (transformer)
        """
        model.save('%s/%s.hd5'%(self.dirName,self.sModelName))
        print('model dumped  in %s/%s.hd5' % (self.dirName,self.sModelName))        
        
        #max_features,max_sentence_len, self.nbClasses,self.tag_vector , node_transformer
        cPickle.dump((self.bMultiType,self.maxngram,self.max_features,self.max_sentence_len,self.nbClasses,self.tag_vector,self.node_transformer),gzip.open('%s/%s.%s'%(self.dirName,self.sModelName,self.sAux),'wb'))
        print('aux data dumped in %s/%s.%s' % (self.dirName,self.sModelName,self.sAux))        
        
    def loadModels(self):
        """
            load models and aux data
        """
        self.model = load_model(os.path.join(self.dirName,self.sModelName+'.hd5'))
        print('model loaded: %s/%s.hd5' % (self.dirName,self.sModelName))  
        try:
            self.bMultiType,self.maxngram,self.max_features,self.max_sentence_len, self.nbClasses,self.tag_vector , self.node_transformer = cPickle.load(gzip.open('%s/%s.%s'%(self.dirName,self.sModelName,self.sAux),'r'))
        except:
            self.maxngram,self.max_features,self.max_sentence_len, self.nbClasses,self.tag_vector , self.node_transformer = cPickle.load(gzip.open('%s/%s.%s'%(self.dirName,self.sModelName,self.sAux),'r'))
            self.bMultiType = False
        print('aux data loaded: %s/%s.%s' % (self.dirName,self.sModelName,self.sAux))        
        print("ngram: %s\tmaxfea=%s\tpadding=%s\tnbclasses=%s" % (self.maxngram,self.max_features,self.max_sentence_len, self.nbClasses))
        print("multitype model:%s"%(self.bMultiType))
        
    def training(self,traindata):
        """
            training
        """
        train_X,_ = traindata #self.load_data(self.lTrain)
        
        self.initTransformeur()
        
        fX= [item  for sublist in train_X  for item in sublist ]
        self.node_transformer.fit(fX)
#         
        lX,lY = self.prepareTensor(traindata)

#         print lX.shape
#         print lY.shape

        model = Sequential()
        reg= L1L2(l1=0.001, l2=0.0)

        model.add(Masking(mask_value=0., input_shape=(self.max_sentence_len, self.max_features)))
        model.add(Bidirectional(LSTM(self.hiddenSize,return_sequences = True,bias_regularizer=reg))) 
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.nbClasses, activation='softmax')))
        #keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['categorical_accuracy']  )
        print (model.summary())
        _ = model.fit(lX, lY, epochs = self.nbEpochs,batch_size = self.batch_size, verbose = 1,validation_split = 0.1, shuffle=True)
        
        del lX,lY
        
        auxdata = self.max_features,self.max_sentence_len,self.nbClasses,self.tag_vector,self.node_transformer
        
        return model, auxdata

    def training_multitype(self,traindata):
        """
            training
        """
        train_X,_ = traindata #self.load_data(self.lTrain)
        
        self.initTransformeur()
        fX= [item  for sublist in train_X  for item in sublist ]
        self.node_transformer.fit(fX)
#         
        lX,(lY,lY2) = self.prepareTensor_multitype(traindata)

#         print lX.shape
#         print lY.shape

        inputs = Input(shape=(self.max_sentence_len, self.max_features))

        x = Masking(mask_value=0)(inputs)
        x = Bidirectional(LSTM(self.hiddenSize,return_sequences = True))(x) 
        x = Dropout(0.5)(x)
    
        out1 = TimeDistributed(Dense(self.nbClasses, activation='softmax'),name='BIES')(x)
        out2 = TimeDistributed(Dense(self.nbClasses, activation='softmax'),name='Label')(x)

        model = Model(input = inputs,output = [out1,out2])
        
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['categorical_accuracy']  )
        print (model.summary())
        _ = model.fit(lX, [lY,lY2], epochs = self.nbEpochs,batch_size = self.batch_size, verbose = 1,validation_split = 0.33, shuffle=True)
        
        del lX,lY
        
        auxdata = self.max_features,self.max_sentence_len,self.nbClasses,self.tag_vector,self.node_transformer
        
        return model, auxdata

    def prepareTensor_multitype(self,annotated):
        lx,ly = annotated 
        
        lX = list()
        lY1= list()
        lY2= list()
        
        # = np.array()
        for x,y in zip(lx,ly):
            words = self.node_transformer.transform(x)
            wordsvec = []
            elem_tags1 = []
            elem_tags2 = []

            for ix,ss in enumerate(words):
                wordsvec.append(ss)
                elem_tags1.append(list(self.tag_vector[y[ix][0]]))
                elem_tags2.append(list(self.tag_vector[y[ix][1]]))
        
            nil_X = np.zeros(self.max_features)
            nil_Y = np.array(self.tag_vector['NIL'])
            pad_length = self.max_sentence_len - len(wordsvec)
            lX.append( wordsvec +((pad_length)*[nil_X]) )
            lY1.append( elem_tags1 + ((pad_length)*[nil_Y]) )        
            lY2.append( elem_tags2 + ((pad_length)*[nil_Y]) )        

        del lx
        del ly
            
        lX=np.array(lX)
        lY1=np.array(lY1)
        lY2=np.array(lY2)
        
        return lX,(lY1,lY2)
    
    def prepareTensor(self,annotated):
        lx,ly = annotated 
        
        lX = list()
        lY= list()
        
        # = np.array()
        for x,y in zip(lx,ly):
            words = self.node_transformer.transform(x)
            wordsvec = []
            elem_tags = []
            for ix,ss in enumerate(words):
                wordsvec.append(ss)
                elem_tags.append(list(self.tag_vector[y[ix]]))
        
            nil_X = np.zeros(self.max_features)
            nil_Y = np.array(self.tag_vector['NIL'])
            pad_length = self.max_sentence_len - len(wordsvec)
            lX.append( wordsvec +((pad_length)*[nil_X]) )
            lY.append( elem_tags + ((pad_length)*[nil_Y]) )        

        del lx
        del ly
            
        lX=np.array(lX)
        lY=np.array(lY)
        
        return lX,lY
        
        
    def testModel(self,testdata):
        """
            test model
        """
        
        lX,lY= self.prepareTensor(testdata)

#         print lX.shape
#         print lY.shape

        scores = self.model.evaluate(lX,lY,verbose=True)
        print(list(zip(self.model.metrics_names,scores)))
        
        test_x, _ = testdata
        
        y_pred = self.model.predict(lX)
        for i,_ in enumerate(lX): 
            pred_seq = y_pred[i]
            pred_tags = []
            #pad_length = self.max_sentence_len - len(test_x[i])
            for class_prs in pred_seq:
                class_vec = np.zeros(self.nbClasses, dtype=np.int32)
                class_vec[ np.argmax(class_prs) ] = 1
                if tuple(class_vec.tolist()) in self.tag_vector:
                    pred_tags.append((self.tag_vector[tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)]))
            print(test_x[i],pred_tags[:len(test_x[i])])
        
    def testModel_Multitype(self,testdata):
        """
            test model
        """
        
        lX,(lY,lY2) = self.prepareTensor_multitype(testdata)

        scores = self.model.evaluate(lX,[lY,lY2],verbose=True)
        print(list(zip(self.model.metrics_names,scores)))
        
        test_x, _ = testdata
        
        y_pred1,y_pred2 = self.model.predict(lX)
        for i,_ in enumerate(lX):
            for pred_seq in [y_pred1[i],y_pred2[i]]:
                pred_tags = []
                for class_prs in pred_seq:
                    class_vec = np.zeros(self.nbClasses, dtype=np.int32)
                    class_vec[ np.argmax(class_prs) ] = 1
#                     print class_prs[class_prs >0.1]
                    if tuple(class_vec.tolist()) in self.tag_vector:
#                         print(self.tag_vector[tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)])
                        pred_tags.append((self.tag_vector[tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)]))
                print(test_x[i],pred_tags[:len(test_x[i])])
        
    def prepareOutput_multitype(self,lToken,lLTags):
        """
            format final output with MultiType
            first level: BIES segmentation
            remaining levels: label
            
            assumption: no contradiction between layers 
        """
        chunk=[]
        lChunk=[]
        curTag=None
        
        for itok,tok in enumerate(lToken):
#             print(tok,lLTags[0][itok],lLTags[1][itok])
            BIES,_ = lLTags[0][itok]
            offset = itok
            tag,score2 = lLTags[1][itok]
            if tag != curTag:
                if chunk !=[]:
                    lChunk.append((chunk,curTag))
                curTag=tag
                chunk= [(offset,tok,score2)]
            elif BIES == 'B':
                if chunk !=[]:
                    lChunk.append((chunk,curTag))
                curTag=tag
                chunk= [(offset,tok,score2)]
            elif BIES in ['I','E']:
                chunk.append((offset,tok,score2))
            elif BIES == 'S':
                if chunk !=[]:
                    lChunk.append((chunk,curTag))
                curTag=tag
                chunk= [(offset,tok,score2)]    
            
                
        if chunk !=[]:
            lChunk.append((chunk,tag))
        
        lRes=[]
        for (lList,label) in lChunk:
#             tok = " ".join(map(lambda x,tok,y: tok,lList))
            tok = " ".join(x[1] for x in lList)
            toffset = min(x[0] for x in lList),max(x[0] for x in lList)
            lScore = (x[2] for x in lList)
#             print toffset,tok,label,lScore
            lRes.append((toffset,tok,label,list(lScore)))

        return lRes       
 
        
    def prepareOutput(self,lToken, lTags):
        """
            format final output
        """
        
        chunk=[]
        lChunk=[]
        curTag=None
        for offset,(tok, (tag,score)) in enumerate(list(zip(lToken,lTags))):
#             print(tok.encode('utf-8'), tag)
            try:BIES,tag = tag.split('_')
            ## comp with old model structure
            except ValueError: BIES,tag,_ = tag.split('_') 
            if tag != curTag:
                if chunk !=[]:
                    lChunk.append((chunk,curTag))
                curTag=tag
                chunk= [(offset,tok,score)]
            elif BIES == 'B':
                if chunk !=[]:
                    lChunk.append((chunk,curTag))
                curTag=tag
                chunk= [(offset,tok,score)]
            elif BIES in ['I','E']:
                chunk.append((offset,tok,score))
            elif BIES == 'S':
                if chunk !=[]:
                    lChunk.append((chunk,curTag))
                curTag=tag
                chunk= [(offset,tok,score)]    
                
        if chunk !=[]:
            lChunk.append((chunk,tag))
        
        lRes=[]
        for (lList,label) in lChunk:
            tok = " ".join(x[1] for x in lList)
            toffset = min(x[0] for x in lList),max(x[0] for x in lList)
            lScore = (x[2] for x in lList)            
#             tok = " ".join(map(lambda x,tok,y: tok,lList))
#             toffset = (min(map(lambda offset,x,y: offset,lList)),max(map(lambda offset,x,y: offset,lList)))
#             lScore = (map(lambda offset,_,score: score,lList))
            lRes.append((toffset,tok,label,list(lScore)))
        
        return lRes
            
    def predict_multiptype(self,lsent):
        """
            predict over a set of sentences (unicode)
        """
    
        lRes= []
        for mysent in lsent :
    #         print self.tag_vector
            if len(mysent.split())> self.max_sentence_len:
                print ('max sent length: %s'%self.max_sentence_len)
                continue
            allwords= self.node_transformer.transform(mysent.split())
#             print mysent.split()
#             n=len(mysent.split())
            wordsvec = []
            for w in allwords:
                wordsvec.append(w)
            lX = list()
            nil_X = np.zeros(self.max_features)
            pad_length = self.max_sentence_len - len(wordsvec)
            lX.append( wordsvec +((pad_length)*[nil_X]) )
            lX=np.array(lX)
#             print(pad_length*[nil_X] + wordsvec, self.max_sentence_len)
#             assert pad_length*[nil_X] + wordsvec >= self.max_sentence_len
            y_pred1,y_pred2 = self.model.predict(lX)
            for i,_ in enumerate(lX):
#                 pred_seq = y_pred[i]
                l_multi_type_results = []
                for pred_seq in [y_pred1[i],y_pred2[i]]:
                    pred_tags = []
                    pad_length = self.max_sentence_len - len(allwords)
                    for class_prs in pred_seq:
                        class_vec = np.zeros(self.nbClasses, dtype=np.int32)
                        class_vec[ np.argmax(class_prs) ] = 1
    #                     print class_prs[class_prs >0.1]
                        if tuple(class_vec.tolist()) in self.tag_vector:
                            #print self.tag_vector[tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)]
                            pred_tags.append((self.tag_vector[tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)]))
                    l_multi_type_results.append(pred_tags[:len(allwords)])
                    print(l_multi_type_results) 
                lRes.append(self.prepareOutput_multitype(mysent.split(),l_multi_type_results))

        return lRes
        
    def predict(self,lsent):
        """
            predict over a set of sentences (unicode)
        """
    
        lRes= []
        for mysent in lsent :
    #         print self.tag_vector
            if len(mysent.split())> self.max_sentence_len:
                print ('max sent length: %s'%self.max_sentence_len)
                continue
            allwords= self.node_transformer.transform(mysent.split())
#             print mysent.split()
#             n=len(mysent.split())
            wordsvec = []
            for w in allwords:
                wordsvec.append(w)
            lX = list()
            nil_X = np.zeros(self.max_features)
            pad_length = self.max_sentence_len - len(wordsvec)
            lX.append( wordsvec +((pad_length)*[nil_X]) )
            lX=np.array(lX)
#             assert pad_length*[nil_X] + wordsvec >= self.max_sentence_len
            y_pred = self.model.predict(lX)
            for i,_ in enumerate(lX):
                pred_seq = y_pred[i]
                pred_tags = []
                pad_length = self.max_sentence_len - len(allwords)
                for class_prs in pred_seq:
                    class_vec = np.zeros(self.nbClasses, dtype=np.int32)
                    class_vec[ np.argmax(class_prs) ] = 1
#                     print class_prs[class_prs >0.1]
                    if tuple(class_vec.tolist()) in self.tag_vector:
                        pred_tags.append((self.tag_vector[tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)]))
#                 print zip(mysent.encode('utf-8').split(),pred_tags[pad_length:])
#                 lRes.append((mysent.split(),pred_tags[pad_length:]))   
                lRes.append(self.prepareOutput(mysent.split(),pred_tags[:len(allwords)]))

        return lRes
    
    def gridSearch(self):
        """
            perform grid search training
            assume epochs,ngram, nbfeatures   as N,N
            assume testing data for cross valid 
        """
    
        
    def run(self):
        """
            
        """
        if self.bGridSearch:
            self.gridSearch()
            
            
        if self.bMultiType and self.bTraining:
            lX, lY = self.load_data_Multitype2(self.lTrain)
#             print(lY)
            model, other = self.training_multitype((lX,lY))
            # store
            self.storeModel(model,other)
            del lX, lY
            del self.node_transformer
            del model            
            
        if self.bTraining and not self.bMultiType:
            lX, lY = self.load_data(self.lTrain)
            model, other = self.training((lX,lY))
            # store
            self.storeModel(model,other)
            del lX, lY
            del self.node_transformer
            del model
            
        if self.bTesting:
            self.loadModels()
            if self.bMultiType:
                lX,lY = self.load_data_for_testing_Multitype(self.lTest)
                res = self.testModel_Multitype((lX,lY))
            else:
                lX,lY = self.load_data_for_testing(self.lTest)
                res = self.testModel((lX,lY))

                       
        if self.bPredict:
            # which input  format: [unicode]
            self.loadModels()
            lsent = [self._sent]
            print (lsent)
            if self.bMultiType:
                lres = self.predict_multiptype(lsent)
            else:
                lres = self.predict(lsent)
            for r in lres:
                print (r)

if __name__ == '__main__':
    
    cmp = DeepTagger()
    cmp.parser = OptionParser(usage="", version="1.0")
    cmp.parser.description = "BiLSTM approach for NER"
    cmp.parser.add_option("--name", dest="name",  action="store", type="string", help="model name")
    cmp.parser.add_option("--dir", dest="dirname",  action="store", type="string", help="directory to store model")

    cmp.parser.add_option("--training", dest="training",  action="append", type="string", help="training data")
    cmp.parser.add_option("--ml", dest="multitype",  action="store_true",default=False, help="multi type version")
        
    cmp.parser.add_option("--hidden", dest="hidden",  action="store", type="int", help="hidden layer dimension")    
    cmp.parser.add_option("--batch", dest="batchSize",  action="store", type="int", help="batch size")    

    cmp.parser.add_option("--epochs", dest="nbEpochs",  action="store", type="int", help="nb epochs for training")    
    cmp.parser.add_option("--ngram", dest="ngram",  action="store", type="int", help="ngram size")    
    cmp.parser.add_option("--nbfeatures", dest="nbfeatures",  action="store", type="int", help="nb features")    

    cmp.parser.add_option("--testing", dest="testing",  action="append", type="string", help="test data")    
    cmp.parser.add_option("--run", dest="predict",  action="store", type="string", help="string to be categorized")    

    (options, args) = cmp.parser.parse_args()
    #Now we are back to the normal programmatic mode, we set the component parameters
    cmp.setParams(options)
    #This component is quite special since it does not take one XML as input but rather a series of files.
    #doc = cmp.loadDom()
    doc = cmp.run()    
