# -*- coding: utf-8 -*-

"""
    taggerChrono.py

    task: recognition of a chronological sequence of dates
        
    H. Déjean
    
    copyright Naver labs Europe 2018
    READ project 

    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import sys,os
from io import open
from optparse import OptionParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import  load_model, Model
from keras.layers  import Bidirectional, Input, Add,Masking, Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
# from keras.regularizers import L1L2
import numpy as np

import pickle 
import gzip


from contentProcessing.attentiondecoder import AttentionDecoder

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

        self.lnbClasses = None
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

        self.bAttentionLayer= False
        self.bMultiType = False
        # mapping vector
        self.ltag_vector=[]


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
        
        if dParams.attention:
            self.bAttentionLayer=True
        
        
    def initTransformeur(self):
        # lowercase = False ??   True by default
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
            
            X                   Y1    Y2
            Sa 17 Okt 06 uhr    17    10
        """
        self.lnbClasses = []
        self.lClasses = []
        nbc = 2
        for i in range(0,nbc):
            self.lnbClasses.append(0)  
            self.lClasses.append([])
            self.ltag_vector.append({})

        max_seq = 30
        lTmp=[]               
        for fname in lFName:
            f=open(fname,encoding='utf-8')
            x=[]
            iseq = 0
            for l in f:
                iseq += 1
                l = l.strip()
                if l[:2] == '# ':continue  # comments
                if l =='EOS' or iseq == max_seq:
                    lTmp.append(x)
                    self.max_sentence_len = max(self.max_sentence_len,len(x))
                    x=[]
                    iseq = 0
                else:
                    try:
                        la=l.split('\t')
                        b1=la[-1]
                        b2=la[-2]
                    except  ValueError:
                        #print 'cannot find value and label in: %s'%(l)
                        continue
                    assert len(la) != 0 
                    if b1 not in self.lClasses[0]:
                        self.lClasses[0].append(b1)
                    if b2 not in self.lClasses[1]:
                        self.lClasses[1].append(b2)                        
#                     print (la[0],(b1,b2))
                    x.append((la[0],(b1,b2)))
        
            if x != []:
                lTmp.append(x)
            f.close()
        
        for i in [0,1]: 
            self.lnbClasses[i] = len(self.lClasses[i]) + 1 
            for tag_class_id,b in enumerate(self.lClasses[i]):
                one_hot_vec = np.zeros(self.lnbClasses[i], dtype=np.int32)
                one_hot_vec[tag_class_id] = 1
                self.ltag_vector[i][b] = tuple(one_hot_vec)
                self.ltag_vector[i][tuple(one_hot_vec)] = b
                            
            # Add nil class
            if 'NIL' not in self.ltag_vector[i]:
                self.lClasses[i].append('NIL')
                one_hot_vec = np.zeros(self.lnbClasses[i], dtype=np.int32)
                one_hot_vec[self.lnbClasses[i]-1] = 1
                self.ltag_vector[i]['NIL'] = tuple(one_hot_vec)
                self.ltag_vector[i][tuple(one_hot_vec)] = 'NIL'
        
        
        # more than 1 sequence
        assert len(lTmp) > 1
        
        #     shuffle(lTmp)
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
            f=open(fname,encoding='utf-8')
            x=[]
            for l in f:
                l = l.strip()
                if l[:2] == '# ':continue  # comments
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
        pickle.dump((self.bMultiType,self.maxngram,self.max_features,self.max_sentence_len,self.lnbClasses,self.ltag_vector,self.node_transformer),gzip.open('%s/%s.%s'%(self.dirName,self.sModelName,self.sAux),'wb'))
        print('aux data dumped in %s/%s.%s' % (self.dirName,self.sModelName,self.sAux))        
        
    def loadModels(self):
        """
            load models and aux data
        """
        if self.bAttentionLayer:
            self.model = load_model(os.path.join(self.dirName,self.sModelName+'.hd5'),custom_objects={"AttentionDecoder": AttentionDecoder})
        else:
            self.model = load_model(os.path.join(self.dirName,self.sModelName+'.hd5'))

        print('model loaded: %s/%s.hd5' % (self.dirName,self.sModelName))  
        try:
            self.bMultiType,self.maxngram,self.max_features,self.max_sentence_len, self.lnbClasses,self.ltag_vector , self.node_transformer = pickle.load(gzip.open('%s/%s.%s'%(self.dirName,self.sModelName,self.sAux),'r'))
        except:
            self.maxngram,self.max_features,self.max_sentence_len, self.lnbClasses,self.ltag_vector , self.node_transformer = pickle.load(gzip.open('%s/%s.%s'%(self.dirName,self.sModelName,self.sAux),'r'))
            self.bMultiType = False
        print('aux data loaded: %s/%s.%s' % (self.dirName,self.sModelName,self.sAux))        
        print("ngram: %s\tmaxfea=%s\tpadding=%s\tnbclasses=%s" % (self.maxngram,self.max_features,self.max_sentence_len, self.lnbClasses))
        print("multitype model:%s"%(self.bMultiType))
        print (self.model.summary())
        


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

#         print (lX.shape)
#         print (lY.shape)
#         print (lY2.shape)



        inputs = Input(shape=(self.max_sentence_len, self.max_features))

        x = Masking(mask_value=0)(inputs)
        x = Bidirectional(LSTM(self.hiddenSize,return_sequences = True, dropout=0.5,activation='relu'), merge_mode='concat')(x)
#         x3 = TimeDistributed(Dense(self.hiddenSize, activation='relu'))(x)
        x = TimeDistributed(Dense(self.hiddenSize, activation='relu'))(x)

#         x = TimeDistributed(Dense(64, activation='relu'))(x)
#         x = Bidirectional(LSTM(self.hiddenSize,return_sequences = True,dropout=0.25,activation='relu'), merge_mode='concat')(x)
#         x = TimeDistributed(Dense(64, activation='relu'))(x)
#         x = Bidirectional(LSTM(self.hiddenSize,return_sequences = True, dropout=0.5,activation='relu'), merge_mode='concat')(x)
#         x = TimeDistributed(Dense(64, activation='relu'))(x)        
#         x = Concatenate()([x3,x2])
        out1 = TimeDistributed(Dense(self.lnbClasses[0], activation='softmax'),name='M')(x)
        out2 = TimeDistributed(Dense(self.lnbClasses[1], activation='softmax'),name='D')(x)

        model = Model(input = inputs,output = [out1,out2])
        
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['categorical_accuracy']  )
        print (model.summary())
        _ = model.fit(lX, [lY,lY2], epochs = self.nbEpochs,batch_size = self.batch_size, verbose = 1,validation_split = 0.1, shuffle=True)
        
        del lX,lY,lY2
        
        auxdata = self.max_features,self.max_sentence_len,self.lnbClasses,self.ltag_vector,self.node_transformer
        
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
                elem_tags1.append(list(self.ltag_vector[0][y[ix][0]]))
                elem_tags2.append(list(self.ltag_vector[1][y[ix][1]]))
        
            nil_X = np.zeros(self.max_features)
            nil_Y1 = np.array(self.ltag_vector[0]['NIL'])
            nil_Y2 = np.array(self.ltag_vector[1]['NIL'])
            pad_length = self.max_sentence_len - len(wordsvec)
            lX.append( wordsvec +((pad_length)*[nil_X]) )
            lY1.append( elem_tags1 + ((pad_length)*[nil_Y1]) )        
            lY2.append( elem_tags2 + ((pad_length)*[nil_Y2]) )        

        del lx
        del ly
            
        lX=np.array(lX)
        lY1=np.array(lY1)
        lY2=np.array(lY2)
        
        return lX,(lY1,lY2)
        
    def testModel_Multitype(self,testdata):
        """
            test model
        """
        
        lX,(lY,lY2) = self.prepareTensor_multitype(testdata)

        scores = self.model.evaluate(lX,[lY,lY2],verbose=True)
        #print(list(zip(self.model.metrics_names,scores)))
        
        test_x, _ = testdata
        
        y_pred1,y_pred2 = self.model.predict(lX)
        for i,_ in enumerate(lX):
            for iy,pred_seq in enumerate([y_pred1[i],y_pred2[i]]):
                pred_tags = []
                for class_prs in pred_seq:
                    class_vec = np.zeros(self.lnbClasses[iy], dtype=np.int32)
                    class_vec[ np.argmax(class_prs) ] = 1
#                     print class_prs[class_prs >0.1]
                    if tuple(class_vec.tolist()) in self.ltag_vector[iy]:
#                         print(self.tag_vector[tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)])
                        pred_tags.append((self.tag_vector[tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)]))
                print(test_x[i],pred_tags[:len(test_x[i])])
        
    def prepareOutput_multitype(self,lToken,lLTags):
        """
            format final output with MultiType
        """

        lRes= []        
        for itok,seq in enumerate(lToken):
#             print (seq,lLTags[0][itok],lLTags[1][itok])
            tag1,tag2 = lLTags[0][itok],lLTags[1][itok]
            lRes.append([((itok,itok),seq,tag1,tag2)])
#             lRes.append((toffset,tok,label,list(lScore)))

        return lRes       
 
        
            
    def predict_multiptype(self,lsent):
        """
            predict over a set of sentences (unicode)
        """
    
        lRes= []
        for mysent in [lsent] :
            if len(mysent)> self.max_sentence_len:
                print ('max sent length: %s'%self.max_sentence_len)
                continue
#             allwords= self.node_transformer.transform(mysent.split())
            allwords= self.node_transformer.transform(mysent)

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
#             print (self.model.summary())
            for i,_ in enumerate(lX):
#                 pred_seq = y_pred[i]
                l_multi_type_results = []
                for iy,pred_seq in enumerate([y_pred1[i],y_pred2[i]]):
                    pred_tags = []
                    pad_length = self.max_sentence_len - len(allwords)
                    for class_prs in pred_seq:
                        class_vec = np.zeros(self.lnbClasses[iy], dtype=np.int32)
                        class_vec[ np.argmax(class_prs) ] = 1
                        if tuple(class_vec.tolist()) in self.ltag_vector[iy]:
#                             print (iy,tuple(class_vec.tolist()),self.ltag_vector[iy][tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)])
                            pred_tags.append((self.ltag_vector[iy][tuple(class_vec.tolist())],class_prs[np.argmax(class_prs)]))
                    l_multi_type_results.append(pred_tags[:len(allwords)])
#                     print(mysent,l_multi_type_results) 
                lRes.append(self.prepareOutput_multitype(mysent,l_multi_type_results))

        return lRes
        
    def run(self):
        """
            
        """
        if self.bGridSearch:
            pass
#             self.gridSearch()
            
            
        if self.bMultiType and self.bTraining:
            lX, lY = self.load_data_Multitype(self.lTrain)
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
            lsent= ['Janer','February','Dez ','April','June' ,'July','September' ]
#             lsent = ['30 Jäner ','31 Janiuer ','5 Jager','So 9 Faber 1835', '15 Februar 1835', '25 FAber 138','6 jn ','5 marg','16 decz ']
            #lsent = ['30 Janer ','31 Jäner ','5 Febre', '15 Mär 1835', '25 März 138','6 Marz ','5 Gpmil','16 Apmil ', '5 15. Aprmil','2 Juni','5 Juni']
            #lsent = ['160. Nov. frühe 10 Uhr 187', '6. Dezember Fruth 1 Uhr','ledig. Erich. 5 Uhr', '12. Dez. Hirsch ½ 9 Uhr','15. Jäner Neuötting',' 16. Jan. In. de.']
            #lsent = ['31te̳ Augus','. Okt 1866','e oo', '4 nov']
            #lsent= [ 'Nov.','30Janr 1867','5 Febr. 1867.','dbre','feb','18 Ir 1867'                ]
#             lsent=['jan','feb','mar','apr','jun','sept','okt','dez']
#             lsent= ['1','2','3','4','5','6','7','8','9','10','11','12','13']
#             lsent= [chr(x) for x in range(97,106)] 

            print (lsent)
            if self.bMultiType:
                lres = self.predict_multiptype(lsent)
            else:
                lres = self.predict(lsent)
            for r in lres:
                print (r)

if __name__ == '__main__':
    
    cmp = DeepTagger()
    cmp.parser = OptionParser(usage="", version="0.1")
    cmp.parser.description = "BiLSTM approach for NER"
    cmp.parser.add_option("--name", dest="name",  action="store", type="string", help="model name")
    cmp.parser.add_option("--dir", dest="dirname",  action="store", type="string", help="directory to store model")

    cmp.parser.add_option("--training", dest="training",  action="append", type="string", help="training data")
    cmp.parser.add_option("--ml", dest="multitype",  action="store_true",default=False, help="multi type version")
        
    cmp.parser.add_option("--hidden", dest="hidden",  action="store", type="int", help="hidden layer dimension")    
    cmp.parser.add_option("--batch", dest="batchSize",  action="store", type="int", help="batch size")    

    cmp.parser.add_option("--epochs", dest="nbEpochs",  action="store", type="int", default=2,help="nb epochs for training")    
    cmp.parser.add_option("--ngram", dest="ngram",  action="store", type="int", default=2,help="ngram size")    
    cmp.parser.add_option("--nbfeatures", dest="nbfeatures",  action="store", type="int",default=128, help="nb features")    

    cmp.parser.add_option("--testing", dest="testing",  action="append", type="string", help="test data")    
    cmp.parser.add_option("--run", dest="predict",  action="store", type="string", help="string to be categorized")    
    cmp.parser.add_option("--att", dest="attention",  action="store_true", default=False, help="add attention layer")    

    (options, args) = cmp.parser.parse_args()
    #Now we are back to the normal programmatic mode, we set the component parameters
    cmp.setParams(options)
    #This component is quite special since it does not take one XML as input but rather a series of files.
    #doc = cmp.loadDom()
    doc = cmp.run()    
