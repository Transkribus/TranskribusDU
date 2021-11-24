# -*- coding: utf-8 -*-
"""
    create embeddings from a pagexml graph
    
    Hervé Déjean
    
    python c:\\Users/hdejean/git/DocumentUnderstanding/TranskribusDU/util/embeddings.py --ext pxml --lvl Word --emb bert-base-multilingual-cased 1394\tst\col --out 1394\tst.bert-base-multilingual-cased.pkl

    xlm-mlm-100-1280
"""
import torch
from optparse import OptionParser
import sys,os
from glob import glob
from lxml import etree
import pickle, gzip
import numpy as np
from transformers import pipeline

from transformers import AutoTokenizer
# from transformers import LayoutLMForTokenClassification




    
def tokenizeBERT(lfilename):
    """
    just tokenise using a HF tokenizer
    
    shape: X,NBTOKEN,DIMTOKEN
    """
    assert len(lfilename) > 0
    tok = AutoTokenizer.from_pretrained(options.emb)
    lXemb=[]
    nbfiles = len(lfilename)
    for ifile,filename in enumerate(lfilename):
#         if ifile != 42 : continue
        print (f'[{ifile}/{nbfiles}] processing {filename}')
        lt = extractText(filename,options.taglevel) 
#         lt= ['on peut payer', 'par chèques','ou pas',' ou encore' ,'tata topo']

        max_length=40
        lX=np.zeros((len(lt),max_length))
        for i,word in  enumerate(lt):       
            # are SEP and CLS needed  ????  NO
            # tok.pad_token_id  
            word_tokens = tok.tokenize(word)
            input_ids = tok.convert_tokens_to_ids(word_tokens)
            input_ids.extend([tok.pad_token_id]* (max_length-len(input_ids)))
            ids = torch.tensor(input_ids).unsqueeze(0)
#             print (word,ids)
            lX[i]=ids
            lXemb.append(lX)    
    
        # store embeddings in model ?
    print (f'stored in {options.outfile}')
    # is gzip uefull?
    pickle.dump(lXemb,gzip.open(options.outfile,'wb'))
    

def mapTOKXML(ltext,tok,bLower=False):
    """
        map tokenizer indexes and text words
        
        skip 0,-1 (101  102 ids)
    """
    
    
    dMapping = {}
    # 101 index 
    curLen=1
    ilen=0
    for i,text in enumerate(ltext):
        if bLower:text=text.lower 
        if i ==0:lindexes= tok.encode(text)
        else: lindexes= tok.encode(f" {text}")
#         print(i,ilen,text,lindexes)
        curLen+=len(lindexes[1:-1])
        dMapping[i]=curLen
        ilen+=len(lindexes)-2
    return dMapping,ilen
    
    
def extractText(filename,tagname):
    """
        return the list of text from pagexml
        
        DU toolkit extract Word een with empty text:
    """
    
    xml = etree.parse(filename)
    # sort Y then X ? 
    #ltext = xml.getroot().findall(f".//pc:{tagname}//pc:Unicode", {"pc":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"})
    ltext = xml.getroot().findall(f".//pc:{tagname}//pc:TextEquiv", {"pc":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"})
#     for i,t in enumerate(ltext):
#         print (f'{i}#{t[0].text}#')
    assert len(ltext) == len([ x[0].text  if x[0].text is not None else 'EMPTY' for x in ltext])
    
#     print (len([ x[0].text  if x[0].text is not None else 'EMPTY' for x in ltext]))
#     return [ x.text for x in ltext if x.text is not None ]

    return [ x[0].text.strip()  if x[0].text is not None else 'EMPTY' for x in ltext]

def testload(lfilename): 

    for filename in lfilename:
        emb = pickle.load(gzip.open(filename[:-len(options.extension)]+'.pkl'))
        print (emb.shape)


        
def statTokenizer(lfilename,options):
    """
        get some stats wrt tokinzer
    """    
    lt=[]
    for i,filename in enumerate(lfilename):
        _lt= extractText(filename,options.taglevel)
        lt.extend(_lt)
        d,nbtok = mapTOKXML(extractText(filename,options.taglevel),options.emb)
        print (len(_lt),nbtok)
    _,ilen = mapTOKXML(lt,options.emb)
    print (len(lt),ilen)
    
    
def BPEmb(lTokens,dim):
    """
    torchnlp.word_to_vector.BPEmb(language='en', dim=300, merge_ops=50000, **kwargs)
    from torchnlp.word_to_vector import BPEmb  # doctest: +SKIP
    vectors = BPEmb(dim=25)  # doctest: +SKIP
    subwords = "▁mel ford shire".split()  # doctest: +SKIP
    vectors[subwords]  # doctest: +SKIP    
    """
    from torchnlp.word_to_vector import BPEmb  # doctest: +SKIP
    vectors = BPEmb(dim=dim)
    print (vectors.shape)    
    return vectors[lTokens]


def GloVe(lTokens):
    """
    need  icu-tokenizer
        
    """
    
    from torchnlp.word_to_vector import GloVe 
    _foo = []
    vectors = GloVe() 
    for t in lTokens:print(t,vectors[t].shape, vectors[t][:10])
    return [_foo.append(vectors[t]) for t in lTokens ]

def FastText(lTokens):
    """    
       pip install icu-tokenizer
        
    """
    
    from torchnlp.word_to_vector import FastText 
    _foo = []
    vectors = FastText() 
    for t in lTokens:print(t,vectors[t])
    return [_foo.append(vectors[t]) for t in lTokens ]

def noncontextual(lfilename,options):
    """
    
    
    from torchnlp.word_to_vector import FastText  # doctest: +SKIP
    vectors = FastText()  # doctest: +SKIP
    vectors['hello']  # doctest: +SKIP
        
    """
    
    assert len(lfilename)>0
    nbfiles = len(lfilename)
    lXemb = []
    for ifile,filename in enumerate(lfilename):
        print (f'[{ifile}/{nbfiles}] processing {filename}')
        lt = extractText(filename,options.taglevel) 
        if options.bFastText:
            lXemb=FastText(lt)
            print (lXemb[0].shape)
        elif options.bGloVe > 0:
            lXemb= GloVe(lt)           
        elif options.BPEmb > 0:
            lXemb= BPEmb(lt,options.BPEmb)
    
    # store embeddings in model ?
    print (f'stored in {options.outfile}')
    # is gzip uefull?
    pickle.dump(lXemb,gzip.open(options.outfile,'wb'))
#     testload(lfilename)
        
    return     
    
def main(lfilename,options):
    print (options)
    if options.BBERT:
        tokenizeBERT(lfilename)
        return
    if options.bFastText or  options.bGloVe or options.BPEmb > 0:
        noncontextual(lfilename,options)
        return
        
    assert len(lfilename)>0
    tok = AutoTokenizer.from_pretrained(options.emb)
    fe=pipeline('feature-extraction',model=options.emb,tokenizer=tok)
#     fe.save_pretrained(".")
    lXemb=[]
    nbfiles = len(lfilename)
    for ifile,filename in enumerate(lfilename):
#         if ifile != 42 : continue
        print (f'[{ifile}/{nbfiles}] processing {filename}')
        lt = extractText(filename,options.taglevel) 
#         lt= ['on peut payer', 'par chèques','ou pas',' ou encore' ,'tata topo']
        dMapping,ilen = mapTOKXML(lt,tok)
#         print (lt)
#         print (len(lt))
        print (f'length: {len(lt)} #tokens: {ilen}  max:{fe.model.config.max_position_embeddings} ')
        ldMapping=[]
                    # camembert! 514!!
        if ilen >= fe.model.config.max_position_embeddings - (1+2) or options.singleton:
            if options.singleton:n=1 #2?
            else: n = 100
            lfinal = [lt[i:i + n] for i in range(0, len(lt), n)] 
#             print (lfinal, [len(s) for s in lfinal])
            [ldMapping.append(mapTOKXML(sub,tok)[0]) for sub in lfinal]
            lx = [np.array(fe(" ".join(sub))[0]) for sub in lfinal ] 
#             print ("shape subtaokens:",[x.shape for x in lx],ldMapping) 
        else:
            ldMapping.append(dMapping)
            sfullsequence = " ".join(lt)
            lx = [ np.array(fe(sfullsequence)[0]) ]
            print (lx[0].shape)
            lfinal = [lt]
#         print ( len(dMapping) , [x.shape for x in lx] )
#         assert len(dMapping) == sum( [(x.shape[0]-2)  for x in lx])

        # empty graphs are skipped in the DU toolkit
        if len(lt)>0:
            lX=np.zeros((len(lt),fe.model.config.hidden_size))
    #         for multitoken nodes: sum? avg, last token 
            for i,l in enumerate(lfinal):
                # skip 101
                curpos=1 
                for j,x in enumerate(l):
#                     print (i,j,lfinal[i][j],
#                            curpos,
#                            ldMapping[i][j],
# #                            lx[i].shape,
#                            lx[i][curpos:ldMapping[i][j] ].shape)
                    #sum
                    lX[i+j] = sum(lx[i][curpos:ldMapping[i][j] ])
                    #last 
                    #lX[i+j] = lx[i][curpos:ldMapping[i][j] ][-1]
                    curpos = ldMapping[i][j]
                print (curpos,lx[i].shape[0]-1)
                assert curpos == lx[i].shape[0]-1
            
            print (lX.shape)
            lXemb.append(lX)
            
    # store embeddings in model ?
    print (f'stored in {options.outfile}')
    # is gzip uefull?
    pickle.dump(lXemb,gzip.open(options.outfile,'wb'))
#     testload(lfilename)
        
    return 

if __name__ == "__main__":
    
    version = "v.01"
    sUsage="""
Usage: %s <INPUTFOLDER>   
    
""" % (sys.argv[0])

    parser = OptionParser(usage=sUsage)
    parser.add_option("--lower", dest='lower',  action="store_true"
                        , help="lowercase string")       
    parser.add_option("--singleton", dest='singleton',  action="store_true"
                        , help="embedding from node only (no context)")      
    parser.add_option("--outfile", dest='outfile',  type='string',action="store"
                        , help="file name where embeddings are stored")   
    parser.add_option("--ext", dest='extension',  type='string',action="store"
                        , help="file extension")        
    parser.add_option("--lvl", dest='taglevel',  type='string',action="store"
                        , help="tag used for text extraction")     
    parser.add_option("--emb", dest='emb',  type='string',action="store"
                        , default='roberta-large',help="embeddings:bert, RoBERTa")   
    parser.add_option("--FastText", dest='bFastText',  action="store_true"
                      ,default=False,help="FastText embeddings")
    parser.add_option("--bert", dest='BBERT',  action="store_true"
                      ,default=False,help="BERT toknization")    
    parser.add_option("--glove", dest='bGloVe',  action="store_true"
                      ,default=False,help="GloVe embeddings")            
    parser.add_option("--BPEmb", dest='BPEmb',  action="store", type='int'
                      ,default=0,help="FastText embeddings")   
    (options, args) = parser.parse_args()
    
    try:
        folder = args[0]
    except ValueError:
        sys.stderr.write(sUsage)
        sys.exit(1)
    
    # in DU_task:            lsFilename = sorted(glob.iglob(os.path.join(sDir, sPattern)))  

    lsFile = sorted([s for s in glob(os.path.join(folder, '*')) if s.endswith(options.extension)])
#     lsFile = ['MAY\MENU_ANNOTATION_BATCHMAY2020\Part_1_ner\michelin-grenoble-scan0434_rotated.pdf.ner.xml']
#     lsFile=['MAY\MENU_ANNOTATION_BATCHMAY2020\Part_1_ner\michelin-paris-c7RKOc6tsdDG9Q8w.pdf-0.ner.xml']
#     lsFile = ['1394/trn/col/Maylis-images-ByAlphabet-French-1.gif.ner.pxml']
    #lsFile = ['1394/tst/col/Michelin-crawl-strasbourg-VuSq21UhGckfKuuw.pdf.ner.pxml']
    
    main(lsFile,options)
    
    print("Done.")    
    