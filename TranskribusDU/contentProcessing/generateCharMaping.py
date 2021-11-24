# make an empty character dictionary
import sys,os
from optparse import OptionParser
from flair.data import Dictionary
import pickle
import glob
import collections

class charMapper():
    
    def __init__(self):
        self.sDocPattern = "*.txt"
        self.sDocIgnorePattern = "*.mpxml"
    
    def loadFiles(self,lsDir):
        counter = collections.Counter()
        processed = 0
        
        print("%d folders: %s" % (len(lsDir), lsDir))
      
        for sDir in lsDir:
            print("--- FOLDER ", sDir)
            files = [filename for filename in glob.iglob(os.path.join(sDir, "**", self.sDocPattern), recursive=True) 
                                        if os.path.isfile(filename) and not(filename.endswith(self.sDocIgnorePattern))]        
            print(files)
            for file in files:
                print(file)
            
                with open(file, 'r', encoding='utf-8') as f:
                    tokens = 0
                    for line in f:
            
                        processed += 1            
                        chars = list(line)
                        tokens += len(chars)
            
                        # Add chars to the dictionary
                        counter.update(chars)
        
                    # comment this line in to speed things up (if the corpus is too large)
                    # if tokens > 50000000: break
        print(processed)
        return counter
    
    def run(self,files, output):

        char_dictionary: Dictionary = Dictionary()
                
        counter = self.loadFiles(files)
        
        total_count = 0
        for letter, count in counter.most_common():
            total_count += count
        print(total_count)

        
        sum = 0
        idx = 0
        for letter, count in counter.most_common():
            sum += count
            percentile = (sum / total_count)
        
            # comment this line in to use only top X percentile of chars, otherwise filter later
            # if percentile < 0.00001: break
        
            char_dictionary.add_item(letter)
            idx += 1
            print('%d\t%s\t%7d\t%7d\t%f' % (idx, letter, count, sum, percentile))
        
        print(char_dictionary.item2idx)
    
        self.storeMap(char_dictionary,output)
    
    def storeMap(self,char_dictionary,output):
        
        with open(output, 'wb') as f:
            mappings = {
                'idx2item': char_dictionary.idx2item,
                'item2idx': char_dictionary.item2idx
            }
            pickle.dump(mappings, f)
    
    
    
if __name__ == "__main__":
    sUsage="usage: %s <output> <col-dir>" % sys.argv[0]

    parser = OptionParser(usage=sUsage)
    
#     parser.add_option("--layout", dest='sLayout',  action="store", type="string"
#                       , default="cell"
#                       , help="Layout Structured used :cell, row,column, textline,...")   
    (options, args) = parser.parse_args()

    try:
        sOutput     = args[0]
        lsDir       = args[1:]
        assert len(lsDir)
    except:
        print(sUsage % sys.argv[0])
        exit(1)
        
    
#     doer = XML_to_Text(XML_to_Text.iColumn)
    doer = charMapper()

    doer.run(lsDir, sOutput)
    print("Done")
        