#!/usr/bin/env python
# coding: utf-8

# python read_embedding.py
# from read_embedding import glove_embedding, w2v_embedding

import codecs
from collections import defaultdict 
import os
from tqdm import tqdm

class glove_embedding:
    def __init__(self):
        path = "./%s/%s" % ("CKIP_Word_Embedding", "Glove_CNA_ASBC_300d.vec")
        assert os.path.exists(path), "找不到 %s" % path
        self.d = defaultdict(self.false)

        with codecs.open(path, "r", encoding="utf8") as f:
            print("start to read %s in " % path)
            for line in f:
                line = line.strip().split()
                key = line[0]
                value = list(map(float, line[1:]))
                self.d[key] = value        

    def false(self):
        # return False
        return [0]*300

    def __call__(self, str_):
        assert type(str_) == str, "input 要是 string"
        return self.d[str_] 
    
class w2v_embedding:
    def __init__(self):
        path = "./%s/%s" % ("CKIP_Word_Embedding", "w2v_CNA_ASBC_300d.vec")
        assert os.path.exists(path), "找不到 %s" % path
        self.d = defaultdict(self.false)
        
        with open(path, "r", errors='ignore') as f: 
            print("start to read %s in " % path)
            for line in f:
                line = line.strip().split()
                key = line[0]
                value = list(map(float, line[1:]))
                self.d[key] = value    
            
                
    def false(self):
        # return False
        return [0]*300
    
    def __call__(self, str_):
        assert type(str_) == str, "input 要是 string"
        return self.d[str_] 

                  
if __name__ == "__main__" :
    test1 = glove_embedding()
    print("test1(\"韓國\"):", test1("韓國") )

    test1 = w2v_embedding()
    print("test1(\"韓國\"):", test1("韓國") )