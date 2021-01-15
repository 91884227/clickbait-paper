# python ckip.py
# from ckip import ckip

# import tool
import tensorflow as tf
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from IPython.display import clear_output
clear_output()
import os

class ckip:
    def __init__(self):
        print("prepare ws pos ner")
        assert os.path.exists("./ckiptagger_data"), "ckiptagger_data 不在同層目錄"
        self.ws = WS("./ckiptagger_data")
        self.pos = POS("./ckiptagger_data")
        self.ner = NER("./ckiptagger_data")
        clear_output() 
        
    def __call__(self, str_):
        assert type(str_) == str, "input 要是 string"
        return( self.ws([str_])[0] )


if __name__ == "__main__":
    test = ckip()
    print("去吃大便斷句結果", test("去吃大便"))