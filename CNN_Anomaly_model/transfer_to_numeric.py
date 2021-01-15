# cd ./Clickbait_project/CNN_Anomaly_model/
# python transfer_to_numeric.py --w 1 --t True

# import tool
import json
import os
from tqdm import tqdm

from ckip import ckip
from read_embedding import glove_embedding, w2v_embedding

# import argparse

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--w", default = 1, type=int)
parser.add_argument("--t", default = False, type=bool)
args = parser.parse_args()

def data_preprocessing(data_name):
    path_ = "./資料集/%s" % data_name
    with open(path_) as json_file:
        data = json.load(json_file)

    if( args.t == True):
        data = data[:3]    
    
    buf = [ckip_class(i) for i in tqdm(data)]
    
    buf = [ [word_f(j) for j in i] for i in buf]
    
    if( args.w == 1):
        save_path = "./數值資料/Glove/%s" % data_name
    elif( args.w == 2):
        save_path = "./數值資料/w2v/%s" % data_name
    
    with open(save_path , 'w') as outfile:
        json.dump(buf, outfile)

if __name__ == "__main__":
    print("read embedding in ....\n")
    if( args.w == 1):
        word_f = glove_embedding()
    elif( args.w == 2):
        word_f = w2v_embedding()
    else:
        assert False, "--w 必須是1或2"  
        
    ckip_class = ckip()
    
    # mkdir
    if( args.w == 1):
        os.mkdir("./數值資料/Glove")
    elif( args.w == 2):
        os.mkdir("./數值資料/w2v")
        
    # find all data
    data_path = [i for i in os.listdir("./資料集") if i.endswith(".json")]

    for i in data_path:
        print("start to deal with %s " % i)
        data_preprocessing(i )