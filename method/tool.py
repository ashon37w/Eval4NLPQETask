import sys
import numpy as np
import scipy.stats
import os
import pandas as pd
import csv

def load_scores(input_file):
    with open(input_file) as f:
        return np.array([float(n) for n in f.read().split("\n") if n != ""])


def get_zip(argv):
    input_file = argv
    zip_name = input_file.split('/')[-1]+'.zip'
    f=open('description.txt','w+')
    f.close()
    os.system(f'cp {input_file} seg.scores')
    os.system(f'zip -r {zip_name}  seg.scores description.txt')
    os.system(f'rm seg.scores')
  
if __name__ == "__main__":
    train = []
    df_train = pd.read_csv("/H1/SharedTask2023/data/en_de/train_en_de.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    t = 100 / (max(df_train['mqm']) - min(df_train['mqm']))
    for (src, hyp, mqm) in zip(df_train['SRC'], df_train['HYP'], df_train['mqm']):
        train.append((src, hyp, str(int(100 + mqm * t))))