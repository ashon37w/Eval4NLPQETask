'''
This baseline implements the direct assessment prompt by
Kocmi and Federmann, Large Language Models Are State-of-the-Art Evaluators of Translation Quality. ArXiv: 2302.14520
for open source LLMs with MT and summarization
'''
import shutil
import os
import guidance, torch
import pandas as pd
import csv
from tqdm import tqdm
from model_dict import load_from_catalogue
import random
from tool import get_zip
import numpy as np
import zipfile
def get_zip(zip_file_name,file_list):
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_name in file_list:
            arcname=os.path.basename(file_name)
            zipf.write(file_name,arcname=arcname)
def linear_transform(score,mi,mx):
    return (score-mi)/(mx-mi)*100

def linear_transform10000(score,mi,mx):
    return (score-mi)/(mx-mi)*10000
def Zscore_transform(lis):
    sum1=sum(lis)
    le=len(lis)
    mean=sum1/le
def read_data(path1):
    ans1=[]
    normalized_scores=[]
    with open(path1,'r') as f:
        lines=f.readlines()
        for line in lines:
            stp=float(line)
            ans1.append(stp)
        mx1=max(ans1)
        mi1=min(ans1)
        normalized_scores=[linear_transform(score,mi1,mx1) for score in ans1]
    return ans1

def read_data10000(path1):
    ans1=[]
    normalized_scores=[]
    with open(path1,'r') as f:
        lines=f.readlines()
        for line in lines:
            stp=float(line)
            ans1.append(stp)
        mx1=max(ans1)
        mi1=min(ans1)
        normalized_scores=[linear_transform10000(score,mi1,mx1) for score in ans1]
    return normalized_scores

if __name__ == "__main__":

    folder_path='/H1/SharedTask2023/baselines/maxsbert/oop_p9_result'
    if os.path.exists(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    items=os.listdir(folder_path)
    lis_nam=[]
    for item in items:
        lis_nam.append(folder_path+'/'+item)
    get_zip('/H1/SharedTask2023/baselines/test_zips/oop_p9_maxbert_result.zip',lis_nam)


    folder_path='/H1/SharedTask2023/baselines/maxsbert/oop_p4_result'
    if os.path.exists(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    items=os.listdir(folder_path)
    lis_nam=[]
    for item in items:
        lis_nam.append(folder_path+'/'+item)
    get_zip('/H1/SharedTask2023/baselines/test_zips/oop_p4_maxbert_result.zip',lis_nam)



    folder_path='/H1/SharedTask2023/baselines/maxsbert/oop_p10_result'
    if os.path.exists(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    items=os.listdir(folder_path)
    lis_nam=[]
    for item in items:
        lis_nam.append(folder_path+'/'+item)
    get_zip('/H1/SharedTask2023/baselines/test_zips/oop_p10_maxbert_result.zip',lis_nam)