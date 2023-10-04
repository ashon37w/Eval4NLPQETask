import guidance, torch
import pandas as pd
import csv
from tqdm import tqdm
from model_dict import load_from_catalogue
import random


def get_rank(lis):
    rank_list=[]
    sorted_values=sorted(lis)
    for i, value in enumerate(lis):
        rank=sorted_values.index(value)+1
        rank_list.append(rank)
    return rank_list

def read_data(dir):
    data=[]
    with open(dir,'r') as f:
        lines=f.readlines()
        for line in lines:
            data.append(float(line))
    return data
    
    
if __name__ == "__main__":
    random.seed(42)
    n=2
    lists=[[] for _ in range(n)]
    lists[0]=get_rank(read_data('/H1/SharedTask2023/translate_prob/results/nh.zhen.tgtonly.logprob'))
    lists[1]=get_rank(read_data('/H1/SharedTask2023/prob_var/results/op.zhen.prompt4.logprobs'))
    #lists[2]=get_rank(read_data('/H1/SharedTask2023/prob_var/results/wz.zhen.prompt3.logprobs'))
    #lists[3]=get_rank(read_data('/H1/SharedTask2023/prob_var/results/wz.zhen.prompt4.logprobs'))
    #lists[4]=get_rank(read_data('/H1/SharedTask2023/prob_var/results/wz.zhen.prompt5.logprobs'))
    quan=[0.25,1]
    le=max(lists[0])
    scores=[]
    for i in range(le):
        scores.append(float(0))
    for i in tqdm(range(n)):
        for j in range(le):
            scores[j]+=(quan[i]*lists[i][j])
            
    with open('/H1/SharedTask2023/baselines/rank_results/nhtgto_opp4_ensemble4_zhen_logprobs.scores', 'w') as f:
        for score in scores:
            f.write(str(score) + "\n")
        
    get_zip('/H1/SharedTask2023/baselines/rank_results/nhtgto_opp4_ensemble4_zhen_logprobs.scores')