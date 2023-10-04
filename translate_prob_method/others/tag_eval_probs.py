import sys
import numpy as np
import scipy.stats
import os


def get_tag_problist(path):
    with open(path,'r') as f:
        rl = f.readlines()
    prob_res = []
    for i in range(len(rl)):
        sc = rl[i].strip().split('\t')
        now_score = []
        for it in sc:
            if len(it)<1:
                continue
            prob,tag = it.split(',')
            now_score.append((float(prob),tag))
        prob_res.append(now_score)
    return prob_res


def get_tag_prob_score(now_score,tags_weight,avg=False):
    if len(now_score)<1:
        return 0
    res = []
    for probs in now_score:
        prob,tag = probs
        weight = tags_weight.get(tag,1)
        res.append(prob*weight)
    if avg:
        return sum(res)/len(res)
    return sum(res)

def get_eval_score(save_path,prob_path,tags_weight):
    prob_res = get_tag_problist(prob_path)
    score_res = [get_tag_prob_score(x,tags_weight) for x in prob_res]
    with open(save_path,'w+') as f:
        f.write('\n'.join([str(x) for x in score_res]))
    os.system('python flcomp.py '+ save_path)


if __name__ == "__main__":
    prob_path = './prob_results/nh.zhen.postag.logprob'
    tags_weights=[{'NOUN':1,'VERB':1,'ADV':1,'ADJ':1},
                  {'NOUN':10,'VERB':10,'ADV':5,'ADJ':5},
                 {'NOUN':10,'VERB':10,'ADV':2,'ADJ':2},
                 {'NOUN':2,'VERB':2,'ADV':5,'ADJ':5},
                 {'NOUN':10,'VERB':10,'ADV':10,'ADJ':10}
                 ]
    for i in range(len(tags_weights)):
        get_eval_score(f'./results/nh.zhen.postagv{i}.logprob',prob_path,tags_weights[i])
        
