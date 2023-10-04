import sys
import numpy as np
import scipy.stats
import os




def get_eval_res(score1,score2):
    res_scores = np.array(score1)
    ref_scores = np.array(score2)
    results = scipy.stats.kendalltau(res_scores, ref_scores)[0]
    return results


def get_problist(path):
    with open(path,'r') as f:
        rl = f.readlines()
    prob_res = []
    for i in range(0,len(rl)-1,2):
        sc = rl[i].strip().split('\t')
        now_score = []
        for it in sc:
            if len(it)<1:
                continue
            x = it.split(',')
            now_score.append((int(x[0]),float(x[1])))
        prob_res.append({'score':now_score,'mqm':float(rl[i+1].strip())})
    return prob_res



def get_dev_problist(path):
    with open(path,'r') as f:
        rl = f.readlines()
    prob_res = []
    for i in range(len(rl)):
        sc = rl[i].strip().split('\t')
        now_score = []
        for it in sc:
            if len(it)<1:
                continue
            now_score.append(float(it))
        prob_res.append(now_score)
    return prob_res



def get_score(now_score,sc_w,avg=False,use_prob=False):
    if len(now_score)<1:
        return 0
    res = []
    for idx,probs in now_score:
        if idx == -1:
            sc = sc_w[2]
        elif idx<3:
            sc = sc_w[0]
        else:
            sc = sc_w[1]
        if use_prob and idx>-1:
            sc = probs
        res.append(sc)
    if avg:
        return sum(res)/len(res)
    return sum(res)


def get_prob_score(now_score,avg=False):
    if len(now_score)<1:
        return 0
    res = []
    for probs in now_score:
        sc = probs
        res.append(sc)
    if avg:
        return sum(res)/len(res)
    return sum(res)


def get_score(pth):
    save_path = pth.split('/')[-1]
    if os.path.exists('test_results/'+save_path):
        return
    print(save_path)
    
    prob_res = get_dev_problist(pth)
    score_res = [get_prob_score(x,avg=False) for x in prob_res]
    with open(save_path,'w+') as f:
        f.write('\n'.join([str(x) for x in score_res]))
    os.system('python test_flcomp.py '+ save_path)


if __name__ == "__main__":
    for it in os.listdir('./test_prob'):
        get_score('test_prob/'+it)
    
