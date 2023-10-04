import guidance
from model_dict import load_from_catalogue
import re
import pandas as pd
import csv
from tqdm import tqdm
import sys
import numpy as np
import scipy.stats
import argparse

modelname = "../models/Nous-Hermes-13b"
# modelname = "../models/WizardLM-13B-V1.1-GPTQ"

# modelname = "../models/orca_mini_v3_7b"
# modelname = "../models/OpenOrca-Platypus2-13B"



def get_logproblist(gmodel,tokenizer,src, hyp,source_lang,target_lang):
    scores = []
    hyp_re = [tokenizer.decode(x) for x in tokenizer.encode(hyp)]
 
    for i in range(1,len(hyp_re)):
        hyp_st = ' '.join(hyp_re[:i])
        hyp_ed = hyp_re[i]
        s = '\n'.join([
            "### Instruction:",
            f"Translate the following {source_lang} sentence into {target_lang}.",
            f'{source_lang} source: "{src}"',
            "### Response:",
            # f'English translation: {hyp_st} {{{{gen "translation" logprobs=3 pattern="{hyp_ed} .*" max_tokens={i+2} }}}}'
            f'{target_lang} translation: {hyp_st} {{{{gen "translation" logprobs=50000 max_tokens=1 }}}}'
        ])
        res = guidance(s,llm=gmodel)()
        reslogprobs = res['translation_logprobs'][0]
        # print(reslogprobs)
        # keys in reslogprobs require post processing
        lplist = list(reslogprobs)
        lplist_res = [x[1:] for x in lplist]
        # print(lplist)
        now_score = (-1,0)
        if hyp_ed in lplist_res:
            idx = lplist_res.index(hyp_ed)
            now_score = (idx,reslogprobs[lplist[idx]])
        scores.append(now_score)
    return scores


def get_eval_res(score1,score2):
    res_scores = np.array(score1)
    ref_scores = np.array(score2)
    results = scipy.stats.kendalltau(res_scores, ref_scores)[0]
    return results


def get_train_set_probs(path,source_lang,target_lang,save_path):
    inputs = []
    mqm_res = []
    df_source = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp,mqm) in zip(df_source['SRC'], df_source['HYP'], df_source['mqm']):
        inputs.append((src, hyp,mqm))
        mqm_res.append(mqm)
    
    scores_res = []
    cnt = 0
    for src, hyp,mqm in tqdm(inputs):
        scores = get_logproblist(src,
                                 hyp,
                                 source_lang=source_lang,
                                 target_lang=target_lang)
        # print(scores)
        scores_res.append((scores,mqm))
    
    wl = []
    for scores,mqm in scores_res:
        wl.append('\t'.join([f'{x[0]},{x[1]}' for x in scores])+'\n')
        wl.append(str(mqm)+'\n')
    with open(save_path,'w+') as f:
        f.writelines(wl)

def get_dev_set_probs(gmodel,tokenizer,path,source_lang,target_lang,save_path,st=0,ed=10050):
    inputs = []
    df_source = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['HYP']):
        inputs.append((src, hyp))
    
    scores_res = []
    cnt = 0
    for src, hyp in tqdm(inputs[st:ed+1]):
        scores = get_logproblist(gmodel,tokenizer,
                                 src,hyp,
                                 source_lang=source_lang,
                                 target_lang=target_lang)
        if cnt<3:
            print(scores)
        cnt += 1
        scores_res.append(scores)
    
    wl = []
    for scores in scores_res:
        wl.append('\t'.join([f'{x[0]},{x[1]}' for x in scores])+'\n')
        wl.append('0\n')
    with open(save_path,'w+') as f:
        f.writelines(wl)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/H1/SharedTask2023/data/zh_en/dev_zh_en.tsv")
    parser.add_argument("--save_dir", type=str, default="/H1/SharedTask2023/evaluation/results")
    parser.add_argument("--model_name_or_path", type=str, default="../models/Nous-Hermes-13b")
    parser.add_argument("--st", type=int, default=0)
    parser.add_argument("--ed", type=int, default=10050)
    args = parser.parse_args()
    print('data:',args.data_dir)
    print('save:','./prob_results/'+args.save_dir)
    print('model:',args.model_name_or_path)
    print(f'{args.st} -> {args.ed}')
    model, tokenizer, u_prompt, a_prompt = load_from_catalogue(args.model_name_or_path)
    tokenizer.padding_side = 'left'
    gmodel = guidance.llms.Transformers(model, tokenizer=tokenizer, trust_remote_code=True)

    get_dev_set_probs(gmodel,tokenizer,args.data_dir,
                      source_lang = 'Chinese',target_lang='English',
                      save_path='./prob_results/'+args.save_dir,
                      st=args.st,ed=args.ed)
    