'''
This baseline implements the direct assessment prompt by
Kocmi and Federmann, Large Language Models Are State-of-the-Art Evaluators of Translation Quality. ArXiv: 2302.14520
for open source LLMs with MT and summarization
'''

import guidance, torch
import pandas as pd
import csv
from tqdm import tqdm
from model_dict import load_from_catalogue
import random
from tool import get_zip
import numpy as np
class DirectAssessment:
    def __init__(self, model, tokenizer, **kwargs):
        self.model = guidance.llms.Transformers(
            model, tokenizer=tokenizer, trust_remote_code=True, **kwargs
        )
        self.set_model(self.model)

    def set_model(self, model):
        self.model = model
        guidance.llms.Transformers.cache.clear()


    def direct_assessment_mt_block(
        self,
        hyp,
        gt,
        prompt_placeholder="",
        response_placeholder="",
        source_lang="en",
        target_lang="de",
        demo_src="",
        demo_hyp="",
        demo_score="",
        as1=0,
        as2=0
    ):
        return "\n".join(
            [
                prompt_placeholder,
                f"Please score the following translation from {source_lang} to {target_lang}",
                "with respect to the source sentence on a continuous scale from 0 to 100, ",
                'where a score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".',
                "As reference, there are two other models' scores provided.",
                #f'{source_lang} source: "{demo_src[0]}"',
                #f'{target_lang} translation: "{demo_hyp[0]}"',
                #f"Score: {demo_score[0]}",
                #f"Score the following translation from {source_lang} to {target_lang} with respect to",
                # "the source sentence on a continuous scale from -100 to 0, where a score of negative one hundred means",
                # '"no meaning preserved" and score of zero means "perfect meaning and grammar".',
                # f"Based on the given source, identify the major and minor errors in this translation. ",
                # "Note that Major errors refer to actual translation or grammatical errors, ",
                # "and Minor errors refer to smaller imperfections, and purely subjective opinions about the translation. ",
                # "Then count the number of major and minor errors and compute the final score for this translation. ",
                # "Deduct 5 points for each major error. Deduct 1 point for each minor error. If the translation has no errors, its score will be 0.",
                #f"Score the following {target_lang} sentence on a continuous scale from 0 to 100, where a score of zero means",
                #'"grammatically incorrect and bad-written" and score of one hundred means "grammatically correct and well-written".',
                #f'{source_lang} source: "{demo_src[0]}"',
                #f'{target_lang} translation: "{demo_hyp[0]}"',
                
                #f'{target_lang} sentence: "{demo_hyp[0]}"',
                # response_placeholder,
                #f"Score: {demo_score[0]}",
                # prompt_placeholder,
                # f'{source_lang} source: "{demo_src[1]}"',
                # f'{target_lang} translation: "{demo_hyp[1]}"',
                # f'{target_lang} sentence: "{demo_hyp[1]}"',
                # # response_placeholder,
                # f"Score: {demo_score[1]}",
                # prompt_placeholder,
                # f'{source_lang} source: "{demo_src[2]}"',
                # f'{target_lang} translation: "{demo_hyp[2]}"',
                # f'{target_lang} sentence: "{demo_hyp[2]}"',
                # # response_placeholder,
                # f"Score: {demo_score[2]}",
                # prompt_placeholder,
                # f'{source_lang} source: "{demo_src[3]}"',
                # f'{target_lang} translation: "{demo_hyp[3]}"',
                # f'{target_lang} sentence: "{demo_hyp[3]}"',
                # # response_placeholder,
                # f"Score: {demo_score[3]}",
                # prompt_placeholder,
                # f'{source_lang} source: "{demo_src[4]}"',
                # f'{target_lang} translation: "{demo_hyp[4]}"',
                # f'{target_lang} sentence: "{demo_hyp[4]}"',
                # # response_placeholder,
                # f"Score: {demo_score[4]}",
                # prompt_placeholder,
                f'{source_lang} source: "{gt}"',
                f'{target_lang} translation: "{hyp}"',
                #f'{target_lang} sentence: "{hyp}"',
                #f'[Score1]: The first model give the translation a score of {as1}',
                #f'[Score2]: The second model give the translation a score of {as2}',
                f'[Score1]: {as1}',
                f'[Score2]: {as2}',
                response_placeholder,
                "Score: {{gen 'score' pattern='(0|100|[1-9]?[0-9])'}}",
                #"Score: {{gen 'score' pattern='^(100(\.0{1,2})?|\d{1,2}(\.\d{1,2})?)$'}}"
                #"Score: {{gen 'score' pattern='(-\\d+(\\.\\d+)?)'}}",
                #"Score: {{gen 'score' pattern='(-100|-(?:[1-9]|[1-9][0-9])|0)'}}"
                # prompt_placeholder,
                # f"Rewrite the following text into high-quality text with its core information:{demo_src} In other words, {demo_hyp}",
                # f"Score: {demo_score}",
                # f"Rewrite the following text into high-quality text with its core information:{gt} In other words, {hyp}",
                # response_placeholder,
                # "Score: {{gen 'score' pattern='(100|[1-9]?[0-9])'}}",

            ]
        )

    def direct_assessment_summ_block(
        self,
        hyp,
        gt,
        prompt_placeholder="",
        response_placeholder="",
    ):
        return "\n".join(
            [
                prompt_placeholder,
                f"Score the summarization with respect to the summarized document",
                "on a continuous scale from 0 to 100, where a score of zero means",
                '"irrelevant, factually incorrect and not readable" and score of one hundred means',
                '"relevant, factually correct, good readability".',
                f'Source text: "{gt}"',
                f'Summary: "{hyp}"',
                response_placeholder,
                "Score: {{gen 'score' pattern='(100|[1-9]?[0-9])'}}",
            ]
        )

    def prompt_model(
        self,
        gt,
        hyp,
        mt = True,
        prompt_placeholder=None,
        response_placeholder=None,
        target_lang="German",
        source_lang="English",
        demo_src = None,
        demo_hyp = None,
        demo_score = None,
        verbose=False,
        as1=None,
        as2=None
    ):
        if mt:
            prompt = self.direct_assessment_mt_block(
            gt=gt,
            hyp=hyp,
            response_placeholder=response_placeholder,
            prompt_placeholder=prompt_placeholder,
            target_lang=target_lang,
            source_lang=source_lang,
            demo_src=demo_src,
            demo_hyp=demo_hyp,
            demo_score=demo_score,
            as1=as1,
            as2=as2
        )
        else:
            prompt = self.direct_assessment_summ_block(
            gt=gt,
            hyp=hyp,
            response_placeholder=response_placeholder,
            prompt_placeholder=prompt_placeholder
        )

        if verbose:
            print(prompt)

        guidance_prompt = guidance(prompt, llm=self.model)
        res = guidance_prompt()

        torch.cuda.empty_cache()
        return res.text, res["score"]

def get_mqm(tuple_item):
    return int(tuple_item[2])

def split_list(lst,k):
    avg=len(lst)//k
    remainder=len(lst)%k
    result=[]
    
    for i in range(k):
        start=i*avg+min(i,remainder)
        end=start+avg+(1 if i<remainder else 0)
        result.append(lst[start:end])
    return result   

def split_list_by_mqm_range(lst, k):
    sorted_lst=sorted(lst, key=get_mqm)
    min_mqm=get_mqm(sorted_lst[0])
    max_mqm=get_mqm(sorted_lst[-1])
    mqm_range=(max_mqm-min_mqm)/k
    result=[[] for _ in range(k)]

    for item in sorted_lst:
        mqm=get_mqm(item)
        sublist_index=int((mqm-min_mqm) // mqm_range)
        sublist_index=min(sublist_index,k-1)
        result[sublist_index].append(item)

    return result

def linear_transform(score,mi,mx):
    return (score-mi)/(mx-mi)*100


def Zscore_transform(lis):
    sum1=sum(lis)
    le=len(lis)
    mean=sum1/le
    

if __name__ == "__main__":
    #modelname = "../models/Nous-Hermes-13b"
    #modelname = "../models/WizardLM-13B-V1.1-GPTQ"
    modelname = "../models/OpenOrca-Platypus2-13B"
    #modelname = "../models/orca_mini_v3_7b"
    
    #modelname = "NousResearch/Nous-Hermes-13b"
    #modelname = "TheBloke/guanaco-65B-GPTQ"
    #modelname = "TheBloke/WizardLM-13B-V1.1-GPTQ"
    #modelname = "Open-Orca/OpenOrca-Platypus2-13B"
    #modelname = "psmathur/orca_mini_v3_7b"
    model, tokenizer, u_prompt, a_prompt = load_from_catalogue(modelname)
    BPG = DirectAssessment(model=model, tokenizer=tokenizer)
    
    random.seed(42)
    
    train = []
    df_train = pd.read_csv("/H1/SharedTask2023/data/en_de/train_en_de.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    # for (src, hyp, mqm) in zip(df_train['SRC'], df_train['HYP'], df_train['mqm']):
    #     train.append((src, hyp, int(100 + mqm)))
    t = 100 / (max(df_train['mqm']) - min(df_train['mqm']))
    for (src, hyp, mqm) in zip(df_train['SRC'], df_train['HYP'], df_train['mqm']):
        train.append((src, hyp, int(100 + mqm * t)))
    
    sorted_train = sorted(train,key=get_mqm)
    normalized_scores1=[]
    normalized_scores2=[]
    subtrains = split_list(sorted_train,5)
    ans1=[]
    ans2=[]
    inputs = []
    df_source = pd.read_csv("/H1/SharedTask2023/data/en_de/dev_en_de.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['HYP']):
        inputs.append((src, hyp))
    with open('/H1/SharedTask2023/prob_var/results/nh.ende.prompt9.logprobs','r') as f:
        lines=f.readlines()
        for line in lines:
            stp=float(line)
            a=int(stp)
            ans1.append(stp)
        mx1=max(ans1)
        mi1=min(ans1)
        normalized_scores1=[linear_transform(score,mi1,mx1) for score in ans1]
    with open('/H1/SharedTask2023/translate_prob/results/wz.ende.tgtonly.logprob','r') as f:
        lines=f.readlines()
        for line in lines:
            stp=float(line)
            a=int(stp)
            ans2.append(stp)
        mx2=max(ans2)
        mi2=min(ans2)
        normalized_scores2=[linear_transform(score,mi2,mx2) for score in ans2]
    scores = []
    cnt=-1
    for (src, hyp) in tqdm(inputs):
            demo_src, demo_hyp, demo_score = [], [], []
            for _, subtrain in enumerate(subtrains):
                j = random.randint(0, len(subtrain) - 1)
                d_src, d_hyp, d_score = subtrain[j]
                demo_src.append(d_src)
                demo_hyp.append(d_hyp)
                demo_score.append(d_score)
            cnt+=1
            _, score = BPG.prompt_model(
            gt=src,
            hyp=hyp,
            prompt_placeholder=u_prompt,
            response_placeholder=a_prompt,
            demo_src=demo_src,
            demo_hyp=demo_hyp,
            demo_score=demo_score,
            verbose=True,
            as1=normalized_scores1[cnt],
            as2=normalized_scores2[cnt]
        )
            scores.append(score)
            print(score)
    
    with open('/H1/SharedTask2023/baselines/mulit_agent_results/op_prompt3_wztgt_nh9_ende_ensemble_logprobs.scores', 'w') as f:
        for score in scores:
            f.write(score + "\n")
        
    get_zip('/H1/SharedTask2023/baselines/mulit_agent_results/op_prompt3_wztgt_nh9_ende_ensemble_logprobs.scores')
    
    random.seed(42)
    train = []
    df_train = pd.read_csv("/H1/SharedTask2023/data/zh_en/train_zh_en.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    # for (src, hyp, mqm) in zip(df_train['SRC'], df_train['HYP'], df_train['mqm']):
    #     train.append((src, hyp, int(100 + mqm)))
    t = 100 / (max(df_train['mqm']) - min(df_train['mqm']))
    for (src, hyp, mqm) in zip(df_train['SRC'], df_train['HYP'], df_train['mqm']):
        train.append((src, hyp, int(100 + mqm * t)))
    
    sorted_train = sorted(train,key=get_mqm)
    
    subtrains = split_list(sorted_train,5)
    ans1=[]
    ans2=[]
    normalized_scores1=[]
    normalized_scores2=[]
    inputs = []
    df_source = pd.read_csv("/H1/SharedTask2023/data/zh_en/dev_zh_en.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['HYP']):
        inputs.append((src, hyp))
    with open('/H1/SharedTask2023/prob_var/results/nh.zhen.prompt9.logprobs','r') as f:
        lines=f.readlines()
        for line in lines:
            stp=float(line)
            a=int(stp)
            ans1.append(stp)
        mx1=max(ans1)
        mi1=min(ans1)
        normalized_scores1=[linear_transform(score,mi1,mx1) for score in ans1]
    with open('/H1/SharedTask2023/translate_prob/results/wz.zhen.tgtonly.logprob','r') as f:
        lines=f.readlines()
        for line in lines:
            stp=float(line)
            a=int(stp)
            ans2.append(stp)
        mx2=max(ans2)
        mi2=min(ans2)
        normalized_scores2=[linear_transform(score,mi2,mx2) for score in ans2]
    scores = []
    cnt=-1
    for (src, hyp) in tqdm(inputs):
            demo_src, demo_hyp, demo_score = [], [], []
            for _, subtrain in enumerate(subtrains):
                j = random.randint(0, len(subtrain) - 1)
                d_src, d_hyp, d_score = subtrain[j]
                demo_src.append(d_src)
                demo_hyp.append(d_hyp)
                demo_score.append(d_score)
            cnt+=1
            _, score = BPG.prompt_model(
            gt=src,
            hyp=hyp,
            prompt_placeholder=u_prompt,
            response_placeholder=a_prompt,
            target_lang="English",
            source_lang="Chinese",
            demo_src=demo_src,
            demo_hyp=demo_hyp,
            demo_score=demo_score,
            verbose=True,
            as1=normalized_scores1[cnt],
            as2=normalized_scores2[cnt]
        )
            scores.append(score)
    
    with open('/H1/SharedTask2023/baselines/mulit_agent_results/op_prompt3_wztgt_nh9_zhen_ensemble_logprobs.scores', 'w') as f:
        for score in scores:
            f.write(score + "\n")
        
    get_zip('/H1/SharedTask2023/baselines/mulit_agent_results/op_prompt3_wztgt_nh9_zhen_ensemble_logprobs.scores')