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
        as2=0,
        as3=0,
        as4=0,
        as5=0,
        as6=0,
        as7=0,
        as8=0,
        as9=0,
        as10=0
    ):
        return "\n".join(
            [
                prompt_placeholder,
                f"Please score the following translation from {source_lang} to {target_lang}",
                "with respect to the source sentence on a continuous scale from 0 to 100, ",
                'where a score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".',
                "As reference, there are five other models' scores provided.",
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
                f'[Score3]: {as3}',
                f'[Score4]: {as4}',
                f'[Score5]: {as5}',
                response_placeholder,
                "Score: {{gen 'score' pattern='(0|100|[1-9]?[0-9])'}}"
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
        as2=None,
        as3=None,
        as4=None,
        as5=None,
        as6=None,
        as7=None,
        as8=None,
        as9=None,
        as10=None
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
            as2=as2,
            as3=as3,
            as4=as4,
            as5=as5,
            as6=as6,
            as7=as7,
            as8=as8,
            as9=as9,
            as10=as10
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
    folder_path='/H1/SharedTask2023/baselines/test_results/ensemble_oop1tooop10_cha'
    if os.path.exists(folder_path):
        pass
    else:
        os.mkdir(folder_path)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
    random.seed(42)
    normalized_scores=[[] for _ in range(10)]
    inputs = []
    df_source = pd.read_csv("/H1/SharedTask2023/data/test/mt_en_de_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs.append((src, hyp))
    
    normalized_scores[0]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p1.3sbert.logprobs/en_de.scores')
    normalized_scores[1]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p2.3sbert.logprobs/en_de.scores')
    normalized_scores[2]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p3.3sbert.logprobs/en_de.scores')
    normalized_scores[3]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p4.3sbert.logprobs/en_de.scores')
    normalized_scores[4]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p5.3sbert.logprobs/en_de.scores')
    normalized_scores[5]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p6.3sbert.logprobs/en_de.scores')
    normalized_scores[6]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p7.3sbert.logprobs/en_de.scores')
    normalized_scores[7]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p8.3sbert.logprobs/en_de.scores')
    normalized_scores[8]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p9.3sbert.logprobs/en_de.scores')
    normalized_scores[9]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p10.3sbert.logprobs/en_de.scores')
    scores=[]
    for i in range(len(inputs)):
        sum=0.0
        for j in range(10):
            sum+=normalized_scores[j][i]
        mean=sum/10.0
        differences=[]
        for j in range(10):
            differences.append( (abs(normalized_scores[j][i]-mean),j) )
        differences.sort(key=lambda differences: differences[0],reverse=True)
        ans=0.0
        for j in range(5):
            value,index=differences[j]
            ans+=normalized_scores[index][i]
        scores.append(str(ans))
    prompts=[]
    cnt=-1
    with open(folder_path+'/en_de.scores', 'w') as f:
        for score in scores:
            f.write(score + "\n")
    with open(folder_path+'/en_de.description', 'w') as f:
        sentence='This method uses mathematical method to fuse the previous good results'
        f.write(sentence+"\n")
    with open(folder_path+'/en_de.prompts', 'w') as f:
        for i in range(len(scores)):
            f.write("mathematical method"+"\n")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
    random.seed(42)
    normalized_scores=[[] for _ in range(10)]
    inputs = []
    df_source = pd.read_csv("/H1/SharedTask2023/data/test/mt_en_zh_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs.append((src, hyp))
    
    normalized_scores[0]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p1.3sbert.logprobs/en_zh.scores')
    normalized_scores[1]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p2.3sbert.logprobs/en_zh.scores')
    normalized_scores[2]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p3.3sbert.logprobs/en_zh.scores')
    normalized_scores[3]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p4.3sbert.logprobs/en_zh.scores')
    normalized_scores[4]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p5.3sbert.logprobs/en_zh.scores')
    normalized_scores[5]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p6.3sbert.logprobs/en_zh.scores')
    normalized_scores[6]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p7.3sbert.logprobs/en_zh.scores')
    normalized_scores[7]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p8.3sbert.logprobs/en_zh.scores')
    normalized_scores[8]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p9.3sbert.logprobs/en_zh.scores')
    normalized_scores[9]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p10.3sbert.logprobs/en_zh.scores')
    scores=[]
    for i in range(len(inputs)):
        sum=0.0
        for j in range(10):
            sum+=normalized_scores[j][i]
        mean=sum/10.0
        differences=[]
        for j in range(10):
            differences.append( (abs(normalized_scores[j][i]-mean),j) )
        differences.sort(key=lambda differences: differences[0],reverse=True)
        ans=0.0
        for j in range(5):
            value,index=differences[j]
            ans+=normalized_scores[index][i]
        scores.append(str(ans))
    prompts=[]
    cnt=-1
    with open(folder_path+'/en_zh.scores', 'w') as f:
        for score in scores:
            f.write(score + "\n")
    with open(folder_path+'/en_zh.description', 'w') as f:
        sentence='This method uses mathematical method to fuse the previous good results'
        f.write(sentence+"\n")
    with open(folder_path+'/en_zh.prompts', 'w') as f:
        for i in range(len(scores)):
            f.write("mathematical method"+"\n")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
    random.seed(42)
    normalized_scores=[[] for _ in range(10)]
    inputs = []
    df_source = pd.read_csv("/H1/SharedTask2023/data/test/mt_en_es_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs.append((src, hyp))
    
    normalized_scores[0]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p1.3sbert.logprobs/en_es.scores')
    normalized_scores[1]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p2.3sbert.logprobs/en_es.scores')
    normalized_scores[2]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p3.3sbert.logprobs/en_es.scores')
    normalized_scores[3]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p4.3sbert.logprobs/en_es.scores')
    normalized_scores[4]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p5.3sbert.logprobs/en_es.scores')
    normalized_scores[5]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p6.3sbert.logprobs/en_es.scores')
    normalized_scores[6]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p7.3sbert.logprobs/en_es.scores')
    normalized_scores[7]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p8.3sbert.logprobs/en_es.scores')
    normalized_scores[8]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p9.3sbert.logprobs/en_es.scores')
    normalized_scores[9]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p10.3sbert.logprobs/en_es.scores')
    scores=[]
    for i in range(len(inputs)):
        sum=0.0
        for j in range(10):
            sum+=normalized_scores[j][i]
        mean=sum/10.0
        differences=[]
        for j in range(10):
            differences.append( (abs(normalized_scores[j][i]-mean),j) )
        differences.sort(key=lambda differences: differences[0],reverse=True)
        ans=0.0
        for j in range(5):
            value,index=differences[j]
            ans+=normalized_scores[index][i]
        scores.append(str(ans))
    prompts=[]
    cnt=-1
    with open(folder_path+'/en_es.scores', 'w') as f:
        for score in scores:
            f.write(score + "\n")
    with open(folder_path+'/en_es.description', 'w') as f:
        sentence='This method uses mathematical method to fuse the previous good results'
        f.write(sentence+"\n")
    with open(folder_path+'/en_es.prompts', 'w') as f:
        for i in range(len(scores)):
            f.write("mathematical method"+"\n")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
    items=os.listdir(folder_path)
    lis_nam=[]
    for item in items:
        lis_nam.append(folder_path+'/'+item)
    get_zip('/H1/SharedTask2023/baselines/test_zips/ensemble_oop1tooop10_cha.zip',lis_nam)
    

