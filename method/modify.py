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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    folder_path='/H1/SharedTask2023/baselines/maxsbert/oop_p4_result'
    if os.path.exists(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    random.seed(42)
    inputs=[[] for _ in range(3)]
    df_source=pd.read_csv("/H1/SharedTask2023/data/test/mt_en_de_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs[0].append((src, hyp))
    df_source=pd.read_csv("/H1/SharedTask2023/data/test/mt_en_zh_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs[1].append((src, hyp))
    df_source=pd.read_csv("/H1/SharedTask2023/data/test/mt_en_es_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs[2].append((src, hyp))
    normalized_scores=[[] for _ in range(3)]
    normalized_scores_3sbert=[[] for _ in range(3)]
    normalized_scores[0]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p4.maxsbert.logprobs/en_de.scores')
    normalized_scores[1]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p4.maxsbert.logprobs/en_zh.scores')
    normalized_scores[2]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p4.maxsbert.logprobs/en_es.scores')
    normalized_scores_3sbert[0]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p4.3sbert.logprobs/en_de.scores')
    normalized_scores_3sbert[1]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p4.3sbert.logprobs/en_zh.scores')
    normalized_scores_3sbert[2]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p4.3sbert.logprobs/en_es.scores')
    scores=[[] for _ in range(3)]
    prompts=[[] for _ in range(3)]
    for src,hyp in inputs[0]:
        prompts[0].append("Help me to translate "+src+" into German: {tgt seq}")
    for src,hyp in inputs[1]:
        prompts[1].append("Help me to translate "+src+" into Chinese: {tgt seq}")
    for src,hyp in inputs[2]:
        prompts[2].append("Help me to translate "+src+" into Spanish: {tgt seq}")
    
    for i in range(len(normalized_scores[0])):
        if abs(normalized_scores[0][i]-0.0)<=1e-3:
            normalized_scores[0][i]=normalized_scores_3sbert[0][i]
    for i in range(len(normalized_scores[1])):
        if abs(normalized_scores[1][i]-0.0)<=1e-3:
            normalized_scores[1][i]=normalized_scores_3sbert[1][i]
    for i in range(len(normalized_scores[2])):
        if abs(normalized_scores[2][i]-0.0)<=1e-3:
            normalized_scores[2][i]=normalized_scores_3sbert[2][i]

    with open(folder_path+'/en_de.scores', 'w') as f:
        for score in normalized_scores[0]:
            f.write(str(score) + "\n")
    with open(folder_path+'/en_de.description', 'w') as f:
        sentence='This submission uses the translation prompt and generates results using the OpenOrca-Platypus2-13B, which obtains sentence-level QE scores with the probability of each word being generated on the translation side.'
        f.write(sentence+"\n")
    with open(folder_path+'/en_de.prompts', 'w') as f:
        for prompt in prompts[0]:
            f.write(prompt+"\n")
    with open(folder_path+'/en_zh.scores', 'w') as f:
        for score in normalized_scores[1]:
            f.write(str(score) + "\n")
    with open(folder_path+'/en_zh.description', 'w') as f:
        sentence='This submission uses the translation prompt and generates results using the OpenOrca-Platypus2-13B, which obtains sentence-level QE scores with the probability of each word being generated on the translation side.'
        f.write(sentence+"\n")
    with open(folder_path+'/en_zh.prompts', 'w') as f:
        for prompt in prompts[1]:
            f.write(prompt+"\n")
    with open(folder_path+'/en_es.scores', 'w') as f:
        for score in normalized_scores[2]:
            f.write(str(score) + "\n")
    with open(folder_path+'/en_es.description', 'w') as f:
        sentence='This submission uses the translation prompt and generates results using the OpenOrca-Platypus2-13B, which obtains sentence-level QE scores with the probability of each word being generated on the translation side.'
        f.write(sentence+"\n")
    with open(folder_path+'/en_es.prompts', 'w') as f:
        for prompt in prompts[2]:
            f.write(prompt+"\n")
    # {src lang} source: {src seq}; {tgt lang} translation: {tgt seq}
    # The {tgt lang} translation of {src lang} is: {src seq}


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    folder_path='/H1/SharedTask2023/baselines/maxsbert/oop_p9_result'
    if os.path.exists(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    random.seed(42)
    inputs=[[] for _ in range(3)]
    df_source=pd.read_csv("/H1/SharedTask2023/data/test/mt_en_de_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs[0].append((src, hyp))
    df_source=pd.read_csv("/H1/SharedTask2023/data/test/mt_en_zh_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs[1].append((src, hyp))
    df_source=pd.read_csv("/H1/SharedTask2023/data/test/mt_en_es_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs[2].append((src, hyp))
    normalized_scores=[[] for _ in range(3)]
    normalized_scores_3sbert=[[] for _ in range(3)]
    normalized_scores[0]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p9.maxsbert.logprobs/en_de.scores')
    normalized_scores[1]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p9.maxsbert.logprobs/en_zh.scores')
    normalized_scores[2]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p9.maxsbert.logprobs/en_es.scores')
    normalized_scores_3sbert[0]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p9.3sbert.logprobs/en_de.scores')
    normalized_scores_3sbert[1]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p9.3sbert.logprobs/en_zh.scores')
    normalized_scores_3sbert[2]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p9.3sbert.logprobs/en_es.scores')
    scores=[[] for _ in range(3)]
    prompts=[[] for _ in range(3)]
    for src,hyp in inputs[0]:
        prompts[0].append("English source: "+src+"; German translation: {tgt seq}")
    for src,hyp in inputs[1]:
        prompts[1].append("English source: "+src+"; Chinese translation: {tgt seq}")
    for src,hyp in inputs[2]:
        prompts[2].append("English source: "+src+"; Spanish translation: {tgt seq}")
    #{src lang} source: {src seq}; {tgt lang} translation: {tgt seq}
    for i in range(len(normalized_scores[0])):
        if abs(normalized_scores[0][i]-0.0)<=1e-3:
            normalized_scores[0][i]=normalized_scores_3sbert[0][i]
    for i in range(len(normalized_scores[1])):
        if abs(normalized_scores[1][i]-0.0)<=1e-3:
            normalized_scores[1][i]=normalized_scores_3sbert[1][i]
    for i in range(len(normalized_scores[2])):
        if abs(normalized_scores[2][i]-0.0)<=1e-3:
            normalized_scores[2][i]=normalized_scores_3sbert[2][i]

    with open(folder_path+'/en_de.scores', 'w') as f:
        for score in normalized_scores[0]:
            f.write(str(score) + "\n")
    with open(folder_path+'/en_de.description', 'w') as f:
        sentence='This submission uses the translation prompt and generates results using the OpenOrca-Platypus2-13B, which obtains sentence-level QE scores with the probability of each word being generated on the translation side.'
        f.write(sentence+"\n")
    with open(folder_path+'/en_de.prompts', 'w') as f:
        for prompt in prompts[0]:
            f.write(prompt+"\n")
    with open(folder_path+'/en_zh.scores', 'w') as f:
        for score in normalized_scores[1]:
            f.write(str(score) + "\n")
    with open(folder_path+'/en_zh.description', 'w') as f:
        sentence='This submission uses the translation prompt and generates results using the OpenOrca-Platypus2-13B, which obtains sentence-level QE scores with the probability of each word being generated on the translation side.'
        f.write(sentence+"\n")
    with open(folder_path+'/en_zh.prompts', 'w') as f:
        for prompt in prompts[1]:
            f.write(prompt+"\n")
    with open(folder_path+'/en_es.scores', 'w') as f:
        for score in normalized_scores[2]:
            f.write(str(score) + "\n")
    with open(folder_path+'/en_es.description', 'w') as f:
        sentence='This submission uses the translation prompt and generates results using the OpenOrca-Platypus2-13B, which obtains sentence-level QE scores with the probability of each word being generated on the translation side.'
        f.write(sentence+"\n")
    with open(folder_path+'/en_es.prompts', 'w') as f:
        for prompt in prompts[2]:
            f.write(prompt+"\n")



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    folder_path='/H1/SharedTask2023/baselines/maxsbert/oop_p10_result'
    if os.path.exists(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    random.seed(42)
    inputs=[[] for _ in range(3)]
    df_source=pd.read_csv("/H1/SharedTask2023/data/test/mt_en_de_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs[0].append((src, hyp))
    df_source=pd.read_csv("/H1/SharedTask2023/data/test/mt_en_zh_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs[1].append((src, hyp))
    df_source=pd.read_csv("/H1/SharedTask2023/data/test/mt_en_es_test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs[2].append((src, hyp))
    normalized_scores=[[] for _ in range(3)]
    normalized_scores_3sbert=[[] for _ in range(3)]
    normalized_scores[0]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p10.maxsbert.logprobs/en_de.scores')
    normalized_scores[1]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p10.maxsbert.logprobs/en_zh.scores')
    normalized_scores[2]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p10.maxsbert.logprobs/en_es.scores')
    normalized_scores_3sbert[0]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.ende.p10.3sbert.logprobs/en_de.scores')
    normalized_scores_3sbert[1]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enzh.p10.3sbert.logprobs/en_zh.scores')
    normalized_scores_3sbert[2]=read_data('/H1/SharedTask2023/translate_prob/test_results/oop.tst.enes.p10.3sbert.logprobs/en_es.scores')
    scores=[[] for _ in range(3)]
    prompts=[[] for _ in range(3)]
    for src,hyp in inputs[0]:
        prompts[0].append("The German translation of English is: "+src)
    for src,hyp in inputs[1]:
        prompts[1].append("The Chinese translation of English is: "+src)
    for src,hyp in inputs[2]:
        prompts[2].append("The Spanish translation of English is: "+src)
    # The {tgt lang} translation of {src lang} is: {src seq}
    for i in range(len(normalized_scores[0])):
        if abs(normalized_scores[0][i]-0.0)<=1e-3:
            normalized_scores[0][i]=normalized_scores_3sbert[0][i]
    for i in range(len(normalized_scores[1])):
        if abs(normalized_scores[1][i]-0.0)<=1e-3:
            normalized_scores[1][i]=normalized_scores_3sbert[1][i]
    for i in range(len(normalized_scores[2])):
        if abs(normalized_scores[2][i]-0.0)<=1e-3:
            normalized_scores[2][i]=normalized_scores_3sbert[2][i]

    with open(folder_path+'/en_de.scores', 'w') as f:
        for score in normalized_scores[0]:
            f.write(str(score) + "\n")
    with open(folder_path+'/en_de.description', 'w') as f:
        sentence='This submission uses the translation prompt and generates results using the OpenOrca-Platypus2-13B, which obtains sentence-level QE scores with the probability of each word being generated on the translation side.'
        f.write(sentence+"\n")
    with open(folder_path+'/en_de.prompts', 'w') as f:
        for prompt in prompts[0]:
            f.write(prompt+"\n")
    with open(folder_path+'/en_zh.scores', 'w') as f:
        for score in normalized_scores[1]:
            f.write(str(score) + "\n")
    with open(folder_path+'/en_zh.description', 'w') as f:
        sentence='This submission uses the translation prompt and generates results using the OpenOrca-Platypus2-13B, which obtains sentence-level QE scores with the probability of each word being generated on the translation side.'
        f.write(sentence+"\n")
    with open(folder_path+'/en_zh.prompts', 'w') as f:
        for prompt in prompts[1]:
            f.write(prompt+"\n")
    with open(folder_path+'/en_es.scores', 'w') as f:
        for score in normalized_scores[2]:
            f.write(str(score) + "\n")
    with open(folder_path+'/en_es.description', 'w') as f:
        sentence='This submission uses the translation prompt and generates results using the OpenOrca-Platypus2-13B, which obtains sentence-level QE scores with the probability of each word being generated on the translation side.'
        f.write(sentence+"\n")
    with open(folder_path+'/en_es.prompts', 'w') as f:
        for prompt in prompts[2]:
            f.write(prompt+"\n")
# The {tgt lang} translation of {src lang} is: {src seq}