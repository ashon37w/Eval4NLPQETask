import torch
import tqdm
import json
import time
import random
import argparse
import pandas as pd
import csv
import re
import os
from transformers import AutoTokenizer
from bm25_demos import create_bm25_database
from log_prob_utils import load_hf_lm_and_tokenizer, generate_completions, get_placeholder

in_tag_cut_dict = {'English':"en_core_web_sm",'Germany':"de_core_news_sm"}


def get_demos(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    bm25_database, corpus = create_bm25_database('demos/cn_dev.txt','demos/en_dev.txt',tokenizer)
    
    inputs = []
    df_source = pd.read_csv("/H1/SharedTask2023/data/zh_en/dev_zh_en.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['HYP']):
        inputs.append((src, hyp))

    tokenized_src_lines = [tokenizer.tokenize(src) for src,hyp in inputs]
    bm25_top_ns = []
    for line in tqdm.tqdm(tokenized_src_lines, desc="BM25 Querying"):
        bm25_top_ns.append(bm25_database.get_top_n(line, corpus, n=args.demo_num))
    demo_res = []
    for demos in bm25_top_ns:
        demo_res.append([x.split(" ||| ") for x in demos])
    return demo_res


def get_sentbert_demos(args,use_cache=True):
    if use_cache:
        if not os.path.exists(args.sent_bert_res):
            print('warning: no cache')
        else:
            demos_lst = [line.strip() for line in open(args.sent_bert_res, "r", encoding="utf-8").readlines()]
            demos_res = []
            for st in demos_lst:
                now_demos = [x.split(" ||| ") for x in st.split('\t')[:args.demo_num]]
                demos_res.append(now_demos)
            return demos_res
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    model = SentenceTransformer('../models/all-MiniLM-L6-v2')
    inputs = []
    df_source = pd.read_csv(args.data_dir, sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs.append((src, hyp))
    src_corpus = [line.strip() for line in open(args.sent_bert_src, "r", encoding="utf-8").readlines()]
    tgt_corpus = [line.strip() for line in open(args.sent_bert_tgt, "r", encoding="utf-8").readlines()]

    sentences = [x[0] for x in inputs] + src_corpus
    embeddings = model.encode(sentences)
    embeddings = [x.reshape(1,-1) for x in embeddings]
    demo_embed = [(emb,src,tgt) for emb,src,tgt in zip(embeddings[len(inputs):],src_corpus,tgt_corpus)]
    demo_res = []
    for i in tqdm.tqdm(range(len(inputs))):
        src,hyp = inputs[i]
        now_embed = embeddings[i]
        sim_res = [(cosine_similarity(now_embed,emb)[0][0],src,tgt) for emb,src,tgt in demo_embed]
        sim_res.sort(key=lambda x:x[0],reverse=True)
        demo_res.append([(src,tgt) for emb,src,tgt in sim_res[:5]])
    with open(args.sent_bert_res,'w+') as f:
        for now in demo_res:
            f.write('\t'.join([f'{x} ||| {y}' for x,y in now])+'\n')
    demo_res = [x[:args.demo_num] for x in demo_res]
    return demo_res


def eval_hf_model(args, model, tokenizer, prompts, save_path=None,in_tag_cut=None):
    stop_sequnce = tokenizer.encode("\n\n", add_special_tokens=False)[-2:]
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=512,
        batch_size=1,
        stop_id_sequences=[[stop_sequnce]] ,
        generate_name = args.save_dir,
        in_tag_cut = in_tag_cut
    )
    
    if save_path:
        fout = open(save_path, "w+")
        for output in outputs:
            if args.postag:
                now_wl = '\t'.join([f'{prob:.6f},{tag}' for prob,tag in output])
            else:
                now_wl = '\t'.join([f'{x:.6f}' for x in output])
            fout.write(now_wl + "\n")        

            
def get_prompt(source_lang, target_lang, src, now_demo, holder):
    demo_s = []
    for ds,dh in now_demo:
        demo_s+= [
            holder['user'],
            f"Help me to translate {ds} from {source_lang} into {target_lang}:",
            holder['assistant'],
            f'{dh}'
        ]
    res =  "\n".join(demo_s + [
            holder['user'],
            f"Help me to translate {src} from {source_lang} into {target_lang}:",
            holder['assistant']
        ])
    return res
            
            
def get_max_prompt(tokenizer,source_lang, target_lang, src, hyp, now_demo, holder):
    demo_s = []
    prompt =  [
        holder['user'],
        f"Help me to translate {src} from {source_lang} into {target_lang}:",
        holder['assistant']
    ]
    for ds,dh in now_demo:
        demo_prompt = [
            holder['user'],
            f"Help me to translate {ds} from {source_lang} into {target_lang}:",
            holder['assistant'],
            f'{dh}'
        ]
        res =  "\n".join(demo_s + demo_prompt + prompt)
        if len(tokenizer.encode(res+hyp))>=1950:
            break
        demo_s += demo_prompt
    res =  "\n".join(demo_s + prompt)
    return res
            
def get_eval(args):
    source_lang = args.source_lang
    target_lang = args.target_lang
    save_path = './test_prob/'+args.save_dir
    
    inputs = []
    df_source = pd.read_csv(args.data_dir, sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['TGT']):
        inputs.append((src, hyp))
    if args.bm_25:
        demos = get_demos(args)
    elif args.sent_bert:
        demos = get_sentbert_demos(args)
    else:
        demos = [[] for _ in inputs]
    
    holder = get_placeholder(args.model_name_or_path)
    prompts = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    for dt,now_demo in tqdm.tqdm(zip(inputs,demos)):
        src,hyp = dt
        if args.max_demo:
            prompts.append({'prompt':get_max_prompt(tokenizer,source_lang, target_lang, src, hyp, now_demo, holder),'hyp':hyp})
        else:
            prompts.append({'prompt':get_prompt(source_lang, target_lang, src, now_demo, holder),'hyp':hyp})
    print('prompts:')
    for it in prompts[:2]:
        print(it['prompt'],end='\n\n')

    if args.model_name_or_path:
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path, 
            device_map="balanced_low_0" if torch.cuda.device_count() > 0 else "auto",
            gptq_model=args.gptq,
            use_llama = args.llama
        )
    in_tag_cut = in_tag_cut_dict[target_lang] if args.postag else None
    task_perf = eval_hf_model(
                args, 
                model, 
                tokenizer,  
                prompts,
                save_path = save_path,
                in_tag_cut = in_tag_cut
            )


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/test/mt_en_zh_test.tsv")
    parser.add_argument("--save_dir", type=str, default="nh.enzh.test.logprob")
    parser.add_argument("--model_name_or_path", type=str, default="../models/Nous-Hermes-13b")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--demo_num", type=int, default=20)
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--llama", action="store_true")
    parser.add_argument("--postag", action="store_true")
    parser.add_argument("--bm_25", action="store_true")
    parser.add_argument("--sent_bert", action="store_true")
    parser.add_argument("--max_demo", action="store_true")
    parser.add_argument("--sent_bert_src", type=str, default='demos/zhen_en_dev.txt')
    parser.add_argument("--sent_bert_tgt", type=str, default='demos/zhen_zh_dev.txt')
    parser.add_argument("--sent_bert_res", type=str, default='demos/enzh_sbert.txt')
    parser.add_argument("--source_lang", type=str, default="English")
    parser.add_argument("--target_lang", type=str, default="Chinese")
    args = parser.parse_args()
    print('save:','./prob_results/'+args.save_dir)
    print('model:',args.model_name_or_path)
    print(f'demo:{args.demo_num}')
    get_eval(args)