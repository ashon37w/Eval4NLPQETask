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
from utils import load_hf_lm_and_tokenizer, generate_completions, get_placeholder

fixed_demos=[("波黑奥委会主席：坚信中国有能力在特殊时期办好冬奥盛会-新华网","President of the Olympic Committee of Bosnia and Herzegovina: Firmly believe that China has the ability to successfully host the Winter Olympic Games in a special period - Xinhua Net"),
       ("美学是以对美的本质及其意义的研究为主题的学科。","Aesthetics is a subject with the theme of the study of the essence and significance of beauty."),
       ("美三航母集结中国周边，最强大武器罕见访问关岛。中国近年来在南沙群岛修建大批军事设施。","The three US aircraft carriers assembled the most powerful weapons around China and rarely visited Guam. China has built a large number of military facilities in the Nansha Islands in recent years."),
       ("那家餐厅一直都说在处理中","That restaurant has always said that it is being processed"),
       ("除了在经济发展、经济增长及东亚经济领域取得的成就外，他还是一位富有远见的教育领导者。","In addition to his achievements in economic development, economic growth and the East Asian economy, he is also a visionary education leader.")]

in_tag_cut_dict = {'English':"en_core_web_sm",'Germany':"de_core_news_sm"}

# Translate the following {source_lang} sentence into {target_lang}.
# Translate {src seq} into {tgt lang}: {tgt seq}
# Please translate {src seq} into {tgt lang}: {tgt seq}
# Help me to translate {src seq} into {tgt lang}: {tgt seq}
# Translate {src seq} from {src lang} into {tgt lang}
# Please translate {src seq} from {src lang} into {tgt lang}: {tgt seq}
# Help me to translate {src seq} from {src lang} into {tgt lang}: {tgt seq}
# {src lang}: {src seq}; {tgt lang}: {tgt seq}
# {src lang} source: {src seq}; {tgt lang} translation: {tgt seq}
# The {tgt lang} translation of {src lang} is: {src seq}


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
        if not os.path.exists('demos/zhen_sbert.txt'):
            print('no cache')
        else:
            demos_lst = [line.strip() for line in open('demos/zhen_sbert.txt', "r", encoding="utf-8").readlines()]
            demos_res = []
            print(len(demos_lst))
            for st in demos_lst:
                now_demos = [x.split(" ||| ") for x in st.split('\t')[:args.demo_num]]
                demos_res.append(now_demos)
            return demos_res
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    model = SentenceTransformer('../models/all-MiniLM-L6-v2')
    inputs = []
    df_source = pd.read_csv("/H1/SharedTask2023/data/zh_en/dev_zh_en.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['HYP']):
        inputs.append((src, hyp))
    src_corpus = [line.strip() for line in open('demos/cn_dev.txt', "r", encoding="utf-8").readlines()]
    tgt_corpus = [line.strip() for line in open('demos/en_dev.txt', "r", encoding="utf-8").readlines()]
    print("sent_bert")
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
    with open('demos/zhen_sbert.txt','w+') as f:
        for now in demo_res:
            f.write('\t'.join([f'{x} ||| {y}' for x,y in now])+'\n')
    demo_res = [x[:args.demo_num] for x in demo_res]
    return demo_res


def eval_hf_model(args, model, tokenizer, prompts, save_path=None,in_tag_cut=None):

    if args.in_cot:
        stop_sequnce = None
    else:
        stop_sequnce = tokenizer.encode("\n\n", add_special_tokens=False)[-2:]

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=512,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1,
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


def get_eval(args):
    print('get_eval')
    # source_lang = 'English'
    # target_lang = 'Germany'
    source_lang = 'Chinese'
    target_lang = 'English'
    # save_path = os.path.join(args.save_dir , 'dev_ende.batchres.seg.scores')
    if args.postag:
        print('pos tagger:',in_tag_cut_dict[target_lang])
    save_path = './prob_results/'+args.save_dir
    task_prompt = "\n".join(
            [
                f"Translate the following {source_lang} sentence into {target_lang}."
            ])
    
    task_examples=[]
    inputs = []
    
    df_source = pd.read_csv("/H1/SharedTask2023/data/zh_en/dev_zh_en.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['HYP']):
        inputs.append((src, hyp))
    if args.bm_25:
        demos = get_demos(args)
    elif args.sent_bert:
        demos = get_sentbert_demos(args)
    else:
        demos = []
        for i in range(len(inputs)):
            demos.append(fixed_demos[:args.demo_num])
    print('get example\n')
    
    holder = get_placeholder(args.model_name_or_path)
    prompts = []
    for dt,now_demo in zip(inputs[args.st:args.ed+1],demos[args.st:args.ed+1]):
        src,hyp = dt
        demo_s = []
        for ds,dh in now_demo:
            demo_s+= [
                    holder['user'],
                    task_prompt,
                    f'{source_lang} source: {ds}',
                    holder['assistant'],
                    f'{target_lang} translation: {dh}'
            ]
        demo_s = '\n'.join(demo_s)
        s =  "\n".join(
            [
                demo_s,
                holder['user'],
                task_prompt,
                f'{source_lang} source: {src}',
                holder['assistant'],
                f'{target_lang} translation: '
            ])
        prompts.append({'prompt':s,'hyp':hyp})
    
    print('prompts:')
    for it in prompts[:2]:
        print(it['prompt'],end='\n\n')

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path, 
            load_in_8bit=args.load_in_8bit, 
            device_map="balanced_low_0" if torch.cuda.device_count() > 0 else "auto",
            gptq_model=args.gptq,
            use_llama = args.llama
        )
    print('get model device:',"balanced_low_0" if torch.cuda.device_count() > 1 else "auto")
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
    parser.add_argument("--save_dir", type=str, default="nh.zhen.test.logprob")
    # parser.add_argument("--model_name_or_path", type=str, default="../models/Nous-Hermes-13b")
    parser.add_argument("--model_name_or_path", type=str, default="../models/OpenOrca-Platypus2-13B")
    # parser.add_argument("--model_name_or_path", type=str, default="../models/WizardLM-13B-V1.1-GPTQ")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--in_cot", action="store_true")
    parser.add_argument("--max_num_examples_per_task", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--demo_num", type=int, default=3)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--llama", action="store_true")
    parser.add_argument("--postag", action="store_true")
    parser.add_argument("--bm_25", action="store_true")
    parser.add_argument("--sent_bert", action="store_true")
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--st", type=int, default=0)
    parser.add_argument("--ed", type=int, default=10499)
    args = parser.parse_args()
    print('save:','./prob_results/'+args.save_dir)
    print('model:',args.model_name_or_path)
    print(f'{args.st} -> {args.ed}')
    print(f'demo:{args.demo_num}')
    get_eval(args)
    

