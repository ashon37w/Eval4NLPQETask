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

in_tag_cut_dict = {'English':"en_core_web_sm",'Germany':"de_core_news_sm"}

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
    source_lang = 'English'
    target_lang = 'Germany'
    if args.postag:
        print('pos tagger:',in_tag_cut_dict[target_lang])
    save_path = './prob_results/'+args.save_dir
    task_examples=[]
    inputs = []
    df_source = pd.read_csv("/H1/SharedTask2023/data/en_de/dev_en_de.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['HYP']):
        inputs.append((src, hyp))
    demos = []
    for i in range(len(inputs)):
        demos.append("")
    print('get example\n')
    
    holder = get_placeholder(args.model_name_or_path)
    prompts = []
    for dt,now_demo in zip(inputs[args.st:args.ed+1],demos[args.st:args.ed+1]):
        src,hyp = dt
        s =  "\n".join(
            [
                holder['assistant'],
            ])
        prompts.append({'prompt':"",'hyp':hyp})
    
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
    parser.add_argument("--save_dir", type=str, default="nh.ende.demo.logprob")
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
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--st", type=int, default=0)
    parser.add_argument("--ed", type=int, default=10499)
    args = parser.parse_args()
    print('save:','./demo_prob_results/'+args.save_dir)
    print('model:',args.model_name_or_path)
    print(f'{args.st} -> {args.ed}')
    print(f'demo:{args.demo_num}')
    get_eval(args)
    

