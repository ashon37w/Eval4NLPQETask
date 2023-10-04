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

from utils import load_hf_lm_and_tokenizer, generate_completions, get_placeholder

demos=[("波黑奥委会主席：坚信中国有能力在特殊时期办好冬奥盛会-新华网","President of the Olympic Committee of Bosnia and Herzegovina: Firmly believe that China has the ability to successfully host the Winter Olympic Games in a special period - Xinhua Net"),
       ("美学是以对美的本质及其意义的研究为主题的学科。","Aesthetics is a subject with the theme of the study of the essence and significance of beauty."),
       ("美三航母集结中国周边，最强大武器罕见访问关岛。中国近年来在南沙群岛修建大批军事设施。","The three US aircraft carriers assembled the most powerful weapons around China and rarely visited Guam. China has built a large number of military facilities in the Nansha Islands in recent years."),
       ("那家餐厅一直都说在处理中","That restaurant has always said that it is being processed"),
       ("除了在经济发展、经济增长及东亚经济领域取得的成就外，他还是一位富有远见的教育领导者。","In addition to his achievements in economic development, economic growth and the East Asian economy, he is also a visionary education leader.")]



def eval_hf_model(args, model, tokenizer, prompts, save_path=None):

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
        generate_name = args.save_dir
    )
    
    if save_path:
        fout = open(save_path, "w+")
        for output in outputs:
            now_wl = '\t'.join([f'{x:.6f}' for x in output])
            fout.write(now_wl + "\n")        


def get_eval(args):
    print('get_eval')
    # source_lang = 'English'
    # target_lang = 'Germany'
    source_lang = 'Chinese'
    target_lang = 'English'
    # save_path = os.path.join(args.save_dir , 'dev_ende.batchres.seg.scores')
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
    print('get example\n')
    holder = get_placeholder(args.model_name_or_path)
    prompts = []
    for src, hyp in inputs[args.st:args.ed+1]:
        demo_s = []
        for ds,dh in demos[:args.demo_num]:
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
            gptq_model=args.gptq
        )
    print('get model device:',"balanced_low_0" if torch.cuda.device_count() > 1 else "auto")

    task_perf = eval_hf_model(
                args, 
                model, 
                tokenizer,  
                prompts,
                save_path=save_path
            )



if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="nh.zhen.demo.logprob")
    parser.add_argument("--model_name_or_path", type=str, default="../models/OpenOrca-Platypus2-13B")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--in_cot", action="store_true")
    parser.add_argument("--max_num_examples_per_task", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--demo_num", type=int, default=1)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--st", type=int, default=0)
    parser.add_argument("--ed", type=int, default=10499)
    args = parser.parse_args()
    print('save:','./demo_prob_results/'+args.save_dir)
    print('model:',args.model_name_or_path)
    print(f'{args.st} -> {args.ed}')
    print(f'demo:{args.demo_num}')
    get_eval(args)
    

