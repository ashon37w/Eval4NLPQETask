from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
import pandas as pd
import csv
import tqdm

def create_bm25_database(train_file_src, train_file_tgt, tokenizer):
    src_corpus = [line.strip() for line in open(train_file_src, "r", encoding="utf-8").readlines()]
    tgt_corpus = [line.strip() for line in open(train_file_tgt, "r", encoding="utf-8").readlines()]
    corpus = [line[0] + " ||| " + line[1] for line in zip(src_corpus, tgt_corpus)]
    corpus = list(set(corpus))
    tokenized_src_corpus = [tokenizer.tokenize(line.split(" ||| ")[0]) for line in corpus]
    bm25_database = BM25Okapi(tokenized_src_corpus)
    return bm25_database, corpus

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../models/Nous-Hermes-13b")
    # tokenizer = AutoTokenizer.from_pretrained("../models/WizardLM-13B-V1.1-GPTQ")
    bm25_database, corpus = create_bm25_database('demos/deen_en_dev.txt','demos/deen_de_dev.txt',tokenizer)
    
    inputs = []
    df_source = pd.read_csv("/H1/SharedTask2023/data/en_de/dev_en_de.tsv", sep="\t", quoting=csv.QUOTE_NONE)
    for (src, hyp) in zip(df_source['SRC'], df_source['HYP']):
        inputs.append((src, hyp))

    tokenized_src_lines = [tokenizer.tokenize(src) for src,hyp in inputs]
    bm25_top_ns = []
    for line in tqdm.tqdm(tokenized_src_lines, desc="BM25 Querying"):
        bm25_top_ns.append(bm25_database.get_top_n(line, corpus, n=1))
    with open('./demos/dev_ende_demo.txt','w+') as f:
        for line in bm25_top_ns:
            src,hyp = line[0].split(" ||| ")
            f.write(f'{src}\t{hyp}\n')