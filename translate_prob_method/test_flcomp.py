import sys
import numpy as np
import scipy.stats
import os


prompt_lst = ["Translate the following {source_lang} sentence into {target_lang}.\n{source_lang} source: {src seq}\n{target_lang} translation:{tgt seq}",
"Translate {src seq} into {tgt lang}: {tgt seq}",
"Please translate {src seq} into {tgt lang}: {tgt seq}",
"Help me to translate {src seq} into {tgt lang}: {tgt seq}",
"Translate {src seq} from {src lang} into {tgt lang}",
"Please translate {src seq} from {src lang} into {tgt lang}: {tgt seq}",
"Help me to translate {src seq} from {src lang} into {tgt lang}: {tgt seq}",
"{src lang}: {src seq}; {tgt lang}: {tgt seq}",
"{src lang} source: {src seq}; {tgt lang} translation: {tgt seq}",
"The {tgt lang} translation of {src lang} is: {src seq}"]


def load_scores(input_file):
    with open(input_file) as f:
        return np.array([float(n) for n in f.read().split("\n") if n != ""])


def main(argv):
    _, input_file = argv
    dir_name = input_file.split('/')[-1]
    os.system(f'mkdir test_results/{dir_name}')
    type_name = ""
    if 'ende' in dir_name:
        type_name = 'en_de'
    elif 'enzh' in dir_name:
        type_name = 'en_zh'
    elif 'enes' in dir_name:
        type_name = 'en_es'
    prompt = ''
    for i in range(10):
        if f'p{i+1}.' in dir_name:
            prompt = prompt_lst[i]
    f=open(f'test_results/{dir_name}/{type_name}.description','w+')
    f.close()
    f=open(f'test_results/{dir_name}/{type_name}.explanations','w+')
    f.close()
    f=open(f'test_results/{dir_name}/{type_name}.prompts','w+')
    f.write(prompt)
    f.close()
    os.system(f'cp {input_file} test_results/{dir_name}/{type_name}.scores')



# Run
if __name__ == '__main__':
    main(sys.argv)
