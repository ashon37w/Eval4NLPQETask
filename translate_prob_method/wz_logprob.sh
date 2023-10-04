python get_probs.py --sent_bert --save_dir "wz.ende.3sbert.p2.logprobs" --model_name_or_path "../models/WizardLM-13B-V1.1-GPTQ" --demo_num 3 --gptq
python tm_probs.py --sent_bert --save_dir "wz.zhen.3sbert.p9.logprobs" --model_name_or_path "../models/WizardLM-13B-V1.1-GPTQ" --demo_num 3 --gptq
