import torch
from transformers import StoppingCriteria,LogitsProcessorList,LogitsProcessor
import tqdm
from torch.nn.functional import log_softmax
import numpy as np
import spacy


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)

def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        torch_dtype="auto",
        gptq_model=False,
        use_fast_tokenizer=False,
        padding_side="left",
        use_llama=False
    ):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM, LlamaForCausalLM

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            use_safetensors=True,
            trust_remote_code=True,
            use_triton=False,
            quantize_config=None,
            inject_fused_attention=True
        )
    elif use_llama:
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            if torch.cuda.is_available():
                model = model.cuda()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
   
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        
    return model, tokenizer


class ForcedPrefixLogitsProcessor(LogitsProcessor):
    def __init__(self, out_token_ids):
        self.out_token_ids = out_token_ids
        self.pre_len = len(out_token_ids)
        self.cnt = -1
        self.scores_list = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.cnt += 1
        if self.cnt<1 or self.cnt>=self.pre_len:
            return scores
        now_prob = log_softmax(scores[0])
        now_idx = self.out_token_ids[self.cnt]
        self.scores_list.append(now_prob[now_idx].detach().cpu().item())
        scores[0][self.out_token_ids[self.cnt]] = 1000000
        return scores


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True, disable_tqdm=False, generate_name="",in_tag_cut=False,**generation_kwargs):
    generations = []
    if in_tag_cut:
        spacy_nlp = spacy.load(in_tag_cut)

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions "+generate_name)
    
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts)):
        batch_prompts = prompts[i:i+1]
        tokenized_prompts = tokenizer([x['prompt'] for x in batch_prompts], 
                                      padding="longest", 
                                      return_tensors="pt", 
                                      add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()
        hyp_res = batch_prompts[0]['hyp']
        if not isinstance(hyp_res,str):
            hyp_res = str(hyp_res)
        hyp_tokens = tokenizer.encode(hyp_res)
        if in_tag_cut:
            hyp_tokens = []
            hyp_tag = []
            hyp_doc = spacy_nlp(hyp_res)
            hyp_doc = [(x.text,x.pos_) for x in hyp_doc]
            
            for it in hyp_doc:
                now_token = tokenizer.encode(it[0])
                hyp_tokens = hyp_tokens + now_token
                hyp_tag = hyp_tag + [it[1]]*len(now_token)
        logits_processor  = LogitsProcessorList([
            ForcedPrefixLogitsProcessor(hyp_tokens)
        ])
        
        try:
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                max_new_tokens = len(hyp_tokens)+15,
                output_scores = True,
                logits_processor=logits_processor,
                temperature=0
            )
            input_len = batch_input_ids.shape[1]
            out_seq = outputs.sequences[0][input_len+1:]
            probs = logits_processor[0].scores_list
            if in_tag_cut:
                generations.append([(prob,tag) for prob,tag in zip(probs,hyp_tag)])
            else:
                generations.append(probs)
        except Exception as e:
            print("Error when generating completions for:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            generations.append([])
        torch.cuda.empty_cache()
        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)
    return generations


def get_placeholder(path):
    model_name = path.split('/')[-1]
    holder_dict = {
        'Nous-Hermes-13b':{'user':"### Instruction:",'assistant':"### Response:"},
        'WizardLM-13B-V1.1-GPTQ':{'user':"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n USER: ",'assistant':"Assistant: "},
        'OpenOrca-Platypus2-13B':{'user':"### Instruction:",'assistant':"### Response:"}
    }
    holder = holder_dict.get(model_name,{'user':"### Instruction:",'assistant':"### Response:"})
    return holder
