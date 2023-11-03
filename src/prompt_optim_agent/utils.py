import os
import time
import openai 
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from glob import glob
import google.generativeai as palm

def openai_key_config(api_key=None, key_file = '../api_keys.txt'):
    if api_key is not None and api_key.startswith('AIza'):
        palm.configure(api_key=api_key)
        return
    
    if api_key is not None:
        print(f'api_key: {api_key}')
        openai.api_key = api_key.strip()
        return
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Please enter your API key in {key_file}")
    api_key = open(key_file).readlines()[0].strip()
    print(f'api_key: {api_key}')
    openai.api_key = api_key

CHAT_COMPLETION_MODELS = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314']
COMPLETION_MODELS =  ['text-davinci-003', 'text-davinci-002','code-davinci-002']
OPENSOURCE_MODELS = ['flan-t5']
PALM_MODELS = ['models/chat-bison-001']
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

def create_logger(logging_dir, name, log_mode='train'):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    if log_mode == "test":
        name += "-test"
    else:
        name += "-train"
    logging_dir = os.path.join(logging_dir, name)
    num = len(glob(logging_dir+'*'))
    
    logging_dir += '-'+f'{num:03d}'+".log"
    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}")]
    )
    logger = logging.getLogger('prompt optimization agent')
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("datasets").setLevel(logging.CRITICAL)
    return logger

def batch_forward_chatcompletion(batch_prompts, model='gpt-3.5-turbo', temperature=0, max_tokens=1024):
    responses = []
    for prompt in batch_prompts:
        messages = [{"role": "user", "content": prompt},]
        response = gpt_chat_completion(messages=messages, model=model, temperature=temperature)
        responses.append(response['choices'][0]['message']['content'].strip())
    return responses

def batch_forward_chatcompletion_palm(batch_prompts, model='models/chat-bison-001', temperature=0, max_tokens=1024):
    responses = []
    for prompt in batch_prompts:
        response = gpt_palm_completion(messages=prompt, temperature=temperature, model=model)

        if response.last is None:
            responses.append("N/A: no answer")
            continue

        responses.append(response.last.strip())
    return responses

def batch_forward_completion(batch_prompts, model='text-davinci-003', temperature=0, max_tokens=1024):
    gpt_output = gpt_completion(prompt=batch_prompts, model=model, temperature=temperature, max_tokens=max_tokens)['choices']
    responses = []
    for response in gpt_output:
        responses.append(response['text'].strip())
    return responses

def gpt_palm_completion(messages, temperature, model):
    backoff_time = 1
    while True:
        try:
            return palm.chat(messages=messages, temperature=temperature, model=model)
        except:
            print(f' Sleeping {backoff_time} seconds...')
            time.sleep(backoff_time)
            backoff_time *= 1.5

def gpt_chat_completion(**kwargs):
    backoff_time = 1
    while True:
        try:
            return openai.ChatCompletion.create(**kwargs)
        except openai.error.OpenAIError:
            print(openai.error.OpenAIError, f' Sleeping {backoff_time} seconds...')
            time.sleep(backoff_time)
            backoff_time *= 1.5

def gpt_completion(**kwargs):
    backoff_time = 1
    while True:
        try:
            return openai.Completion.create(**kwargs)
        except openai.error.OpenAIError:
            print(openai.error.OpenAIError, f' Sleeping {backoff_time} seconds...')
            time.sleep(backoff_time)
            backoff_time *= 1.5

def print_dict(d, space='  '):
    s = ''
    for k in d:
        s+=f'{space}{k}:\n{space}{d}\n'
    return s

def batch_forward_flant5(batch_prompts, model='flan-t5', temperature=0, max_tokens=500):
    flant5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    flant5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    responses = []
    for prompt in batch_prompts:
        input_ids = flant5_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        outputs = flant5_model.generate(input_ids, max_length=500, bos_token_id=0)
        response = flant5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)
    return responses

def print_search_result_dict(result_dict):
    print('\n--------------  state test result  -----------------')
    for i in range(len(result_dict['state'])):
        prompt = result_dict['state'][i]
        test_acc = result_dict['test_acc'][i]
        prompt_len = len(prompt.split(' '))
        print(f'{i} | test_acc: {test_acc:.3f}  | length: {prompt_len} | prompt: {prompt.strip()}')
    print('\n--------------  batch gd result  -----------------')
    for i in range(len(result_dict['forward_acc'])):
        prompt_before = result_dict['state'][i]
        prompt_after = result_dict['state'][i+1]
        forward_acc = result_dict['forward_acc'][i]
        forward_correct = result_dict['forward_correct'][i]
        re_forward_acc = result_dict['re_forward_acc'][i]
        re_forward_correct = result_dict['re_forward_correct'][i]
        print(f'batch {i}')
        print(f'prompt_before: {prompt_before}\nforward_acc   : {forward_acc} correct:{forward_correct}')
        print(f'prompt_after: {prompt_after}\nre_forward_acc: {re_forward_acc} correct:{re_forward_correct}')