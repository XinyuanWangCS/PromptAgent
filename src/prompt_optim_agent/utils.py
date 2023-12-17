import os
import time
import openai 
import logging
from glob import glob
import google.generativeai as palm
from datetime import datetime
import pytz

# Supported models
CHAT_COMPLETION_MODELS = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314']
COMPLETION_MODELS =  ['text-davinci-003', 'text-davinci-002','code-davinci-002']
PALM_MODELS = ['models/chat-bison-001']

def get_pacific_time():
    current_time = datetime.now()
    pacific = pytz.timezone('US/Pacific')
    pacific_time = current_time.astimezone(pacific)
    return pacific_time
    
def api_key_config(api_key):
    if api_key is not None and api_key.startswith('AIza'):
        # PaLM2 keys start with AIza
        print(f'Set PaLM2 API: {api_key}')
        palm.configure(api_key=api_key.strip())
        return
    
    # set up key from command
    if api_key is not None:
        print(f'Set OpenAI API: {api_key}')
        openai.api_key = api_key.strip()
        return
    
    raise ValueError(f"api_key error: {api_key}")


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

def batch_forward_chatcompletion(batch_prompts, model='gpt-3.5-turbo', temperature=0):
    """
    Input a batch of prompts to openai chat API and retrieve the answers.
    """
    responses = []
    for prompt in batch_prompts:
        messages = [{"role": "user", "content": prompt},]
        response = gpt_chat_completion(messages=messages, model=model, temperature=temperature)
        responses.append(response['choices'][0]['message']['content'].strip())
    return responses

def batch_forward_chatcompletion_palm(batch_prompts, model='models/chat-bison-001', temperature=0):
    """
    Input a batch of prompts to PaLM chat API and retrieve the answers.
    """
    responses = []
    for prompt in batch_prompts:
        response = gpt_palm_completion(messages=prompt, temperature=temperature, model=model)

        if response.last is None:
            responses.append("N/A: no answer")
            continue

        responses.append(response.last.strip())
    return responses

def batch_forward_completion(batch_prompts, model='text-davinci-003', temperature=0):
    """
    Input a batch of prompts to openai completion API and retrieve the answers.
    """
    gpt_output = gpt_completion(prompt=batch_prompts, model=model, temperature=temperature)['choices']
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

