import os
from datetime import datetime
from tqdm import tqdm

from .world_model.prompts import *
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tasks import *
from .utils import *
import json
import pandas as pd

def eval_prompts_in_xls(task_name, excel_dir, index, seed, log_examples=True, post_instruction=False, eval_model='gpt-3.5-turbo', batch_size=1, temperature=0, max_tokens=2048, log_dir='logs/prompt_test_logs', api_key=None, openai_key_txt_file='api_keys.txt', data_dir=None, **kwargs) :
    '''
        evaluate cur_prompt on task testing dataset
    '''
    openai_key_config(api_key, openai_key_txt_file)
    data = pd.read_excel(excel_dir).iloc[index]
    exp_name = data['Task']
    log_dir = os.path.join(log_dir, exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    train_size, eval_size, test_size = data['train_size'], data['eval_size'], data['test_size']
    prompts = {'human_prompt': data['human prompt'], 
               'APE_prompt': data['APE prompt'], 
               'ours_prompt': data['ours prompt']}
    
    logger = create_logger(log_dir, f'{exp_name}', log_mode='test')
    if data_dir is not None:
        task = get_task(task_name)(seed=seed, train_size=train_size,  eval_size=eval_size, test_size=test_size, post_instruction=post_instruction, data_dir=data_dir)
    else:
        task = get_task(task_name)(seed=seed, train_size=train_size,  eval_size=eval_size, test_size=test_size, post_instruction=post_instruction)
   
    build_forward_prompts_func = task.build_forward_prompts_completion
    if eval_model in COMPLETION_MODELS:
        batch_forward_func = batch_forward_completion
    elif eval_model in CHAT_COMPLETION_MODELS:
        batch_forward_func = batch_forward_chatcompletion
    elif eval_model in OPENSOURCE_MODELS:
        batch_forward_func  = batch_forward_flant5
    elif eval_model in PALM_MODELS:
        batch_forward_func = batch_forward_chatcompletion_palm
    else:
        raise ValueError(f"Model {eval_model} not supported.")
    
    test_dataloader = task.get_dataloader('test', batch_size=1)
    
    logger.info(f'exp_name: {exp_name}')
    logger.info(f'Task: {task.task_name}, test set size: {int(len(test_dataloader)*batch_size)}, shuffle: {False}')
    logger.info(f'post_instruction: {post_instruction}')
    logger.info(f'Eval model: {eval_model}, batch_size: {batch_size}, temperature: {temperature}, max_tokens:{max_tokens}')
    
    test_metrics = {}
    for i, prompt_key in enumerate(prompts):
        logger.info(f'------------------- prompt {i}\t{prompt_key}-------------------')
        eval_prompt = prompts[prompt_key]
        logger.info(f'prompt: {eval_prompt}')
        
        all_questions = []
        all_labels = []
        all_preds = []
        all_chats = []
        pbar = tqdm(test_dataloader, leave=False)
        count = 0
        for batch in pbar:
            batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
            responses = batch_forward_func(batch_prompts, model=eval_model, temperature=temperature, max_tokens=max_tokens)
            preds = task.batch_clean_responses(responses)
            labels = task.clean_labels(batch['answer'])
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_questions.extend(batch['question'])
            
            metric = task.cal_metric(all_preds, all_labels, all_questions)
            if not isinstance(metric, tuple):
                pbar.set_postfix_str(f"Test Metric: {metric:.4f}")
            else:
                pbar.set_postfix_str(f"Test Metrics: {metric}")
                
            for i in range(len(batch['answer'])):
                all_chats.append({
                    'question': batch['question'][i],
                    'prompt': batch_prompts[i],
                    'response': responses[i],
                    'gt_answer':batch['answer'][i],
                    'label':labels[i],
                    'pred':preds[i],

                })
                if log_examples:
                    prompt = batch_prompts[i]
                    response = responses[i]
                    label = labels[i]
                    pred = preds[i]
                    logger.info(f'-------- example {count} --------')
                    
                    logger.info(f'Input:\n{prompt}\n')
                    logger.info(f'Response:\n{response}\n')
                    logger.info(f'Pred: {pred}  Label: {label}  Correct: {pred==label}')
                    count += 1
                    if not isinstance(metric, tuple):
                        logger.info(f"Test Metric: {metric:.4f}")
                    else:
                        logger.info(f"Test Metrics: {metric}")
                    logger.info('-------------------------------')
    
        metric = task.cal_metric(all_preds, all_labels, all_questions)
        logger.info('--------------------------------------------')
        if not isinstance(metric, tuple):
            logger.info(f"Test Metric: {metric:.4f}")
        else:
            logger.info(f"Test Metrics: {metric}")
        logger.info('--------------------------------------------')
        test_metrics[prompt_key] = metric
        
    for key in test_metrics:
        logger.info('--------------------------------------------')
        logger.info(f'{key}: {test_metrics[key]}')
        logger.info(f'{key}: {prompts[key]}')
    
    return test_metrics

def eval_all_nodes_in_json(task_name, exp_name, json_dir, output_dir, seed, log_examples=True, post_instruction=False, train_size=None, eval_size=None, test_size=None, eval_model='gpt-3.5-turbo', batch_size=1, temperature=0, max_tokens=2048, log_dir='logs/prompt_test_logs', api_key=None, openai_key_txt_file='api_keys.txt', data_dir=None, **kwargs) :
    '''
        evaluate cur_prompt on task testing dataset
    '''
    with open(json_dir, 'r') as f:
        data = json.load(f)
    all_nodes = data['all_nodes']
    
    log_dir = os.path.join(log_dir, exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    openai_key_config(api_key, openai_key_txt_file)
    logger = create_logger(log_dir, f'{exp_name}', log_mode='test')
    if data_dir is not None:
        task = get_task(task_name)(seed=seed, train_size=train_size,  eval_size=eval_size, test_size=test_size, post_instruction=post_instruction, data_dir=data_dir)
    else:
        task = get_task(task_name)(seed=seed, train_size=train_size,  eval_size=eval_size, test_size=test_size, post_instruction=post_instruction)
   
    build_forward_prompts_func = task.build_forward_prompts_completion
    if eval_model in COMPLETION_MODELS:
        batch_forward_func = batch_forward_completion
    elif eval_model in CHAT_COMPLETION_MODELS:
        batch_forward_func = batch_forward_chatcompletion
    elif eval_model in OPENSOURCE_MODELS:
        batch_forward_func  = batch_forward_flant5
    else:
        raise ValueError(f"Model {eval_model} not supported.")
    
    test_dataloader = task.get_dataloader('test', batch_size=1)
    
    logger.info(f'exp_name: {exp_name}')
    logger.info(f'Task: {task.task_name}, test set size: {int(len(test_dataloader)*batch_size)}, shuffle: {False}')
    logger.info(f'post_instruction: {post_instruction}')
    logger.info(f'Eval model: {eval_model}, batch_size: {batch_size}, temperature: {temperature}, max_tokens:{max_tokens}')
    
    test_metrics = []
    for i, node in enumerate(all_nodes):
        node_id = node['id']
        logger.info(f'------------------- node {i}\t{node_id}-------------------')
        eval_prompt = node['prompt']
        
        all_questions = []
        all_labels = []
        all_preds = []
        all_chats = []
        pbar = tqdm(test_dataloader, leave=False)
        count = 0
        for batch in pbar:
            batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
            responses = batch_forward_func(batch_prompts, model=eval_model, temperature=temperature, max_tokens=max_tokens)
            preds = task.batch_clean_responses(responses)
            labels = task.clean_labels(batch['answer'])
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_questions.extend(batch['question'])
            
            metric = task.cal_metric(all_preds, all_labels, all_questions)
            if not isinstance(metric, tuple):
                pbar.set_postfix_str(f"Test Metric: {metric:.4f}")
            else:
                pbar.set_postfix_str(f"Test Metrics: {metric}")
                
            for i in range(len(batch['answer'])):
                all_chats.append({
                    'question': batch['question'][i],
                    'prompt': batch_prompts[i],
                    'response': responses[i],
                    'gt_answer':batch['answer'][i],
                    'label':labels[i],
                    'pred':preds[i],

                })
                if log_examples:
                    prompt = batch_prompts[i]
                    response = responses[i]
                    label = labels[i]
                    pred = preds[i]
                    logger.info(f'-------- example {count} --------')
                    
                    logger.info(f'Input:\n{prompt}\n')
                    logger.info(f'Response:\n{response}\n')
                    logger.info(f'Pred: {pred}  Label: {label}  Correct: {pred==label}')
                    count += 1
                    if not isinstance(metric, tuple):
                        logger.info(f"Test Metric: {metric:.4f}")
                    else:
                        logger.info(f"Test Metrics: {metric}")
                    logger.info('-------------------------------')
    
        metric = task.cal_metric(all_preds, all_labels, all_questions)
        logger.info('--------------------------------------------')
        if not isinstance(metric, tuple):
            logger.info(f"Test Metric: {metric:.4f}")
        else:
            logger.info(f"Test Metrics: {metric}")
        logger.info('--------------------------------------------')
        test_metrics.append(metric)
        
    for key in data.keys():
        if len(data[key]) != 0:
            if isinstance(data[key][0], dict):
                for i in range(len(data[key])):
                    data[key][i]['test_metric'] = test_metrics[data[key][i]['id']]
            elif isinstance(data[key][0], list):
                for path_i in range(len(data[key])):
                    for i in range(len(data[key][path_i])):
                        data[key][path_i][i]['test_metric'] = test_metrics[data[key][path_i][i]['id']]
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'data.json'), 'w') as f:
            json.dump(data, f, indent=4)  
    return test_metrics
    
def eval(task_name, exp_name, eval_prompt, prompt_file, path, node, seed, log_examples=True, post_instruction=False, train_size=None, val_size=None, eval_size=None, test_size=None, eval_model='gpt-3.5-turbo', batch_size=1, temperature=0, max_tokens=2048, log_dir='logs/prompt_test_logs', api_key=None, openai_key_txt_file='api_keys.txt', data_dir=None, **kwargs) :
    '''
        evaluate cur_prompt on task testing dataset
    '''
    if prompt_file is not None:
        with open(prompt_file, 'r') as file:
            eval_prompt = file.read()
            
    log_dir = os.path.join(log_dir, exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    openai_key_config(api_key, openai_key_txt_file)
    logger = create_logger(log_dir, f'{exp_name}', log_mode='test')
    if data_dir is not None:
        task = get_task(task_name)(seed=seed, train_size=train_size, val_size=val_size, eval_size=eval_size, test_size=test_size, post_instruction=post_instruction, data_dir=data_dir)
    else:
        task = get_task(task_name)(seed=seed, train_size=train_size, val_size=val_size, eval_size=eval_size, test_size=test_size, post_instruction=post_instruction)
   
    test_dataloader = task.get_dataloader('test', batch_size=1)
    
    build_forward_prompts_func = task.build_forward_prompts_completion
    if eval_model in COMPLETION_MODELS:
        batch_forward_func = batch_forward_completion
    elif eval_model in CHAT_COMPLETION_MODELS:
        batch_forward_func = batch_forward_chatcompletion
    elif eval_model in OPENSOURCE_MODELS:
        batch_forward_func  = batch_forward_flant5
    else:
        raise ValueError(f"Model {eval_model} not supported.")
    
    logger.info(f'exp_name: {exp_name}')
    logger.info(f'path: {path}  node: {node}')
    logger.info(f'eval_prompt: {eval_prompt}\n')
    logger.info(f'prompt example: \n{build_forward_prompts_func(["example_question"], eval_prompt)[0]}\n')
    
    logger.info(f'Task: {task.task_name}, test set size: {int(len(test_dataloader)*batch_size)}, shuffle: {False}')
    logger.info(f'post_instruction: {post_instruction}')
    logger.info(f'Eval model: {eval_model}, batch_size: {batch_size}, temperature: {temperature}, max_tokens:{max_tokens}')
    
    all_questions = []
    all_labels = []
    all_preds = []
    all_chats = []
    
    pbar = tqdm(test_dataloader, leave=False)
    count = 0
    for batch in pbar:
        batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
        responses = batch_forward_func(batch_prompts, model=eval_model, temperature=temperature, max_tokens=max_tokens)
        preds = task.batch_clean_responses(responses)
        labels = task.clean_labels(batch['answer'])
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_questions.extend(batch['question'])
        
        metric = task.cal_metric(all_preds, all_labels, all_questions)
        if not isinstance(metric, tuple):
            pbar.set_postfix_str(f"Test Metric: {metric:.4f}")
        else:
            pbar.set_postfix_str(f"Test Metrics: {metric}")
            
        for i in range(len(batch['answer'])):
            all_chats.append({
                'question': batch['question'][i],
                'prompt': batch_prompts[i],
                'response': responses[i],
                'gt_answer':batch['answer'][i],
                'label':labels[i],
                'pred':preds[i],

            })
            if log_examples:
                prompt = batch_prompts[i]
                response = responses[i]
                label = labels[i]
                pred = preds[i]
                logger.info(f'-------- example {count} --------')
                
                logger.info(f'Input:\n{prompt}\n')
                logger.info(f'Response:\n{response}\n')
                logger.info(f'Pred: {pred}  Label: {label}  Correct: {pred==label}')
                count += 1
                if not isinstance(metric, tuple):
                    logger.info(f"Test Metric: {metric:.4f}")
                else:
                    logger.info(f"Test Metrics: {metric}")
                logger.info('-------------------------------')
    
    metric = task.cal_metric(all_preds, all_labels, all_questions)
    logger.info('--------------------------------------------')
    if not isinstance(metric, tuple):
        logger.info(f"Test Metric: {metric:.4f}")
    else:
        logger.info(f"Test Metrics: {metric}")
    logger.info('--------------------------------------------')
    return {
            'metric':metric,
            'all_chats':all_chats, 
            'all_labels':all_labels, 
            'all_preds':all_preds
            }

def eval_instruction(task, eval_prompt, eval_model='text-davinci-003', batch_size=10, temperature=0, max_tokens=512) :
    '''
        evaluate cur_prompt on task testing dataset
    '''
    print(f'Task: {task.task_name}: {task.task_type}, testset_size: {task.test_set_size}')
    print(f'eval_prompt: {eval_prompt}')
    print(f'Eval model: {eval_model}, batch_size: {batch_size}, temperature: {temperature}')
    test_data = task.get_test_set()
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    all_labels = []
    all_preds = []
    all_chats = []
    build_forward_prompts_func = task.build_forward_prompts_completion
    if eval_model in COMPLETION_MODELS:
        batch_forward_func = batch_forward_completion
    elif eval_model in CHAT_COMPLETION_MODELS:
        batch_forward_func = batch_forward_chatcompletion
    elif eval_model in OPENSOURCE_MODELS:
        batch_forward_func  = batch_forward_flant5
    else:
        raise ValueError(f"Model {eval_model} not supported.")
    
    for batch in tqdm(test_dataloader):
        batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
        responses = batch_forward_func(batch_prompts, model=eval_model, temperature=temperature, max_tokens=max_tokens)
        preds = task.batch_clean_responses(responses)
        labels = task.clean_labels(batch['answer'])
        for i in range(batch_size):
            all_chats.append({
                'question': batch['question'][i],
                'prompt': batch_prompts[i],
                'response': responses[i],
                'gt_answer':batch['answer'][i],
                'label':labels[i],
                'pred':preds[i],

            })
        all_preds.extend(preds)
        all_labels.extend(labels)
 
    acc = (np.array(all_labels) == np.array(all_preds)).sum()*1.0 / len(all_preds)
    print(f'Accuracy: {acc:.2f}')
    return {
            'acc':acc,
            'all_chats':all_chats, 
            'all_labels':all_labels, 
            'all_preds':all_preds
            }


def eval_instruction_with_loader(task, eval_prompt, dataloader, model='gpt-3.5-turbo', temperature=0, max_tokens=1024, record_outputs=True):
    '''
        evaluate cur_prompt on task testing dataset
    '''
    
    build_forward_prompts_func = task.build_forward_prompts_completion
    if model in COMPLETION_MODELS:
        batch_forward_func = batch_forward_completion
    elif model in CHAT_COMPLETION_MODELS:
        batch_forward_func = batch_forward_chatcompletion
    elif model in OPENSOURCE_MODELS:
        batch_forward_func  = batch_forward_flant5
    else:
        raise ValueError(f"Model {model} not supported.")
    
    all_questions = []
    all_labels = []
    all_preds = []
    all_prompts = []
    all_responses = []
    eval_output = {}
    
    pbar = tqdm(dataloader, leave=False)
    for batch in pbar:
        batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
        responses = batch_forward_func(batch_prompts, model=model, temperature=temperature, max_tokens=max_tokens)
        preds = task.batch_clean_responses(responses)
        labels = task.clean_labels(batch['answer'])
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_questions.extend(batch['question'])
        if record_outputs:
            all_prompts.extend(batch_prompts)
            all_responses.extend(responses)
        metric = task.cal_metric(all_preds, all_labels, all_questions)
        if not isinstance(metric, tuple):
            pbar.set_postfix_str(f"Test Metric: {metric:.4f}")
        else:
            pbar.set_postfix_str(f"Test Metrics: {metric}")
    
    if record_outputs:
        eval_output['model_inputs'] =  all_prompts
        eval_output['model_responses'] =  all_responses
        eval_output['preds'] =  all_preds
        eval_output['labels'] =  all_labels
    eval_output['correct'] =  task.cal_correct(all_preds, all_labels)    
    metric = task.cal_metric(all_preds, all_labels, all_questions)
    return metric, eval_output
    

def print_test_result(test_result):
    acc = test_result['acc']
    print(f'Acc: {acc}')
    for i, chat in enumerate(test_result['all_chats']):
        print(f'{i} --------------------------')
        for k in chat:
            if k in ['label', 'pred']:
                print(f'\033[91m{k}: \033[0m{chat[k]}')
            else:
                print(f'\033[91m{k}:\n\033[0m{chat[k]}')
