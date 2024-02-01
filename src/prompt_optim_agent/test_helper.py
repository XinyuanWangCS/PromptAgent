import os
from tqdm import tqdm
import time
from datetime import  timedelta
from .world_model.prompts import *
from tasks import *
from .utils import *
from .language_model import get_language_model

def eval(
    base_model_type,
    base_model_name,
    temperature = 0.0,
    task_name = None,
    eval_prompt=None, 
    prompt_file=None, 
    post_instruction=False,
    seed=None,   
    train_size=None, 
    eval_size=None, 
    test_size=None, 
    batch_size=1, 
    log_dir='logs/prompt_test_logs', 
    log_examples=True,
    data_dir=None, 
    api_key = None,
    **kwargs):
    
    '''
        Evaluate prompt on task testing dataset
    '''
    
    if prompt_file is not None:
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as file:
                eval_prompt = file.read()
        else:
            raise ValueError(f"prompt_file path doesn't exist: {prompt_file}")
            
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = create_logger(log_dir, task_name, log_mode='test')
    
    task = get_task(task_name)(
        train_size=train_size, 
        eval_size=eval_size, 
        test_size=test_size, 
        seed=seed, 
        post_instruction=post_instruction, 
        data_dir=data_dir)
    
    test_dataloader = task.get_dataloader('test', batch_size=1)
    
    base_model = get_language_model(base_model_type)(
            model = base_model_name,
            temperature = temperature,
            api_key=api_key,
        )
    build_forward_prompts_func = task.build_forward_prompts_completion
    batch_forward_func = base_model.batch_forward_func
    
    logger.info(f'task_name: {task_name}')
    logger.info(f'eval_prompt: {eval_prompt}\n')
    logger.info(f'testset size: {int(len(test_dataloader)*batch_size)}, shuffle: {False}, post_instruction: {post_instruction}')
    logger.info(f'prompt example: \n{build_forward_prompts_func(["example_question"], eval_prompt)[0]}\n')

    all_questions = []
    all_labels = []
    all_preds = []
    all_chats = []
    start_time = time.time()
    
    pbar = tqdm(test_dataloader, leave=False)
    count = 0
    for batch in pbar:
        batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
        responses = batch_forward_func(batch_prompts)
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
    end_time = time.time()
    exe_time = str(timedelta(seconds=end_time-start_time)).split('.')[0]
    logger.info(f'\nDone! Excution time: {exe_time}')
    return {
            'metric':metric,
            'all_chats':all_chats, 
            'all_labels':all_labels, 
            'all_preds':all_preds
            }

def eval_instruction_with_loader(task, eval_prompt, base_model, dataloader,  temperature=0, record_outputs=True):
    '''
        evaluate cur_prompt on task testing dataset
    '''
    
    build_forward_prompts_func = task.build_forward_prompts_completion
    batch_forward_func = base_model.batch_forward_func
    
    all_questions = []
    all_labels = []
    all_preds = []
    all_prompts = []
    all_responses = []
    eval_output = {}
    
    pbar = tqdm(dataloader, leave=False)
    for batch in pbar:
        batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
        responses = batch_forward_func(batch_prompts)
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
    
