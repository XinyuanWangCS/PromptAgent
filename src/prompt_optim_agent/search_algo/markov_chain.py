from tqdm import tqdm
from prompt_optim_agent.test_helper import *
from prompt_optim_agent.utils import *
from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader

from .base_algo import SearchAlgo

class MarkovChain(SearchAlgo):
    def __init__(self, 
                 task,
                 world_model, 
                 logger=None, 
                 
                 train_shuffle = True,
                 train_batch_size: int = 8,
                 
                 seed=0, 
                 print_log=True,
                 eval_last_batch=True,
                 test_every_step=True,
                 skip_all_correct_batch=True,
                 **kwargs) -> None:
        self.task = task
        self.world_model = world_model
        self.dataloader = self.task.get_dataloader('train', 
                                                    batch_size=train_batch_size, 
                                                    shuffle=train_shuffle)

        self.states = []
        self.logger = logger
        self.print_log = print_log if logger is not None else False
        self.seed = seed
        self.eval_last_batch =eval_last_batch
        self.test_every_step = test_every_step
        self.skip_all_correct_batch = skip_all_correct_batch

    def search(self, init_state, iteration_num = 10):
        result_dict = defaultdict(list)
        
        self.states.append(init_state)
        cur_state = init_state
        iter_bar = range(iteration_num)
        iter_bar = tqdm(iter_bar) #if not self.print_log else iter_bar
        data_iterator = iter(self.dataloader)

        result_dict['state'].append(init_state)
        if self.test_every_step:
            test_acc = eval_cur_prompt(self.task, eval_prompt=cur_state, 
                                        eval_model=self.world_model.gradient_descent.pred_model, 
                                        temperature=self.world_model.gradient_descent.forward_temperature, 
                                        logger=self.logger)
            result_dict['test_acc'].append(test_acc)

        for i in iter_bar:
            self.logger.info(f'\n-----------------------------------------------------')
            self.logger.info(f'---------------  Iteration: {i}  --------------------')
            batch = next(data_iterator)
            optimized_prompts, grad_result_dict = self.world_model.gradient_descent(batch, cur_state) 
            
            if self.skip_all_correct_batch and grad_result_dict['acc']==1:
                self.logger.info(f'\n-----------------------------------------------------')
                self.logger.info('all correct: skip updating cur_prompt')
                self.logger.info(f'\n-----------------------------------------------------\n')
            else:
                cur_state = optimized_prompts[0] #TODO: beam with reward

            # log batch result
            result_dict['forward_acc'].append(grad_result_dict['acc'])
            result_dict['forward_correct'].append(grad_result_dict['correct'])
            result_dict['state'].append(cur_state)
            self.states.append(cur_state)

            if self.eval_last_batch:
                self.logger.info('---------------   Eval last batch   ---------------')
                self.logger.info(f'new prompt: {cur_state}')
                forward_output = self.world_model.gradient_descent.forward(batch, cur_state)
                result_dict['re_forward_acc'].append(forward_output['acc']) 
                result_dict['re_forward_correct'].append(forward_output['correct']) 
                
            if self.test_every_step:
                test_acc = eval_cur_prompt(self.task, eval_prompt=cur_state, 
                                           eval_model=self.world_model.gradient_descent.pred_model, 
                                           temperature=self.world_model.gradient_descent.forward_temperature, 
                                           logger=self.logger)
                result_dict['test_acc'].append(test_acc)

        self.log_result_dict(result_dict=result_dict)
        return self.states, result_dict#, self.states # best states

def eval_cur_prompt(task, eval_prompt, eval_model, temperature, batch_size=10, max_tokens=2048, logger=None) :
    '''
        evaluate cur_prompt on task testing dataset
    '''
    test_data = task.get_test_set()
    if logger is None:
        print(f'eval_prompt: {eval_prompt}\nmodel: {eval_model} testset size: {len(test_data)} batch_size: {batch_size}, temperature: {temperature}')
    else:
        logger.info('\n----------------  test  --------------------')
        logger.info(f'eval_prompt: {eval_prompt}\nbatch_size: {batch_size} | temperature: {temperature} | eval_model: {eval_model}')
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
    
    for batch in test_dataloader:
        batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
        responses = batch_forward_func(batch_prompts, model=eval_model, temperature=temperature, max_tokens=max_tokens)
        preds = task.batch_clean_responses(responses)
        labels = task.clean_labels(batch['answer'])
        for i in range(batch_size):
            all_chats.append({
                'question': batch['question'][i],
                'prompt': batch_prompts[i],
                'response': responses[i],
                'pred':preds[i],
                'gt_answer':batch['answer'][i],
                'label':labels[i],
            })
        all_preds.extend(preds)
        all_labels.extend(labels)

    acc = (np.array(all_labels).astype(int) == np.array(all_preds).astype(int)).sum()*1.0 / len(all_preds)
    if logger is None:
        print(f'Accuracy: {acc:.3f}')
        print(f'all_labels: {all_labels}')
        print(f'all_preds: {all_preds}')
    else:
        logger.info(f'Accuracy: {acc:.3f}')
        logger.info(f'all_labels: {all_labels}')
        logger.info(f'all_preds: {all_preds}')
        logger.info('-------------------------------------------')
    return acc

def log_result_dict(self, result_dict):
        self.logger.info('\n--------------  state test result  -----------------')
        if self.test_every_step:
            for i in range(len(result_dict['state'])):
                prompt = result_dict['state'][i]
                test_acc = result_dict['test_acc'][i]
                prompt_len = len(prompt.split(' '))
                self.logger.info(f'{i} | test_acc: {test_acc:.3f}  | length: {prompt_len} | prompt: {prompt.strip()}')
        else:
            for i in range(len(result_dict['state'])):
                prompt = result_dict['state'][i]
                prompt_len = len(prompt.split(' '))
                self.logger.info(f'{i} | length: {prompt_len} | prompt: {prompt.strip()}')

        self.logger.info('\n--------------  batch gd result  -----------------')
        if self.eval_last_batch:
            for i in range(len(result_dict['forward_acc'])):
                prompt_before = result_dict['state'][i]
                prompt_after = result_dict['state'][i+1]
                forward_acc = result_dict['forward_acc'][i]
                forward_correct = result_dict['forward_correct'][i]
                re_forward_acc = result_dict['re_forward_acc'][i]
                re_forward_correct = result_dict['re_forward_correct'][i]
                self.logger.info(f'batch {i}')
                self.logger.info(f'prompt_before: {prompt_before}\nforward_acc   : {forward_acc} correct:{forward_correct}')
                self.logger.info(f'prompt_after: {prompt_after}\nre_forward_acc: {re_forward_acc} correct:{re_forward_correct}')
        else:
            for i in range(len(self.states)):
                state = result_dict['state'][i]
                if isinstance(state, str):
                    prompt = state
                    forward_acc = result_dict['forward_acc'][i]
                    forward_correct = result_dict['forward_correct'][i]
                    self.logger.info(f'batch {i}')
                    self.logger.info(f'prompt: {prompt}\nforward_acc   : {forward_acc} correct:{forward_correct}')
                elif isinstance(state, tuple):
                    for j, prompt in enumerate(state):
                        forward_acc = result_dict['forward_acc'][i][j]
                        forward_correct = result_dict['forward_correct'][i][j]
                        self.logger.info(f'batch {i} state {i}')
                        self.logger.info(f'<{j}> prompt: {prompt}\nforward_acc   : {forward_acc} correct:{forward_correct}')

