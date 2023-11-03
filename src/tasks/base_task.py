# define task prompts for various datasets
import re
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import random
from abc import ABC, abstractmethod
import string
import numpy as np

class BaseDataset(Dataset):
    '''
        dataset:
            [{'question':question, 'answer':answer}]
    '''
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

class Base(ABC):
    @abstractmethod
    def load_task_dataset(self):
        '''
        <Task Specific>
        dataset:
            all examples: 
                [{'question':question, 'answer':answer}]
            or
            default split: 
                {'train':[{'question':question, 'answer':answer}], 'test':[{'question':question, 'answer':answer}]}
    '''
        pass
    
    def get_split_task_dataset(self, origin_dataset, seed, train_size, eval_size, test_size, base_shuffle=True):
        if isinstance(origin_dataset, dict):
            train_set, eval_set, test_set = self.split_dict_dataset(
                                                origin_dataset, 
                                                seed=seed, 
                                                train_size=train_size,
                                                eval_size=eval_size,
                                                test_size=test_size,
                                                base_shuffle=base_shuffle
                                                )
        elif isinstance(origin_dataset, list):
            train_set, eval_set, test_set = self.split_list_dataset(
                                                origin_dataset, 
                                                seed=seed, 
                                                train_size=train_size,
                                                eval_size=eval_size,
                                                test_size=test_size,
                                                base_shuffle=base_shuffle
                                                )
        else:
            raise ValueError(f'Dataset type {type(origin_dataset)} is not supported.')
        dataset = dict(train=train_set, eval=eval_set, test=test_set)
        return dataset
    
    def split_dict_dataset(self, dataset, seed, train_size, eval_size, test_size, base_shuffle=True):
        train_set = dataset['train']
        self.all_train_set = dataset['train']
        if 'test' in dataset.keys():
            test_set = dataset['test']
        elif 'validation' in dataset.keys():
            test_set = dataset['validation']
        else:
            raise ValueError('No test / validation split in the dataset dict')
        
        if base_shuffle and seed is not None:
            print(f'shuffle dataset seed {seed}')
            random.seed(seed)
            random.shuffle(train_set)
            random.shuffle(test_set)
        
        eval_set = train_set[-eval_size:]
        train_set = train_set[:train_size]
        test_set = test_set[:test_size]
        return train_set, eval_set, test_set
    
    def split_list_dataset(self, dataset, seed, train_size, eval_size, test_size, base_shuffle=True):
        print(f'dataset size: {len(dataset)}')
        if base_shuffle and seed is not None:
            print(f'shuffle dataset seed {seed}')
            random.seed(seed)
            random.shuffle(dataset)
        
        # Split the dataset
        if test_size is None:
            test_set = dataset[train_size+eval_size:]
            dataset = dataset[:train_size+eval_size]
        else:
            test_set = dataset[:test_size]
            dataset = dataset[test_size:]
        self.all_train_set = dataset
        train_set = dataset[:train_size]
        eval_set = dataset[-eval_size:]
        
        return train_set, eval_set, test_set
    
    def build_task_dataset(self, dataset, TaskDataset=None):
        return TaskDataset(dataset=dataset)
    
    def get_random_dataloader(self, size, batch_size, shuffle=False):
        if self.TaskDataset is None:
            self.TaskDataset = BaseDataset
        
        random.shuffle(self.all_train_set)
        dataset = self.build_task_dataset(self.all_train_set[:size], TaskDataset=self.TaskDataset)
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_dataloader(self, split, batch_size, shuffle=False):
        if self.TaskDataset is None:
            self.TaskDataset = BaseDataset
            
        if split not in self.dataset.keys():
            raise ValueError(f'Dataset split {split} does not exist.')
        
        dataset = self.build_task_dataset(self.dataset[split], TaskDataset=self.TaskDataset)
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    @abstractmethod
    def build_forward_prompts_completion(self, questions, cur_propmt):
        '''
            task specific
        '''
        pass

    @abstractmethod
    def clean_labels(self, answers):
        '''
            task specific
        '''
        return answers
    
    @abstractmethod
    def clean_response(self, response):
        '''
            task specific
        '''
        return response
    
    def batch_clean_responses(self, responses):
        if not isinstance(responses, list):
            responses = list(responses)
        batch_answers = []
        for response in responses:
            batch_answers.append(self.clean_response(response))
        return batch_answers
    
class BaseTask(Base):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None, 
                 
                 task_name = 'base_task',
                 task_description = "Basic task",
                 data_dir=None,  #json file
                 seed=None,  
                 post_instruction=True, 
                 TaskDataset=BaseDataset,
                 option_num=5, 
                 **kwargs):
        super().__init__()
        self.task_name = task_name    
        self.task_discription = task_description
        self.data_dir = data_dir
        self.seed = seed
        self.train_size = train_size
        self.test_size = test_size
        self.eval_size = eval_size
        self.post_instruction = post_instruction
        self.TaskDataset = TaskDataset
        self.option_num = option_num
        
        origin_dataset = self.load_task_dataset(data_dir=data_dir)
        origin_dataset = self.transform_format(origin_dataset)
        self.dataset = self.get_split_task_dataset(origin_dataset=origin_dataset, 
                                                   seed=seed, 
                                                   train_size=train_size, 
                                                   eval_size=eval_size,
                                                   test_size=test_size, 
                                                   base_shuffle=True)
        self.train_size = self.dataset['train']
        self.eval_size = self.dataset['eval']
        self.test_size = self.dataset['test']
        print(f'train_size set: {len(self.train_size)}')
        print(f'eval_size set: {len(self.eval_size)}')
        print(f'test_size set: {len(self.test_size)}')
        self.answer_format_prompt = "At the end show the answer option bracketed between <answer> and </answer>."
    
    def get_split_task_dataset(self, origin_dataset, seed, train_size, eval_size, test_size, base_shuffle=True):
        return super().get_split_task_dataset(
            origin_dataset=origin_dataset, 
            seed=seed, 
            train_size=train_size, 
            eval_size=eval_size,
            test_size=test_size, 
            base_shuffle=base_shuffle)
    
    def load_task_dataset(self, data_dir):
        '''
            <task specific>
        '''
        dataset = self._load_json_file(data_dir)
        return dataset
    
    def _load_json_file(self, data_dir):
        if not (os.path.exists(data_dir) and data_dir.endswith('.json')):
            raise ValueError(f'json file {data_dir} does not exist.')
        
        with open(data_dir, 'r') as file:
            data = json.load(file)
        return data
    
    def get_dataloader(self, split, batch_size, shuffle=False):
        return super().get_dataloader(split, batch_size, shuffle)
    
    def get_dataset_size(self, split='test'):
        return len(self.dataset[split])
    
    def transform_format(self, original_examples):
        '''
            <task specific>
        '''
        examples = []
        for example in original_examples['examples']:
            question = example['question']
            answer = example['answer']

            formatted_example = {
                'question': question,
                'answer': answer
            }
            examples.append(formatted_example)
        
        return examples
    
    def build_forward_prompts_completion(self, questions, cur_propmt):
        '''
            <task specific>
        '''
        prompts = []
        if self.post_instruction:
            for question in questions:
                prompts.append(f'{question}\n{cur_propmt}')
        else:
            for question in questions:
                prompts.append(f'{cur_propmt}\n{question}\n{self.answer_format_prompt}')#
        
        return prompts

    def clean_labels(self, labels):
        '''
            <task specific>
        '''
        return labels
    
    def clean_response(self, response):
        '''
            <task specific>
        '''
        letters = string.ascii_uppercase[:self.option_num] + string.ascii_lowercase[:self.option_num]
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"
        match = re.findall(clean_pattern, response.lower())
        if len(match) == 0:
            return 'N/A: Format error'

        answer = re.search(r"\([" + letters + r"]\)", match[-1])
        if answer is not None:
            return answer.group(0)[1].upper()
        answer = re.search(r"[" + letters + r"]", match[-1])
        if answer is None:
            return 'N/A: Format error'
        return answer[0].upper()
    
    def batch_clean_responses(self, responses):
        return super().batch_clean_responses(responses)
    
    def cal_correct(self, preds, labels):
        '''
            <task specific>
        '''
        return list(np.array((np.array(preds) == np.array(labels))).astype(int))
    
    def cal_metric(self, preds, labels, questions=None):
        '''
            <task specific>
            return 1 number / tuple of metrics
        '''
        correct = self.cal_correct(preds=preds, labels=labels)
        return np.mean(correct)
    
    def process_gradient_descent_output(self, gradient_descent_output):
        return gradient_descent_output