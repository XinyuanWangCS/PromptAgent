# define task prompts for various datasets
import re
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import random
import string
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

class BaseTask():
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None, 
                 
                 task_name = 'base_task',
                 data_dir=None, 
                 seed=None,  
                 post_instruction=False, 
                 TaskDataset=BaseDataset,
                 option_num=5, 
                 **kwargs):
        
        """
            Base class for each tasks.
            
            args:
                option_num: max number of options in the task
                post_instruction: True: answer + prompt | False: prompt + answer
                data_dir: dir of task data's json file
                
            The BaseTask is designed for direct asnwer matching tasks, 
            single choice selection tasks, multi-choice selection tasks.
            
            If requiring new tasks form, or different selection tasks,
            several parts need to be implemented:
            
                function: cal_correct
                function: cal_metric
                function: clean_labels
                function: clean_response
                property: self.answer_format_prompt
                property: option_num
                function: build_forward_prompts_completion (optional)
                
            These two need to be implemented if your data is not organised
            as the requirement of load_task_dataset
                function: transform_format (optional)
                function: load_task_dataset (optional)
                
                
        """
        self.task_name = task_name    
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
        """
            answer_format_prompt: 
                It is appended after the task question to help the model extract the prediction.
                It should match your prediction extraction method in function "clean_response".
        """
        
    def load_task_dataset(self, data_dir):
        """
        <Task Specific>
        This is a default function for loading task dataset from json files. It can be re-implemented in the task.py files.
        
        The output dataset can be either a list of question answer pairs or a dict with a default train-test split:
            all examples: 
                [{'question':question, 'answer':answer}]
            or
            default split: 
                {'train':[{'question':question, 'answer':answer}], 'test':[{'question':question, 'answer':answer}]}
        """
        dataset = self._load_json_file(data_dir)
        
        examples = []
        for example in dataset['examples']:
            question = example['question']
            answer = example['answer']

            formatted_example = {
                'question': question,
                'answer': answer
            }
            examples.append(formatted_example)
            
        return examples
    
    def transform_format(self, dataset):
        """
        <task specific>
        This function is to transform the dataset question's format for 
        a better input form for the base_model.
        For example, for each data point:
        "question": Who is better? 
        transform_format 
        -> Who is better? Options: A. Allen B.Jack
        
        It can be re-implemented in the task.py files.
        
        Do nothing if your "question" in the dataset can be directly fed to 
        the base_model to make predictions.
        """
        return dataset
    
    def cal_correct(self, preds, labels, data_type = "str"):
        '''
        <task specific>
        The function of comparing the predictions and labels.
        
        data_type: str | set
            str: preds, labels are List(str)
            set: preds, labels are List(set)
            
        Every time a batch data is sampled and predicted, by comparing with
        the labels, PromptAgent collect the errors.
        Function called at: prompt_optim_agent/world_model/gradient_descent.py line 54
        
        '''
        if data_type == "set":
            comparisons = []
            for p, l in zip(preds, labels):
                if p == l:
                    comparisons.append(1)
                else:
                    comparisons.append(0)
                return comparisons
        else:
            return list(np.array((np.array(preds) == np.array(labels))).astype(int))
    
    def cal_metric(self, preds, labels, questions=None):
        '''
        <task specific>
        Calculate the evaluation metric, e.g. Accuracy, F1 score.
        "question" is for NCBI calculating F1 score.
        return a number / tuple of metrics
        
        This function is for calculating the reward of MCTS.
        '''
        correct = self.cal_correct(preds=preds, labels=labels)
        return np.mean(correct)
    

    def clean_labels(self, labels):
        '''
        <task specific>
        Transfer the form of the task ground-truth answers to List(set) 
        or List(str) that fit the input requirement of function "cal_correct"
        
        Do nothing if the data is alreadly loaded that way.
        '''
        return labels
    
    def clean_response(self, response):
        '''
        <task specific>
        Extract the prediction from base_model's response,
        so that the output form batch_clean_responses fit
        the input requirement of function "cal_correct"
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
        """
            Extract preds from responses.
        """
        if not isinstance(responses, list):
            responses = list(responses)
        batch_answers = []
        for response in responses:
            batch_answers.append(self.clean_response(response))
        return batch_answers
    
    def build_forward_prompts_completion(self, questions, cur_propmt):
        '''
        Optional: <task specific>
        The format of combining question and prompts.
        '''
        prompts = []
        if self.post_instruction:
            for question in questions:
                prompts.append(f'{question}\n{cur_propmt}')
        else:
            for question in questions:
                prompts.append(f'{cur_propmt}\n{question}\n{self.answer_format_prompt}')#
        
        return prompts
    
    
    def get_split_task_dataset(self, origin_dataset, train_size=None, eval_size=150, test_size=0, seed=None, base_shuffle=True):
        """
        Split the dataset into training set, eval set and testing set.
        Support either a list of question answer pairs or a dict with a default train-test split.
        train_set and eval_set may have overlap.
        """
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
    
    def split_dict_dataset(self, dataset, train_size=None, eval_size=150, test_size=0, seed=None, base_shuffle=True):
        train_set = dataset['train']

        test_set = []
        if 'test' in dataset.keys():
            test_set = dataset['test']
        elif 'validation' in dataset.keys():
            test_set = dataset['validation']
        elif 'valid' in dataset.keys():
            test_set = dataset['valid']
        
        if base_shuffle and seed is not None:
            if seed is not None:
                print(f'shuffle dataset seed {seed}')
                random.seed(seed)
            random.shuffle(train_set)
        
        eval_set = train_set[-eval_size:]
        if train_size is not None:
            train_set = train_set[:train_size]
        test_set = test_set[:test_size]
        return train_set, eval_set, test_set
    
    def split_list_dataset(self, dataset, train_size=None, eval_size=150, test_size=0, seed=None, base_shuffle=True):
        if base_shuffle and seed is not None:
            if seed is not None:
                print(f'shuffle dataset seed {seed}')
                random.seed(seed)
            random.shuffle(dataset)
        
        test_set = dataset[:test_size]
        dataset = dataset[test_size:]

        if train_size is not None:
            train_set = dataset[:train_size]
        else:
            train_set = dataset
        eval_set = dataset[-eval_size:]
        
        return train_set, eval_set, test_set
    
    def _load_json_file(self, data_dir):
        if not (os.path.exists(data_dir) and data_dir.endswith('.json')):
            raise ValueError(f'json file {data_dir} does not exist.')
        
        with open(data_dir, 'r') as file:
            data = json.load(file)
        return data
    
    def build_task_dataset(self, dataset, TaskDataset=None):
        return TaskDataset(dataset=dataset)
    
    def get_dataloader(self, split, batch_size, shuffle=False):
        if self.TaskDataset is None:
            self.TaskDataset = BaseDataset
            
        if split not in self.dataset.keys():
            raise ValueError(f'Dataset split {split} does not exist.')
        
        dataset = self.build_task_dataset(self.dataset[split], TaskDataset=self.TaskDataset)
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_dataset_size(self, split='test'):
        return len(self.dataset[split])
    
    def process_gradient_descent_output(self, gradient_descent_output):
        return gradient_descent_output