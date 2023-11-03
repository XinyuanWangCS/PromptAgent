# define task prompts for various datasets
import re
from datasets import load_dataset
from .base_task import BaseTask
import random

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'gad', 
                 task_discription = "",
                 seed=None, 
                 
                 post_instruction=True, 
                 **kwargs):
        super().__init__(task_name=task_name,
                         task_discription=task_discription,
                         seed=seed, 
                         train_size=train_size, 
                         eval_size=eval_size,
                         test_size = test_size, 
                         post_instruction=post_instruction,
                         )

        self.answer_format_prompt = '\nA:'
        for split in self.dataset:
            count = 0
            for example in self.dataset[split]:
                count += example['answer'] == 'yes'
            print(f'{split}: {count}/{len(self.dataset[split])}')
        
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('bigbio/gad')
        answer_dict = {1:'yes', 0:'no'}
        question_format = "Text: {sentence}\nOptions:\n- Yes\n- No"
        new_dataset = dict(train=[], test=[])

        for example in dataset['train']:
            question_str = question_format.format(
                sentence=example['sentence'], 
                )
            new_dataset['train'].append(dict(question=question_str, answer=answer_dict[example['label']]))

        for example in dataset['test']:
            question_str = question_format.format(
                sentence=example['sentence'], 
                )
            new_dataset['test'].append(dict(question=question_str, answer=answer_dict[example['label']]))


        return new_dataset
    
    def transform_format(self, data):
        return data
    
    def clean_response(self, response):
        clean_pattern = r"\b(yes|no)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]
    
        return "N/A: format error."
    