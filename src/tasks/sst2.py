# define task prompts for various datasets
import re
from datasets import load_dataset
from .base_task import BaseTask

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'ss2', 
                 task_discription = "sentiment analysis",
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

        self.answer_format_prompt = 'Is it:\n(A) positive\n(B) negative\n'
        self.options = {"positive":"A", "negative": "B"}
    
    def build_forward_prompts_completion(self, questions, cur_propmt):
        prompts = []
        if self.post_instruction:
            for question in questions:
                prompts.append(f'{question}\n{cur_propmt}')
        else:
            for question in questions:
                prompts.append(f'{cur_propmt}\nSentence: {question}\n{self.answer_format_prompt}')#
        return prompts
    
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('glue','sst2')
        
        new_dataset = dict(train=[], test=[])
        for example in dataset['train']:
            new_dataset['train'].append(dict(question=example['sentence'], answer='A' if example['label']==1 else 'B'))
        for example in dataset['validation']:
            new_dataset['test'].append(dict(question=example['sentence'], answer='A' if example['label']==1 else 'B'))
            
        return new_dataset

    def transform_format(self, data):
        return data
    
    
    def clean_response(self, response):
        clean_pattern = r"\((A|B)\)"
        match = re.findall(clean_pattern, response)
        if len(match) != 0:
            return match[-1]
        
        clean_pattern = "positive|negative"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return self.options[match[-1]]
        
        clean_pattern = "a|b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1].upper()
        return "N/A: format error."
    