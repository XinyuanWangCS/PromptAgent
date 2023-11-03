# define task prompts for various datasets
import re
from datasets import load_dataset
from .base_task import BaseTask

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'subjective', 
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

        self.answer_format_prompt = ''
    
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('SetFit/subj')
        question_format = "Text: {text}\nIs the preceding text objective or subjective?\nOptions:\n- Objective\n- Subjective"
        new_dataset = dict(train=[], test=[])
        for example in dataset['train']:
            question_str = question_format.format(
                text=example['text'], 
                )
            new_dataset['train'].append(dict(question=question_str, answer=example['label_text']))
        for example in dataset['test']:
            question_str = question_format.format(
                text=example['text'], 
                )
            new_dataset['test'].append(dict(question=question_str, answer=example['label_text']))
            
        return new_dataset
    
    def transform_format(self, data):
        return data
    
    def clean_response(self, response):
        clean_pattern = r"\b(objective|subjective)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]
    
        return "N/A: format error."
    