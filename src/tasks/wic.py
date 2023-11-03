# define task prompts for various datasets
import re
from datasets import load_dataset
from .base_task import BaseTask

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'wic', 
                 task_discription = "word in context",
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

        self.answer_format_prompt = 'A:'
    
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('super_glue','wic')
        question_format = "Word: {word}\nSentence1: {sentence1}\nSentence2: {sentence2}\nQ: Does the word has the same meaning the these two sentences?\nOptions:\n- True\n- False\n"
        new_dataset = dict(train=[], test=[])
        for example in dataset['train']:
            question_str = question_format.format(
                word=example['word'], 
                sentence1 = example['sentence1'],
                sentence2=example['sentence2']
                )
            new_dataset['train'].append(dict(question=question_str, answer='true' if example['label']==1 else 'false'))
        for example in dataset['validation']:
            question_str = question_format.format(
                word=example['word'], 
                sentence1 = example['sentence1'],
                sentence2=example['sentence2']
                )
            new_dataset['test'].append(dict(question=question_str, answer='true' if example['label']==1 else 'false'))
            
        return new_dataset
    
    def transform_format(self, data):
        return data
    
    def clean_response(self, response):
        clean_pattern = r"\btrue\b|\bfalse\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]
    
        return "N/A: format error."
    