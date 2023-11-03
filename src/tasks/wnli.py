# define task prompts for various datasets
import re
from datasets import load_dataset
from .base_task import BaseTask

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'wnli', 
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
    
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('glue','wnli')
        answer_dict = {1:'true', 0:'false'}
        question_format = "Sentence1: {sentence1}\nSentence2: {sentence2}\nQ: Does the preceding second sentence entails the first one?\nOptions:\n- True\n- False"
        new_dataset = dict(train=[], test=[])
        for example in dataset['train']:
            question_str = question_format.format(
                sentence1=example['sentence1'], 
                sentence2 = example['sentence2'],
                )
            new_dataset['train'].append(dict(question=question_str, answer=answer_dict[example['label']]))
        for example in dataset['validation']:
            question_str = question_format.format(
                sentence1=example['sentence1'], 
                sentence2 = example['sentence2'],
                )
            new_dataset['test'].append(dict(question=question_str, answer=answer_dict[example['label']]))
            
        return new_dataset
    
    def transform_format(self, data):
        return data
    
    def clean_response(self, response):
        clean_pattern = r"\b(true|false)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]
    
        return "N/A: format error."
    