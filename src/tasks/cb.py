# define task prompts for various datasets
import re
from datasets import load_dataset
from .base_task import BaseTask

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'cb', 
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
        dataset = load_dataset('super_glue','cb')
        answer_dict = {0:'entailment', 1:'contradiction', 2:'neutral'}
        question_format = "Premise: {premise}\nHypothesis: {hypothesis}\nThat is the relationship between the preceding premise and the hypothesis?\nOptions:\n- Contradiction\n- Neutral\n- Entailment"
        new_dataset = dict(train=[], test=[])
        for example in dataset['train']:
            question_str = question_format.format(
                premise=example['premise'], 
                hypothesis = example['hypothesis'],
                )
            new_dataset['train'].append(dict(question=question_str, answer=answer_dict[example['label']]))
        for example in dataset['validation']:
            question_str = question_format.format(
                premise=example['premise'], 
                hypothesis = example['hypothesis'],
                )
            new_dataset['test'].append(dict(question=question_str, answer=answer_dict[example['label']]))
            
        return new_dataset
    
    def transform_format(self, data):
        return data
    
    def clean_response(self, response):
        clean_pattern = r"\b(entailment|contradiction|neutral)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]
    
        return "N/A: format error."
    