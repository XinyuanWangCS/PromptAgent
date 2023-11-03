# define task prompts for various datasets
import re
from datasets import load_dataset
from .base_task import BaseTask

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'gsm8k', 
                 task_discription = "math reasoning task",
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

        self.answer_format_prompt = 'At the end show the answer bracketed between <answer> and </answer>.'

    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('gsm8k', 'main')
        return dataset

    def transform_format(self, original_examples):
        dataset = dict(train=[], test=[])
        for example in original_examples['train']:
            dataset['train'].append(example)
        for example in original_examples['test']:
            dataset['test'].append(example)
        return dataset
    
    def clean_labels(self, answers):
        labels = []
        for answer in answers:
            answer = answer.split('####')[-1].split('.')[0].strip()
            if ',' in answer:
                answer = answer.replace(',', '')
            labels.append(answer)
        return labels
    
    def clean_response(self, response):
        '''
            Clean one answer from one response
            If response is a list, find the number following "The answer: "
            If response is a string, find the last <a> anything </a>
            Find all the int numbers and return the last one
        '''
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"
        answer = re.findall(clean_pattern, response.lower())
        if len(answer)==0:
            clean_pattern = r"-?\d+(?:,\d+)*(?:\.\d+)?"
            answer = re.findall(clean_pattern, response)
            if len(answer) == 0:
                return 'N/A'
            else:
                answer = answer[-1]
                if ',' in answer:
                    answer = answer.replace(',','')
                if '.' in answer:
                    answer = answer.split('.')[-2]
                return answer
            
        response = answer[-1]
        
        clean_pattern = r"-?\d+(?:,\d+)*(?:\.\d+)?"
        answer = re.findall(clean_pattern, response)
        if len(answer) == 0:
            return 'N/A'
        else:
            answer = answer[-1]
            if ',' in answer:
                answer = answer.replace(',','')
            if '.' in answer:
                answer = answer.split('.')[-2]
        return answer
    