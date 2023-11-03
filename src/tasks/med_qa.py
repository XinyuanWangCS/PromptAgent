# define task prompts for various datasets
import re
from datasets import load_dataset
from .base_task import BaseTask
import string
class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'med_qa', 
                 task_discription = "domain",
                 seed=None, 
                 
                 post_instruction=True, 
                 **kwargs):
        self.options = {}
        super().__init__(task_name=task_name,
                         task_discription=task_discription,
                         seed=seed, 
                         train_size=train_size, 
                         eval_size=eval_size,
                         test_size = test_size, 
                         post_instruction=post_instruction,
                         )

        self.answer_format_prompt = 'At the end show the answer bracketed between <answer> and </answer>.'
    
    # def load_task_dataset(self, **kwargs):
    #     dataset = load_dataset("bigbio/med_qa")
    #     choices = dataset['choices']
    #     answer_dict = {choices[i]: chr(65 + i) for i in range(len(choices))}
    #     options_str = "\n".join(["- " + choice for choice in choices])
    #     question_format = "{question}\nOptions:\n" + options_str
    #     new_dataset = dict(train=[], test=[])
    #     for example in dataset['train']:
    #         question_str = question_format.format(
    #             question=example['question'], 
    #             )
    #         new_dataset['train'].append(dict(question=question_str, answer=answer_dict[example['label']]))
    #     for example in dataset['test']:
    #         question_str = question_format.format(
    #             question=example['question'], 
    #             )
    #         new_dataset['test'].append(dict(question=question_str, answer=answer_dict[example['label']]))
            
    #     return new_dataset
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset("bigbio/med_qa")
        new_dataset = dict(train=[], test=[])

        def process_split(split_name):
            for example in dataset[split_name]:
                # Extract choices and answer key from the example
                choices = [option['value'] for option in example['options']]
                for i, option in enumerate(choices):
                    self.options[option.lower()] = f'{chr(65 + i)}'
                answer_key = example['answer_idx']
                
                answer_dict = {option['value']: option['key'] for option in example['options']}
                
                # Construct the question format with letters in front of options
                options_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
                question_format = "{question}\nOptions:\n" + options_str
                question_str = question_format.format(question=example['question'])
                
                # Append to the new dataset
                new_dataset[split_name].append(dict(question=question_str, answer=answer_dict[example['answer']]))

        process_split('train')
        process_split('test')

        return new_dataset


    
    def transform_format(self, data):
        return data
    

    def clean_response(self, response):
        letters = string.ascii_uppercase[:self.option_num] + string.ascii_lowercase[:self.option_num]
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"
        match = re.findall(clean_pattern, response.lower())
        if len(match) == 0 or not match[-1].strip():
            pattern_str = '|'.join([re.escape(option) for option in self.options])
            backup_match = re.findall(pattern_str, response, re.IGNORECASE)

            if backup_match:
                return self.options[backup_match[-1].lower()]
            else:
                return 'N/A: Format error'

        answer = re.search(r"\([" + letters + r"]\)", match[-1])
        if answer is not None:
            return answer.group(0)[1].upper()
        answer = re.search(r"[" + letters + r"]", match[-1])
        if answer is None:
            return 'N/A: Format error'
        return answer[0].upper()