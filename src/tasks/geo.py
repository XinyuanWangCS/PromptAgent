# define task prompts for various datasets
from .base_task import BaseDataset, BaseTask
import re
import string

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None,  
                 
                 task_name = "bigbench",
                 task_description = "task from bigbench",
                 data_dir='',  
                 seed=None, 
                 
                 post_instruction=True, 
                 TaskDataset=BaseDataset,
                 option_num=5, 
                 **kwargs):
        self.options = {}
        super().__init__(
                        task_name = task_name,  
                        task_description = task_description, 
                        data_dir=data_dir,
                        seed = seed,
                        train_size = train_size,
                        eval_size=eval_size,
                        test_size = test_size,
                        post_instruction = post_instruction,
                        TaskDataset=TaskDataset,
                        option_num=option_num,
                        )
        self.answer_format_prompt = "At the end show the answer option bracketed between <answer> and </answer>."+"\n"
        
    def load_task_dataset(self, data_dir):
        '''
            <task specific>
        '''
        json_data = self._load_json_file(data_dir)
        self.task_description = json_data['description']
        max_example = max(json_data['examples'], key=lambda x: len(x['target_scores']))
        self.option_num = len(max_example['target_scores'])
        return json_data
    
    def transform_format(self, data):
        original_examples = data['examples']

        examples = []
        # Extracting input and target scores
        for example in original_examples:
            question = example['input']
            if 'task_prefix' in data.keys():
                task_prefix = data['task_prefix'].strip()
                question = task_prefix+"\n"+question
            
            target_scores = example['target_scores']
            
            # Generating options and answer
            options = list(target_scores.keys())
            answer = [chr(65 + i) for i, option in enumerate(options) if target_scores[option] == 1][0]
            for i, option in enumerate(options):
                self.options[option.lower()] = f'{chr(65 + i)}'
            options = [f'({chr(65 + i)}) {option}' for i, option in enumerate(options)]
            options_str = 'Options:\n'+'\n'.join(options)
            question_str = question+'\n'+options_str+'\n'
            
            # Formatting the output
            formatted_example = {
                'question': question_str,
                'answer': answer
            }
            examples.append(formatted_example)
        
        return examples
    
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
        pattern_str = '|'.join([re.escape(option) for option in self.options])
        backup_match = re.findall(pattern_str, response, re.IGNORECASE)
        if backup_match:
            return self.options[backup_match[-1].lower()]
        answer = re.search(r"\([" + letters + r"]\)", match[-1])
        if answer is not None:
            return answer.group(0)[1].upper()
        answer = re.search(r"[" + letters + r"]", match[-1])
        if answer is None:
            return 'N/A: Format error'
        return answer[0].upper()