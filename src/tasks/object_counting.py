# define task prompts for various datasets
from .base_task import BaseDataset, BaseTask
import re
import string

number_to_word_dict = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21
}

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None,  
                 
                 task_name = "object_counting",
                 task_description = "object_counting",
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
        self.answer_format_prompt = ""
        
    def load_task_dataset(self, data_dir):
        '''
            <task specific>
        '''
        json_data = self._load_json_file(data_dir)
        self.task_description = json_data['description']
        return json_data
    
    def transform_format(self, data):
        original_examples = data['examples']

        examples = []
        # Extracting input and target scores
        for example in original_examples:
            question = example['input']
            answer = example['target']
            formatted_example = {
                'question': question,
                'answer': answer[-1]
            }
            examples.append(formatted_example)
        return examples
    
    def clean_response(self, response):
        integer_pattern = r"\d+"
        matches = re.findall(integer_pattern, response)
        if len(matches) != 0:
            return str(matches[-1])
        extended_pattern = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one)\b"
        matches = re.findall(extended_pattern, response)
        if len(matches) != 0:
            return str(number_to_word_dict[matches[-1]])
        else:
            return "N/A: format error."
