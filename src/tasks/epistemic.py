# define task prompts for various datasets
import re
from datasets import load_dataset
from .base_task import BaseTask

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None,  
                 
                 task_name = "epistemic",
                 task_description = "task from bigbench",
                 data_dir='',  
                 seed=None, 
                 
                 post_instruction=True, 
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
                        )

        self.answer_format_prompt = "\nA:"
    
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
            target_scores = example['target_scores']
            
            # Generating options and answer
            options = list(target_scores.keys())
            
            answer = [option.lower() for i, option in enumerate(options) if target_scores[option] == 1][0]

            options_str = 'Options:\n- entailment\n- non-entailment'
            question_str = "Identify the relation between the following premises and hypotheses, choosing from the options 'entailment' or 'non-entailment'.\n"+question+"\n"+options_str+'\n'
            
            # Formatting the output
            formatted_example = {
                'question': question_str,
                'answer': answer
            }
            examples.append(formatted_example)
        
        return examples
    
    def clean_response(self, response):
        clean_pattern = r"\b(entailment|non-entailment)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]
    
        return "N/A: format error."
    