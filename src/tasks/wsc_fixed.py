# define task prompts for various datasets
import re
from datasets import load_dataset
from .base_task import BaseTask

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'wsc_fixed', 
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

        self.answer_format_prompt = 'A:'
    
    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        # converts text as done in T5.
        text = example['text']
        text = self._mark_span(
            text, example['span1_text'], example['span1_index'], '*')
        # Compensate for 2 added "words" added in previous step.
        span2_index = example['span2_index'] + 2 * \
            int(example['span1_index'] < example['span2_index'])
        text = self._mark_span(text, example['span2_text'], span2_index, '#')
        return text
    
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('super_glue','wsc.fixed')
        question_format = "Sentence: {text}\nQ: Do * {span1_text} * and # {span2_text} # refer to the same thing?\nOptions:\n- True\n- False\n"
        new_dataset = dict(train=[], test=[])
        for example in dataset['train']:
            text = self.preprocessor(example=example)
            question_str = question_format.format(
                span1_text=example['span1_text'], 
                span2_text = example['span2_text'],
                text=text
                )
            new_dataset['train'].append(dict(question=question_str, answer='true' if example['label']==1 else 'false'))
        for example in dataset['validation']:
            text = self.preprocessor(example=example)
            question_str = question_format.format(
                span1_text=example['span1_text'], 
                span2_text = example['span2_text'],
                text=text
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
    