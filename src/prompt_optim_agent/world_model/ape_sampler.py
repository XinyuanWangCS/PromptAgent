from .prompts import *
from .prompts.ape_prompts import *
from ..utils import *
import re
import numpy as np

class APESampler():
    def __init__(self, 
                 task, 
                 
                 pred_model, 
                 optim_model,
                 forward_temperature=0, 
                 optim_temperature = 0,
                 max_tokens=2048,
                 logger = None,
                 num_new_prompts = 1,
                 ):

        self.task = task
        self.pred_model = pred_model
        self.optim_model = optim_model
        self.forward_temperature = forward_temperature
        self.optim_temperature = optim_temperature
        self.max_tokens = max_tokens
        self.logger = logger
        self.num_new_prompts = num_new_prompts
        
        self.forward_generation_tempelate = forward_generation_tempelate 
        self.example_tempelate = example_tempelate
        self.resample_tempelate = resample_tempelate
    
    def _get_examples_str(self, batch):
        batch_example_strings = []
        for i in range(len(batch['question'])):
            batch_example_strings.append(
                self.example_tempelate.format(
                    input=batch['question'][i],
                    output=batch['answer'][i]
                    )
                )
        return ''.join(batch_example_strings)

    def _optim_model_completion(self, model_input):
        messages = [{"role": "user", "content": model_input},]
        response = gpt_chat_completion(messages=messages, 
                                       model=self.optim_model, 
                                       temperature=self.optim_temperature)['choices'][0]['message']['content'].strip()
        return response
        
    def sample(self, prompt):
        resample_prompt = self.resample_tempelate.format(instruction=prompt)
        resample_prompt = self._optim_model_completion(resample_prompt)
        return resample_prompt
    
    def init_sample(self, batch):
        example_str = self._get_examples_str(batch=batch)
        input_prompt = self.forward_generation_tempelate.format(examples=example_str)
        init_sample_prompt = self._optim_model_completion(input_prompt)
        return init_sample_prompt