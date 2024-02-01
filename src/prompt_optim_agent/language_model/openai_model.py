import openai
import time

CHAT_COMPLETION_MODELS = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314', 'gpt-4-1106-preview']
COMPLETION_MODELS =  ['text-davinci-003', 'text-davinci-002','code-davinci-002']

class OpenAIModel():
    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float,
        **kwargs):
        
        self.model = model
        self.temperature = temperature
        self._api_key_config(api_key)
        
        if model in COMPLETION_MODELS:
            self.batch_forward_func = self.batch_forward_completion
            self.generate = self.gpt_completion
        elif model in CHAT_COMPLETION_MODELS: 
            self.batch_forward_func = self.batch_forward_chatcompletion
            self.generate = self.gpt_chat_completion
        else:
            raise ValueError(f"Model {model} not supported.")
        
    def _api_key_config(self, api_key):
        # set up key from command
        if api_key is not None:
            print(f'Set OpenAI API: {api_key}')
            openai.api_key = api_key.strip()
        else:
            raise ValueError(f"api_key error: {api_key}")
    
    def batch_forward_chatcompletion(self, batch_prompts):
        """
        Input a batch of prompts to openai chat API and retrieve the answers.
        """
        responses = []
        for prompt in batch_prompts:
            response = self.gpt_chat_completion(prompt=prompt)
            responses.append(response)
        return responses
    
    def batch_forward_completion(self, batch_prompts):
        """
        Input a batch of prompts to openai completion API and retrieve the answers.
        """
        gpt_output = self.gpt_completion(
            prompt=batch_prompts)['choices']
        responses = []
        for response in gpt_output:
            responses.append(response)
        return responses
    
    def gpt_chat_completion(self, prompt):
        messages = [{"role": "user", "content": prompt},]
        backoff_time = 1
        while True:
            try:
                return openai.ChatCompletion.create(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature)['choices'][0]['message']['content'].strip()
            except openai.error.OpenAIError:
                print(openai.error.OpenAIError, f' Sleeping {backoff_time} seconds...')
                time.sleep(backoff_time)
                backoff_time *= 1.5

    def gpt_completion(self, prompt):
        backoff_time = 1
        while True:
            try:
                return openai.Completion.create(
                    prompt=prompt,
                    model=self.model,
                    temperature=self.temperature)['text'].strip()
            except openai.error.OpenAIError:
                print(openai.error.OpenAIError, f' Sleeping {backoff_time} seconds...')
                time.sleep(backoff_time)
                backoff_time *= 1.5

        
    
        
        