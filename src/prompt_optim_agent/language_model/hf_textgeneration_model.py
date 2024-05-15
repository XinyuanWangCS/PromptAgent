from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HFTextGenerationModel():
    def __init__(
        self,
        model_name: str,
        temperature: float,
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
        **kwargs):
        
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.do_sample = True if temperature != 0 else False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, truncate=True, padding=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        
    def batch_forward_func(self, batch_prompts):
        model_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        model_output = self.model.generate(**model_inputs, do_sample=self.do_sample, temperature=self.temperature,
        max_new_tokens=1024)
        responses = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        print(responses)
        return responses

    def generate(self, input):
        model_inputs = self.tokenizer([input], return_tensors="pt", padding=True, truncation=True).to(self.device)
        model_output = self.model.generate(**model_inputs, do_sample=self.do_sample, temperature=self.temperature,
        max_new_tokens=1024)
        responses = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        print("HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(len(responses))
        print(responses)
        return responses[0]
        
    
        
        