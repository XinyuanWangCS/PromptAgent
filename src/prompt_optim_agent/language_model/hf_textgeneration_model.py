from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HFTextGenerationModel():
    def __init__(
        self,
        model_name: str,
        temperature: float,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs):
        
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.do_sample = True if temperature != 0 else False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, truncate=True, padding=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.generate = self.batch_forward_func
        
    def batch_forward_func(self, batch_prompts):
        for p in batch_prompts:
            print(len(p))
        model_inputs = self.tokenizer(batch_prompts, return_tensors="pt").to(self.device)
        model_output = self.model.generate(**model_inputs, do_sample=self.do_sample, temperature=self.temperature)
        responses = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        print(responses)
        return responses


        
    
        
        