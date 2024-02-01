import ctranslate2
import transformers
import torch

class CTranslateModel():
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        model_path: str = "/home/xinyuan/workspace/download_models/Mistral-7B-Instruct-v0.2_int8_float16",
        temperature: float = 0,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs):
        
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_length = max_length
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, truncate=True, padding=True)
        self.model = ctranslate2.Generator(model_path, device=device)
        
    def batch_forward_func(self, batch_prompts):
        responses = []
        for prompt in batch_prompts:
            responses.append(self.generate(prompt=prompt))
        return responses
    
    def generate(self, prompt):
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(prompt))
        results = self.model.generate_batch([tokens], sampling_temperature=0, max_length=self.max_length, include_prompt_in_result=False)
        output = self.tokenizer.decode(results[0].sequences_ids[0])
        return output
        


        
    
        
        