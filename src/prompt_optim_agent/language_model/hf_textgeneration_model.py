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
        
    def batch_forward_func(self, batch_prompts):
        model_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        model_output = self.model.generate(
            **model_inputs, 
            do_sample=self.do_sample,
            temperature=self.temperature,
            max_new_tokens=1024,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        generated_sequences = model_output.sequences[:, model_inputs['input_ids'].size(1):]
        responses = self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        
        return [response.strip() for response in responses]

    def generate(self, input):
        model_inputs = self.tokenizer([input], return_tensors="pt", padding=True, truncation=True).to(self.device)
        model_output = self.model.generate(
            **model_inputs,
            do_sample=self.do_sample,
            temperature=self.temperature,
            max_new_tokens=2048,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        generated_sequence = model_output.sequences[0, model_inputs['input_ids'].size(1):]
        response = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)

        if response == "" or response is None:
            return ""
        return response[0]
            
    
        
        