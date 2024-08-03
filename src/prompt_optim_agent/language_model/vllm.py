from vllm import LLM, SamplingParams
from typing import Optional, List
class VllmModel():
    def __init__(
        self,
        model_name: str,
        temperature: float,
        gpu_ids: Optional[List[int]] = None,
        **kwargs):

        self.temperature = temperature
        self.do_sample = True if temperature != 0 else False
        self.model_name = model_name
        
        if gpu_ids is None:
            gpu_ids = [0]
        self.llm = LLM(model=model_name, gpu_ids=gpu_ids)
        
    def batch_forward_func(self, batch_prompts):
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=2048,
            repetition_penalty=1.2
        )
        responses = self.llm.generate(batch_prompts, sampling_params)
        return [response.outputs[0].text for response in responses]

    def generate(self, input):
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=1024,
            repetition_penalty=1.2
        )
        responses = self.llm.generate([input], sampling_params)
        return responses[0].outputs[0].text
            