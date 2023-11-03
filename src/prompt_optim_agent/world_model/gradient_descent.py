from .prompts import *
from .prompts.world_model_prompts import *
from ..utils import *
import re
import numpy as np

class GradientDescent():
    def __init__(self, 
                 task, 
                 prompt_length_limit,
                 
                 pred_model, 
                 optim_model,
                 forward_temperature=0, 
                 optim_temperature = 0,
                 max_tokens=2048,
                 print_log = True,
                 logger = None,
                 num_new_prompts = 1,
                 
                 gradient_prompt_tempelate = None,
                 optimize_prompt_tempelate = None,
                 error_example_template = None):

        self.task = task
        self.pred_model = pred_model
        self.optim_model = optim_model
        self.forward_temperature = forward_temperature
        self.optim_temperature = optim_temperature
        self.max_tokens = max_tokens
        self.logger = logger
        self.print_log = print_log if logger is not None else False
        self.num_new_prompts = num_new_prompts
        self.prompt_length_limit = prompt_length_limit
        
        self.use_correct_examples = False
        
        self.optimize_prompt_tempelate = optimize_prompt_tempelate_gamma2 if optimize_prompt_tempelate is None else optimize_prompt_tempelate
        self.gradient_prompt_tempelate = gradient_prompt_tempelate_gamma0 if gradient_prompt_tempelate is None else gradient_prompt_tempelate
        self.error_example_template    = error_example_template_gamma1 if error_example_template is None else error_example_template
        
        self.self_eval_prompt = self_eval_prompt_template_v6
        self.example_temlate = example_template_v0
        
        self.correct_example_template = correct_example_template_beta0
        self.correct_gradient_prompt_tempelate = correct_gradient_prompt_tempelate_alpha1
        self.trajectory_sample_template = trajectory_sample_template_gamma0
        
        self._build_forward_prompts_func = task.build_forward_prompts_completion
        if pred_model in COMPLETION_MODELS:
            self._batch_forward_func = batch_forward_completion
        elif pred_model in CHAT_COMPLETION_MODELS: 
            self._batch_forward_func = batch_forward_chatcompletion
        elif pred_model in OPENSOURCE_MODELS:
            self._batch_forward_func = batch_forward_flant5
        else:
            raise ValueError(f"Model {pred_model} not supported.")
        

    def forward(self, batch, cur_prompt):
        batch_size = len(batch['question'])
        batch_prompts =self._build_forward_prompts_func(batch['question'], cur_prompt)
        responses = self._batch_forward_func(batch_prompts, model=self.pred_model, temperature=self.forward_temperature, max_tokens=self.max_tokens)
        preds = self.task.batch_clean_responses(responses)
        
        labels = self.task.clean_labels(batch['answer'])
        correct = self.task.cal_correct(preds, labels)
        if np.mean(correct) == 1:
            return dict(acc=-1)
        
        batch_logs = []
        for i in range(batch_size):
            batch_logs.append({
                'cur_prompt': cur_prompt,
                'question': batch['question'][i],
                'model_input': batch_prompts[i],
                'gt_answer':batch['answer'][i],
                'model_response': responses[i],
                'label':labels[i],
                'pred':preds[i],
                })
        
        forward_output = {
            'cur_prompt': cur_prompt,
            'correct':correct,
            'examples':batch_logs, 
            'acc':np.mean(correct)
            }
        
        if self.print_log:
            log_str = forward_log_tempelate_v1.format(cur_prompt=cur_prompt,
                                                   batch_prompts=batch_prompts,
                                                   responses=responses,
                                                   preds=preds,
                                                   labels=labels,
                                                   correct=forward_output['correct'],
                                                   acc=forward_output['acc'])

            self.logger.info(log_str)
        return forward_output
    
    def _get_batch_examples_str(self, batch):
        batch_example_strings = []
        for i in range(len(batch['question'])):
            batch_example_strings.append(self.example_temlate.format(index=i+1,
                                                                     question=batch['question'][i],
                                                                     label=batch['answer'][i]))
        return ''.join(batch_example_strings)
    
    def _clean_self_eval_score(self, response):
        return re.findall(r'\d+', response)[-1]
    
    def self_evaluate(self, prompt_state, evaluate_helper_data):
        
        cur_prompt = prompt_state.prompt
        parent_prompt = evaluate_helper_data['gradient_descent_output']['cur_prompt']
        batch_examples_str = self._get_batch_examples_str(evaluate_helper_data['action'])
        error_example_str = self._get_error_examples(evaluate_helper_data['gradient_descent_output'])
        correct_example_str = self._get_correct_examples(evaluate_helper_data['gradient_descent_output'])
        self_eval_prompt = self.self_eval_prompt.format(
            parent_prompt = parent_prompt,
            cur_prompt = cur_prompt,
            examples_str=batch_examples_str,
            correct_str = correct_example_str,
            error_str = error_example_str
            )
        
        response = self._optim_model_completion(self_eval_prompt)
        score = float(self._clean_self_eval_score(response))
        if self.print_log:
            log_str = self_evaluate_log_tempelate.format(self_eval_prompt=self_eval_prompt,
                                                        response=response,
                                                        self_eval_score=f'{score}')
            self.logger.info(log_str)
        self_eval_output = dict(self_eval_prompt=self_eval_prompt,
                                response=response,
                                score=score)
        return score, self_eval_output
    
    def _get_error_examples(self, forward_output): 
        error_examples = []
        count = 0
        for i, example in enumerate(forward_output['examples']):
            if forward_output['correct'][i]==0:
                count += 1
                error_examples.append(self.error_example_template.format(
                    index=count, 
                    question=example['model_input'],
                    label=example['label'], 
                    response=example['model_response'],
                    prediction=example['pred']))
            elif forward_output['correct'][i]==1:
                continue
            else:
                raise ValueError(f'_get_error_examples: invalid correct number {i} {forward_output}.')
        error_string = ''.join(error_examples)
        return error_string
    
    def _get_correct_examples(self, forward_output): 
        correct_examples = []
        count = 0
        for i, example in enumerate(forward_output['examples']):
            if forward_output['correct'][i]==1:
                count += 1
                correct_examples.append(self.correct_example_template.format(
                    index=count, 
                    question=example['model_input'],
                    label=example['label'], 
                    response=example['model_response'],
                    prediction=example['pred']))
            elif forward_output['correct'][i]==0:
                continue
            else:
                raise ValueError(f'_get_correct_examples: invalid correct number {i} {forward_output}.')
        correct_string = ''.join(correct_examples)
        return correct_string

    def _optim_model_completion(self, model_input):
        messages = [{"role": "user", "content": model_input},]
        response = gpt_chat_completion(messages=messages, 
                                       model=self.optim_model, 
                                       temperature=self.optim_temperature)['choices'][0]['message']['content'].strip()
        return response

    def _build_prompt_trajectory_str(self, prompts):
        prompt_path_str = ""
        prompt_path_str_tempelate = "({index}) {prompt}\n"
        for i, prompt in enumerate(prompts):
            prompt_path_str += prompt_path_str_tempelate.format(index=i,prompt=prompt)
        return prompt_path_str
    
    def sample_next_prompt_from_trajectory(self, prompts):
        trajectory_str = self._build_prompt_trajectory_str(prompts=prompts)
        optim_model_input = self.trajectory_sample_template.format(trajectory_str=trajectory_str)
        response = self._optim_model_completion(optim_model_input)
        optimized_prompt = self._clean_optim_response(response)
        return optimized_prompt[0]
        
    def cal_gradient(self, cur_prompt, error_string, correct_string=None):
        correct_gradient = None
        if self.use_correct_examples:
            correct_gradient_prompt = self.correct_gradient_prompt_tempelate.format(cur_prompt=cur_prompt, 
                                                                    correct_string=correct_string)
            correct_gradient = self._optim_model_completion(correct_gradient_prompt)
        
        gradient_prompt = self.gradient_prompt_tempelate.format(cur_prompt=cur_prompt, 
                                                                error_string=error_string)
        gradient = self._optim_model_completion(gradient_prompt)
        
        if self.print_log:
            if self.use_correct_examples:
                log_str = correct_gradient_log_tempelate.format(gradient_prompt=correct_gradient_prompt,
                                                        gradient=correct_gradient)

                self.logger.info(log_str)
            
            log_str = gradient_log_tempelate.format(gradient_prompt=gradient_prompt,
                                                    gradient=gradient)

            self.logger.info(log_str)

        return gradient, correct_gradient

    def _clean_optim_response(self, optim_response):
        pattern = r'<START>(.*?)<END>'
        matches = re.findall(pattern=pattern, string=optim_response, flags=re.DOTALL)
        for i, m in enumerate(matches):
            matches[i] = m.strip()
        return matches

    def optimize(self, cur_prompt, error_str, gradient, trajectory_prompts, correct_string, correct_gradient, steps_per_gradient):
        optimize_prompt = self.optimize_prompt_tempelate.format(cur_prompt=cur_prompt, 
                                                                correct_string=correct_string,
                                                                correct_gradient=correct_gradient,
                                                                error_str=error_str, 
                                                                gradient=gradient, 
                                                                trajectory_prompts=trajectory_prompts,
                                                                prompt_length_limit=self.prompt_length_limit,
                                                                steps_per_gradient=steps_per_gradient)
        response = self._optim_model_completion(optimize_prompt)
        optimized_prompt = self._clean_optim_response(response)
        if self.print_log:
            log_str = optimize_log_tempelate.format(optimize_prompt=optimize_prompt,
                                                    response=response,
                                                    optimized_prompt=optimized_prompt)
            self.logger.info(log_str)

        return optimized_prompt
    
    def gradient_descent_step(self, cur_prompt, batch, helper_data):
        
        self.logger.info(f'cur_prompt: {cur_prompt}')

        gradient_descent_output = self.forward(batch=batch, cur_prompt=cur_prompt)
        if gradient_descent_output['acc']==-1:
            return gradient_descent_output
        
        if self.use_correct_examples:
            correct_string = self._get_correct_examples(gradient_descent_output)
        else:
            correct_string = None
            
        error_string = self._get_error_examples(gradient_descent_output)
        
        gradient, correct_gradient = self.cal_gradient(cur_prompt=cur_prompt, error_string=error_string, correct_string=correct_string)
        
        trajectory_prompts = helper_data['trajectory_prompts']
        trajectory_prompts = self._build_prompt_trajectory_str(trajectory_prompts)
        
        optimized_prompts = self.optimize(cur_prompt=cur_prompt, 
                                          error_str=error_string, 
                                          gradient=gradient, 
                                          trajectory_prompts=trajectory_prompts,
                                          correct_string=correct_string, 
                                          correct_gradient=correct_gradient, 
                                          steps_per_gradient=self.num_new_prompts)
        
        gradient_descent_output['error_string'] = error_string
        gradient_descent_output['gradient'] = gradient
        gradient_descent_output['optimized_prompts'] = optimized_prompts
        return gradient_descent_output
    
    def __call__(self, batch, cur_prompt, helper_data=None):
        gradient_descent_output = self.gradient_descent_step(cur_prompt=cur_prompt, batch=batch, helper_data=helper_data)
        return gradient_descent_output