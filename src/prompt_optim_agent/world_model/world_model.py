from .gradient_descent import *
from typing import NamedTuple
from ..test_helper import eval_instruction_with_loader
from typing import Generic
from ..search_algo.base_algo import State, Action, Trace, PromptState, EvaluateMetrics
from ..search_algo.mcts import MCTSNode
from ..utils import gpt_chat_completion

class WorldModel(Generic[State, Action]):
    def __init__(self,
                 task,
                 logger,
                 
                 prompt_length_limit,
                 # model
                 pred_model: str,
                 optim_model: str,
                 pred_temperature: float, 
                 optim_temperature: float,
                 
                 eval_methods,
                 max_tokens=2048,
                 num_new_prompts = 2,
                 
                 train_shuffle = True,
                 train_batch_size: int = 5,
                 test_batch_size: int = 1,
                 eval_batch_size: int = 1,
                 **kwargs) -> None:
        
        self.task = task
        self.logger = logger
        self.pred_model = pred_model
        self.pred_temperature=pred_temperature
        self.optim_model = optim_model
        self.optim_temperature = optim_temperature
        self.eval_methods = eval_methods
        self.max_tokens=max_tokens
        self.num_new_prompts = num_new_prompts
        
        self.train_shuffle = train_shuffle
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.eval_batch_size = eval_batch_size
        
        self.train_dataloader = self.task.get_dataloader('train', 
                                                        batch_size=train_batch_size, 
                                                        shuffle=train_shuffle)
        self.train_data_iterator = self._infinite_data_loader(self.train_dataloader)
        
        self.test_dataloader = self.task.get_dataloader('test', 
                                                        batch_size=test_batch_size, 
                                                        shuffle=False)
        self.eval_dataloader = self.task.get_dataloader('eval', 
                                                        batch_size=eval_batch_size, 
                                                        shuffle=False)
        self.gradient_descent = GradientDescent(task=self.task, 
                                                logger=self.logger, 
                                                pred_model=pred_model, 
                                                optim_model=optim_model, 
                                                num_new_prompts = num_new_prompts,
                                                forward_temperature=pred_temperature, 
                                                optim_temperature = optim_temperature,
                                                max_tokens=max_tokens,
                                                prompt_length_limit=prompt_length_limit)
        self.log_vars()
        
    def log_vars(self):
        self.logger.info('----------------- World Model --------------------------')
        ignored_print_vars = ['task', 'logger', 'train_dataloader','train_data_iterator','test_dataloader','eval_dataloader', 'gradient_descent']
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars: continue
            var_value = vars_dict[var_name]
            self.logger.info(f'{var_name} : {var_value}')
        
    def _infinite_data_loader(self, data_loader):
        while True:
            for batch in data_loader:
                yield batch
                
    def get_train_batch(self):
        return next(self.train_data_iterator)
    
    def _get_trajectory_prompts(self, node: MCTSNode):
        trajectory_prompts = []
        temp_node = node
        while True:
            trajectory_prompts.append(temp_node.state.prompt)
            if temp_node.parent is not None:
                temp_node = temp_node.parent
            else:
                break
        return trajectory_prompts[::-1]
                
    
    def _gradient_descent_step(self, node: MCTSNode, batch):
        '''
            state: PromptState
            batch: batch data
        '''
        trajectory_prompts = self._get_trajectory_prompts(node=node)
        helper_data = dict(trajectory_prompts=trajectory_prompts)
        
        gradient_descent_output = self.gradient_descent(batch, node.state.prompt, helper_data) 
        if gradient_descent_output['acc']==-1:
            return [], gradient_descent_output
        
        node.state.update_state_record_with_examples(
            correct=gradient_descent_output['correct'], 
            choice=EvaluateMetrics.CUR_BATCH_ACC
            )
        
        new_nodes = []
        for prompt in gradient_descent_output['optimized_prompts']:
            new_prompt_state = PromptState(prompt=prompt, eval_methods=self.eval_methods)
            
            child_node = MCTSNode(
                state=new_prompt_state, 
                action=gradient_descent_output['gradient'], 
                parent=node)
            new_nodes.append(child_node)
        
        return new_nodes, gradient_descent_output
    
    def step(self, node:MCTSNode, batch):
        new_nodes, gradient_descent_output = self._gradient_descent_step(node=node, batch=batch)
        return new_nodes, gradient_descent_output
    
    def build_root(self, init_prompt):
        init_state = PromptState(prompt=init_prompt, eval_methods=self.eval_methods)
        for method in init_state.eval_methods:
            init_state.state_record[method] = 0.0
        node = MCTSNode(state=init_state, action=None, parent=None)
        root_threshold = self.evaluate_node_on_eval_set(node=node)
        node.threshold = root_threshold
        return node
    
    
    
    def evaluate_child_node(self, node:MCTSNode):
        '''if EvaluateMetrics.PARENT_BATCH_ACC in self.eval_methods:
            node.prompt_state.update_state_record_item(EvaluateMetrics.PARENT_BATCH_ACC, 
                                                  evaluate_helper_data['gradient_descent_output']['acc'])
          '''  
        if EvaluateMetrics.PARENT_EVAL_DATA_ACC in self.eval_methods:
            node.state.update_state_record_item(
                EvaluateMetrics.PARENT_EVAL_DATA_ACC, 
                node.parent.state[EvaluateMetrics.EVAL_DATA_ACC])
            
        '''if EvaluateMetrics.CUR_BATCH_ACC in self.eval_methods:
            batch = evaluate_helper_data['batch']
            forward_output = self.gradient_descent.forward(batch=batch, cur_prompt=prompt_state.prompt) 
            node.prompt_state.update_state_record_with_examples(
                correct=forward_output['correct'], 
                 choice=EvaluateMetrics.CUR_BATCH_ACC
                 )'''
        
        if EvaluateMetrics.EVAL_DATA_ACC in self.eval_methods:
            evaludate_output = self.evaluate_prompt(prompt=node.state.prompt)
            node.state.update_state_record_with_examples(
                correct=evaludate_output['correct'], 
                choice=EvaluateMetrics.EVAL_DATA_ACC
                )
            node.threshold = node.state.get_threshold()

        '''if EvaluateMetrics.SELF_EVAL_SCORE in self.eval_methods:
            self.self_evaluate(prompt_state=node.state, evaluate_helper_data=evaluate_helper_data)'''
    
    def sample_single_state_trajectory(self, prompts, depth):
        new_prompt_states = []
        for _ in range(depth):
            new_prompt = self.gradient_descent.sample_next_prompt_from_trajectory(prompts=prompts)
            prompts.append(new_prompt)
            new_prompt_state = PromptState(prompt=new_prompt, eval_methods=self.eval_methods)
            new_prompt_states.append(new_prompt_state)
        return new_prompt_states
        
    def evaluate_node_on_eval_set(self, node:MCTSNode):
        
        evaludate_output = self.evaluate_prompt(prompt=node.state.prompt)
        node.state.update_state_record_with_examples(
            correct=evaludate_output['correct'], 
            choice=EvaluateMetrics.EVAL_DATA_ACC
            )
        self.logger.info(f'Evaluate node: {node.id}\tEval acc: {node.state.state_record[EvaluateMetrics.EVAL_DATA_ACC]}')
        return node.state.state_record[EvaluateMetrics.EVAL_DATA_ACC]
        
    def self_evaluate(self, prompt_state:PromptState, evaluate_helper_data, norm=5.0):
        self_eval_score, _ = self.gradient_descent.self_evaluate(prompt_state=prompt_state,
                                                       evaluate_helper_data=evaluate_helper_data)
        self_eval_score /= norm
        prompt_state.update_state_record_item(item_name='self_eval_score', item_value=self_eval_score)
        return self_eval_score
    
    def test_prompt(self, prompt):
        metric, eval_output = eval_instruction_with_loader(task=self.task, 
                                           eval_prompt=prompt,
                                           dataloader=self.test_dataloader,
                                           model=self.pred_model,
                                           temperature=self.pred_temperature,
                                           max_tokens=self.max_tokens)
        return metric, eval_output
    
    
    def evaluate_prompt(self, prompt):
        self.logger.info(f'prompt: {prompt}')
        metric, eval_output = eval_instruction_with_loader(task=self.task, 
                                           eval_prompt=prompt,
                                           dataloader=self.eval_dataloader,
                                           model=self.pred_model,
                                           temperature=self.pred_temperature,
                                           max_tokens=self.max_tokens)
        correct = eval_output['correct']
        evaludate_output = dict(
            metric=metric,
            correct = correct,
            acc = np.mean(correct)
        )
        
        return evaludate_output
    
    def is_terminal(self, state: PromptState): #TODO
        if state.get_score() < 0:
            return True
        return False
    
    def _optim_model_completion(self, prompt):
        messages = [{"role": "user", "content": prompt},]
        response = gpt_chat_completion(messages=messages, model=self.optim_model, temperature=self.optim_temperature)['choices'][0]['message']['content'].strip()
        return response
    
    @staticmethod
    def init_state(init_prompt) -> PromptState:
        return PromptState(prompt=init_prompt)