from .gradient_descent import *
from ..test_helper import eval_instruction_with_loader
from typing import Generic
from ..search_algo.base_algo import State, Action
from ..search_algo.beam_search import BeamNode

class BeamSearchWorldModel(Generic[State, Action]):
    def __init__(
        self,
        task,
        logger,
        
        # model
        base_model: str,
        optim_model: str,
        
        prompt_length_limit:int,
        num_new_prompts = 3,
        train_shuffle = True,
        train_batch_size: int = 5,
        test_batch_size: int = 1,
        eval_batch_size: int = 1,
        **kwargs
        ) -> None:
        
        self.task = task
        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model

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
                                                base_model=base_model, 
                                                optim_model=optim_model, 
                                                num_new_prompts = num_new_prompts,
                                                prompt_length_limit=prompt_length_limit)
    def _infinite_data_loader(self, data_loader):
        while True:
            for batch in data_loader:
                yield batch
                
    def get_train_batch(self):
        return next(self.train_data_iterator)
    
    def _get_trajectory_prompts(self, node: BeamNode):
        trajectory_prompts = []
        temp_node = node
        while True:
            trajectory_prompts.append(temp_node.prompt)
            if temp_node.parent is not None:
                temp_node = temp_node.parent
            else:
                break
        return trajectory_prompts[::-1]
                
    
    def _gradient_descent_step(self, node: BeamNode, batch):
        '''
            state: PromptState
            batch: batch data
        '''
        trajectory_prompts = self._get_trajectory_prompts(node=node)
        helper_data = dict(trajectory_prompts=trajectory_prompts)
        
        gradient_descent_output = self.gradient_descent(batch, node.prompt, helper_data) 
        if gradient_descent_output['acc']==-1:
            return [], gradient_descent_output
        
        new_nodes = []
        for prompt in gradient_descent_output['optimized_prompts']:
            child_node = BeamNode(
                prompt=prompt, 
                action=gradient_descent_output['gradient'], 
                parent=node)
            new_nodes.append(child_node)
        
        return new_nodes, gradient_descent_output
    
    def step(self, node:BeamNode, batch):
        new_nodes, gradient_descent_output = self._gradient_descent_step(node=node, batch=batch)
        return new_nodes, gradient_descent_output
    
    def build_root(self, init_prompt):
        node = BeamNode(prompt=init_prompt, action=None, parent=None)
        self.evaluate_node(node=node)
        return node
    
    def evaluate_node(self, node:BeamNode):
        self.logger.info(f'node: {node.id}\tprompt: {node.prompt}')
        evaludate_output = self.evaluate_prompt(prompt=node.prompt)
        node.eval_metric = evaludate_output['metric']
        self.logger.info(f'eval_metric: {node.eval_metric}')
    
    def test_prompt(self, prompt):
        metric, eval_output = eval_instruction_with_loader(task=self.task, 
                                           eval_prompt=prompt,
                                           dataloader=self.test_dataloader,
                                           model=self.base_model,
                                           )
        return metric, eval_output
    
    
    def evaluate_prompt(self, prompt):
        metric, eval_output = eval_instruction_with_loader(task=self.task, 
                                           eval_prompt=prompt,
                                           dataloader=self.eval_dataloader,
                                           model=self.base_model,
                                           )
        correct = eval_output['correct']
        evaludate_output = dict(
            metric=metric,
            correct = correct,
            acc = np.mean(correct)
        )
        
        return evaludate_output

