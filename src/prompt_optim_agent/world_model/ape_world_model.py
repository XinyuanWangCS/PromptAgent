from .ape_sampler import APESampler
from ..test_helper import eval_instruction_with_loader
from ..search_algo.ape import APENode
import numpy as np

class APEWorldModel():
    def __init__(
        self,
        task,
        logger,

        # model
        pred_model: str,
        optim_model: str,
        pred_temperature: float, 
        optim_temperature: float,

        max_tokens=2048,
        num_new_prompts = 3,
        eval_set_size = 50,
        
        train_shuffle = True,
        train_batch_size: int = 5,
        test_batch_size: int = 1,
        eval_batch_size: int = 1,
        **kwargs
    ) -> None:
        
        self.task = task
        self.logger = logger
        self.pred_model = pred_model
        self.optim_model = optim_model
        self.pred_temperature=pred_temperature
        self.num_new_prompts = num_new_prompts
        self.eval_batch_size = eval_batch_size
        self.eval_set_size = eval_set_size
        if not isinstance(self.pred_temperature, float):
            raise ValueError('temperature is not a float')
        self.optim_temperature = optim_temperature
        self.max_tokens=max_tokens
        
        self.train_dataloader = self.task.get_dataloader('train', 
                                                        batch_size=train_batch_size, 
                                                        shuffle=train_shuffle)
        self.train_data_iterator = self._infinite_data_loader(self.train_dataloader)
        
        self.test_dataloader = self.task.get_dataloader('test', 
                                                        batch_size=test_batch_size, 
                                                        shuffle=False)
        self.eval_dataloader = self.task.get_random_dataloader(size=self.eval_set_size, batch_size=self.eval_batch_size, shuffle=False)
        self.samplar = APESampler(
            task=self.task, 
            logger=self.logger, 
            pred_model=pred_model, 
            optim_model=optim_model, 
            num_new_prompts = num_new_prompts,
            forward_temperature=pred_temperature, 
            optim_temperature = optim_temperature,
            max_tokens=max_tokens)
    
    def _infinite_data_loader(self, data_loader):
        while True:
            for batch in data_loader:
                yield batch
                
    def get_train_batch(self):
        return next(self.train_data_iterator)
    
    def update_eval_dataloader(self):
        self.logger.info('Resample eval set.')
        self.eval_dataloader = self.task.get_random_dataloader(size=self.eval_set_size, batch_size=self.eval_batch_size, shuffle=False)
    
    def init_sample(self, batch):
        new_nodes = []
        for i in range(self.num_new_prompts):
            new_prompt = self.samplar.init_sample(batch=batch)
            
            new_nodes.append(
                APENode(prompt=new_prompt, parent=None)
            )
            self.logger.info(f'node: {new_nodes[-1].id} prompt: {i}\n{new_prompt}')

        return new_nodes
    
    def sample(self, node:APENode):
        new_nodes = []
        for _ in range(self.num_new_prompts):
            new_prompt = self.samplar.sample(prompt=node.prompt)
            new_nodes.append(
                APENode(prompt=new_prompt, parent=node)
            )
        return new_nodes
    
    def evaluate_ape_nodes(self, nodes):
        for node in nodes:
            self.logger.info(f'eval node: {node.id}\nprompt: {node.prompt}')
            evaludate_output = self.evaluate_prompt(prompt=node.prompt)
            node.eval_metric = evaludate_output['metric']
            self.logger.info(f'eval metric: {node.eval_metric}')
            self.logger.info(f'------------------')
    
    def test_prompt(self, prompt):
        metric, eval_output = eval_instruction_with_loader(task=self.task, 
                                           eval_prompt=prompt,
                                           dataloader=self.test_dataloader,
                                           model=self.pred_model,
                                           temperature=self.pred_temperature,
                                           max_tokens=self.max_tokens)
        return metric, eval_output
    
    
    def evaluate_prompt(self, prompt):
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
