from prompt_optim_agent.test_helper import *
from prompt_optim_agent.utils import *

import itertools
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Generic, Optional, List

from .base_algo import SearchAlgo, State, Action
from .action_agent import DataAgent

class APENode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, 
                 prompt: str, 
                 parent: "Optional[APENode]" = None,
                 ):

        self.id = next(APENode.id_iter)
        self.prompt = prompt
        self.test_metric = 0.
        self.eval_metric = 0. 

        self.parent = parent
        self.children: 'Optional[list[APENode]]' = []

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
    
    def to_dict(self):
        if self.parent is None:
            p_id = -1
        else:
            p_id = self.parent.id
        
        return {
            'id': self.id,
            'depth':self.depth,
            'parent':p_id,
            'eval_metric': self.eval_metric,
            'test_metric': self.test_metric,
            'prompt':self.prompt,
        }

class APE(SearchAlgo, Generic[State, Action]):

    def __init__(
        self, 
        task,
        world_model, 
        # dataset
        train_shuffle:bool,
        batch_size: int,
        
        expand_width = 5,
        depth_limit: int = 8,
        filtered_prompt_num = 5,
        # log
        logger=None, 
        log_dir = None,
        test_all_nodes = False,
        **kwargs) -> None:
        
        self.task = task
        self.world_model = world_model
        self.logger = logger
        self.log_dir = log_dir

        self.nodes:List[APENode] = [] 
        self.all_nodes = []
        self.expand_width = expand_width
        self.depth_limit = depth_limit
        
        self.filtered_prompt_num = filtered_prompt_num
        self.test_all_nodes = test_all_nodes


    def search(self, **kwargs):
        nodes = []
        self.logger.info(f'\n---------------------  generate init prompts ------------------------')
        for _ in range(self.expand_width):
            batch = self.world_model.get_train_batch()
            new_nodes = self.world_model.init_sample(batch = batch)
            self.world_model.evaluate_ape_nodes(nodes = new_nodes)
            nodes.extend(new_nodes)
        nodes = sorted(nodes, key=lambda node: self._sort_helper(node.eval_metric), reverse=True)[:self.filtered_prompt_num]
        self.all_nodes.extend(nodes)
        self.nodes = nodes
        
        self.logger.info(f'\n---------------------  generate init prompts ------------------------')
        for i in range(self.depth_limit):
            self.world_model.update_eval_dataloader()
            self.logger.info(f'----------------  iteration {i} ----------------')
            nodes = []
            for node in self.nodes:
                sampled_nodes = self.world_model.sample(node)
                self.world_model.evaluate_ape_nodes(nodes = sampled_nodes)
                nodes.extend(sampled_nodes)
            nodes = sorted(nodes, key=lambda node: self._sort_helper(node.eval_metric), reverse=True)[:self.filtered_prompt_num]
            self.all_nodes.extend(nodes)
            self.nodes = nodes
        
        output = self.prepare_output()
        self.draw(output['all_paths'])
        self.output_to_json(output=output)
        
        return self.nodes, output

    def __call__(self, **kwargs):
        APENode.reset_id()

        nodes, output = self.search()
        
        return nodes, output 

    def _sort_helper(self, metric):
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric
        
    def test_and_log_node(self, node:APENode, eval=False, eval_type='test'):
        if eval:
            if eval_type == 'test':
                test_metric, eval_output = self.world_model.test_prompt(node.prompt)
            else:
                raise ValueError(f'eval_type {eval_type} is not supported.')
            node.test_metric = test_metric
        if node.parent is not None:
            self.logger.info(f'node {node.id}:    parent: {node.parent.id} | depth: {node.depth} | eval: {node.eval_metric} | test: {node.test_metric}\nprompt: {node.prompt}')
        else:
            self.logger.info(f'node {node.id}:    parent: N/A | depth: {node.depth} | eval: {node.eval_metric} | test: {node.test_metric}\nprompt: {node.prompt}')
        self.logger.info(f'---------------------')
            
    def test_and_log_nodes(self, nodes, eval=False):
        for node in nodes:
            self.test_and_log_node(node=node, eval=eval)
    
    def draw(self, paths: List[List[APENode]]):
        offset = np.linspace(-0.1, 0.1, len(paths))
        
        fig, ax = plt.subplots()
        colors = plt.cm.jet(np.linspace(0, 1, len(paths)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for idx, path in enumerate(paths):
            depths = np.array([node.depth for node in path]) + offset[idx]
            
            metrics = [node.test_metric[0] if isinstance(node.test_metric, tuple) else node.test_metric for node in path]
            ids = [node.id for node in path]

            ax.plot(depths, metrics, color=colors[idx], marker='o')  
            for d, r, id_ in zip(depths, metrics, ids):
                ax.annotate(str(id_), (d, r))

        ax.set_title("Test Metric")
        ax.set_xlabel("Depth")
        ax.set_ylabel("Test")
        plt.savefig(os.path.join(self.log_dir, 'test_metric.png'), bbox_inches='tight')
        
        fig, ax = plt.subplots()
        colors = plt.cm.jet(np.linspace(0, 1, len(paths)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for idx, path in enumerate(paths):
            depths = np.array([node.depth for node in path]) + offset[idx]
            eval_metrics = [self._sort_helper(node.eval_metric) for node in path]
            ids = [node.id for node in path]

            ax.plot(depths, eval_metrics, color=colors[idx], marker='o')  
            for d, r, id_ in zip(depths, eval_metrics, ids):
                ax.annotate(str(id_), (d, r))

        ax.set_title("Eval Metric")
        ax.set_xlabel("Depth")
        ax.set_ylabel("Metric")
        plt.savefig(os.path.join(self.log_dir, 'eval_metric.png'), bbox_inches='tight')
    
    def prepare_output(self):
        # test and log nodes
        self.logger.info(f'\n---------------------  test nodes ------------------------')
        self.test_and_log_nodes(nodes=self.nodes, eval=True)
        # prepare output
        paths_nodes = []
        
        for i, node in enumerate(self.nodes):
            path = []
            while node.parent is not None:
                path.append(node)
                node = node.parent
            path = path[::-1]
            paths_nodes.append(path)
            self.logger.info(f'---------------------  path {i} ------------------------')
            self.test_and_log_nodes(path, eval=False)
        print(paths_nodes)
        print('*************')
        

        best_path = sorted(paths_nodes, key=lambda path: self._sort_helper(path[-1].eval_metric), reverse=True)[0]
        best_node = sorted(self.all_nodes, key=lambda node: self._sort_helper(node.eval_metric), reverse=True)[0]

        
        self.logger.info(f'---------------------  best path ------------------------')
        self.test_and_log_nodes(best_path, eval=True)
            
        self.logger.info(f'---------------------  best path node------------------------')
        self.test_and_log_node(best_path[-1], eval=False)
        
        self.logger.info(f'---------------------  best global node------------------------')
        self.test_and_log_node(best_node, eval=True) 
        
        
        return dict(
            all_paths = paths_nodes,
            best_path = best_path,
            best_path_node=[best_path[-1]],
            best_global_node=[best_node]
        )
    
    def output_to_json(self, output):
        data_to_save = {}
        paths = []
        for path in output['all_paths']:
            paths.append([node.to_dict() for node in path])
        data_to_save['all_paths'] = paths
        
        for key in output:
            if key != "all_paths":
                data_to_save[key] = [node.to_dict() for node in output[key]]
        with open(os.path.join(self.log_dir, 'data.json'), 'w') as f:
            json.dump(data_to_save, f, indent=4)