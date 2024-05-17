# The MCTS algorithm code is adapted from Reasoning with Language Model is Planning with World Model
# https://github.com/Ber666/llm-reasoners

import os
import json
import itertools
import numpy as np
from copy import deepcopy
from typing import Generic, Optional
from .base_algo import SearchAlgo, State, Action

class MCTSNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, 
                 prompt: str, 
                 action: Optional[Action], 
                 parent: "Optional[MCTSNode]" = None,
                 ):
        """
        A node in the MCTS search tree

        :param prompt: the current state
        :param action: the action of the last optimization step, 
            i.e., the state transition prompt from parent node to current node
        :param parent: the parent node, None if root of the tree
        """
        self.id = next(MCTSNode.id_iter)

        self.prompt = prompt
        self.action = action
        self.parent = parent
        self.is_terminal = False
        
        self.children: 'Optional[list[MCTSNode]]' = []
        self.cum_rewards: list[float] = []
        self.reward = 0.0
        self.test_metric = -1.0
        self.uct = 0.0
        
        self.visited = 0
        self.expand_times = 0
        
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
    
    def calc_q(self, x):
        return np.mean(x)
    
    def cal_reward(self):
        return self.reward
    
    @property
    def Q(self) -> float:
        if len(self.cum_rewards) == 0:
            return self.reward
        else:
            return self.calc_q(self.cum_rewards)
        
    def to_dict(self):
        return {
            'id': self.id,
            'depth':self.depth,
            'parent':-1 if self.parent is None else self.parent.id,
            'visited':self.visited,
            'expand_times':self.expand_times,
            'q':self.Q,
            'uct':self.uct,
            'prompt': self.prompt,
            'reward': self.reward,
            'test_metric':self.test_metric
        }
        
class MCTS(SearchAlgo, Generic[State, Action]):

    def __init__(
        self, 
        task,
        world_model, 
        
        #mcts arguments
        expand_width = 3,
        w_exp: float = 2.5,
        depth_limit: int = 8,
        min_depth: int = 2,
        iteration_num: int = 12,
        
        # log
        log = True,
        logger=None, 
        log_dir = None,
        **kwargs) -> None:
        
        """
        MCTS search algorithm

        :param task: the specific task
        :param world_model: the MCTS world model for state transition
        :param expand_width: number of batches to be sampled
        :param w_exp: the weight of mcts exploration
        :param depth_limit: the max depth of a single MCTS path
        :param iteration_num: number of MCTS iterations
        :param logger: logger
        :param log_dir: logger directory to save the results
        """
        
        self.task = task
        self.world_model = world_model
        
        self.expand_width = expand_width
        self.depth_limit = depth_limit
        self.w_exp = w_exp
        self.iteration_num = iteration_num
        self.min_depth = min_depth # Apply early stop only when depth is larger than min_depth
        
        self.mcts_threshold = 0.0 # The highest reward node globally
        self.min_threshold = 0.0 # The root node's reward as a min threshold
        
        # output
        self.log = log
        self.logger = logger
        self.log_dir = log_dir
        self.k = 1 # top-k reward nodes
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.nodes:list[MCTSNode] = [] 
        
        self.log_vars()
    

    def simulate_choice(self, x):
        return np.argmax(x)
    
    def increase_threshold(self, threshold):
        if threshold > self.mcts_threshold:
            self.mcts_threshold = threshold
    
    def cal_cum_reward(self, rewards):
        return np.sum(rewards)
    
    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.depth >= self.depth_limit
    
    def early_stop(self, node: MCTSNode):
        return node.reward > self.mcts_threshold and node.depth > self.min_depth
    
    def _is_terminal_with_min_threshold(self, node: MCTSNode):
        if node.parent is None:
            min_threshold = self.min_threshold
        else:
            min_threshold = (self.min_threshold + node.parent.reward) / 2
        return node.reward < min_threshold and node.depth > self.min_depth
    
    def is_terminal_node(self, node: MCTSNode):
        return self._is_terminal_with_depth_limit(node) or self._is_terminal_with_min_threshold(node) or node.is_terminal
    
    def _uct(self, node: MCTSNode) -> float:
        if node.parent is None:
            N_parent = 0
        else:
            N_parent = len(node.parent.cum_rewards)
        return node.Q + self.w_exp * np.sqrt(np.log(N_parent+1) / max(1, len(node.cum_rewards)))

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=self._uct)
    
    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        """
        Selection: 
            From root node, keep selecting child node based on UCT
        """
        
        path = []
        while True:
            path.append(node)
            node.visited += 1
            if len(node.children) == 0 or self.is_terminal_node(node):
                return path

            node = self._uct_select(node)
            if self.log: 
                self.logger.info(f'Select node {node.id}: depth {node.depth}, \
                                 reward: {node.reward:.4f} utc: {self._uct(node=node)}')
    
    def _expand(self, node: MCTSNode):
        """
        Expansion: 
            Sample batches of data and perform state transition on the given node.
            Generate new child nodes and calculate their temporary reward.
        """
        if self.log: self.logger.info(f'Expanding:')
        if self.is_terminal_node(node):
            node.is_terminal = True
            return
        
        if self.log: self.logger.info(f"Expanding: node: {node.id}, depth {node.depth}, reward: {node.reward:.4f}")
        
        i = 0
        node.expand_times += 1
        while i < self.expand_width:
            batch = self.world_model.get_train_batch() #sample batch data
            children, gradient_descent_output = self.world_model.step(node, batch) 
            # optim step: sample new child nodes using one batch
            
            i += 1
            for child_node in children: # There could be multiple children in one optim step (num_new_prompts>1)
                self.world_model.evaluate_child_node(node=child_node)
                child_node.reward = child_node.cal_reward()
                child_node.is_terminal = self.is_terminal_node(child_node)
        
            self.nodes.extend(children)
            node.children.extend(children)
        
        if self.log:
            for child in node.children:
                self.logger.info(f'child_node {child.id} (reward:{child.reward:.4f}, reward: {child.reward:.4f})')
        
    
    def _simulate(self, path: list[MCTSNode]):
        """
        Simulation: simulate the last node in the selected path, stop if reaching terminal or early stop.
        """
        
        if self.log: self.logger.info(f'Simulating:')
        node = path[-1]

        while True:
            if self.early_stop(node):
                node.is_terminal = self.is_terminal_node(node)
                self.increase_threshold(node.reward)
                if self.log: self.logger.info(
                    f"Early Stop: node {node.id}, reward: {node.reward}. \
                    MCTS threshold increases to {self.mcts_threshold}. Stop simulating.\n")
                return
            
            self.increase_threshold(node.reward)
            
            if self.is_terminal_node(node):
                return
            
            if len(node.children) == 0:
                self._expand(node)
            
            rewards = [child.reward for child in node.children]
            if len(rewards) != 0:
                node = node.children[self.simulate_choice(rewards)]
            else:
                node.is_terminal = True
                
            node.visited += 1
            path.append(node)
            
    
    def _back_propagate(self, path: list[MCTSNode]):
        """
        Back Propagation: Update the cumulated rewards of each node in the path.
        """
        
        if self.log: self.logger.info(f'Back propagating:')
        
        rewards = []
        cum_rewards = []
        
        for node in reversed(path):
            rewards.append(node.reward)
            cum_reward = self.cal_cum_reward(rewards[::-1])
            cum_rewards.append(cum_reward)
            node.cum_rewards.append(cum_reward)
            if self.log: self.logger.info(
                f'node {node.id}: depth {node.depth}, \
                new cum_reward: {node.cum_rewards[-1]:.4f}')
            
        cum_rewards = cum_rewards[::-1]
        return cum_rewards
    
    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        """
        MCTS iteration: Selection, Expansion, Simulation, Back-Propagation
        """
        path = self._select(node)
        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            self._simulate(path)
        cum_rewards = self._back_propagate(path)
                    
        return path, cum_rewards

    def search(self, init_state: str):

        self.root = self.world_model.build_root(init_state)
        self.root.reward = self.root.cal_reward()
        self.nodes.append(self.root)
        
        if self.min_threshold == 0:
            self.min_threshold = self.root.reward
            self.increase_threshold(self.root.reward)

        self.trace_in_each_iter = []
        for i in range(self.iteration_num):
            if self.log: self.logger.info(
                f'---------------------  iteration {i} ------------------------')
            
            path, cum_rewards = self.iterate(self.root)
            self.trace_in_each_iter.append(deepcopy(path))
        
        mcts_output = self.prepare_output()
        self.output_to_json(mcts_output=mcts_output)
        return self.trace_in_each_iter, mcts_output

    def __call__(self,
                 init_state: str,
                 **kwargs):
        
        MCTSNode.reset_id()

        iteration_paths, mcts_outputs = self.search(init_state)
        
        return iteration_paths, mcts_outputs


    #################################################################################
    #                        Log and Evaluate Helper Functions                      #
    #################################################################################

    def eval_and_log_node(self, node:MCTSNode, eval=False, log_metric=False, eval_type='test'):
        if node.parent is not None:
            self.logger.info(f'node {node.id}:    \
                             parent: {node.parent.id} | depth: {node.depth} | visited: {node.visited} | expand_times: {node.expand_times}  | terminal: {node.is_terminal} | children: {len(node.children)}')
        else:
            self.logger.info(f'node {node.id}:    parent: N/A | depth: {node.depth} | visited: {node.visited} | expand_times: {node.expand_times} | terminal: {node.is_terminal} | children: {len(node.children)}')
        self.logger.info(f'   reward: {node.reward:.4f} | Q: {node.Q:.4f} | uct: {self._uct(node):.4f} | cum_rewards: {node.cum_rewards}')
        self.logger.info(f'   prompt: {node.prompt}')
        
        if eval:
            if eval_type == 'test':
                test_metric, eval_output = self.world_model.test_prompt(node.prompt)
            else:
                raise ValueError(f'eval_type {eval_type} is not supported.')
            node.test_metric = test_metric
        if log_metric:
            if not isinstance(node.test_metric, tuple):
                self.logger.info(f'   {eval_type} metric: {node.test_metric:.4f}')
            else:
                self.logger.info(f'   {eval_type} metric: {node.test_metric}')
        self.logger.info(f'---------------------')
    
    def log_vars(self):
        self.logger.info('-------------------- MCTS -----------------------')
        ignored_print_vars = ['task', 'log_dir', 'logger', 'trace_in_each_iter', 'root', 'nodes']
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars: continue
            var_value = vars_dict[var_name]
            self.logger.info(f'{var_name} : {var_value}')
        self.logger.info('-------------------------------------------')
       
    def log_path(self, path, eval=False, log_metric=False):
        for node in path:
            self.eval_and_log_node(node=node, eval=eval, log_metric=log_metric)
    
    def log_nodes(self, nodes, eval=False, log_metric=False, eval_type='test'):
        for i, node in enumerate(nodes):
            self.eval_and_log_node(node, eval=eval, log_metric=log_metric, eval_type=eval_type) 
        self.logger.info('\n')
        
    def log_paths(self, paths, eval=False, log_metric=False, eval_type='test'):
        for i, path in enumerate(paths):
            self.logger.info(f'\n----------------  path {i} ------------------') 
            for node in path:
                self.eval_and_log_node(node, eval=eval, log_metric=log_metric, eval_type=eval_type)

    def _sort_helper(self, metric):
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric
    
    def prepare_output(self):
        self.logger.info(f'\n---------------------  all iteration paths ------------------------') 
        self.log_paths(self.trace_in_each_iter)
        self.logger.info(f'\n---------------------  all nodes ------------------------') 
        self.log_nodes(self.nodes)
        
        # prepare output
        paths_nodes = []
        paths_ids = []
        paths_qs = []
        paths_rewards = []
        paths_ucts = []
        
        for i, path in enumerate(self.trace_in_each_iter):
            path_nodes = []
            path_ids = []
            path_qs = []
            path_rewards = []
            path_ucts = []
            
            for node in path:
                path_ids.append(node.id)
                
            for id in path_ids:
                node = self.nodes[id]
                uct = self._uct(node)
                node.uct = uct
                path_ucts.append(uct)
                path_nodes.append(node)
                path_qs.append(node.Q)
                path_rewards.append(node.reward)

            paths_nodes.append(path_nodes)
            paths_ids.append(path_ids)
            paths_qs.append(path_qs)
            paths_rewards.append(path_rewards)
            paths_ucts.append(path_ucts)
            
            self.logger.info(f'path {i}: {path_ids} ')
            self.logger.info(f'mean values:   path_uct: {np.mean(path_ucts):.4f} | path_q: {np.mean(path_qs):.4f} | path_reward: {np.mean(path_rewards):.4f}')
            self.logger.info(f'path_ucts:  {path_ucts}')
            self.logger.info(f'paths_qs :  {paths_qs}')
            self.logger.info(f'path_reward : {path_rewards}')
            self.logger.info('---------------------------')
        
        qs_rank = np.argsort([np.mean(row) for row in paths_qs])[::-1].tolist()
        rewards_rank = np.argsort([np.mean(row) for row in paths_rewards])[::-1].tolist()

        best_q_path = paths_nodes[qs_rank[0]]
        best_reward_path = paths_nodes[rewards_rank[0]]
        top_k_reward_nodes = sorted(self.nodes, key=lambda node: node.reward, reverse=True)[:self.k]

        if len(self.world_model.test_dataloader) != 0:
            self.logger.info(f'\n----------------  test nodes ------------------') 
            test_nodes_set = set(best_q_path + best_reward_path + top_k_reward_nodes)
            for node in self.nodes:
                if node in test_nodes_set:
                    self.eval_and_log_node(node, eval=True, log_metric=True, eval_type='test') 
        
            self.logger.info(f'\n----------------  top_k_reward_nodes------------------') 
            for node in top_k_reward_nodes:
                self.eval_and_log_node(node, eval=False, log_metric=True, eval_type='test')

            self.logger.info(f'\n----------------  best_reward_path------------------') 
            for node in best_reward_path:
                self.eval_and_log_node(node, eval=False, log_metric=True, eval_type='test')
        
        selected_node = sorted(best_reward_path, key=lambda node: self._sort_helper(node.reward), reverse=True)[0]
        
        return dict(
            all_paths = paths_nodes,
            all_nodes = self.nodes,
            best_q_path = best_q_path,
            best_reward_path = best_reward_path,
            top_k_reward_nodes=top_k_reward_nodes,
            best_reward_path_last_node = [best_reward_path[-1]],
            best_reward_path_selected_node = [selected_node],
        )
    
    def output_to_json(self, mcts_output):
        data_to_save = {}
        paths = []
        for path in mcts_output['all_paths']:
            paths.append([node.to_dict() for node in path])
        data_to_save['all_paths'] = paths
        
        for key in mcts_output:
            if key != "all_paths":
                data_to_save[key] = [node.to_dict() for node in mcts_output[key]]
        with open(os.path.join(self.log_dir, 'data.json'), 'w') as f:
            json.dump(data_to_save, f, indent=4)
            