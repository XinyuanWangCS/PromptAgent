from prompt_optim_agent.test_helper import *
from prompt_optim_agent.utils import *

import math
import itertools
from tqdm import trange
from copy import deepcopy
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Generic, Optional

from .base_algo import SearchAlgo, State, Action, PromptState
from .action_agent import DataAgent

class MCTSNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, 
                 state: Optional[State], 
                 action: Optional[Action], 
                 parent: "Optional[MCTSNode]" = None,
                 instant_reward: float = 0., 
                 is_terminal: bool = False, 
                 threshold = 0.
                 ):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param instant_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        self.id = next(MCTSNode.id_iter)

        self.cum_rewards: list[float] = []
        self.reward = self.instant_reward = instant_reward

        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[MCTSNode]]' = []
        self.metric = 0
        
        self.threshold = threshold
        self.visited = 0
        self.expand_times = 0
        self.is_best = False
        self.uct = 0
        
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
    
    def calc_q(self, x):
        return np.mean(x)
    
    def to_dict(self):
        if self.parent is None:
            p_id = -1
        else:
            p_id = self.parent.id
        
        return {
            'id': self.id,
            'depth':self.depth,
            'parent':p_id,
            'visited':self.visited,
            'reward': self.reward,
            'q':self.Q,
            'uct':self.uct,
            'prompt': self.state.prompt,
            'eval_metric':self.state.get_threshold(),
            'test_metric':self.metric
        }
        
    @property
    def Q(self) -> float:
        if len(self.cum_rewards) == 0:
            return self.instant_reward
        else:
            return self.calc_q(self.cum_rewards)

class MCTS(SearchAlgo, Generic[State, Action]):

    def __init__(
        self, 
        task,
        world_model, 
        
        #mcts arguments
        min_threshold:float = 0.,
        threshold_increase: float = 0.03,
        init_threshold_increase:float = 0.01,
        expand_width = 2,
        w_exp: float = 2.5,
        depth_limit: int = 8,
        min_depth: int = 2,
        n_iters: int = 12,
        
        # log
        logger=None, 
        log_dir = None,
        test_all_nodes = False,
        **kwargs) -> None:
        
        self.task = task
        self.world_model = world_model
        self.logger = logger
        self.log_dir = log_dir
        
        self.expand_width = expand_width
        self.depth_limit = depth_limit
        self.w_exp = w_exp
        self.min_depth = min_depth
        self.n_iters = n_iters
        
        self.min_threshold = min_threshold
        self.init_threshold_increase = init_threshold_increase
        self.threshold_increase = threshold_increase
        self.threshold = min_threshold
        
        # output
        self.k = 3
        self.test_all_nodes = test_all_nodes
        
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.nodes:list[MCTSNode] = [] 
        
        self.log_vars()
    
    def log_vars(self):
        self.logger.info('-------------------- MCTS -----------------------')
        ignored_print_vars = ['task', 'log_dir', 'logger', 'trace_in_each_iter', 'root', 'nodes']
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars: continue
            var_value = vars_dict[var_name]
            self.logger.info(f'{var_name} : {var_value}')
        self.logger.info('-------------------------------------------')
        
    def simulate_choice(self, x):
        return np.argmax(x)
    
    def increase_threshold(self, threshold, threshold_increase=None):
        if threshold_increase is None:
            self.threshold = threshold + self.threshold_increase
        else:
            self.threshold = threshold + threshold_increase
    
    def cal_cum_reward(self, rewards, decay=0.8):
        return np.sum(rewards)
    
    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.depth >= self.depth_limit
    
    def _is_terminal_with_threshold(self, node: MCTSNode):
        #return False
        return node.threshold > self.threshold
    
    def _is_terminal_with_min_threshold(self, node: MCTSNode):
        if node.parent is None:
            min_threshold = self.min_threshold
        else:
            min_threshold = (self.min_threshold + node.parent.threshold) / 2
        return node.threshold < min_threshold and node.depth > self.min_depth
    
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
        '''
        select:
            find a path with highest uct reward
        '''
        self.logger.info('--------------  select  ---------------')
        path = []
        while True:
            path.append(node)
            node.visited += 1
            if len(node.children) == 0 or self.is_terminal_node(node):
                self.logger.info(f'Node {node.id} is leaf node. Finish selecting.\n')
                return path

            node = self._uct_select(node)
            self.logger.info(f'select node {node.id}: depth {node.depth}, reward: {node.reward:.4f} utc: {self._uct(node=node)}')
    
    
    def _expand(self, node: MCTSNode):
        '''
            expand: 
                add children to the node
                each child has: instant_reward, parent, action, is_terminal
                assign reward: mean of children's instant_reward
                
            If node is_terminal: return
            actions: sample different batches of data
            for each action, get num_new_prompts new prompt_states
            for each prompt state
                score the state
                build one new node as one child: state=prompt_state, instant_reward=score, action=batch   
        '''
        
        self.logger.info(f'------------------  expand node {node.id} ---------------------')
        self.logger.info(f'node {node.id}: depth {node.depth}, instant_reward: {node.instant_reward:.4f}, reward: {node.reward:.4f}')
        self._log_one_record(node.state)
        #self.world_model.update_eval_dataloader()
        if self.is_terminal_node(node):
            node.is_terminal = True
            self.logger.info(f"Node {node.id} (is_terminal:{node.is_terminal}, depth: {node.depth}, threshold:{node.threshold:.4f}) is terminal. Stop expanding.")
            self.logger.info('------------------------------')
        
        node.expand_times += 1
        
        i = 0
        while i < self.expand_width:
            batch = self.world_model.get_train_batch()
            self.logger.info(f'------- expanding batch {i} ---------')
            children, gradient_descent_output = self.world_model.step(node, batch)
            if gradient_descent_output['acc'] == -1:
                self.logger.info('All correct, sample new batch.')
                continue
            i += 1
            
            for child_node in children:
                self.world_model.evaluate_child_node(node=child_node) #also update threshold
                child_node.instant_reward = child_node.state.get_threshold()
                child_node.reward = child_node.state.get_score()
                child_node.is_terminal = self.is_terminal_node(child_node)
                self.logger.info(f'****************')
                self.logger.info(f'MCTS min thershold: {self.min_threshold:.4f}')
                self.logger.info(f'MCTS threshold: {self.threshold:.4f}')
                self.logger.info(f'New node {child_node.id}: depth {child_node.depth}, threshold: {child_node.threshold:.4f}')
                self.logger.info(f'****************')
        
            self.nodes.extend(children)
            node.children.extend(children)
        
        self.logger.info(f'node {node.id} (reward:{node.reward:.4f}, instant_reward: {node.instant_reward:.4f}) has {len(node.children)} children:')
        for child in node.children:
             self.logger.info(f'child_node {child.id} (reward:{child.reward:.4f}, instant_reward: {child.instant_reward:.4f})')
        
    
    def _simulate(self, path: list[MCTSNode]):
        '''
        simulate: 
            start from path's last node
            
            find the node with largest instant_reward as node
            append this node to th path
            continue until is_terminal
        '''
        self.logger.info(f'-------------  simulate ---------------')
        node = path[-1]

        while True:
            if self._is_terminal_with_threshold(node):
                self.increase_threshold(node.threshold)
                node.is_best = True
                node.is_terminal = self.is_terminal_node(node)
                self.logger.info(f"Node {node.id}(threshold: {node.threshold}) is leaf node. MCTS threshold increases to {self.threshold}. Stop simulating.\n")
                return
                
            if self.is_terminal_node(node):
                self.logger.info(f"Node {node.id} is terminal node. Stop simulating.\n")
                return
            
            if len(node.children) == 0:
                self._expand(node)
                
            for child_node in node.children:
                child_node.is_terminal = self.is_terminal_node(child_node)
                self.logger.info(f'new node {child_node.id}: depth {child_node.depth}, instant_reward: {child_node.instant_reward:.4f}, reward: {child_node.reward:.4f}, is_terminal: {child_node.is_terminal}')
                self._log_one_record(child_node.state)
                self.logger.info(f'-----------------')
            
            children_ids = [child.id for child in node.children]
            rewards = [child.reward for child in node.children]
            node = node.children[self.simulate_choice(rewards)]
            
            node.visited += 1
            path.append(node)
            
            self.logger.info(f'children ids:          {children_ids}')
            self.logger.info(f'children rewards: {rewards}')
            self.logger.info(f'choose: node {node.id}')
            self.logger.info('--------------')
            
    
    def _back_propagate(self, path: list[MCTSNode]):
        '''
            Each node's cum_rewards add the list[reward from this node to the last node in the path]
        '''
        self.logger.info(f'---------------------  back_propagate ------------------------')
        rewards = []
        cum_rewards = []
        for node in reversed(path):
            rewards.append(node.reward)
            self.logger.info(f'node {node.id}: depth {node.depth}, instant_reward: {node.instant_reward:.4f}, reward: {node.reward:.4f}')
            self.logger.info(f'cum_rewards    : {node.cum_rewards}')
            cum_reward = self.cal_cum_reward(rewards[::-1])
            cum_rewards.append(cum_reward)
            node.cum_rewards.append(cum_reward)
            self.logger.info(f'new cum_rewards: {node.cum_rewards}')
        cum_rewards = cum_rewards[::-1]
        self.logger.info(f'------------------')
        self.logger.info(f'back_propagate cum_rewards: {cum_rewards}\n')
        return cum_rewards
    
    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        path = self._select(node)
        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            self._simulate(path)
        cum_rewards = self._back_propagate(path)
                    
        return path, cum_rewards

    def search(self, init_state: str, iteration_num: int =None):
        if iteration_num is not None:
            self.n_iters = iteration_num
        
        self.root = self.world_model.build_root(init_state)
        self.root.instant_reward = self.root.state.get_threshold()
        self.root.reward = self.root.state.get_score()
        
        if self.min_threshold == 0:
            self.min_threshold = self.root.threshold

        self.increase_threshold(self.root.threshold, threshold_increase=self.init_threshold_increase)
        self.nodes.append(self.root)
        
        self.trace_in_each_iter = []
            
        for i in range(self.n_iters):
            self.logger.info(f'---------------------  iteration {i} ------------------------')
            
            path, cum_rewards = self.iterate(self.root)
            
            self.logger.info(f'----------  iteration {i} result path -----------')
            self.logger.info(f'cum_rewards: {cum_rewards}')
            self.log_path(path, eval=False, log_metric=False)
            self.trace_in_each_iter.append(deepcopy(path))
        
        self.log_untested_output()
        mcts_output = self.prepare_output()
        self.output_to_json(mcts_output=mcts_output)
        self.draw(mcts_output['paths_to_draw'])
        self.draw(mcts_output['all_paths'], plot_names=['all_paths_eval'])
        return self.trace_in_each_iter, mcts_output
    
    def log_untested_output(self):
        # log path result
        self.logger.info(f'\n---------------------  all iteration paths ------------------------') 
        for i, path in enumerate(self.trace_in_each_iter):
            self.logger.info(f'\n----------------  path {i} ------------------') 
            for node in path:
                self.eval_and_log_node(node, eval=False, log_metric=False, eval_type='test')
        
        self.logger.info(f'\n---------------------  all nodes ------------------------') 
        for i, node in enumerate(self.nodes):
            self.eval_and_log_node(node, eval=False, log_metric=False, eval_type='test') 
        self.logger.info('\n')

    def __call__(self,
                 init_prompt: str,
                 n_iters: int = None,
                 **kwargs):
        MCTSNode.reset_id()

        iteration_paths, mcts_outputs = self.search(init_prompt, n_iters)
        
        return iteration_paths, mcts_outputs

    def eval_and_log_node(self, node:MCTSNode, eval=True, log_metric=True, eval_type='test'):
        if node.parent is not None:
            self.logger.info(f'node {node.id}:    parent: {node.parent.id} | depth: {node.depth} | visited: {node.visited} | expand_times: {node.expand_times} | best: {node.is_best} | terminal: {node.is_terminal} | children: {len(node.children)}')
        else:
            self.logger.info(f'node {node.id}:    parent: N/A | depth: {node.depth} | visited: {node.visited} | expand_times: {node.expand_times} | best: {node.is_best} | terminal: {node.is_terminal} | children: {len(node.children)}')
        self.logger.info(f'   Q: {node.Q:.4f} | reward: {node.reward:.4f} | uct: {self._uct(node):.4f} | instant_reward: {node.instant_reward:.4f} | cum_rewards: {node.cum_rewards}')
        self._log_one_record(node.state)
        if eval:
            if eval_type == 'test':
                metric, eval_output = self.world_model.test_prompt(node.state.prompt)
            else:
                raise ValueError(f'eval_type {eval_type} is not supported.')
            node.metric = metric
        if log_metric:
            if not isinstance(node.metric, tuple):
                self.logger.info(f'   {eval_type} metric: {node.metric:.4f}')
            else:
                self.logger.info(f'   {eval_type} metric: {node.metric}')
        self.logger.info(f'---------------------')
            
    def log_path(self, path, eval=False, log_metric=False):
        for node in path:
            self.eval_and_log_node(node=node, eval=eval, log_metric=log_metric)
    
    def _log_one_record(self, prompt_state:PromptState, index=None, blank="   "):
        prompt   = prompt_state.prompt
        score    = prompt_state.get_score()
        total_acc = prompt_state.state_record['total_acc']
        #parent_acc = prompt_state.state_record['parent_batch_acc']
        cur_batch_acc   = prompt_state.state_record['cur_batch_acc']
        eval_data_acc = prompt_state.state_record['eval_data_acc']
        #self_eval_score = prompt_state.state_record['self_eval_score']
        total_example_num = prompt_state.state_record['total_example_num']
        if index is not None:
            self.logger.info(f"{blank}prompt {index}: {prompt}")
        else:
            self.logger.info(f"{blank}prompt: {prompt}")
        self.logger.info(f'{blank}score={score:.3f}       | total_acc={total_acc:.3f} | total_example_num={total_example_num}')
        self.logger.info(f'{blank}eval_data_acc={eval_data_acc:.3f} | cur_batch_acc={cur_batch_acc:.3f}')
    
    
    def _sort_helper(self, metric):
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric
    
    def prepare_output(self):
        # test and log nodes
        self.logger.info(f'\n--------------------- Test all nodes ------------------------') 
        if self.test_all_nodes:
            for i, node in enumerate(self.nodes):
                self.eval_and_log_node(node, eval=True, log_metric=True, eval_type='test') 
        else:
            for i, node in enumerate(self.nodes):
                self.eval_and_log_node(node, eval=False, log_metric=True, eval_type='test') 
        # prepare output
        paths_nodes = []
        paths_ids = []
        paths_qs = []
        paths_rewards = []
        paths_ucts = []
        paths_test_metrics = []
        
        for i, path in enumerate(self.trace_in_each_iter):
            path_nodes = []
            path_ids = []
            path_qs = []
            path_rewards = []
            path_ucts = []
            path_test_metrics = []
            
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
                path_test_metrics.append(node.metric)

            paths_nodes.append(path_nodes)
            paths_ids.append(path_ids)
            paths_qs.append(path_qs)
            paths_rewards.append(path_rewards)
            paths_ucts.append(path_ucts)
            paths_test_metrics.append(path_test_metrics)
            
            self.logger.info(f'path {i}: {path_ids} ')
            self.logger.info(f'mean values:   path_uct: {np.mean(path_ucts):.4f} | path_q: {np.mean(path_qs):.4f} | path_reward: {np.mean(path_rewards):.4f}')
            self.logger.info(f'test_metrics : {path_test_metrics}')
            self.logger.info(f'path_ucts:  {path_ucts}')
            self.logger.info(f'paths_qs :  {paths_qs}')
            self.logger.info(f'path_reward : {path_rewards}')
            self.logger.info('---------------------------')
        
        qs_rank = np.argsort([np.mean(row) for row in paths_qs])[::-1].tolist()
        rewards_rank = np.argsort([np.mean(row) for row in paths_rewards])[::-1].tolist()

        best_q_path = paths_nodes[qs_rank[0]]
        best_reward_path = paths_nodes[rewards_rank[0]]
        
        top_k_reward_nodes = sorted(self.nodes, key=lambda node: node.reward, reverse=True)[:self.k]
        top1_node = top_k_reward_nodes[0]
        
        top1_node_path = []
        for path in paths_nodes:
            if top1_node in path:
                top1_node_path = path
                if top1_node == path[-1]:
                    break

        
        if not self.test_all_nodes:
            test_nodes_set = set(best_q_path + best_reward_path + top_k_reward_nodes)
            for node in self.nodes:
                if node in test_nodes_set:
                    self.eval_and_log_node(node, eval=True, log_metric=True, eval_type='test')
        # log path rank   
        
        top_k_test_nodes = sorted(self.nodes, key=lambda node: self._sort_helper(node.metric), reverse=True)[:self.k]
        selected_node = sorted(best_reward_path, key=lambda node: self._sort_helper(node.reward), reverse=True)[0]
        best_nodes = []
        best_nodes.append(selected_node)
        best_nodes.append(best_reward_path[-1])
        best_nodes.append(top_k_reward_nodes[0])
        best_nodes.append(top_k_test_nodes[0])

                
        self.logger.info(f'\n----------------  top_k_reward_nodes------------------') 
        for node in top_k_reward_nodes:
            self.eval_and_log_node(node, eval=False, log_metric=True, eval_type='test')

        self.logger.info(f'\n----------------  best_reward_path------------------') 
        for node in best_reward_path:
            self.eval_and_log_node(node, eval=False, log_metric=True, eval_type='test')
            
        return dict(
            all_paths = paths_nodes,
            all_nodes = self.nodes,
            top_k_reward_nodes = top_k_reward_nodes,
            top_k_test_nodes=top_k_test_nodes,
            best_q_path = best_q_path,
            best_reward_path = best_reward_path,
            best_nodes=best_nodes,
            paths_to_draw = [best_reward_path]
        )
    
    def output_to_json(self, mcts_output):
        data_to_save = {}
        paths = []
        for path in mcts_output['all_paths']:
            paths.append([node.to_dict() for node in path])
        data_to_save['all_paths'] = paths
        
        paths = []
        for path in mcts_output['paths_to_draw']:
            paths.append([node.to_dict() for node in path])
        data_to_save['paths_to_draw'] = paths
        
        for key in mcts_output:
            if key != "all_paths" and key != 'paths_to_draw':
                data_to_save[key] = [node.to_dict() for node in mcts_output[key]]
        with open(os.path.join(self.log_dir, 'data.json'), 'w') as f:
            json.dump(data_to_save, f, indent=4)
            
    def draw(self, paths, plot_names=['test', 'reward', 'uct', 'eval']): #'all_paths_eval'
        offset = np.linspace(-0.1, 0.1, len(paths))
        
        if 'test' in plot_names:
            fig, ax = plt.subplots()
            colors = plt.cm.jet(np.linspace(0, 1, len(paths)))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            for idx, path in enumerate(paths):
                depths = np.array([node.depth for node in path]) + offset[idx]
                
                metrics = [node.metric[0] if isinstance(node.metric, tuple) else node.metric for node in path]
                ids = [node.id for node in path]

                ax.plot(depths, metrics, color=colors[idx], marker='o')  
                for d, r, id_ in zip(depths, metrics, ids):
                    ax.annotate(str(id_), (d, r))

            ax.set_title("Test Metric")
            ax.set_xlabel("Depth")
            ax.set_ylabel("Test")
            plt.savefig(os.path.join(self.log_dir, 'test_metric.png'), bbox_inches='tight')
        
        if 'reward' in plot_names:
            fig, ax = plt.subplots()
            colors = plt.cm.jet(np.linspace(0, 1, len(paths)))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            for idx, path in enumerate(paths):
                depths = np.array([node.depth for node in path]) + offset[idx]
                rewards = [node.reward for node in path]
                ids = [node.id for node in path]

                ax.plot(depths, rewards, color=colors[idx], marker='o')  
                for d, r, id_ in zip(depths, rewards, ids):
                    ax.annotate(str(id_), (d, r))

            ax.set_title("Rewards")
            ax.set_xlabel("Depth")
            ax.set_ylabel("Rewards")
            plt.savefig(os.path.join(self.log_dir, 'rewards.png'), bbox_inches='tight')
        
        if 'uct' in plot_names:
            fig, ax = plt.subplots()
            colors = plt.cm.jet(np.linspace(0, 1, len(paths)))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            for idx, path in enumerate(paths):
                depths = np.array([node.depth for node in path]) + offset[idx]
                ucts = [node.uct for node in path]
                ids = [node.id for node in path]

                ax.plot(depths, ucts, color=colors[idx], marker='o')  
                for d, r, id_ in zip(depths, ucts, ids):
                    ax.annotate(str(id_), (d, r))

            ax.set_title("UCT")
            ax.set_xlabel("Depth")
            ax.set_ylabel("UCT")
            plt.savefig(os.path.join(self.log_dir, 'ucts.png'), bbox_inches='tight')
        
        if 'eval' in plot_names:
            fig, ax = plt.subplots()
            colors = plt.cm.jet(np.linspace(0, 1, len(paths)))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            for idx, path in enumerate(paths):
                depths = np.array([node.depth for node in path]) + offset[idx]
                instant_rewards = [node.instant_reward for node in path]
                ids = [node.id for node in path]

                ax.plot(depths, instant_rewards, color=colors[idx], marker='o')  
                for d, r, id_ in zip(depths, instant_rewards, ids):
                    ax.annotate(str(id_), (d, r))

            ax.set_title("Eval Metric")
            ax.set_xlabel("Depth")
            ax.set_ylabel("Metric")
            plt.savefig(os.path.join(self.log_dir, 'eval_metric.png'), bbox_inches='tight')
            
        if 'all_paths_eval' in plot_names:
            fig, ax = plt.subplots()
            colors = plt.cm.jet(np.linspace(0, 1, len(paths)))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            for idx, path in enumerate(paths):
                depths = np.array([node.depth for node in path]) + offset[idx]
                instant_rewards = [node.instant_reward for node in path]
                ids = [node.id for node in path]

                ax.plot(depths, instant_rewards, color=colors[idx], marker='o')  
                for d, r, id_ in zip(depths, instant_rewards, ids):
                    ax.annotate(str(id_), (d, r))

            ax.set_title("Eval Metric All Paths")
            ax.set_xlabel("Depth")
            ax.set_ylabel("Metric")
            plt.savefig(os.path.join(self.log_dir, 'eval_metric_all_paths.png'), bbox_inches='tight')