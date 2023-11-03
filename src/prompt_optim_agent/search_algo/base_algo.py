from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np

State = TypeVar("State")
Action = TypeVar("Action")
Trace = tuple[list[State], list[Action]]

class EvaluateMetrics:
    TOTAL_ACC = 'total_acc'
    CUR_BATCH_ACC = 'cur_batch_acc'
    PARENT_BATCH_ACC = 'parent_batch_acc'
    EVAL_DATA_ACC = 'eval_data_acc'
    SELF_EVAL_SCORE = 'self_eval_score'
    PARENT_EVAL_DATA_ACC = 'parent_eval_data_acc'
    
class PromptState():
    def __init__(self,
                 prompt: str,
                 eval_methods,
                 score: float = -1.0,
                 state_record: dict = None
                ):
        self.prompt = prompt
        self.eval_methods = eval_methods
        self.score = score
        self.weight_dict = {
            EvaluateMetrics.PARENT_BATCH_ACC:-0.5,
            EvaluateMetrics.TOTAL_ACC:1,
            EvaluateMetrics.CUR_BATCH_ACC:1,
            EvaluateMetrics.PARENT_EVAL_DATA_ACC:-0.5,
            EvaluateMetrics.EVAL_DATA_ACC:1,
            EvaluateMetrics.SELF_EVAL_SCORE:1,
            }
        if state_record is None:
            self.state_record = dict(
                                    cur_batch_acc = -1.0,
                                    cur_batch_correct = [],
                                    
                                    eval_data_acc=-1.0, 
                                    eval_data_correct = [],
                                    
                                    parent_batch_acc = -1.0,
                                    
                                    total_acc=-1.0, 
                                    correct = [],
                                    
                                    parent_eval_data_acc = -1.0,
                                    self_eval_score = -1.0,
                                    total_example_num = 0,
                                    ) 
        else:
            self.state_record = state_record
    
    def get_threshold(self):
        threshold = self.state_record[EvaluateMetrics.EVAL_DATA_ACC]
        return threshold
    
    
    def get_score(self):
        total_weight = 0.0
        score = 0.0
        for method in self.eval_methods:
            total_weight += self.weight_dict[method]
            score += self.weight_dict[method] * self.state_record[method]
        score /= total_weight
        self.score = score
        return score
        
    def update_state_record_item(self, item_name, item_value):
        if item_name not in self.state_record.keys():
            raise ValueError(f'{item_name} does not exist in state record.' )
        self.state_record[item_name] = item_value
    
    def update_state_record_with_examples(self, correct, choice): 
        '''
            choice: cur_batch, eval_data
        '''
        if choice == EvaluateMetrics.CUR_BATCH_ACC:
            self.state_record['cur_batch_correct'] = correct
            self.state_record['cur_batch_acc'] =  np.mean(self.state_record['cur_batch_correct'])
        elif choice == EvaluateMetrics.EVAL_DATA_ACC:
            self.state_record['eval_data_correct'] = correct
            self.state_record['eval_data_acc'] =  np.mean(self.state_record['eval_data_correct'])
        else:
            raise ValueError(f'{choice} does not exist in state record.')
        
        self.state_record['correct'].extend(correct)
        self.state_record['total_example_num'] += len(correct)
        self.state_record['total_acc'] =  np.mean(self.state_record['correct'])
        
class SearchAlgo(ABC):
    def __init__(self, 
                 task,
                 world_model, 
                 action_agent,
                 logger=None, 
                 seed=0, 
                 print_log=True,
                 test_every_step=True,
                 depth_limit = None,
                 ) -> None:
        self.task = task
        self.world_model = world_model
        self.action_agent = action_agent
        self.states = []
        self.logger = logger
        self.print_log = print_log if logger is not None else False
        self.seed = seed
        self.test_every_step = test_every_step
        self.depth_limit = depth_limit

    @abstractmethod
    def search(self):
        pass
    
    def get_states(self):
        return self.states
        
    def process_all_correct_batch(self):
        self.logger.info(f'\n-----------------------------------------------------')
        self.logger.info('all correct: skip updating cur_prompt')
        self.logger.info(f'\n-----------------------------------------------------\n')

