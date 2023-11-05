import time
import datetime
from .utils import *
from tasks import *
from .world_model import get_world_model
from .search_algo import get_search_algo
from .search_algo.base_algo import EvaluateMetrics
from datetime import datetime, timedelta
import pytz

class BaseAgent():
    def __init__(self,
                 task_name: str,
                 search_algo: str,
                 pred_model: str,
                 optim_model: str,
                 pred_temperature: float,
                 optim_temperature: float,
                 batch_size: int,
                 train_size: int,
                 eval_size: int, 
                 test_size: int,
                 seed:int, 
                 train_shuffle:bool,
                 post_instruction,
                 log_dir: str,
                 data_dir: str,
                 test_all_nodes: bool,
                 
                 expand_width: int,
                 num_new_prompts: int,
                 
                 prompt_length_limit: int,
                 min_depth:int,
                 depth_limit: int,
                 
                 iteration_num: int, 
                 threshold_increase: float,
                 init_threshold_increase:float,
                 min_threshold:float,
                 w_exp:float, 
                 
                 eval_methods = None, 
                 # Beam Search
                 beam_width:int = None,
                 **kwargs) -> None:
        
        self.task_name = task_name
        self.train_size = train_size
        self.eval_size = eval_size
        self.test_size = test_size
        self.post_instruction = post_instruction
        if eval_methods is None:
            self.eval_methods = [EvaluateMetrics.EVAL_DATA_ACC,]
        else:
            self.eval_methods = eval_methods
        self.seed = seed

        self.log_dir = log_dir
        self.data_dir = data_dir
        
        self.task = get_task(task_name)(train_size=train_size, 
                                        eval_size=eval_size,
                                        test_size=test_size, 
                                        seed=seed,
                                        post_instruction=post_instruction,
                                        data_dir = data_dir)
        test_set_size = self.task.get_dataset_size('test')
        real_eval_size = self.task.get_dataset_size('eval')
        if data_dir is not None and task_name == "bigbench":
            task_name = task_name + "_" + data_dir.split('/')[-1].split('.')[-2]
        current_time = datetime.now()
        pacific = pytz.timezone('US/Pacific')
        pacific_time = current_time.astimezone(pacific)
        exp_name = f'{pacific_time.strftime("%Y%m%d_%H%M")}-{task_name}-algo_{search_algo}-batch_{batch_size}-train_{train_size}-eval_{real_eval_size}-test_{test_set_size}'
        
        self.log_dir = os.path.join(log_dir, exp_name)
        self.logger = create_logger(self.log_dir, f'{exp_name}', log_mode='train')
        self.logger.info(exp_name)
        self.log_vars()

        self.world_model = get_world_model(search_algo)(
            task=self.task, 
            logger=self.logger, 
            # Model
            pred_model=pred_model,
            pred_temperature= pred_temperature,
            optim_model=optim_model, 
            optim_temperature=optim_temperature,
            # Optim hyperparameter
            num_new_prompts = num_new_prompts,
            prompt_length_limit=prompt_length_limit,
            
            train_shuffle = train_shuffle,
            train_batch_size = batch_size,
            #MCTS
            eval_methods=self.eval_methods,
            )
        
        self.search_algo = get_search_algo(search_algo)(
            task=self.task, 
            world_model=self.world_model, 
            # Data and log
            test_all_nodes=test_all_nodes,
            logger=self.logger,
            log_dir = self.log_dir,
            # Search
            min_depth=min_depth,
            depth_limit=depth_limit,
            expand_width=expand_width,
            # MCTS
            n_iters = iteration_num,
            threshold_increase = threshold_increase,
            init_threshold_increase = init_threshold_increase,
            min_threshold=min_threshold,
            w_exp=w_exp,
            #BeamSearch
            beam_width=beam_width,
            )

        
    def run(self, init_state, iteration_num):
        self.logger.info(f'init_prompt: {init_state}')
        start_time = time.time()
        states, result_dict = self.search_algo.search(init_state=init_state, iteration_num=iteration_num)
        end_time = time.time()
        exe_time = str(timedelta(seconds=end_time-start_time)).split('.')[0]
        self.logger.info(f'\nDone! Iteration: {iteration_num} Excution time: {exe_time}')
        return states, result_dict
    
    def log_vars(self):
        ignored_print_vars = ['logger']
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars: continue
            var_value = vars_dict[var_name]
            self.logger.info(f'{var_name} : {var_value}')
    
    @staticmethod   
    def pacific_time_formatter(record):
        pacific = pytz.timezone('US/Pacific')
        dt = datetime.fromtimestamp(record.created, pacific)
        return dt.strftime('%Y-%m-%d %H:%M:%S')