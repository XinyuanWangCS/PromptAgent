import os
import time
from datetime import  timedelta
from .utils import get_pacific_time, create_logger
from tasks import get_task
from .world_model import get_world_model
from .search_algo import get_search_algo
from .language_model import get_language_model

class BaseAgent():
    def __init__(
        self,
        task_name: str,
        search_algo: str,
        print_log: bool,
        log_dir: str,
        
        init_prompt: str,
        
        task_setting: dict,
        base_model_setting: dict,
        optim_model_setting: dict,
        search_setting: dict,
        world_model_setting: dict
        ) -> None:
        """
        BaseAgent: set up task, logger, search algorithm, world model
        
        :param task_name: the names of .py files in the tasks folder
        :param search_algo: "mcts" or "beam_search"
        :param base_model: the model that answers the
        :param base_temperature: temperature of base_model
        :param optim_model: the optimizer model that gives error feedback and generate new prompts
        :param optim_temperature: temperature of optim_model
        
        :param batch_size: batch size of each optimization step
        :param train_size: training set size
        :param eval_size: the set reserved for reward calculation
        :param test_size: testing set size
        :param train_shuffle: whether to shuffle the training set
        :param seed: the seed for train/test split
        :param post_instruction: whether the optimized prompt is behind the task question or in front of the question 
            (True: question + prompt, False: prompt + question)
            
        :param log_dir: logger directory
        :param data_dir: data file directory (if the data is stored in a file)
        :param expand_width: number of optimization step in each expansion operation
        :param num_new_prompts: number of new prompts sampled in each optimization step
        
        :param min_depth: minimum depth of MCTS (early stop is applied only when depth is deeper than min_depth)
        :param depth_limit: maximum depth of MCTS
        :param iteration_num: iteration number of MCTS
        :param w_exp: the weight between exploitation and exploration, default 2.5

        """
        self.task_name = task_name
        self.search_algo = search_algo
        self. print_log = print_log
        self.log_dir = log_dir
        self.init_prompt =init_prompt
        
        self.task_setting = task_setting
        self.base_model_setting = base_model_setting
        self.optim_model_setting = optim_model_setting
        self.search_setting = search_setting
        self.world_model_setting = world_model_setting
        
        self.task = get_task(task_name)(**task_setting)

        if task_setting["data_dir"] is not None and task_name == "bigbench":
            task_name = task_name + "_" + task_setting["data_dir"].split('/')[-1].split('.')[-2]
        
        exp_name = f'{get_pacific_time().strftime("%Y%m%d_%H%M%S")}-{task_name}-algo_{search_algo}'
        
        self.log_dir = os.path.join(log_dir, exp_name)
        self.logger = create_logger(self.log_dir, f'{exp_name}', log_mode='train')
        self.logger.info(exp_name)
        self.log_vars()
        
        
        self.base_model = get_language_model(
            base_model_setting["model_type"])(**base_model_setting)
        
        self.optim_model = get_language_model(
            optim_model_setting["model_type"])(**optim_model_setting) 
        
        self.world_model = get_world_model(search_algo)(
            task=self.task, 
            logger=self.logger, 
            base_model=self.base_model,
            optim_model=self.optim_model, 
            **world_model_setting
            )
        
        self.search_algo = get_search_algo(search_algo)(
            task=self.task, 
            world_model=self.world_model, 
            logger=self.logger,
            log_dir = self.log_dir,
            **self.search_setting
            )
    
    def run(self):
        """
        Start searching from initial prompt
        """
        self.logger.info(f'init_prompt: {self.init_prompt}')
        start_time = time.time()
        
        states, result_dict = self.search_algo.search(init_state=self.init_prompt)
        end_time = time.time()
        exe_time = str(timedelta(seconds=end_time-start_time)).split('.')[0]
        self.logger.info(f'\nDone!Excution time: {exe_time}')
        return states, result_dict
    
    def log_vars(self):
        """
        Log arguments
        """
        ignored_print_vars = ['logger']
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars: continue
            var_value = vars_dict[var_name]
            self.logger.info(f'{var_name} : {var_value}')

    