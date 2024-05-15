import os
import time
from datetime import  timedelta
from .utils import get_pacific_time, create_logger, parse_model_args
from tasks import get_task
from .world_model import get_world_model
from .search_algo import get_search_algo
from .language_model import get_language_model

class BaseAgent():
    def __init__(
        self,
        task_name: str,
        search_algo: str,
        
        base_model_type:str,
        optim_model_type:str,
        
        batch_size: int,
        train_size: int,
        eval_size: int, 
        test_size: int,
        seed:int, 
        train_shuffle: bool,
        post_instruction: bool,
        log_dir: str,
        data_dir: str,
        
        expand_width: int,
        num_new_prompts: int,
        min_depth:int,
        depth_limit: int,
        iteration_num: int, 
        w_exp:float, 
        print_log: bool,
        **kwargs) -> None:
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
        self.train_size = train_size
        self.eval_size = eval_size
        self.test_size = test_size
        self.post_instruction = post_instruction
        self.seed = seed

        self.log_dir = log_dir
        self.data_dir = data_dir
        
        self.task = get_task(task_name)(train_size=train_size, 
                                        eval_size=eval_size,
                                        test_size=test_size, 
                                        seed=seed,
                                        post_instruction=post_instruction,
                                        data_dir = data_dir)

        if data_dir is not None and task_name == "bigbench":
            task_name = task_name + "_" + data_dir.split('/')[-1].split('.')[-2]
        
        exp_name = f'{get_pacific_time().strftime("%Y%m%d_%H%M%S")}-{task_name}-algo_{search_algo}-batch_{batch_size}-train_{train_size}'
        
        self.log_dir = os.path.join(log_dir, exp_name)
        self.logger = create_logger(self.log_dir, f'{exp_name}', log_mode='train')
        self.logger.info(exp_name)
        self.log_vars()
        self. print_log = print_log
        
        self.logger.info("*****************")
        self.logger.info(kwargs)
        base_args, optim_args = parse_model_args(kwargs=kwargs)
        self.logger.info(base_args)
        self.logger.info(optim_args)
        self.base_model = get_language_model(base_model_type)(**base_args)
        self.optim_model = get_language_model(optim_model_type)(**optim_args) 
        
        self.world_model = get_world_model(search_algo)(
            task=self.task, 
            logger=self.logger, 
            base_model=self.base_model,
            optim_model=self.optim_model, 
            num_new_prompts = num_new_prompts,
            train_shuffle = train_shuffle,
            train_batch_size = batch_size,
            )
        
        self.search_algo = get_search_algo(search_algo)(
            task=self.task, 
            world_model=self.world_model, 
            logger=self.logger,
            log_dir = self.log_dir,
            min_depth=min_depth,
            depth_limit=depth_limit,
            expand_width=expand_width,
            iteration_num = iteration_num,
            w_exp=w_exp,
            )
    
    
    
    def run(self, init_state, iteration_num):
        """
        Start searching from initial prompt
        """
        self.logger.info(f'init_prompt: {init_state}')
        start_time = time.time()
        
        states, result_dict = self.search_algo.search(init_state=init_state, iteration_num=iteration_num)
        end_time = time.time()
        exe_time = str(timedelta(seconds=end_time-start_time)).split('.')[0]
        self.logger.info(f'\nDone! Iteration: {iteration_num} Excution time: {exe_time}')
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

    