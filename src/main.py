import os
import argparse
from prompt_optim_agent import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def config():
    parser = argparse.ArgumentParser(description='Process prompt search agent arguments')
    parser.add_argument('--task_name', type=str,  help='')  
    parser.add_argument('--search_algo', type=str, default='mcts', help='Prompt search algorithm.')    
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size depending on the memory and model size')
    parser.add_argument('--depth_limit', type=int, default=5)
    parser.add_argument('--train_size', type=int, default=150)
    parser.add_argument('--eval_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_shuffle', type=str2bool, default=True, help='Shuffle training set')
    
    #Search
    parser.add_argument('--init_prompt', type=str, default="Let's answer the question.", help='Initial prompt.')
    parser.add_argument('--iteration_num', type=int, default=5, help='Optimize iteration number.')
    parser.add_argument('--expand_width', type=int, default=3)
    parser.add_argument('--num_new_prompts', type=int, default=1)
    parser.add_argument('--prompt_length_limit', type=int, default=300)
    parser.add_argument('--post_instruction', type=str2bool, default=True)
    # MCTS
    parser.add_argument('--min_depth', type=int, default=3)
    parser.add_argument('--w_exp', type=float, default=2.5)
    parser.add_argument('--init_threshold_increase', type=float, default=0)
    parser.add_argument('--threshold_increase', type=float, default=0)
    parser.add_argument('--min_threshold', type=float, default=0.0)
    
    # BeamSearch
    parser.add_argument('--beam_width', type=int, default=3)
    # APE
    parser.add_argument('--eval_set_size', type=int, default=50)
    parser.add_argument('--filtered_prompt_num', type=int, default=5)
    # World Model
    parser.add_argument('--pred_model', type=str, default='gpt-3.5-turbo', help='')
    parser.add_argument('--optim_model', type=str, default='gpt-4', help='Prompt optimizer.') 
    parser.add_argument('--pred_temperature', type=float, default=0.0)
    parser.add_argument('--optim_temperature', type=float, default=1.0)
    
    parser.add_argument('--log_dir', type=str, default='../logs/', help='Log directory.')
    parser.add_argument('--test_all_nodes', type=str2bool, default=False)
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the data file')
    parser.add_argument('--api_key', type=str, default=None, help='openai api key')
    parser.add_argument('--openai_key_txt_file', type=str, default='../api_keys.txt', help='The txt file that contains openai api key.')
    args = parser.parse_args()

    args = vars(args)
    
    openai_key_config(args['api_key'], args['openai_key_txt_file'])
    
    return args


def main(args):
    agent = BaseAgent(**args)
    states, result_dict = agent.run(init_state=args['init_prompt'], iteration_num=args['iteration_num'])
    return

if __name__ == '__main__':
    args = config()
    main(args)