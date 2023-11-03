from prompt_optim_agent.test_helper import eval_prompts_in_xls
import argparse
import os

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
    parser = argparse.ArgumentParser(description='test prompt')
    parser.add_argument('--task_name', type=str)   
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--index', type=int)
    parser.add_argument('--eval_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Log directory.')
    parser.add_argument('--excel_dir', type=str, default=None)
    parser.add_argument('--api_key', type=str, default=None, help='openai api key')
    parser.add_argument('--openai_key_txt_file', type=str, default='../api_keys.txt', help='The txt file that contains openai api key.')
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--post_instruction', type=str2bool, default=False)
    parser.add_argument('--log_examples', type=str2bool, default=True)
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    args = vars(args)
    
    return args

def main(args):
    print(os.getcwd())
    print(args['log_dir'], os.path.exists(args['log_dir']))
    print(args['openai_key_txt_file'], os.path.exists(args['openai_key_txt_file']))
    eval_prompts_in_xls(**args)
    
    
if __name__ == '__main__':
    args = config()
    main(args)