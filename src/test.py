from prompt_optim_agent.test_helper import eval
import argparse

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
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--eval_prompt', type=str, default="Let's solve the problem.")
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--post_instruction', type=str2bool, default=False)
    
    parser.add_argument('--train_size', type=int, default=0)
    parser.add_argument('--eval_size', type=int, default=0)
    parser.add_argument('--test_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--pred_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0)
    
    parser.add_argument('--log_dir', type=str, default='logs/', help='Log directory.')
    parser.add_argument('--log_examples', type=str2bool, default=True)
    parser.add_argument('--data_dir', type=str, default=None)
    
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key or PaLM2 API key')
    args = parser.parse_args()

    args = vars(args)
    
    return args

def main(args):
    eval(**args)
    
if __name__ == '__main__':
    args = config()
    main(args)