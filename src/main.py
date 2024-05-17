import argparse
from prompt_optim_agent import *
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def config():
    parser = argparse.ArgumentParser(description='Process prompt search agent arguments')

    parser.add_argument('--config_dir', type=str, default='./defualt_config.yaml')
   
    args = parser.parse_args()

    args = load_config(args.config_dir)
    return args

def validate_config(config):
    # Basic settings
    assert config['task_name'] is not None, "task_name must be specified"
    assert config['search_algo'] in ['mcts', 'beam_search'], "search_algo must be 'mcts' or 'beam_search'"
    assert isinstance(config['print_log'], bool), "print_log must be a boolean"
    assert config['log_dir'] is not None, "log_dir must be specified"
    assert config['init_prompt'] is not None, "init_prompt must be specified"

    # Task setting
    assert isinstance(config['task_setting']['train_size'], (int, type(None))), "train_size must be an integer or None"
    assert isinstance(config['task_setting']['eval_size'], int), "eval_size must be an integer"
    assert isinstance(config['task_setting']['test_size'], int), "test_size must be an integer"
    assert isinstance(config['task_setting']['seed'], int), "seed must be an integer"
    assert isinstance(config['task_setting']['post_instruction'], bool), "post_instruction must be a boolean"

    # Base model setting
    assert config['base_model_setting']['model_type'] in ['openai', 'palm', 'hf_text2text', 'hf_textgeneration', 'ct_model'], "base_model.model_type must be one of 'openai', 'palm', 'hf_text2text', 'hf_textgeneration', 'ct_model'"
    assert config['base_model_setting']['model_name'] is not None, "base_model.model_name must be specified"
    assert isinstance(config['base_model_setting']['temperature'], float), "base_model.temperature must be a float"
    assert config['base_model_setting']['device'] in [None, 'cuda', 'cpu'] or config['base_model_setting']['device'].startswith('cuda:'), "base_model.device must be None, 'cuda', 'cpu', or 'cuda:x'"
    if config['base_model_setting']['model_type'] in ['openai', 'palm'] and config['base_model_setting']['api_key'] is None:
        raise ValueError("Please set base model's api key")

    # Optim model setting
    assert config['optim_model_setting']['model_type'] in ['openai', 'palm', 'hf_text2text', 'hf_textgeneration', 'ct_model'], "optim_model.model_type must be one of 'openai', 'palm', 'hf_text2text', 'hf_textgeneration', 'ct_model'"
    assert config['optim_model_setting']['model_name'] is not None, "optim_model.model_name must be specified"
    assert isinstance(config['optim_model_setting']['temperature'], float), "optim_model.temperature must be a float"
    assert config['optim_model_setting']['device'] in [None, 'cuda', 'cpu'] or config['optim_model_setting']['device'].startswith('cuda:'), "optim_model.device must be None, 'cuda', 'cpu', or 'cuda:x'"
    if config['optim_model_setting']['model_type'] in ['openai', 'palm'] and config['optim_model_setting']['api_key'] is None:
        raise ValueError("Please set optim model's api key")

    # Search config
    assert isinstance(config['search_setting']['iteration_num'], int), "search.iteration_num must be an integer"
    assert isinstance(config['search_setting']['expand_width'], int), "search.expand_width must be an integer"
    assert isinstance(config['search_setting']['depth_limit'], int), "search.depth_limit must be an integer"
    # MCTS setting
    assert isinstance(config['search_setting']['min_depth'], int), "min_depth must be an integer"
    assert isinstance(config['search_setting']['w_exp'], float), "w_exp must be a float"
    # Beam search setting
    assert isinstance(config['search_setting']['beam_width'], int), "beam_width must be an integer"

    # World model setting
    assert isinstance(config['world_model_setting']['train_shuffle'], bool), "world_model.train_shuffle must be a boolean"
    assert isinstance(config['world_model_setting']['num_new_prompts'], int), "world_model.num_new_prompts must be an integer"
    assert isinstance(config['world_model_setting']['train_batch_size'], int), "world_model.train_batch_size must be an integer"

def main(args):
    agent = BaseAgent(**args)
    agent.run()
    return

if __name__ == '__main__':
    args = config()
    validate_config(args)
    print(args)
    main(args)