# Prompt-Optim-with-RAP
This project aims to automatically optimize prompt using RAP

## Installation
```bash
git clone https://github.com/XinyuanWangCS/Prompt-Optim-with-RAP.git
cd Prompt-Optim-with-RAP
conda create -n prompt_searcher
conda activate prompt_searcher
pip install -r requirement.txt
```


## Run MCTS agent
```bash
python src/main.py --search_algo mcts --pred_model gpt-3.5-turbo --log_dir logs/ --post_instruction False  --train_shuffle True --batch_size 5 --expand_width 3 --num_new_prompts 1 --iteration_num 12 --depth_limit 8 --seed 42 --task_name ?  --data_dir ? --train_size  --eval_size  --test_size  --test_all_nodes False --init_prompt "" --api_key ""
```

## Run MCTS Logical Deduction:
```bash
python src/main.py --task_name bigbench --search_algo mcts --data_dir datasets/logical_deduction_5_objects.json --pred_model gpt-3.5-turbo --openai_key_txt_file api_keys.txt --log_dir logs/ --post_instruction True --init_prompt "Let's think step by step. At the end show the answer option bracketed with <answer> and </answer>." --train_shuffle False --batch_size 5 --expand_width 3 --num_new_prompts 1 --train_size 50 --eval_size 50 --val_size 0 --test_size 100 --iteration_num 4 
```

## Run GSM8K:
```bash
python src/main.py --task_name gsm8k --search_algo mcts --pred_model gpt-3.5-turbo --openai_key_txt_file api_keys.txt --log_dir logs/ --post_instruction True --init_prompt "Let's think step by step. At the end show the answer option bracketed with <answer> and </answer>." --train_shuffle True --batch_size 5 --expand_width 1 --num_new_prompts 3  --train_size 50 --val_size 0 --test_size 100 --iteration_num 6
```
## Run test.py to test prompt
```bash
python src/test.py --task_name bigbench --exp_name "20230913_1609-bigbench_geometric_shapes-algo_mcts-batch_2-train_150-eval_5-test_5" --eval_prompt "your prompt" --api_key "your_api" --data_dir "bigbench json path"
```

## TODO:
1. RAP
2. Huggingface model
